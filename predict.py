"""
End-to-end inference over the test set, processed ONE COUNTRY AT A TIME so
peak memory is bounded by the single largest country's pool (not the sum of
India+US+France). Requires partition_by_country.py to have been run first
(produces dataset/test_partitioned/{source1,source2,source3}__{slug}.tsv).

For each country partition:
  1. Load that country's S2+S3 pool only, build a SingleCountryBlocker
     (near-linear-time inverted-index build).
  2. Stream that country's source1 partition in batches, query the index,
     buffer pairs, periodically score with the model in bulk and append
     results to the (shared, across all countries) output files.
  3. Discard this country's pool/index before moving to the next country.

--resume skips Source-1 ids already written, so a run can be safely
restarted/continued (e.g. across several invocations) without redoing work
or duplicating output rows.
"""
import argparse
import glob
import os
import pickle
import sys
import time

import joblib
import pandas as pd

sys.path.insert(0, "/home/claude/ber/src")
from normalize import normalize_name, normalize_address, normalize_country
from country_token_blocker import SingleCountryBlocker
from features import build_feature_matrix


def load_done_ids(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            eid = line.split("\t", 1)[0].strip()
            if eid:
                done.add(eid)
    return done


def append_rows(path, header, rows):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header + "\n")
        for s1_id, ids in rows:
            f.write(f"{s1_id}\t{','.join(ids)}\n")


def find_country_slugs(part_dir):
    slugs = set()
    for p in glob.glob(f"{part_dir}/source1__*.tsv"):
        slug = os.path.basename(p)[len("source1__"):-len(".tsv")]
        slugs.add(slug)
    return sorted(slugs)


def load_pool_for_country(part_dir, slug):
    frames = []
    for prefix in ("source2", "source3"):
        p = f"{part_dir}/{prefix}__{slug}.tsv"
        if os.path.exists(p):
            frames.append(pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False))
    if not frames:
        return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part-dir", default="/home/claude/ber/dataset/test_partitioned")
    ap.add_argument("--model", default="/home/claude/ber/output/model.joblib")
    ap.add_argument("--out-dir", default="/home/claude/ber/output")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--batch-read-size", type=int, default=5000)
    ap.add_argument("--flush-every", type=int, default=5000)
    ap.add_argument("--max-seconds", type=int, default=0, help="0 = run to completion")
    ap.add_argument("--only-country", default="", help="slug, to process just one country this run")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-cache", action="store_true",
                     help="skip disk caching of the built index (safer for very large pools "
                          "where pickling itself risks OOM; rebuilds every invocation instead)")
    args = ap.parse_args()

    t_start = time.time()
    match_path = f"{args.out_dir}/matching_results.tsv"
    cand_path = f"{args.out_dir}/candidate_pairs.tsv"

    if not args.resume:
        for p in (match_path, cand_path):
            if os.path.exists(p):
                os.remove(p)
    done_ids = load_done_ids(match_path) if args.resume else set()
    print(f"Already-completed Source-1 ids on disk: {len(done_ids)}", flush=True)

    print("Loading model...", flush=True)
    bundle = joblib.load(args.model)
    clf, threshold = bundle["model"], bundle["threshold"]
    print(f"Threshold = {threshold:.3f}", flush=True)

    slugs = find_country_slugs(args.part_dir)
    if args.only_country:
        slugs = [s for s in slugs if s == args.only_country]
    print(f"Country partitions to process: {slugs}", flush=True)

    n_written_total = 0
    stop_all = False

    for slug in slugs:
        if stop_all:
            break
        t_country = time.time()
        print(f"\n=== Country: {slug} ===", flush=True)
        pool = load_pool_for_country(args.part_dir, slug)
        print(f"  pool size: {len(pool)}  ({time.time() - t_country:.1f}s)", flush=True)
        cache_path = f"{args.out_dir}/_blocker_cache_{slug}.pkl"
        if len(pool) == 0:
            blocker = None
        elif (not args.no_cache) and os.path.exists(cache_path):
            print("  loading cached blocker...", flush=True)
            with open(cache_path, "rb") as f:
                blocker = pickle.load(f)
            print(f"  blocker loaded from cache  ({time.time() - t_country:.1f}s)", flush=True)
        else:
            ids = pool["entity_id"].tolist()
            names = [normalize_name(x) for x in pool["business_name"]]
            addrs = [normalize_address(x) for x in pool["business_address"]]
            del pool  # free the raw frame BEFORE building the index (peak-memory critical
                      # for the largest partition), rather than after
            blocker = SingleCountryBlocker(ids, names, addrs)
            print(f"  blocker built  ({time.time() - t_country:.1f}s)", flush=True)
            if not args.no_cache:
                try:
                    tmp_path = cache_path + ".tmp"
                    with open(tmp_path, "wb") as f:
                        pickle.dump(blocker, f, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(tmp_path, cache_path)
                    print(f"  blocker cached to disk  ({time.time() - t_country:.1f}s)", flush=True)
                except MemoryError:
                    print("  WARNING: not enough memory to cache this blocker to disk; "
                          "continuing without cache.", flush=True)
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
        if 'pool' in dir():
            del pool

        s1_path = f"{args.part_dir}/source1__{slug}.tsv"
        if not os.path.exists(s1_path):
            continue

        match_buf, cand_buf = [], []
        pending_recs, pending_empty = [], []

        def flush_pending():
            nonlocal pending_recs, pending_empty, match_buf, cand_buf
            if pending_recs:
                pairs_df = pd.DataFrame(pending_recs, columns=[
                    "source1_entity_id", "candidate_entity_id",
                    "norm_name_1", "norm_address_1", "norm_country_1",
                    "tok_name_score", "tok_blend_score",
                    "norm_name_2", "norm_address_2", "norm_country_2",
                ])
                X = build_feature_matrix(pairs_df)
                pairs_df["prob"] = clf.predict_proba(X)[:, 1]
                cand_grouped = pairs_df.groupby("source1_entity_id")["candidate_entity_id"].apply(list)
                match_grouped = pairs_df[pairs_df["prob"] >= threshold].groupby(
                    "source1_entity_id")["candidate_entity_id"].apply(list)
                for s1 in pairs_df["source1_entity_id"].unique():
                    cand_buf.append((s1, cand_grouped.get(s1, [])))
                    match_buf.append((s1, list(match_grouped[s1]) if s1 in match_grouped.index else []))
            for eid in pending_empty:
                match_buf.append((eid, []))
                cand_buf.append((eid, []))
            pending_recs, pending_empty = [], []

        cnorm = normalize_country(slug.replace("_", " "))  # only used for logging
        n_proc = 0
        for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False,
                                  chunksize=args.batch_read_size):
            for row in chunk.itertuples(index=False):
                if row.entity_id in done_ids:
                    continue
                nn = normalize_name(row.business_name)
                na = normalize_address(row.business_address)
                if blocker is None:
                    pending_empty.append(row.entity_id)
                    continue
                cands = blocker.query_one(nn, na, k=args.k)
                if not cands:
                    pending_empty.append(row.entity_id)
                    continue
                for pid, ns, bs in cands:
                    pending_recs.append((
                        row.entity_id, pid, nn, na, slug,
                        ns, bs,
                        blocker.pool_name[pid], blocker.pool_addr[pid], slug,
                    ))
            n_proc += len(chunk)
            n_pending_s1 = len(pending_empty) + len({r[0] for r in pending_recs})
            if n_pending_s1 >= args.flush_every:
                flush_pending()
                append_rows(match_path, "source1_entity_id\tmatched_entity_ids", match_buf)
                append_rows(cand_path, "source1_entity_id\tcandidate_entity_ids", cand_buf)
                n_written_total += len(match_buf)
                match_buf, cand_buf = [], []
                elapsed = time.time() - t_start
                print(f"  [{slug}] processed {n_proc} rows, {n_written_total} total written "
                      f"({elapsed:.1f}s total elapsed)", flush=True)
            if args.max_seconds and (time.time() - t_start) > args.max_seconds:
                print("Hit --max-seconds budget, stopping (resumable with --resume).", flush=True)
                stop_all = True
                break
        if pending_empty or pending_recs:
            flush_pending()
        if match_buf:
            append_rows(match_path, "source1_entity_id\tmatched_entity_ids", match_buf)
            append_rows(cand_path, "source1_entity_id\tcandidate_entity_ids", cand_buf)
            n_written_total += len(match_buf)

        del blocker
        print(f"  [{slug}] country done in {time.time() - t_country:.1f}s", flush=True)

    print(f"\nDone. Wrote {n_written_total} new rows this run. "
          f"Total elapsed {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
