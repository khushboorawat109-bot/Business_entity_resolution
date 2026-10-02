"""
End-to-end training:
  1. Load train_source1/2/3 + ground truth (optionally sub-sampled via --sample).
  2. Normalize text fields.
  3. Group-split Source-1 entities into train/val (no entity leaks across split).
  4. Block train split's Source-1 rows against the S2+S3 pool (report recall
     ceiling vs ground truth -- this is the metric that tells you if blocking
     is good enough before the classifier even matters).
  5. Build labeled pairs (positive = in ground truth & recovered by blocking,
     negative = blocked-but-not-a-true-match => hard negatives) and features.
  6. Train sklearn HistGradientBoostingClassifier (BSD-licensed, no external
     deps) on train-split pairs.
  7. Score val-split pairs, sweep the decision threshold to maximize macro
     F_0.5 (the competition metric, precision-weighted), report the result.
  8. Persist the fitted model + chosen threshold with joblib.
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

sys.path.insert(0, "/home/claude/ber/src")
from normalize import add_normalized_columns
from country_token_blocker import SingleCountryBlocker
from features import build_feature_matrix, FEATURE_NAMES
from evaluate import macro_f05
from io_utils import parse_match_list
from sample_utils import stream_sample, stream_filter_ids


def build_pool(df2, df3):
    pool = pd.concat([df2, df3], ignore_index=True)
    return pool


def make_pairs_frame(cand_dict, df1_norm, pool_norm, gt_map):
    """
    cand_dict: source1_id -> [(pool_id, name_cos, blend_cos), ...]
    Returns a DataFrame of pairs with normalized fields + tfidf scores + label.
    """
    df1_idx = df1_norm.set_index("entity_id")
    pool_idx = pool_norm.set_index("entity_id")

    recs = []
    for s1_id, cands in cand_dict.items():
        true_set = set(gt_map.get(s1_id, []))
        r1 = df1_idx.loc[s1_id]
        for pool_id, name_cos, blend_cos in cands:
            r2 = pool_idx.loc[pool_id]
            recs.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": pool_id,
                "norm_name_1": r1["norm_name"], "norm_address_1": r1["norm_address"],
                "norm_country_1": r1["norm_country"],
                "norm_name_2": r2["norm_name"], "norm_address_2": r2["norm_address"],
                "norm_country_2": r2["norm_country"],
                "tfidf_name_cos": name_cos, "tfidf_blend_cos": blend_cos,
                "label": int(pool_id in true_set),
            })
    return pd.DataFrame.from_records(recs)


def tune_threshold_for_f05(val_pairs_df, probs, gt_map, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.025)
    val_pairs_df = val_pairs_df.copy()
    val_pairs_df["prob"] = probs
    best_t, best_score = 0.5, -1.0
    all_s1 = val_pairs_df["source1_entity_id"].unique()
    gt_subset = {s1: gt_map.get(s1, []) for s1 in all_s1}
    for t in thresholds:
        preds = {}
        kept = val_pairs_df[val_pairs_df["prob"] >= t]
        grouped = kept.groupby("source1_entity_id")["candidate_entity_id"].apply(list)
        for s1 in all_s1:
            preds[s1] = grouped.get(s1, [])
        score = macro_f05(preds, gt_subset)
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="/home/claude/ber/dataset/train")
    ap.add_argument("--sample", type=int, default=3000,
                     help="Number of Source-1 training entities to sample for this run "
                          "(use a large number / omit blocking sample below for a full run).")
    ap.add_argument("--pool-sample", type=int, default=150000,
                     help="Cap on S2+S3 pool size for dev runs; set very high for full run.")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--min-sim", type=float, default=0.12)
    ap.add_argument("--model-out", default="/home/claude/ber/output/model.joblib")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.RandomState(args.seed)

    print("Streaming Source-1 sample...", flush=True)
    df1 = stream_sample(f"{args.train_dir}/train_source1.tsv", args.sample,
                         random_state=args.seed)
    print(f"Source-1 sample: {len(df1)} entities", flush=True)

    print("Streaming matching ground-truth rows...", flush=True)
    gt = stream_filter_ids(f"{args.train_dir}/train_ground_truth.tsv",
                            "source1_entity_id", set(df1["entity_id"]))
    gt_map = {row.source1_entity_id: parse_match_list(row.matched_entity_ids)
              for row in gt.itertuples(index=False)}
    # entities with no ground-truth row at all (shouldn't happen, but be safe) -> singleton
    for eid in df1["entity_id"]:
        gt_map.setdefault(eid, [])

    needed_ids = set()
    for s1 in df1["entity_id"]:
        needed_ids.update(gt_map.get(s1, []))
    print(f"True match ids required in pool: {len(needed_ids)}", flush=True)

    print("Streaming forced (true-match) pool rows from source2/3...", flush=True)
    forced2 = stream_filter_ids(f"{args.train_dir}/train_source2.tsv", "entity_id", needed_ids)
    forced3 = stream_filter_ids(f"{args.train_dir}/train_source3.tsv", "entity_id", needed_ids)
    forced = pd.concat([forced2, forced3], ignore_index=True)
    fill_n = max(0, args.pool_sample - len(forced))
    print(f"Forced pool rows: {len(forced)}  -> random-filling {fill_n} more", flush=True)

    fill2_n = fill_n // 2
    fill3_n = fill_n - fill2_n
    fill2 = stream_sample(f"{args.train_dir}/train_source2.tsv", fill2_n, random_state=args.seed)
    fill3 = stream_sample(f"{args.train_dir}/train_source3.tsv", fill3_n, random_state=args.seed)
    fill2 = fill2[~fill2["entity_id"].isin(needed_ids)]
    fill3 = fill3[~fill3["entity_id"].isin(needed_ids)]

    pool = pd.concat([forced, fill2, fill3], ignore_index=True).drop_duplicates(subset="entity_id")
    print(f"Pool sample: {len(pool)} candidate records", flush=True)

    print("Normalizing text...", flush=True)
    df1 = add_normalized_columns(df1)
    pool = add_normalized_columns(pool)

    # Group split by Source-1 entity: 80% train / 20% val
    s1_ids = np.array(df1["entity_id"].tolist())
    perm = rng.permutation(len(s1_ids))
    s1_ids = s1_ids[perm]
    n_val = max(1, int(0.2 * len(s1_ids)))
    val_ids = set(s1_ids[:n_val])
    train_ids = set(s1_ids[n_val:])
    df1_train = df1[df1["entity_id"].isin(train_ids)].reset_index(drop=True)
    df1_val = df1[df1["entity_id"].isin(val_ids)].reset_index(drop=True)
    print(f"Train S1: {len(df1_train)}  Val S1: {len(df1_val)}", flush=True)

    def build_blockers(pool_df):
        blockers = {}
        for country, g in pool_df.groupby("norm_country"):
            blockers[country] = SingleCountryBlocker(
                g["entity_id"].tolist(), g["norm_name"].tolist(), g["norm_address"].tolist())
        return blockers

    print("Building per-country blockers...", flush=True)
    blockers = build_blockers(pool)

    print("Blocking (train split)...", flush=True)
    cand_train = {}
    for row in df1_train.itertuples(index=False):
        b = blockers.get(row.norm_country)
        cand_train[row.entity_id] = b.query_one(row.norm_name, row.norm_address, k=args.k) if b else []
    print("Blocking (val split)...", flush=True)
    cand_val = {}
    for row in df1_val.itertuples(index=False):
        b = blockers.get(row.norm_country)
        cand_val[row.entity_id] = b.query_one(row.norm_name, row.norm_address, k=args.k) if b else []

    # --- Recall ceiling check: what fraction of true matches did blocking find?
    def recall_ceiling(cand_dict, ids):
        total_true, found = 0, 0
        for s1 in ids:
            true_set = set(gt_map.get(s1, []))
            if not true_set:
                continue
            found_set = {c[0] for c in cand_dict.get(s1, [])}
            total_true += len(true_set)
            found += len(true_set & found_set)
        return found / total_true if total_true else 1.0

    rc_train = recall_ceiling(cand_train, df1_train["entity_id"])
    rc_val = recall_ceiling(cand_val, df1_val["entity_id"])
    print(f"Blocking recall ceiling -- train: {rc_train:.4f}  val: {rc_val:.4f}", flush=True)

    print("Building labeled pairs...", flush=True)
    train_pairs = make_pairs_frame(cand_train, df1, pool, gt_map)
    val_pairs = make_pairs_frame(cand_val, df1, pool, gt_map)
    print(f"Train pairs: {len(train_pairs)} (pos={train_pairs['label'].sum()})  "
          f"Val pairs: {len(val_pairs)} (pos={val_pairs['label'].sum()})", flush=True)

    print("Building features...", flush=True)
    X_train = build_feature_matrix(train_pairs)
    y_train = train_pairs["label"].values
    X_val = build_feature_matrix(val_pairs)

    print("Training HistGradientBoostingClassifier...", flush=True)
    clf = HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.08,
        l2_regularization=1.0, random_state=args.seed,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    print("Scoring validation + tuning threshold for macro F_0.5...", flush=True)
    val_probs = clf.predict_proba(X_val)[:, 1]
    gt_val_subset = {s1: gt_map.get(s1, []) for s1 in df1_val["entity_id"]}
    best_t, best_score = tune_threshold_for_f05(val_pairs, val_probs, gt_val_subset)
    print(f"Best threshold: {best_t:.3f}  Val macro F_0.5: {best_score:.4f}", flush=True)

    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_ if hasattr(clf, "feature_importances_") else [0]*len(FEATURE_NAMES)),
                          key=lambda x: -x[1]) if hasattr(clf, "feature_importances_") else None

    joblib.dump({"model": clf, "threshold": best_t, "feature_names": FEATURE_NAMES}, args.model_out)
    print(f"Saved model to {args.model_out}", flush=True)

    metrics = {
        "n_train_s1": len(df1_train), "n_val_s1": len(df1_val),
        "blocking_recall_ceiling_train": rc_train, "blocking_recall_ceiling_val": rc_val,
        "n_train_pairs": len(train_pairs), "n_val_pairs": len(val_pairs),
        "best_threshold": float(best_t), "val_macro_f05": float(best_score),
        "elapsed_sec": time.time() - t0,
    }
    with open("/home/claude/ber/output/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
