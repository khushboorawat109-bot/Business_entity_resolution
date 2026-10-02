#!/usr/bin/env python3
"""
Stdlib-only validator for matching_results.tsv and candidate_pairs.tsv against
the rules in the problem statement. Mirrors the competition's own
utils/validate_submission.py so issues are caught locally before submitting.

Usage:
    python3 validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
"""
import argparse
import csv
import sys


def load_ids(path, prefix):
    ids = set()
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row:
                continue
            eid = row[0]
            if eid.startswith(prefix):
                ids.add(eid)
    return ids


def load_id_list_tsv(path):
    """Returns list of (key, [ids...]) preserving row order, plus raw row count."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if header != [header[0], header[1]] or len(header) != 2:
            pass  # header name checked by caller
        for row in reader:
            if len(row) == 1:
                key, ids_field = row[0], ""
            elif len(row) == 2:
                key, ids_field = row
            else:
                key, ids_field = row[0], "\t".join(row[1:])
            ids = [x for x in ids_field.split(",") if x] if ids_field else []
            rows.append((key, ids))
    return rows, header


def validate(matching_path, candidate_path, test_dir):
    issues = []

    s1_ids = load_ids(f"{test_dir}/test_source1.tsv", "S1-")
    s2_ids = load_ids(f"{test_dir}/test_source2.tsv", "S2-")
    s3_ids = load_ids(f"{test_dir}/test_source3.tsv", "S3-")
    valid_cand_ids = s2_ids | s3_ids

    match_rows, match_header = load_id_list_tsv(matching_path)
    cand_rows, cand_header = load_id_list_tsv(candidate_path)

    if match_header[:2] != ["source1_entity_id", "matched_entity_ids"]:
        issues.append(f"matching_results.tsv header is {match_header}, "
                       f"expected ['source1_entity_id', 'matched_entity_ids']")
    if cand_header[:2] != ["source1_entity_id", "candidate_entity_ids"]:
        issues.append(f"candidate_pairs.tsv header is {cand_header}, "
                       f"expected ['source1_entity_id', 'candidate_entity_ids']")

    # --- matching_results.tsv checks
    seen_s1_match = set()
    match_map = {}
    for key, ids in match_rows:
        if key in seen_s1_match:
            issues.append(f"matching_results.tsv: duplicate source1_entity_id row: {key}")
        seen_s1_match.add(key)
        match_map[key] = ids
        if not key.startswith("S1-") or key not in s1_ids:
            issues.append(f"matching_results.tsv: {key} is not a valid test Source-1 id")
        if len(ids) != len(set(ids)):
            issues.append(f"matching_results.tsv: duplicate ids within row for {key}")
        for cid in ids:
            if cid.startswith("S1-"):
                issues.append(f"matching_results.tsv: {key} lists a Source-1 self-match {cid}")
            elif cid not in valid_cand_ids:
                issues.append(f"matching_results.tsv: {key} lists unknown id {cid} "
                               f"(not in test Source-2/3)")

    missing = s1_ids - seen_s1_match
    if missing:
        issues.append(f"matching_results.tsv: {len(missing)} test Source-1 entities missing "
                       f"(e.g. {list(sorted(missing))[:5]})")

    # --- candidate_pairs.tsv checks
    seen_s1_cand = set()
    cand_map = {}
    for key, ids in cand_rows:
        if key in seen_s1_cand:
            issues.append(f"candidate_pairs.tsv: duplicate source1_entity_id row: {key}")
        seen_s1_cand.add(key)
        cand_map[key] = set(ids)
        if not key.startswith("S1-") or key not in s1_ids:
            issues.append(f"candidate_pairs.tsv: {key} is not a valid test Source-1 id")
        if len(ids) != len(set(ids)):
            issues.append(f"candidate_pairs.tsv: duplicate ids within row for {key}")
        for cid in ids:
            if cid not in valid_cand_ids:
                issues.append(f"candidate_pairs.tsv: {key} lists unknown id {cid} "
                               f"(not in test Source-2/3)")

    missing_c = s1_ids - seen_s1_cand
    if missing_c:
        issues.append(f"candidate_pairs.tsv: {len(missing_c)} test Source-1 entities missing "
                       f"(e.g. {list(sorted(missing_c))[:5]})")

    # --- matches must be subset of candidates
    bad_subset = 0
    for key, ids in match_map.items():
        cset = cand_map.get(key, set())
        extra = [i for i in ids if i not in cset]
        if extra:
            bad_subset += 1
            if bad_subset <= 5:
                issues.append(f"WARNING: {key} has matched ids not present in its own "
                               f"candidate list: {extra[:3]} (pipeline bug signal)")
    if bad_subset > 5:
        issues.append(f"WARNING: ... plus {bad_subset - 5} more rows with match-not-in-candidate issues")

    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args()

    issues = validate(args.matching, args.candidate, args.test_dir)
    if not issues:
        print("PASS")
        sys.exit(0)
    else:
        print(f"{len(issues)} issue(s) found:")
        for i, msg in enumerate(issues, 1):
            print(f"{i}. {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
