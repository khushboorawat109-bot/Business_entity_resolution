"""
One-time streaming partition of test_source1/2/3.tsv into per-country files.

This bounds all later memory use to ONE country's data at a time (the
largest, India, rather than the sum of India+US+France), and does the
partitioning itself in a single low-memory pass -- plain line-based I/O,
no pandas, no big lists held in memory.
"""
import csv
import os
import re
import sys

sys.path.insert(0, "/home/claude/ber/src")
from normalize import normalize_country


def _slug(country_norm):
    s = re.sub(r"[^a-z0-9]+", "_", country_norm.strip().lower()) or "unknown"
    return s


def partition_file(src_path, out_dir, prefix):
    """Streams src_path (a source1/2/3.tsv), writes one file per normalized
    country: {out_dir}/{prefix}__{slug}.tsv, preserving the original header
    and rows (unmodified) in each. Returns dict slug -> country_norm."""
    os.makedirs(out_dir, exist_ok=True)
    writers = {}
    files = {}
    slug_to_country = {}
    with open(src_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        country_idx = header.index("country")
        for row in reader:
            if len(row) <= country_idx:
                continue
            cnorm = normalize_country(row[country_idx])
            slug = _slug(cnorm)
            if slug not in writers:
                path = f"{out_dir}/{prefix}__{slug}.tsv"
                fh = open(path, "w", encoding="utf-8", newline="")
                w = csv.writer(fh, delimiter="\t")
                w.writerow(header)
                writers[slug] = w
                files[slug] = fh
                slug_to_country[slug] = cnorm
            writers[slug].writerow(row)
    for fh in files.values():
        fh.close()
    return slug_to_country


def main():
    test_dir = "/home/claude/ber/dataset/test"
    out_dir = "/home/claude/ber/dataset/test_partitioned"
    for prefix, fname in [("source1", "test_source1.tsv"),
                           ("source2", "test_source2.tsv"),
                           ("source3", "test_source3.tsv")]:
        print(f"Partitioning {fname}...", flush=True)
        mapping = partition_file(f"{test_dir}/{fname}", out_dir, prefix)
        print(f"  -> {mapping}", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
