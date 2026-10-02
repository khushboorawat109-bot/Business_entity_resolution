"""IO helpers: reading source/ground-truth TSVs, writing submission TSVs."""
import pandas as pd


def read_source(path, nrows=None):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=nrows)
    return df


def read_ground_truth(path, nrows=None):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=nrows)
    return df


def parse_match_list(s):
    if not s:
        return []
    return [x for x in s.split(",") if x]


def write_id_list_tsv(path, id_col, list_col, rows):
    """
    rows: list of (source1_id, [candidate_id, ...])
    Writes with an exact single-tab header and no quoting, comma-joined lists.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{id_col}\t{list_col}\n")
        for s1_id, ids in rows:
            f.write(f"{s1_id}\t{','.join(ids)}\n")
