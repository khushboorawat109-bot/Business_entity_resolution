"""
Memory-bounded streaming helpers for the multi-GB source files.

Reading these with a single pd.read_csv(...) call materializes the whole
file as Python/pandas string objects, which can be 3-5x the raw file size --
too much for a constrained-memory environment. These helpers process the
file in chunks and never hold more than O(chunksize + n) rows in memory.
"""
import numpy as np
import pandas as pd


def stream_sample(path, n, chunksize=200000, random_state=42, keep_frac_buffer=3):
    """Approx-uniform random sample of n rows from a large TSV, memory-bounded."""
    rng = np.random.RandomState(random_state)
    reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize)
    buf = []
    buf_len = 0
    for chunk in reader:
        buf.append(chunk)
        buf_len += len(chunk)
        if buf_len > n * keep_frac_buffer:
            combined = pd.concat(buf, ignore_index=True)
            keep_n = min(n * keep_frac_buffer, len(combined))
            combined = combined.sample(keep_n, random_state=random_state)
            buf = [combined]
            buf_len = len(combined)
    combined = pd.concat(buf, ignore_index=True) if buf else pd.DataFrame()
    if len(combined) > n:
        combined = combined.sample(n, random_state=random_state)
    return combined.reset_index(drop=True)


def stream_filter_ids(path, id_col, needed_ids, chunksize=200000):
    """Stream a large TSV, keep only rows whose id_col is in needed_ids (a set)."""
    if not needed_ids:
        # still need columns -> read header only
        header = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=0)
        return header
    reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize)
    parts = []
    remaining = set(needed_ids)
    for chunk in reader:
        if not remaining:
            break
        hit = chunk[chunk[id_col].isin(remaining)]
        if len(hit):
            parts.append(hit)
            remaining -= set(hit[id_col].tolist())
    if parts:
        return pd.concat(parts, ignore_index=True)
    header = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=0)
    return header


def stream_head(path, n, chunksize=200000):
    """First n rows only (cheap, deterministic) -- useful for smoke tests."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=n)
