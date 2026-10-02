"""
Pairwise similarity feature engineering for candidate (source1, candidate) pairs.

No third-party fuzzy-matching library is available offline, so string
similarity is built from Python's stdlib `difflib` plus simple set-based
token / character n-gram Jaccard, which are fast enough to vectorize over
batches with list comprehensions and require no compiled deps.
"""
import difflib
import numpy as np


def _char_ngrams(s: str, n: int = 3):
    s = s.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _token_set(s: str):
    return set(s.split()) if s else set()


def _seq_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).quick_ratio()


def pair_features(name_a, addr_a, name_b, addr_b, country_a, country_b,
                   tok_name_score=None, tok_blend_score=None):
    """
    Compute a fixed-length feature vector for one (record_a, record_b) pair.
    name_*/addr_* must already be normalized strings; country_* normalized too.
    tfidf_*_cos are optional precomputed cosine similarities from the
    blocking stage (reused here to avoid recomputation).
    """
    na_tok, nb_tok = _token_set(name_a), _token_set(name_b)
    aa_tok, ab_tok = _token_set(addr_a), _token_set(addr_b)

    name_tok_jac = _jaccard(na_tok, nb_tok)
    addr_tok_jac = _jaccard(aa_tok, ab_tok)

    name_char_jac = _jaccard(_char_ngrams(name_a), _char_ngrams(name_b))
    addr_char_jac = _jaccard(_char_ngrams(addr_a), _char_ngrams(addr_b))

    name_seq = _seq_ratio(name_a, name_b)
    addr_seq = _seq_ratio(addr_a, addr_b)

    name_len_diff = abs(len(name_a) - len(name_b)) / (max(len(name_a), len(name_b), 1))
    addr_len_diff = abs(len(addr_a) - len(addr_b)) / (max(len(addr_a), len(addr_b), 1))

    name_substr = float(bool(name_a) and bool(name_b) and (name_a in name_b or name_b in name_a))
    first_tok_match = float(bool(na_tok) and bool(nb_tok) and
                             (name_a.split()[0] == name_b.split()[0]))

    country_match = float(country_a == country_b and country_a != "")

    feats = [
        name_tok_jac, addr_tok_jac,
        name_char_jac, addr_char_jac,
        name_seq, addr_seq,
        name_len_diff, addr_len_diff,
        name_substr, first_tok_match,
        country_match,
        tok_name_score if tok_name_score is not None else 0.0,
        tok_blend_score if tok_blend_score is not None else 0.0,
    ]
    return feats


FEATURE_NAMES = [
    "name_tok_jaccard", "addr_tok_jaccard",
    "name_char_jaccard", "addr_char_jaccard",
    "name_seq_ratio", "addr_seq_ratio",
    "name_len_diff", "addr_len_diff",
    "name_substr", "first_tok_match",
    "country_match",
    "tok_name_score", "tok_blend_score",
]


def build_feature_matrix(pairs_df):
    """
    pairs_df must have columns: norm_name_1, norm_address_1, norm_name_2,
    norm_address_2, norm_country_1, norm_country_2, and optionally
    tok_name_score, tok_blend_score (else filled with 0).
    Returns a numpy 2D float array, shape (n_pairs, n_features).
    """
    has_name_cos = "tok_name_score" in pairs_df.columns
    has_blend_cos = "tok_blend_score" in pairs_df.columns
    rows = []
    for row in pairs_df.itertuples(index=False):
        rows.append(pair_features(
            row.norm_name_1, row.norm_address_1,
            row.norm_name_2, row.norm_address_2,
            row.norm_country_1, row.norm_country_2,
            tok_name_score=getattr(row, "tok_name_score") if has_name_cos else None,
            tok_blend_score=getattr(row, "tok_blend_score") if has_blend_cos else None,
        ))
    return np.asarray(rows, dtype=np.float32)
