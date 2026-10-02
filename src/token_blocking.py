"""
Inverted-index token blocking -- the algorithmic fix for scale.

The previous blocking.py computed cosine similarity via chunked DENSE
query x pool matrix blocks. That is O(n_query * n_pool) no matter how the
chunking is arranged -- for 1.73M x ~10M records that is computationally
impossible on any single machine.

The standard large-scale ER technique instead builds a hash index:
  token -> [pool ids containing that token]
and only ever compares a query record against pool records it shares at
least one token with, using an IDF-weighted accumulator. Cost scales with
the number of (query token, posting) pairs actually touched, not with
n_query * n_pool -- near-linear in practice once very common tokens
("street", "inc", "private"...) are capped/dropped.
"""
import math
import pickle
from collections import defaultdict, Counter


class TokenIndex:
    def __init__(self, ids, tokenize_fn, max_df_ratio=0.003, max_postings=900):
        """
        ids: list of pool entity_ids
        tokenize_fn: zero-arg callable returning a FRESH generator/iterator of
                     token-lists, one per id in the same order as ids. Called
                     TWICE (df-counting pass, then postings-building pass) so
                     that per-document token sets are never all held in memory
                     at once -- only the running counters/index are resident.
        """
        n = len(ids)
        self.n = n
        df = Counter()
        for toks in tokenize_fn():
            df.update(set(toks))

        max_df = max(20, int(max_df_ratio * max(n, 1)))
        self.keep = {t for t, c in df.items() if 1 <= c <= max_df and t}
        self.idf = {t: math.log(1.0 + n / (1.0 + df[t])) for t in self.keep}
        del df

        index = defaultdict(list)
        for pid, toks in zip(ids, tokenize_fn()):
            seen = set()
            for t in toks:
                if t in self.keep and t not in seen:
                    seen.add(t)
                    index[t].append(pid)

        self.index = {}
        for t, postings in index.items():
            if len(postings) > max_postings:
                postings = postings[:max_postings]
            self.index[t] = postings
        del index

    def query(self, tokens, k=15):
        """tokens: iterable of normalized tokens for one query record.
        Returns list of (pool_id, raw_score, normalized_score)."""
        q_tok_set = set(tokens)
        scores = defaultdict(float)
        for t in q_tok_set:
            postings = self.index.get(t)
            if not postings:
                continue
            w = self.idf[t]
            for pid in postings:
                scores[pid] += w
        if not scores:
            return []
        max_possible = sum(self.idf.get(t, 0.0) for t in q_tok_set) or 1.0
        items = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [(pid, s, s / max_possible) for pid, s in items]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"index": self.index, "idf": self.idf, "n": self.n}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls.__new__(cls)
        obj.index, obj.idf, obj.n = d["index"], d["idf"], d["n"]
        obj.keep = set(obj.idf.keys())
        return obj


def tokenize_for_blocking(norm_name, norm_address):
    """Union of name + address tokens, used both to build and to query the index."""
    toks = []
    if norm_name:
        toks.extend(norm_name.split())
    if norm_address:
        toks.extend(norm_address.split())
    return toks
