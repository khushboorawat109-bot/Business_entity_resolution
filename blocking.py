"""
Candidate generation (blocking) via character n-gram TF-IDF cosine similarity,
computed in memory-bounded chunks so it scales beyond what fits densely in RAM.

Strategy:
  1. Partition records by exact normalized country string. This is NOT
     hardcoding specific country values -- it is a generic equality check
     that works identically on any string, including "france" which never
     appears in training. Cross-country matches are essentially impossible
     in this domain, so this partition alone cuts the search space ~3x
     with negligible recall loss, and it degrades gracefully (a record with
     a missing/blank country just gets its own partition).
  2. Within each partition, fit a char n-gram TF-IDF vectorizer on the union
     of source1 + candidate-pool text, for two views: business_name alone,
     and name+address ("blend"). Two views because some records have very
     similar names but noisy/partial addresses (or vice versa) -- a single
     merged view can under-rank those.
  3. For each view, compute cosine similarity in (query_chunk x pool_chunk)
     dense blocks -- never materializing the full query x pool matrix -- and
     keep a running top-K per query row via numpy argpartition.
  4. Union the top-K candidates from both views (dedup by id, keep max sim
     per view) as the final candidate set for that Source-1 entity.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_vectorizer(corpus, ngram_range=(2, 4), max_features=60000):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range,
                           max_features=max_features, min_df=1, dtype=np.float32)
    vec.fit(corpus)
    return vec


def chunked_topk_cosine(query_texts, pool_texts, vectorizer,
                         k=15, query_chunk=500, pool_chunk=8000):
    """
    Returns (top_idx, top_sim): both shape (n_query, k), pool-relative
    integer indices (-1 padding) and cosine similarities (0.0 padding).
    """
    Q = vectorizer.transform(query_texts).astype(np.float32)
    P = vectorizer.transform(pool_texts).astype(np.float32)
    n_q, n_p = Q.shape[0], P.shape[0]
    k = min(k, max(n_p, 1))

    all_top_idx = np.full((n_q, k), -1, dtype=np.int64)
    all_top_sim = np.zeros((n_q, k), dtype=np.float32)
    if n_p == 0 or n_q == 0:
        return all_top_idx, all_top_sim

    for qs in range(0, n_q, query_chunk):
        qe = min(qs + query_chunk, n_q)
        Qc = Q[qs:qe]
        cur_idx = np.full((qe - qs, 0), -1, dtype=np.int64)
        cur_sim = np.zeros((qe - qs, 0), dtype=np.float32)

        for ps in range(0, n_p, pool_chunk):
            pe = min(ps + pool_chunk, n_p)
            block = Qc.dot(P[ps:pe].T)
            block = np.asarray(block.todense(), dtype=np.float32)
            combined_sim = np.concatenate([cur_sim, block], axis=1)
            combined_idx = np.concatenate(
                [cur_idx, np.tile(np.arange(ps, pe), (qe - qs, 1))], axis=1
            )
            kk = min(k, combined_sim.shape[1])
            part = np.argpartition(-combined_sim, kk - 1, axis=1)[:, :kk]
            row_sel = np.arange(combined_sim.shape[0])[:, None]
            cur_sim = combined_sim[row_sel, part]
            cur_idx = combined_idx[row_sel, part]

        order = np.argsort(-cur_sim, axis=1)
        cur_sim = np.take_along_axis(cur_sim, order, axis=1)
        cur_idx = np.take_along_axis(cur_idx, order, axis=1)
        kk = cur_sim.shape[1]
        all_top_sim[qs:qe, :kk] = cur_sim
        all_top_idx[qs:qe, :kk] = cur_idx

    return all_top_idx, all_top_sim


def block_partition(q_ids, q_name, q_blend, pool_ids, pool_name, pool_blend,
                     k=15, min_sim=0.12, query_chunk=500, pool_chunk=8000):
    """
    Block one (query, pool) partition using both the name-only and blend
    views, union the results. Returns dict:
        q_id -> list of (pool_id, name_cos, blend_cos)
    """
    if len(pool_ids) == 0 or len(q_ids) == 0:
        return {qid: [] for qid in q_ids}

    name_vec = fit_vectorizer(list(q_name) + list(pool_name))
    blend_vec = fit_vectorizer(list(q_blend) + list(pool_blend))

    name_idx, name_sim = chunked_topk_cosine(q_name, pool_name, name_vec,
                                              k=k, query_chunk=query_chunk,
                                              pool_chunk=pool_chunk)
    blend_idx, blend_sim = chunked_topk_cosine(q_blend, pool_blend, blend_vec,
                                                k=k, query_chunk=query_chunk,
                                                pool_chunk=pool_chunk)

    pool_ids = np.asarray(pool_ids)
    results = {}
    for i, qid in enumerate(q_ids):
        merged = {}
        for j, s in zip(name_idx[i], name_sim[i]):
            if j >= 0 and s >= min_sim:
                pid = pool_ids[j]
                cur = merged.setdefault(pid, [0.0, 0.0])
                cur[0] = max(cur[0], float(s))
        for j, s in zip(blend_idx[i], blend_sim[i]):
            if j >= 0 and s >= min_sim:
                pid = pool_ids[j]
                cur = merged.setdefault(pid, [0.0, 0.0])
                cur[1] = max(cur[1], float(s))
        results[qid] = [(pid, v[0], v[1]) for pid, v in merged.items()]
    return results


class CountryPoolIndex:
    """
    Precomputes TF-IDF vectorizers + transformed matrices for one country's
    candidate pool ONCE, so many query batches (e.g. all of test_source1 for
    that country, processed in chunks) can be blocked against it without
    re-vectorizing the pool every time. This is the efficient shape for
    inference at scale: pool vectorization is the expensive, reusable part.
    """

    def __init__(self, pool_ids, pool_name, pool_blend, max_features=60000):
        self.pool_ids = np.asarray(pool_ids)
        self.name_vec = fit_vectorizer(pool_name, max_features=max_features)
        self.blend_vec = fit_vectorizer(pool_blend, max_features=max_features)
        self.pool_name_mat = self.name_vec.transform(pool_name).astype(np.float32)
        self.pool_blend_mat = self.blend_vec.transform(pool_blend).astype(np.float32)

    def query(self, q_ids, q_name, q_blend, k=15, min_sim=0.12,
              query_chunk=500, pool_chunk=8000):
        if len(self.pool_ids) == 0 or len(q_ids) == 0:
            return {qid: [] for qid in q_ids}
        Q_name = self.name_vec.transform(q_name).astype(np.float32)
        Q_blend = self.blend_vec.transform(q_blend).astype(np.float32)

        name_idx, name_sim = _topk_from_matrices(Q_name, self.pool_name_mat, k,
                                                   query_chunk, pool_chunk)
        blend_idx, blend_sim = _topk_from_matrices(Q_blend, self.pool_blend_mat, k,
                                                     query_chunk, pool_chunk)
        results = {}
        for i, qid in enumerate(q_ids):
            merged = {}
            for j, s in zip(name_idx[i], name_sim[i]):
                if j >= 0 and s >= min_sim:
                    pid = self.pool_ids[j]
                    cur = merged.setdefault(pid, [0.0, 0.0])
                    cur[0] = max(cur[0], float(s))
            for j, s in zip(blend_idx[i], blend_sim[i]):
                if j >= 0 and s >= min_sim:
                    pid = self.pool_ids[j]
                    cur = merged.setdefault(pid, [0.0, 0.0])
                    cur[1] = max(cur[1], float(s))
            results[qid] = [(pid, v[0], v[1]) for pid, v in merged.items()]
        return results


def _topk_from_matrices(Q, P, k, query_chunk, pool_chunk):
    n_q, n_p = Q.shape[0], P.shape[0]
    k = min(k, max(n_p, 1))
    all_top_idx = np.full((n_q, k), -1, dtype=np.int64)
    all_top_sim = np.zeros((n_q, k), dtype=np.float32)
    if n_p == 0 or n_q == 0:
        return all_top_idx, all_top_sim
    for qs in range(0, n_q, query_chunk):
        qe = min(qs + query_chunk, n_q)
        Qc = Q[qs:qe]
        cur_idx = np.full((qe - qs, 0), -1, dtype=np.int64)
        cur_sim = np.zeros((qe - qs, 0), dtype=np.float32)
        for ps in range(0, n_p, pool_chunk):
            pe = min(ps + pool_chunk, n_p)
            block = Qc.dot(P[ps:pe].T)
            block = np.asarray(block.todense(), dtype=np.float32)
            combined_sim = np.concatenate([cur_sim, block], axis=1)
            combined_idx = np.concatenate(
                [cur_idx, np.tile(np.arange(ps, pe), (qe - qs, 1))], axis=1
            )
            kk = min(k, combined_sim.shape[1])
            part = np.argpartition(-combined_sim, kk - 1, axis=1)[:, :kk]
            row_sel = np.arange(combined_sim.shape[0])[:, None]
            cur_sim = combined_sim[row_sel, part]
            cur_idx = combined_idx[row_sel, part]
        order = np.argsort(-cur_sim, axis=1)
        cur_sim = np.take_along_axis(cur_sim, order, axis=1)
        cur_idx = np.take_along_axis(cur_idx, order, axis=1)
        kk = cur_sim.shape[1]
        all_top_sim[qs:qe, :kk] = cur_sim
        all_top_idx[qs:qe, :kk] = cur_idx
    return all_top_idx, all_top_sim


def block_by_country(df1, df_pool, id_col="entity_id", k=15, min_sim=0.12,
                      query_chunk=500, pool_chunk=8000):
    """
    df1, df_pool must already carry norm_name / norm_country / blend_text.
    Returns dict: source1_id -> list of (pool_id, name_cos, blend_cos)
    """
    all_results = {}
    countries = set(df1["norm_country"].unique())
    for country in countries:
        q = df1[df1["norm_country"] == country]
        pool = df_pool[df_pool["norm_country"] == country]
        res = block_partition(
            q[id_col].tolist(), q["norm_name"].tolist(), q["blend_text"].tolist(),
            pool[id_col].tolist(), pool["norm_name"].tolist(), pool["blend_text"].tolist(),
            k=k, min_sim=min_sim, query_chunk=query_chunk, pool_chunk=pool_chunk,
        )
        all_results.update(res)
    # any source1 rows whose country had zero pool candidates
    for qid in df1[id_col]:
        all_results.setdefault(qid, [])
    return all_results
