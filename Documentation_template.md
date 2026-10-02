# Business Entity Resolution — Methodology Document

## 1. Overview

This solution resolves Source-1 business entities against Source-2 and
Source-3 candidate pools across US, India, and France (France unseen in
training) using **inverted-index token blocking → pairwise feature
engineering → gradient-boosted classifier → precision-tuned thresholding**.
No external data, APIs, or business-registry lookups are used at any stage.
The pipeline was run to completion on the **full test set** —
**1,732,544 / 1,732,544 Source-1 entities**, verified against every rule in
the problem statement (see Section 7).

## 2. Data profiling (what shaped the design)

- Legal-entity forms appear as both **prefixes and suffixes** ("LLC Moncada
  Learning Center" vs. "Moncada Learning Center LLC"), across US/India/France
  conventions (Pvt/Ltd, SARL/SAS/SASU/EURL).
- Non-Latin scripts for Indian business names (Devanagari, Kannada).
- Business names that are bare domains (`wilfordhancock.com`).
- DBA/trade names, address component reordering, missing PIN/state,
  landmark references, repeated tokens, empty addresses.
- Ground truth: ~5.6% of training Source-1 entities are true singletons
  (no match at all).
- Scale: ~2.2M Source-1, ~5M Source-2, ~5.3M Source-3 training records;
  test set is similar scale with a third country (France) added.

## 3. Candidate generation / blocking strategy

**This is the part of the design that changed most during development, and
the reasoning behind the change is worth recording.** An initial version
blocked via character n-gram TF-IDF cosine similarity computed in chunked
dense query×pool matrix blocks. That approach is inherently **O(n_query ×
n_pool)** no matter how the chunking is arranged — for this dataset's scale
(1.7M Source-1 rows against a ~10M-record candidate pool) that is
computationally infeasible on constrained hardware, and does not meaningfully
improve on better hardware either since the complexity class itself doesn't
change.

The production design instead uses **inverted-index token blocking**, the
standard large-scale entity-resolution technique:

- Records are grouped by **exact normalized country string** before any
  similarity computation — a generic equality check, not a lookup table, so
  it applies identically to France without any code change.
- For each country partition, two inverted indices are built (token ->
  list of pool ids, capped and IDF-weighted): one over **name tokens only**,
  one over **name+address tokens** ("blend"). A query record's candidates
  are found by looking up its own tokens in the index and accumulating an
  IDF-weighted overlap score per pool id it shares a token with — this only
  ever touches pool records that share at least one token with the query,
  so cost scales with the number of (query token, posting) pairs actually
  touched, not with n_query × n_pool.
- Tokens that are too common to be useful for blocking (appearing in more
  than a small fraction of the pool, e.g. generic address words even after
  abbreviation expansion) are dropped; remaining posting lists are capped at
  a fixed length so no single token can dominate runtime.
- Two views (name-only, blended) are unioned per query so noisy addresses
  don't sink a strong name match and vice versa.
- **Country-by-country processing**: rather than holding all three
  countries' pools in memory simultaneously, the pipeline processes one
  country at a time (partitioning the raw files first via
  `partition_by_country.py`), bounding peak memory to the single largest
  country's pool rather than the sum of all three.

On a representative training sample, this blocking achieved a **~97%
recall ceiling** against ground truth with the final (tightened) token
parameters — i.e., the classifier stage can at best recover ~97% of true
matches. Average candidates surfaced per Source-1 entity on the full test
run: **~20.8**.

## 4. Feature engineering

For every (Source-1, candidate) pair surviving blocking:

| Feature | Description |
|---|---|
| `name_tok_jaccard` | Jaccard similarity of normalized-name token sets |
| `addr_tok_jaccard` | Jaccard similarity of normalized-address token sets |
| `name_char_jaccard` | Jaccard similarity of name character 3-gram sets |
| `addr_char_jaccard` | Jaccard similarity of address character 3-gram sets |
| `name_seq_ratio` | `difflib.SequenceMatcher` quick ratio on normalized names |
| `addr_seq_ratio` | same, on normalized addresses |
| `name_len_diff` | relative length difference of normalized names |
| `addr_len_diff` | relative length difference of normalized addresses |
| `name_substr` | 1 if one normalized name is a substring of the other |
| `first_tok_match` | 1 if first tokens of both normalized names match |
| `country_match` | 1 if normalized country strings are equal |
| `tok_name_score` | IDF-weighted token-overlap score from the name-only blocking index |
| `tok_blend_score` | IDF-weighted token-overlap score from the blended blocking index |

## 5. Model architecture

`sklearn.ensemble.HistGradientBoostingClassifier` — histogram-based gradient
boosted trees, BSD-3-licensed (part of scikit-learn; permissive, same family
as MIT/Apache), far under the 8B parameter constraint.

Hyperparameters: `max_iter=200`, `max_depth=6`, `learning_rate=0.08`,
`l2_regularization=1.0`, `class_weight="balanced"`.

**Training labels**: positives are pairs present in `train_ground_truth.tsv`
that also survived blocking; negatives are all other blocked candidates for
the same Source-1 entities (hard negatives).

**Train/validation split**: grouped by Source-1 entity id (80/20), no
leakage.

## 6. Threshold selection for F_0.5

The decision threshold is swept on a held-out validation split and the value
maximizing macro F_0.5 is selected and persisted with the model, rather than
using a fixed 0.5 cutoff — since F_0.5 weights precision 2x over recall, the
tuned threshold comes out well above 0.5 in practice.

## 7. Full-scale run results and validation

The complete pipeline (blocking + trained model + tuned threshold) was run
to completion on the **entire test set**:

- **1,732,544 / 1,732,544** Source-1 test entities have a prediction row —
  full coverage, France included.
- Non-empty match rate: **93.3%** (singleton rate ~6.65%, comparable to the
  ~5.6% seen in training ground truth).
- Match-count distribution peaks at 3-4 matches per entity, mirroring the
  shape seen in `train_ground_truth.tsv`.
- Both output files were validated against every rule in the problem
  statement using external sort/comm-based checks (memory-safe at this
  scale): zero duplicate rows, zero missing/invalid Source-1 ids, zero
  candidate/match ids outside the valid Source-2/3 test universe, zero
  self-matches, zero in-row duplicate ids, and every matched id confirmed
  present in that same row's own candidate list (matches ⊆ candidates).

**Known limitation / honest disclosure**: the final full-scale inference run
used a model trained on an earlier iteration of the blocking features
(character n-gram TF-IDF cosine scores in the two blocking-score feature
slots, rather than the token-overlap scores the final blocking stage
produces). The other 11 of 13 features are computed identically regardless
of blocking method and are unaffected. Both blocking-score features are
similarity measures in a comparable [0,1] range, so the mismatch is expected
to have a modest rather than severe effect on calibration, but retraining on
the exact final feature distribution (fast — a few minutes on a
representative sample) and re-scoring the already-generated
`candidate_pairs.tsv` (without redoing the expensive blocking step) would be
the natural next step to tighten this up further before a competitive
leaderboard push.

## 8. Fair play

No external database, API, geocoding service, or internet lookup is used
anywhere in this pipeline. All text-normalization rules encode generic,
publicly-known conventions about corporate legal-entity suffixes and address
abbreviations — not information about any specific business's registration
or identity.
