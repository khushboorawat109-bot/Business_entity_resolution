# Business Entity Resolution at Scale

A record-linkage pipeline that matches ~1.7M business records across three
noisy data sources (26M+ rows total) — built and run inside a **1-vCPU /
4GB-RAM** environment with a hard 300-second execution limit per command.
Originally built for an ML hackathon; the interesting part turned out to be
the engineering constraints, not the leaderboard.

## The problem

Given business records from three independent sources (name, address,
country — US / India / France, with France unseen at training time), find
every Source-2/Source-3 record that refers to the same real-world business
as each Source-1 record. Classic **entity resolution**: no shared ID, heavy
noise (typos, abbreviations, transliteration, landmark-based addresses,
missing fields), evaluated with **macro F_0.5** (precision weighted 2x over
recall — false merges are penalized hard).

## Why this was harder than a typical ML task

Most of the real work here wasn't modeling — it was making a quadratic
problem tractable under a tiny compute budget, then catching a bug that
only showed up at real scale.

### 1. Blocking had to scale to ~17 trillion possible pairs
Comparing every Source-1 record against every Source-2/3 record directly
(`1.7M × 10M`) is computationally impossible regardless of hardware. The
fix: **inverted-index token blocking** — build a `token → posting list`
index per country partition, so a query only ever touches records sharing
at least one token with it. This turns an O(n×m) problem into something
close to linear in the number of (token, posting) pairs actually touched.

### 2. Memory had to be managed by hand
With ~26M rows across seven files and 4GB of RAM:
- Processed **one country at a time** rather than holding all pools in
  memory simultaneously — bounds peak memory to the single largest
  partition instead of the sum of all three.
- Streaming/chunked file readers (never `pd.read_csv` on the whole file at
  once) for anything in the multi-hundred-MB range.
- A naive caching layer (pickling the built index to disk for fast resume)
  turned out to risk OOM *in both directions* once a single country's pool
  passed a few million records — caught via a truncated-file check, fixed
  with atomic writes and a `--no-cache` fallback for the largest partition.

### 3. A long-running job had to survive a 300-second command limit
The actual full-scale run took many hours of wall-clock compute, but each
individual command could only run for 5 minutes. Built a resumable
checkpoint pattern: every script writes completed rows to disk, tracks
already-done IDs, and can be safely re-invoked to continue exactly where it
left off — the entire 1.7M-row run was done through dozens of chunked,
resumable invocations.

### 4. The real bug: recall collapses at scale in ways small samples hide
Blocking parameters (posting-list length caps, document-frequency
thresholds) were tuned and validated on a 300k-record sample — looked
great (~97% recall ceiling). At the *actual* full scale of the largest
partition (4.7M records), the same settings collapsed to **64% recall**:
aggressive posting-list truncation was silently dropping legitimate
candidates for common-but-still-useful tokens once document frequency hit
real-world scale. Diagnosed by directly measuring recall against the true
full-scale pool (not a sample) and inspecting actual missed-match pairs —
which revealed a second, more subtle issue: candidates with 10 shared
tokens were still getting outranked by a sea of superficially-similar
decoys sharing common address words (city names, "new", "south"), because
raw additive IDF-sum scoring doesn't normalize for how "generic" a shared
token is relative to the full candidate.

**Lesson, stated plainly: never validate a scale-sensitive algorithm's
hyperparameters on a sample and assume they hold at production scale.**
Small-sample validation hid a bug that would have been invisible without
testing against the real data volume.

## Architecture

```
Normalize → Country-partition → Token-blocking (per country) →
Pairwise similarity features → Gradient-boosted classifier →
Precision-tuned threshold → Validated output
```

- **Normalization**: Unicode-safe cleanup, legal-entity-form stripping
  (prefix *and* suffix, across US/India/France conventions), address
  abbreviation expansion, noise-character removal.
- **Blocking**: inverted-index token blocking, two views (name-only,
  name+address blended) unioned per query.
- **Features**: 11 text-only similarity signals (token/char Jaccard,
  sequence-ratio, length deltas, substring/first-token flags, country
  match) — deliberately computed purely from each record pair's own text,
  with no dependency on blocking-stage artifacts, so there's no
  train/inference feature mismatch possible.
- **Model**: `sklearn.HistGradientBoostingClassifier` — a permissively
  licensed, fast, well-understood choice for tabular similarity features;
  no need for anything heavier.
- **Threshold**: swept on a held-out validation split to directly maximize
  macro F_0.5 (not accuracy, not F1) — since false merges are penalized 2x
  harder than misses here, the optimal threshold sits well above 0.5.

## What I'd do differently next time

- Validate blocking recall against the **true** production-scale pool
  before committing to hyperparameters, not a convenient sample.
- Keep blocking parameters **consistent across partitions** rather than
  tuning them ad-hoc per-country under time pressure — inconsistent
  settings made the final failure mode harder to diagnose.
- Budget explicit time for a held-out, cross-partition generalization check
  (train on one country, validate on another) earlier — it's a cheap way
  to catch overfitting-to-validation-distribution before it costs you on
  the real test set.

## Stack

Python · pandas · NumPy · scikit-learn · stdlib only otherwise (no network
access in the build environment, so no external libraries beyond what ships
with a standard data-science install).

## Note on data

Raw competition data is not included in this repository per the
competition's terms — `dataset/` is `.gitignore`d. The code is fully
runnable against any data matching the documented schema (see `src/` and
inline docstrings for exact format expectations).
