# Free BEIR comparison

This benchmark is entirely local and free: BEIR through `ir-datasets`, open
ColBERTv2 and MiniLM checkpoints, this PLAID server, and the vectordb HNSW crate.
It requires no hosted databases or paid datasets. A Hugging Face token is optional.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r benchmark/requirements.txt
cp .env.example .env  # optional: add a free HF_TOKEN for download rate limits
python benchmark/run.py --dataset beir/nfcorpus/test \
  --report-dir benchmark/reports
```

For a smoke test:

```bash
python benchmark/run.py --limit-docs 1000 --limit-queries 50 --centroids 64
```

The JSON report contains nDCG@10 and Recall@10 against identical BEIR qrels,
p50/p95 retrieval latency (embedding time excluded), and on-disk index size.
It also records the engine's Semantic Version, Git revision/dirty state, Rust
version, installed Python package versions, and model IDs. Each invocation
appends one record to `benchmark/reports/v<version>.jsonl`; large indexes remain
under the git-ignored `benchmark/results/` directory.

Run the standard candidate-generation matrix after building an index:

```bash
python benchmark/matrix.py \
  --index benchmark/results/fiqa-10k/plaid \
  --dataset beir/fiqa/test --limit-docs 10000 --limit-queries 100 \
  --report-dir benchmark/reports
```

This appends exact-FDE records for candidate counts 50/100/250/500 and HNSW
records at `ef_search` 32/64/128/256 to the version ledger. Pass `--candidates`
or `--ef-search` to run a smaller matrix. The HNSW graph is built once and
reused for every HNSW grid point.

`benchmark/run.py`, `benchmark/sweep.py`, and `benchmark/validate_scores.py`
load the repository-root `.env` automatically. The only currently supported
credential is the optional `HF_TOKEN`; it is read by Hugging Face tooling and
is never written to reports. Shell environment variables take precedence.

For a limited corpus with complete judgments for every selected query, build a
relevance-preserving slice instead of taking the first documents:

```bash
python benchmark/run.py \
  --dataset beir/fiqa/test --output benchmark/results/fiqa-qrels-10k \
  --limit-docs 10000 --limit-queries 100 --sampling qrels \
  --centroids 256 --candidates 250
```

Then measure the exhaustive compressed-MaxSim ceiling and HNSW candidate recall:

```bash
python benchmark/diagnose.py \
  --index benchmark/results/fiqa-qrels-10k/plaid \
  --dataset beir/fiqa/test --limit-docs 10000 --limit-queries 100 \
  --sampling qrels --candidates 250 --ef-search 256
```

The diagnostic appends one record containing exhaustive, exact-FDE, and HNSW
quality/latency plus candidate recall versus exact FDE. The saved slice manifest
prevents accidentally evaluating an index against a different document sample.

## Embedding cache

All benchmark commands reuse content-addressed embeddings under
`benchmark/cache/`. ColBERT token embeddings are stored as a contiguous float32
matrix plus per-item offsets; MiniLM embeddings are stored as a fixed float32
matrix. Cache keys include IDs and exact text content, model/checkpoint revision,
query/document mode, normalization, and encoder package versions.

Use a different location with `--cache-dir`, or intentionally regenerate a
matching entry with `--refresh-cache`. Cached embeddings and large indexes are
git-ignored; reports record the cache key, checkpoint revision, and hit/miss
state but never embed the vectors themselves.

Warm all four caches without rebuilding either retrieval index:

```bash
python benchmark/cache_embeddings.py \
  --dataset beir/fiqa/test --limit-docs 10000 --limit-queries 100 \
  --sampling qrels
```

Compare uncompressed and PLAID-compressed MaxSim over the identical exact-FDE
candidate sets:

```bash
python benchmark/uncompressed_oracle.py \
  --index benchmark/results/fiqa-qrels-10k/plaid \
  --dataset beir/fiqa/test --limit-docs 10000 --limit-queries 100 \
  --sampling qrels --candidates 250 --scope candidates
```

Use `--scope exhaustive` for the uncompressed encoder ceiling, or `--scope both`
for both tests. Exhaustive mode performs substantially more matrix multiplication;
candidate mode is the fast test that isolates compression from candidate pruning.

Audit the trained projection and ColBERT query/document conventions after model
or dependency changes:

```bash
python benchmark/audit_encoder.py
```

The command fails if the loaded 128-dimensional projection differs from the
checkpoint or if marker tokens, mask expansion, punctuation filtering, maximum
lengths, or output normalization no longer match the expected ColBERTv2 path.

Benchmark cached MiniLM embeddings with exact search and the vectordb HNSW
implementation, without rebuilding the multi-vector index:

```bash
python benchmark/dense_baseline.py \
  --dataset beir/fiqa/test --limit-docs 10000 --limit-queries 100 \
  --sampling qrels --m 16 --ef-search 256
```
