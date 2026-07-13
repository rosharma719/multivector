# Free BEIR comparison

This benchmark is entirely local and free: BEIR through `ir-datasets`, open
ColBERTv2 and MiniLM checkpoints, this PLAID server, and Qdrant embedded mode.
It requires no API keys, hosted databases, or paid datasets.

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
