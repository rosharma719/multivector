# Free BEIR comparison

This benchmark is entirely local and free: BEIR through `ir-datasets`, open
ColBERTv2 and MiniLM checkpoints, this PLAID server, and Qdrant embedded mode.
It requires no API keys, hosted databases, or paid datasets.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r benchmark/requirements.txt
python benchmark/run.py --dataset beir/nfcorpus/test
```

For a smoke test:

```bash
python benchmark/run.py --limit-docs 1000 --limit-queries 50 --centroids 64
```

The JSON report contains nDCG@10 and Recall@10 against identical BEIR qrels,
p50/p95 retrieval latency (embedding time excluded), and on-disk index size.
Results are written under `benchmark/results/`, which is git-ignored.
