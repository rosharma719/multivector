#!/usr/bin/env python3
"""Sweep a completed index without rebuilding or re-encoding documents."""
import argparse, json, subprocess, time
from collections import defaultdict
from pathlib import Path
import ir_datasets, numpy as np
from pylate import models
from run import colbert_encode, http, score

def main():
    p=argparse.ArgumentParser();p.add_argument("--index",type=Path,default=Path("benchmark/results/nfcorpus/plaid"));p.add_argument("--dataset",default="beir/nfcorpus/test");p.add_argument("--centroids",type=int,default=256);p.add_argument("--configured-probes",type=int,default=8);p.add_argument("--backend",choices=["muvera","centroid"],default="muvera");p.add_argument("--probes",type=int,default=8);p.add_argument("--candidates",type=int,default=100);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    dataset=ir_datasets.load(a.dataset);queries=list(dataset.queries_iter());qrels=defaultdict(dict)
    for r in dataset.qrels_iter():
        if r.relevance>0:qrels[r.query_id][r.doc_id]=r.relevance
    queries=[q for q in queries if q.query_id in qrels];model=models.ColBERT(model_name_or_path="colbert-ir/colbertv2.0");vectors=colbert_encode(model,[q.text for q in queries],True,32)
    root=Path(__file__).resolve().parents[1];command=["cargo","run","--release","--","--dimension","128","--centroids",str(a.centroids),"--probes",str(a.configured_probes),"--path",str(a.index),"--listen","127.0.0.1:18080"];server=subprocess.Popen(command,cwd=root,stdout=subprocess.DEVNULL);base="http://127.0.0.1:18080"
    try:
        for _ in range(60):
            try:http(base,"/healthz");break
            except Exception:time.sleep(1)
        run={};latency=[]
        for q,v in zip(queries,vectors):
            body={"vectors":v,"top_k":100,"candidates":a.candidates}
            if a.backend=="centroid":body["probes"]=a.probes
            started=time.perf_counter();result=http(base,"/v1/query",body);latency.append(time.perf_counter()-started);run[q.query_id]=[x["id"] for x in result["matches"]]
    finally:server.terminate();server.wait(timeout=30)
    report={"dataset":a.dataset,"queries":len(queries),"backend":a.backend,"probes":a.probes if a.backend=="centroid" else None,"candidates":a.candidates,**score(run,qrels),"p50_ms":float(np.percentile(latency,50)*1000),"p95_ms":float(np.percentile(latency,95)*1000)};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
