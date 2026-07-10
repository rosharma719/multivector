#!/usr/bin/env python3
"""Free local BEIR benchmark: PLAID+ColBERTv2 vs Qdrant+MiniLM."""
import argparse, json, math, shutil, subprocess, time, urllib.error, urllib.request
from collections import defaultdict
from pathlib import Path
import ir_datasets, numpy as np
from pylate import models
from qdrant_client import QdrantClient, models as qm
from sentence_transformers import SentenceTransformer

def http(base, route, body=None):
    data=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request(base+route,data,{"content-type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=600) as response:return json.load(response)
    except urllib.error.HTTPError as error:
        detail=error.read().decode(errors="replace")
        raise RuntimeError(f"{route} returned HTTP {error.code}: {detail}") from error

def score(run,qrels,k=10):
    ndcg=[];recall=[]
    for qid,relevant in qrels.items():
        ranked=run.get(qid,[])[:k];gains=[relevant.get(doc,0) for doc in ranked]
        dcg=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(gains));ideal=sorted(relevant.values(),reverse=True)[:k]
        idcg=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(ideal));ndcg.append(dcg/idcg if idcg else 0)
        wanted={doc for doc,g in relevant.items() if g>0};recall.append(len(wanted.intersection(ranked))/len(wanted))
    return {f"ndcg@{k}":float(np.mean(ndcg)),f"recall@{k}":float(np.mean(recall))}

def colbert_encode(model,texts,is_query,batch):
    values=model.encode(texts,is_query=is_query,batch_size=batch,show_progress_bar=True)
    return [np.asarray(value,dtype=np.float32).tolist() for value in values]

def size(path):return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",default="beir/nfcorpus/test");p.add_argument("--output",type=Path,default=Path("benchmark/results/nfcorpus"));p.add_argument("--limit-docs",type=int);p.add_argument("--limit-queries",type=int);p.add_argument("--centroids",type=int,default=256);p.add_argument("--probes",type=int,default=8);p.add_argument("--candidates",type=int,default=100);p.add_argument("--batch-size",type=int,default=32);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    dataset=ir_datasets.load(a.dataset);docs=list(dataset.docs_iter())[:a.limit_docs];queries=list(dataset.queries_iter())[:a.limit_queries];doc_ids={d.doc_id for d in docs};query_ids={q.query_id for q in queries};qrels=defaultdict(dict)
    for r in dataset.qrels_iter():
        if r.query_id in query_ids and r.doc_id in doc_ids and r.relevance>0:qrels[r.query_id][r.doc_id]=r.relevance
    queries=[q for q in queries if q.query_id in qrels];texts=[(getattr(d,"title","")+" "+d.text).strip() for d in docs]

    print("Encoding documents with free ColBERTv2 checkpoint")
    colbert=models.ColBERT(model_name_or_path="colbert-ir/colbertv2.0");multi_docs=colbert_encode(colbert,texts,False,a.batch_size);dimension=len(multi_docs[0][0]);all_tokens=[token for doc in multi_docs for token in doc]
    rng=np.random.default_rng(13);n=min(len(all_tokens),max(a.centroids*50,a.centroids));samples=[all_tokens[i] for i in rng.choice(len(all_tokens),n,replace=False)]
    plaid_path=a.output/"plaid";shutil.rmtree(plaid_path,ignore_errors=True);root=Path(__file__).resolve().parents[1]
    command=["cargo","run","--release","--","--dimension",str(dimension),"--centroids",str(a.centroids),"--probes",str(a.probes),"--path",str(plaid_path),"--listen","127.0.0.1:18080"]
    server=subprocess.Popen(command,cwd=root,stdout=subprocess.DEVNULL);base="http://127.0.0.1:18080"
    try:
        for _ in range(300):
            try:http(base,"/healthz");break
            except Exception:time.sleep(1)
        else:raise RuntimeError("PLAID server did not start")
        http(base,"/v1/train",{"vectors":samples,"iterations":20})
        for start in range(0,len(docs),100):http(base,"/v1/vectors/upsert",{"documents":[{"id":d.doc_id,"vectors":v} for d,v in zip(docs[start:start+100],multi_docs[start:start+100])]})
        multi_queries=colbert_encode(colbert,[q.text for q in queries],True,a.batch_size);plaid_run={};plaid_times=[]
        for q,v in zip(queries,multi_queries):
            started=time.perf_counter();result=http(base,"/v1/query",{"vectors":v,"top_k":100,"candidates":a.candidates});plaid_times.append(time.perf_counter()-started);plaid_run[q.query_id]=[x["id"] for x in result["matches"]]
        plaid_stats=http(base,"/v1/stats")
    finally:server.terminate();server.wait(timeout=30)

    print("Encoding documents with free MiniLM checkpoint and indexing embedded Qdrant")
    dense=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2");dense_docs=dense.encode(texts,batch_size=a.batch_size,normalize_embeddings=True,show_progress_bar=True);qpath=a.output/"qdrant";shutil.rmtree(qpath,ignore_errors=True);client=QdrantClient(path=str(qpath));client.create_collection("docs",vectors_config=qm.VectorParams(size=dense_docs.shape[1],distance=qm.Distance.COSINE))
    for start in range(0,len(docs),256):client.upsert("docs",[qm.PointStruct(id=i,vector=dense_docs[i].tolist(),payload={"doc_id":docs[i].doc_id}) for i in range(start,min(start+256,len(docs)))])
    dense_queries=dense.encode([q.text for q in queries],batch_size=a.batch_size,normalize_embeddings=True);dense_run={};dense_times=[]
    for q,v in zip(queries,dense_queries):
        started=time.perf_counter();hits=client.query_points("docs",query=v.tolist(),limit=100).points;dense_times.append(time.perf_counter()-started);dense_run[q.query_id]=[x.payload["doc_id"] for x in hits]
    client.close();report={"dataset":a.dataset,"documents":len(docs),"queries":len(queries),"systems":{"muvera_colbertv2":{**score(plaid_run,qrels),"p50_ms":float(np.percentile(plaid_times,50)*1000),"p95_ms":float(np.percentile(plaid_times,95)*1000),"storage_bytes":size(plaid_path),"index_stats":plaid_stats},"qdrant_minilm":{**score(dense_run,qrels),"p50_ms":float(np.percentile(dense_times,50)*1000),"p95_ms":float(np.percentile(dense_times,95)*1000),"storage_bytes":size(qpath)}}};(a.output/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
