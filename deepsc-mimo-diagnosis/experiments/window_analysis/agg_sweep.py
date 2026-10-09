"""Observation-window aggregation sweep on saved per-sentence fusion probabilities.
Aggregators are combination rules from Kittler et al. (1998); temperature and probability
floor are chosen by nested leave-one-fold-out selection, never on the reported fold."""
import csv, sys, math, json, statistics as st
from collections import defaultdict
import numpy as np
C=["normal","low_snr","imperfect_csi","spatial_correlation","literal","mismatch","ood"]
WINDOWS=[1,2,4,8,16,32,64,128]
rows=[]
for p in sorted(__import__('glob').glob(sys.argv[1]+'/fusion_probs_seed*.csv')):
    with open(p) as h:
        for r in csv.DictReader(h):
            rows.append((int(r['held_out_seed']),r['path'],int(r['index']),C.index(r['label']),
                         [float(r['p_'+c]) for c in C]))
seeds=sorted({r[0] for r in rows})
# group consecutive sentences within a batch
groups=defaultdict(list)
for s,path,i,y,p in rows: groups[(s,path)].append((i,y,p))
G={}
for key,g in groups.items():
    g.sort(); G[key]=(np.array([x[1] for x in g]),np.array([x[2] for x in g]))
def windows(k,seed):
    Y=[];P=[]
    for (s,_),(y,p) in G.items():
        if s!=seed: continue
        n=(len(y)//k)*k
        if n==0: continue
        Y.append(y[:n:k]); P.append(p[:n].reshape(-1,k,7))
    return np.concatenate(Y),np.concatenate(P)
def agg(P,rule,T=1.0,eps=1e-12):
    if rule=='sum': return P.mean(1)
    L=np.log(np.clip(P,eps,1.0))/T
    if rule=='product': return L.mean(1)
    if rule=='median': return np.median(P,1)
    if rule=='vote':
        v=np.zeros((P.shape[0],7)); a=P.argmax(2)
        for j in range(P.shape[1]): v[np.arange(len(a)),a[:,j]]+=1
        return v+1e-6*P.mean(1)   # tie-break by mean probability
    if rule=='trimmed':
        k=P.shape[1]; c=int(0.2*k)
        S=np.sort(P,1); return S[:,c:k-c].mean(1) if k-2*c>0 else P.mean(1)
def ba(y,s):
    pred=s.argmax(1); return float(np.mean([np.mean(pred[y==c]==c) for c in range(7) if np.any(y==c)])), \
        [float(np.mean(pred[y==c]==c)) for c in range(7)]
CANDS=[('sum',1,1e-12),('median',1,1e-12),('vote',1,1e-12),('trimmed',1,1e-12)]+\
      [('product',T,e) for T in (1.0,) for e in (1e-12,1e-6,1e-3,1e-2)]
cache={}
def score(seed,k,c):
    key=(seed,k,c)
    if key not in cache:
        y,P=windows(k,seed); cache[key]=ba(y,agg(P,c[0],c[1],c[2]))
    return cache[key]
out=[]
for k in WINDOWS:
    per={c:[score(s,k,c)[0] for s in seeds] for c in CANDS}
    nested=[];chosen=[]
    for s in seeds:
        others=[t for t in seeds if t!=s]
        best=max(CANDS,key=lambda c:np.mean([score(t,k,c)[0] for t in others]))
        chosen.append(best); nested.append(score(s,k,best)[0])
    base=per[('sum',1,1e-12)]
    dif=[a-b for a,b in zip(nested,base)]
    row={'k':k,'sum_mean':np.mean(base),'sum_std':np.std(base,ddof=1),
         'nested_mean':np.mean(nested),'nested_std':np.std(nested,ddof=1),
         'gain_mean':np.mean(dif),'gain_std':np.std(dif,ddof=1),'gain_pos':sum(d>0 for d in dif),
         'chosen':[f"{c[0]}/eps={c[2]:g}" for c in chosen]}
    row.update({f"{c[0]}/{c[2]:g}":round(float(np.mean(v)),4) for c,v in per.items()})
    # recalls for nested choice pooled mean
    rec=np.mean([score(s,k,ch)[1] for s,ch in zip(seeds,chosen)],0)
    row['recall_nested']={C[i]:round(float(rec[i]),3) for i in range(7)}
    out.append(row)
    print(f"k={k:3d} sum {row['sum_mean']:.3f}±{row['sum_std']:.3f} | nested {row['nested_mean']:.3f}±{row['nested_std']:.3f} gain {row['gain_mean']:+.3f}±{row['gain_std']:.3f} ({row['gain_pos']}/5) chosen {row['chosen']}")
    print("      all:",{kk:v for kk,v in row.items() if '/' in kk})
    print("      recall:",row['recall_nested'])
json.dump(out,open(sys.argv[2],'w'),indent=1,default=float)
