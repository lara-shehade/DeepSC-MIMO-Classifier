"""Aggregate evidence across several channel realizations (batches) of the same persistent impairment."""
import csv,sys,glob,itertools,json
from collections import defaultdict
import numpy as np
C=["normal","low_snr","imperfect_csi","spatial_correlation","literal","mismatch","ood"]
rows=defaultdict(list)
for p in sorted(glob.glob(sys.argv[1]+'/fusion_probs_seed*.csv')):
    for r in csv.DictReader(open(p)):
        run=r['path'].split('/artifacts/')[0]
        rows[(int(r['held_out_seed']),run,int(r['batch_index']))].append((int(r['index']),C.index(r['label']),[float(r['p_'+c]) for c in C]))
runs=defaultdict(dict)
for (s,run,b),g in rows.items():
    g.sort(); runs[(s,run)][b]=(g[0][1],np.array([x[2] for x in g]))
seeds=sorted({k[0] for k in runs})
def evaluate(m,per_batch,rule):
    res=[]
    for s in seeds:
        y=[];sc=[]
        for (ss,run),bat in runs.items():
            if ss!=s: continue
            ids=sorted(bat)
            for start in range(0,len(ids)-m+1,m):     # disjoint groups of m realizations
                chunk=ids[start:start+m]
                P=np.concatenate([bat[b][1][:per_batch] for b in chunk])
                lab=bat[chunk[0]][0]
                sc.append(P.mean(0) if rule=='sum' else np.log(np.clip(P,1e-12,1)).mean(0)); y.append(lab)
        y=np.array(y);pred=np.array(sc).argmax(1)
        rec=[np.mean(pred[y==c]==c) for c in range(7)]
        res.append((np.mean(rec),rec))
    ba=[r[0] for r in res]; rec=np.mean([r[1] for r in res],0)
    return np.mean(ba),np.std(ba,ddof=1),rec,len(y)
for per_batch in [1,8,32,128]:
    for m in [1,2,5,10]:
        for rule in ['sum']:
            mu,sd,rec,n=evaluate(m,per_batch,rule)
            print(f"sentences/realization={per_batch:3d} realizations={m:2d} total={per_batch*m:4d}: BA {mu:.3f}±{sd:.3f}  recalls "+" ".join(f"{C[i][:6]}={rec[i]:.2f}" for i in range(7))+f"  (decisions/fold={n})")

print("\n--- random subsets of m realizations (200 draws per run), expected accuracy ---")
rng=np.random.default_rng(7)
def evaluate_rs(m,per_batch,draws=200):
    res=[]
    for s in seeds:
        hit=defaultdict(list)
        for (ss,run),bat in runs.items():
            if ss!=s: continue
            ids=sorted(bat); lab=bat[ids[0]][0]
            combos=list(itertools.combinations(ids,m))
            pick=combos if len(combos)<=draws else [combos[i] for i in rng.choice(len(combos),draws,replace=False)]
            for ch in pick:
                P=np.concatenate([bat[b][1][rng.permutation(len(bat[b][1]))[:per_batch]] for b in ch])
                hit[lab].append(P.mean(0).argmax()==lab)
        rec=[np.mean(hit[c]) for c in range(7)]
        res.append((np.mean(rec),rec))
    ba=[r[0] for r in res]; rec=np.mean([r[1] for r in res],0)
    return np.mean(ba),np.std(ba,ddof=1),rec
for per_batch in [1,8,32]:
    for m in [1,2,3,5,8]:
        mu,sd,rec=evaluate_rs(m,per_batch)
        print(f"sent/real={per_batch:3d} m={m}: BA {mu:.3f}±{sd:.3f}  "+" ".join(f"{C[i][:6]}={rec[i]:.2f}" for i in range(7)))
