"""Nested class-bias calibration (vector scaling with identity weights, Guo et al. 2017) for windowed diagnosis.
Bias vector b is fit on four folds by maximizing balanced accuracy of windowed decisions (coordinate search),
then applied to the held-out fold. Window = sentences from 1 or several realizations of the same run."""
import csv,sys,glob,itertools
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
    g.sort(); runs[(s,run)][b]=(g[0][1],np.log(np.clip(np.array([x[2] for x in g]),1e-12,1)))
seeds=sorted({k[0] for k in runs})
rng=np.random.default_rng(11)
def make_windows(seed,m,per_batch,draws=60):
    """returns labels and list of (n_sent x 7) log-prob arrays"""
    Y=[];W=[]
    for (s,run),bat in runs.items():
        if s!=seed: continue
        ids=sorted(bat); lab=bat[ids[0]][0]
        combos=list(itertools.combinations(ids,m))
        pick=combos if len(combos)<=draws else [combos[i] for i in rng.choice(len(combos),draws,replace=False)]
        for ch in pick:
            for rep in range(1 if per_batch>=128 else max(1,min(4,128//per_batch))):
                W.append(np.concatenate([bat[b][1][rng.permutation(128)[:per_batch]] for b in ch])); Y.append(lab)
    return np.array(Y),W
def decide(W,b):  # mean of calibrated probabilities (sum rule) after bias on log-probs
    out=[]
    for L in W:
        z=L+b; z=z-z.max(1,keepdims=True); p=np.exp(z); p/=p.sum(1,keepdims=True); out.append(p.mean(0).argmax())
    return np.array(out)
def bacc(y,pred): return np.mean([np.mean(pred[y==c]==c) for c in range(7)]), [np.mean(pred[y==c]==c) for c in range(7)]
def fit_bias(data):
    b=np.zeros(7); grid=np.linspace(-4,4,33)
    best=np.mean([bacc(y,decide(W,b))[0] for y,W in data])
    for it in range(2):
        for c in range(7):
            for g in grid:
                t=b.copy(); t[c]=g
                v=np.mean([bacc(y,decide(W,t))[0] for y,W in data])
                if v>best+1e-9: best=v; b=t
    return b
for m,per in [(1,1),(1,8),(1,32),(5,8),(5,32)]:
    data={s:make_windows(s,m,per) for s in seeds}
    base=[];cal=[];recs=[]
    for s in seeds:
        b=fit_bias([data[t] for t in seeds if t!=s])
        y,W=data[s]
        base.append(bacc(y,decide(W,np.zeros(7)))[0]); v,r=bacc(y,decide(W,b)); cal.append(v); recs.append(r)
    d=np.array(cal)-np.array(base); rec=np.mean(recs,0)
    print(f"m={m} sent/real={per:3d}: uncal {np.mean(base):.3f}±{np.std(base,ddof=1):.3f} | nested bias-cal {np.mean(cal):.3f}±{np.std(cal,ddof=1):.3f} | gain {d.mean():+.3f} ({(d>0).sum()}/5) | "+" ".join(f"{C[i][:6]}={rec[i]:.2f}" for i in range(7)),flush=True)
