"""Diagnostic's own nominal-vs-correlation binary performance (same task as the genie test)."""
import csv,sys,glob
from collections import defaultdict
import numpy as np
C=["normal","low_snr","imperfect_csi","spatial_correlation","literal","mismatch","ood"]
iN,iC=0,3
blocks=defaultdict(list)
for p in sorted(glob.glob(sys.argv[1]+'/fusion_probs_seed*.csv')):
    for r in csv.DictReader(open(p)):
        if r['label'] not in ('normal','spatial_correlation'): continue
        lr=np.log(max(float(r['p_spatial_correlation']),1e-12))-np.log(max(float(r['p_normal']),1e-12))
        blocks[(int(r['held_out_seed']),r['path'])].append((int(r['index']),r['label'],r['severity'],lr))
seeds=sorted({k[0] for k in blocks})
for k in [1,8,32,128]:
    res=[];sev_rec=defaultdict(list)
    for s in seeds:
        hitN=[];hitC=[];bysev=defaultdict(list)
        for (ss,_),g in blocks.items():
            if ss!=s: continue
            g.sort()
            for st in range(0,len(g)-k+1,k):
                ch=g[st:st+k]; dec=np.mean([x[3] for x in ch])>0   # sum of log-ratios (product rule on the two classes)
                if ch[0][1]=='normal': hitN.append(not dec)
                else: hitC.append(dec); bysev[ch[0][2]].append(dec)
        res.append((np.mean(hitN),np.mean(hitC)))
        for sv,v in bysev.items(): sev_rec[sv].append(np.mean(v))
    rn=np.mean([r[0] for r in res]); rc=np.mean([r[1] for r in res]); ba=[(a+b)/2 for a,b in res]
    print(f"k={k:3d}: nominal recall {rn:.3f}, correlation recall {rc:.3f}, binary balanced {np.mean(ba):.3f}±{np.std(ba,ddof=1):.3f} | corr recall by severity "+", ".join(f"{sv}:{np.mean(v):.3f}" for sv,v in sorted(sev_rec.items())))
