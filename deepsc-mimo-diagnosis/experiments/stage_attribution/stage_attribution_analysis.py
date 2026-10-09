import json,statistics as st,itertools,math,sys
d=json.load(open(sys.argv[1]))
C=["normal","low_snr","imperfect_csi","spatial_correlation","literal","mismatch","ood"]
S=["R0","R1","R2","R3"]; seeds=d['physical_seeds']
CM={}
for r in d['folds']: CM[(frozenset(r['stages']),r['held_out_physical_seed'])]=r['confusion_matrix']
subsets=[frozenset(c) for n in range(1,5) for c in itertools.combinations(S,n)]
def f1(cm,k):
    tp=cm[k][k]; p=sum(cm[i][k] for i in range(7)); t=sum(cm[k]); return 2*tp/(p+t)
def rec(cm,k): return cm[k][k]/sum(cm[k])
def mf1(cm): return sum(f1(cm,k) for k in range(7))/7
def coarse(cm):
    G=[[0],[1,2,3],[4,5,6]]; return sum(sum(cm[i][j] for i in g for j in g)/sum(sum(cm[i]) for i in g) for g in G)/3
chance=1/7
def v(sub,seed,metric):
    if not sub: return chance
    return metric(CM[(sub,seed)])
metrics={'macroF1':mf1, **{c:(lambda k:(lambda cm:f1(cm,k)))(k) for k,c in enumerate(C)}}
name=lambda s:'+'.join(x for x in S if x in s)
ms=lambda xs:(st.mean(xs),st.stdev(xs))
out={}
# 1 per-class F1 & recall table
print("=== per-class F1 (mean) ; recall")
print(f"{'subset':12s}"+''.join(f"{c[:6]:>8s}" for c in C)+"   mF1   coarseBA")
for sb in subsets:
    row=[ms([f1(CM[(sb,s)],k) for s in seeds])[0] for k in range(7)]
    print(f"{name(sb):12s}"+''.join(f"{x:8.3f}" for x in row)+f"  {ms([mf1(CM[(sb,s)]) for s in seeds])[0]:.3f}  {ms([coarse(CM[(sb,s)]) for s in seeds])[0]:.3f}")
print()
for sb in subsets:
    row=[ms([rec(CM[(sb,s)],k) for s in seeds])[0] for k in range(7)]
    print(f"{name(sb):12s}"+''.join(f"{x:8.3f}" for x in row))
# 2 Shapley
def shap(stage,seed,metric):
    others=[x for x in S if x!=stage]; n=4; tot=0
    for r in range(4):
        for T in itertools.combinations(others,r):
            T=frozenset(T); w=math.factorial(r)*math.factorial(n-r-1)/math.factorial(n)
            tot+=w*(v(T|{stage},seed,metric)-v(T,seed,metric))
    return tot
print("\n=== Shapley (F1 above chance) per stage, mean±std over folds")
SH={}
for mn,m in metrics.items():
    vals={s_:ms([shap(s_,sd,m) for sd in seeds]) for s_ in S}
    SH[mn]=vals
    full=ms([m(CM[(frozenset(S),sd)]) for sd in seeds])[0]
    print(f"{mn:20s}"+''.join(f"  {s_}:{a:+.3f}±{b:.3f}" for s_,(a,b) in vals.items())+f"  full={full:.3f}")
# 3 Shapley interaction index for pairs
def inter(i,j,seed,metric):
    others=[x for x in S if x not in (i,j)]; n=4; tot=0
    for r in range(3):
        for T in itertools.combinations(others,r):
            T=frozenset(T); w=math.factorial(r)*math.factorial(n-r-2)/math.factorial(n-1)
            tot+=w*(v(T|{i,j},seed,metric)-v(T|{i},seed,metric)-v(T|{j},seed,metric)+v(T,seed,metric))
    return tot
print("\n=== Shapley interaction index (neg = redundant, pos = synergistic)")
for mn,m in metrics.items():
    print(f"{mn:20s}"+''.join(f"  {i}{j}:{ms([inter(i,j,sd,m) for sd in seeds])[0]:+.3f}" for i,j in itertools.combinations(S,2)))
# 4 marginal gain consistency: adding stage to each context
print("\n=== adding stage: #contexts (of 8) improving mean metric; #(context,fold) improving of 40; mean gain")
for mn,m in metrics.items():
    line=f"{mn:20s}"
    for s_ in S:
        others=[x for x in S if x!=s_]; ctx=[frozenset(T) for r in range(4) for T in itertools.combinations(others,r)]
        g=[[v(T|{s_},sd,m)-v(T,sd,m) for sd in seeds] for T in ctx]
        line+=f"  {s_}: {sum(st.mean(x)>0 for x in g)}/8 {sum(y>0 for x in g for y in x)}/40 {st.mean([st.mean(x) for x in g if True]):+.3f}"
    print(line)
# 5 confusion pairs (symmetric rate: (cm[a][b]+cm[b][a])/(n_a+n_b))
pairs=[(0,3),(0,4),(0,2),(1,2),(2,3),(1,3),(4,6),(0,6),(2,4)]
print("\n=== symmetric confusion rate %")
print(f"{'subset':12s}"+''.join(f"{C[a][:4]}-{C[b][:4]:>5s}" for a,b in pairs))
for sb in subsets:
    row=[]
    for a,b in pairs:
        row.append(st.mean([(CM[(sb,s)][a][b]+CM[(sb,s)][b][a])/(sum(CM[(sb,s)][a])+sum(CM[(sb,s)][b]))*100 for s in seeds]))
    print(f"{name(sb):12s}"+''.join(f"{x:10.1f}" for x in row))
# 6 directional confusion for full and singles: where do errors of each class go
print("\n=== row-normalized confusion (pooled folds) for singles and full")
for sb in [frozenset([x]) for x in S]+[frozenset(S),frozenset(["R0","R3"])]:
    tot=[[sum(CM[(sb,s)][i][j] for s in seeds) for j in range(7)] for i in range(7)]
    print(name(sb))
    for i in range(7):
        n=sum(tot[i]); print(f"   {C[i][:8]:8s}"+''.join(f"{tot[i][j]/n*100:6.1f}" for j in range(7)))
# 7 fold-level: per-seed macroF1 for each subset; hardest fold
print("\n=== per-fold macroF1")
for sb in subsets: print(f"{name(sb):12s}"+''.join(f"{mf1(CM[(sb,s)]):7.3f}" for s in seeds))
# 8 rank correlation across folds: does subset ranking stay consistent? Kendall W
import itertools as it
ranks=[]
for s in seeds:
    vals=[mf1(CM[(sb,s)]) for sb in subsets]; order=sorted(range(15),key=lambda i:-vals[i]); r=[0]*15
    for pos,i in enumerate(order): r[i]=pos+1
    ranks.append(r)
m=5;n=15;Rj=[sum(r[i] for r in ranks) for i in range(n)];Rbar=m*(n+1)/2
W=12*sum((x-Rbar)**2 for x in Rj)/(m**2*(n**3-n)); print("\nKendall W across folds:",round(W,3))
json.dump({k:{s_:list(v_) for s_,v_ in vv.items()} for k,vv in SH.items()},open(sys.argv[2],'w'))
