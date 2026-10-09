import numpy as np
rng=np.random.default_rng(1); N=60000
def cov(r): R=np.array([[1,r],[r,1]]); return np.kron(R,R)
def sample(r,n,m):
    hw=(rng.standard_normal((n,m,2,2))+1j*rng.standard_normal((n,m,2,2)))/np.sqrt(2)
    R=np.array([[1,r],[r,1]]); w,v=np.linalg.eigh(R); S=(v*np.sqrt(w))@v.T
    return (S@hw@S).reshape(n,m,4)
def ll(h,C):  # sum over m realizations
    Ci=np.linalg.inv(C); return (-np.einsum('nmi,ij,nmj->nm',h.conj(),Ci,h).real-np.log(np.linalg.det(C))).sum(1)
rs=[0.3,0.6,0.9]
for m in [1,2,3,5,8]:
    a=sample(0,N,m); b=np.concatenate([sample(r,N//3,m) for r in rs])
    mix=lambda h: np.logaddexp.reduce([ll(h,cov(r)) for r in rs],axis=0)-np.log(3)
    r0=np.mean(ll(a,np.eye(4))>mix(a)); r1=np.mean(mix(b)>ll(b,np.eye(4)))
    per={r:np.mean(mix(sample(r,N,m))>ll(sample(r,N,m)*0+sample(r,N,m),np.eye(4))) for r in []}
    print(f"m={m}: genie-H Bayes nominal-vs-correlation: nominal {r0:.3f}, corr {r1:.3f}, balanced {(r0+r1)/2:.3f}")
