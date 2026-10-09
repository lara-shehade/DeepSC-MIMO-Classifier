import numpy as np
rng=np.random.default_rng(0); N=400000
def logpdf(h,Cinv,logdet):  # complex Gaussian CN(0,C), h: N x 4
    q=np.einsum('ni,ij,nj->n',h.conj(),Cinv,h).real
    return -q-logdet
def cov(r):
    R=np.array([[1,r],[r,1]]); return np.kron(R,R)  # vec(R^1/2 Hw R^1/2) covariance = R^T ⊗ R (R symmetric)
def sample(r,n):
    hw=(rng.standard_normal((n,2,2))+1j*rng.standard_normal((n,2,2)))/np.sqrt(2)
    R=np.array([[1,r],[r,1]]); w,v=np.linalg.eigh(R); S=(v*np.sqrt(w))@v.T
    return (S@hw@S).reshape(n,4)
C0=np.eye(4); out={}
for r in [0.3,0.6,0.9]:
    C1=cov(r); i0,i1=np.linalg.inv(C0),np.linalg.inv(C1); d0,d1=np.log(np.linalg.det(C0)),np.log(np.linalg.det(C1))
    a=sample(0,N); b=sample(r,N)
    acc0=np.mean(logpdf(a,i0,d0)>logpdf(a,i1,d1)); acc1=np.mean(logpdf(b,i1,d1)>logpdf(b,i0,d0))
    print(f"rho_s={r}: oracle-H Bayes test nominal-vs-corr: recall nominal {acc0:.3f}, recall corr {acc1:.3f}, balanced {0.5*(acc0+acc1):.3f}")
# mixture of the three severities vs nominal (as in the dataset, equal thirds)
Cs=[cov(r) for r in [0.3,0.6,0.9]]
def mixlog(h): 
    return np.logaddexp.reduce([logpdf(h,np.linalg.inv(C),np.log(np.linalg.det(C))) for C in Cs],axis=0)-np.log(3)
a=sample(0,N); b=np.concatenate([sample(r,N//3) for r in [0.3,0.6,0.9]])
acc0=np.mean(logpdf(a,np.eye(4),0)>mixlog(a)); acc1=np.mean(mixlog(b)>logpdf(b,np.eye(4),0))
print(f"mixture: recall nominal {acc0:.3f}, recall corr {acc1:.3f}, balanced {0.5*(acc0+acc1):.3f}")
