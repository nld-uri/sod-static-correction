"""Tutorial example: SOD as a smooth-trend extractor with full ground truth.
   x = A s + noise, 6 channels, 6 sources s_i = a_i cos(k_i pi t) on the midpoint grid t_j=(j+1/2)/N.
   These are DCT-II vectors, so <s s^T> and <sd sd^T> are exactly diagonal: the exactness criterion of
   SOD-theory holds with no dynamics. Then lambda_i = <s_i^2>/<sd_i^2> (Rice), known in closed form:
     continuous derivative: 1/(k pi)^2 ;  central differences: (dt / sin(k pi dt))^2 (interior points).
   k=1 is a monotone drift (half cycle), k=6 a slow cycle; k=80..450 are fast, large-variance sources.
   Case B replaces the drift by a tanh step, which is correlated with the slow cycle: exactness fails."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'SOD-toolbox', 'python'))
import sodtoolbox as _st
import numpy as np
from scipy.linalg import eigh, eig
rng=np.random.default_rng(3)
N=4000; dt=1/N; t=(np.arange(N)+0.5)*dt
k=np.array([1,6,80,160,300,450]); amp=np.array([1.0,0.5,3.0,2.0,1.5,1.0])
S=amp[:,None]*np.cos(np.pi*k[:,None]*t)
Q1,_=np.linalg.qr(rng.normal(size=(6,6))); Q2,_=np.linalg.qr(rng.normal(size=(6,6)))
A=Q1@np.diag(np.linspace(1,3,6))@Q2            # mixing with cond(A)=3: an oracle unmixing stays well posed
lam_cont=1/(np.pi*k)**2; lam_cd=(dt/np.sin(np.pi*k*dt))**2
def sod(X,lag=0,h=1):
    """SOD with a step-h central difference, v_j=(x_{j+h}-x_{j-h})/(2h dt) (numpy.gradient for h = 1, keeping every
       sample), centred X, optional lag-`lag` symmetric covariances (SOD Toolbox, sodtoolbox.sod). lag >= 2h+1 removes
       white sensor noise from both covariances; with a lag the pencil may be indefinite in fast directions, which
       then sort as roughest. Returns SOVs 1/mu, unit-norm covectors, smooth coordinates."""
    R=_st.sod(X,dt=dt,deriv='gradient' if h==1 else 'central',h=h,lag=lag,center=True,tol=None)
    P=R.Psi/np.linalg.norm(R.Psi,axis=0); return R.sov,P,R.Q/np.linalg.norm(R.Psi,axis=0)
def pod(X): X=X-X.mean(0); w,U=np.linalg.eigh(X.T@X); return X@U[:,::-1]
def corr(a,b): return abs(np.corrcoef(a,b)[0,1])
def score(Y,Sref,nk=2):
    h=(Sref.shape[1]-len(Y))//2; Sr=Sref[:,h:Sref.shape[1]-h] if h else Sref
    return [max(corr(Y[:,j],Sr[i]) for j in range(nk)) for i in (0,1)]
METHODS=[("SOD",0,1),("SOD, h=15",0,15),("SOD, h=15, lag 31",31,15)]
NOISE=[0,1e-3,3e-3,1e-2,3e-2,1e-1]
def run_noise(Sref,label,reps=20):
    X0=(A@Sref).T; rms=np.sqrt((X0**2).mean(0)); Ai=np.linalg.inv(A)
    print(f"--- {label}  (entries: 1-|corr| for s1, s2)")
    print(f"{'noise':>6} "+" ".join(f"{m[0]:>20}" for m in METHODS)+f" {'POD':>16} {'oracle A^-1':>16}")
    rows=[]
    for nz in NOISE:
        r=[]
        for rep in range(reps if nz else 1):
            X=X0+nz*rms*rng.normal(size=X0.shape); v=[]
            for _,lag,h in METHODS: v+=score(sod(X,lag,h)[2],Sref)
            v+=score(pod(X),Sref); Yo=X@Ai.T; v+=[corr(Yo[:,i],Sref[i]) for i in (0,1)]
            r.append(v)
        m=1-np.median(r,0); rows.append([nz,*m])
        print(f"{nz:6.0e} "+" ".join(f"{m[2*i]:9.1e} {m[2*i+1]:9.1e} " for i in range(len(METHODS)+2)))
    return np.array(rows)
if __name__=="__main__":
    print("max offdiag corr <ss>:",np.max(abs(np.corrcoef(S)-np.eye(6))),
          " <vv>:",np.max(abs(np.corrcoef(np.gradient(S,dt,axis=1))-np.eye(6))))
    l,P,Y=sod((A@S).T)
    print("lambda SOD      :",l); print("lambda CD theory:",lam_cd); print("lambda continuous:",lam_cont)
    print("variance share of slow sources:",(amp[:2]**2).sum()/(amp**2).sum())
    print("table entries are 1-|corr|")
    RA=run_noise(S,"case A: exact (DCT sources)")
    SB=S.copy(); SB[0]=np.tanh((t-0.55)/0.08); SB[0]-=SB[0].mean()
    print("case B: corr(s1,s2)=",corr(SB[0],SB[1])," corr(sd1,sd2)=",corr(np.gradient(SB[0]),np.gradient(SB[1])))
    VB=np.gradient(SB,dt,axis=1); lamB=(SB[0]**2).mean()/(VB[0]**2).mean()
    lB,PB,YB=sod((A@SB).T); print("case B clean: lambda1/lamB-1 =",lB[0]/lamB-1," 1-corr:",1-np.array(score(YB,SB)))
    RB=run_noise(SB,"case B: tanh drift, correlated with slow cycle")
    np.savez('figdata_trend.npz',t=t,S=S,SB=SB,A=A,k=k,amp=amp,RA=RA,RB=RB,lam_cd=lam_cd)
