"""SOD-ROM: sensitivity of SOD modes to the retained data rank (POD truncation tolerance), beam, light damping,
   single-point random excitation. Prints POD spectrum, SOD frequencies and best MAC per tolerance, and mode energies."""
import numpy as np, sys; 
from scipy.linalg import eigh
from systems import beam
from romlib import *
rng=np.random.default_rng(11); S=beam(); M,K,B=S['M'],S['K'],S['B']; C=S['C']['light']; dt=1e-3; N=200000
X,V=simulate(M,K,C,B,band_noise(N,dt,250.0,2,rng)*np.r_[1,0],dt)
w2,P=eigh(K,M); w=np.sqrt(w2)
ev=np.linalg.eigvalsh(X.T@X); print("POD eigenvalues / max:",np.array2string(ev[::-1][:16]/ev.max(),precision=1))
for tol in [1e-6,1e-8,1e-10,1e-12,1e-13,1e-14]:
    try:
        Phi,ws=sod(X,V,6,tol)
        mac=[max((Phi[:,i]@M@P[:,j])**2/((Phi[:,i]@M@Phi[:,i])*(P[:,j]@M@P[:,j])) for j in range(10)) for i in range(6)]
        print(f"tol={tol:.0e} rank={np.sum(ev>tol*ev.max())}: SOD freq {np.round(ws,1)}  best MAC {np.round(mac,3)}")
    except Exception as e: print(tol,e)
print("energy fraction of each SOD mode (||phi_i||^2 var(q_i) / total):")
for tol in [1e-10,1e-14]:
    Sx=X.T@X; ww,U=eigh(Sx); U=U[:,ww>tol*ww.max()]; Xp=X@U; Vp=V@U; mu,Psi=eigh(Vp.T@Vp,Xp.T@Xp)
    Q=Xp@Psi; Phi=np.linalg.inv(Psi).T   # Xp = Q Phi^T
    e=np.sum(Q**2,0)*np.sum(Phi**2,0); e/=np.sum(Xp**2)
    print(f"  tol={tol:.0e}: freq {np.round(np.sqrt(mu[:8]),1)}  energy {np.array2string(e[:8],precision=1)}")
