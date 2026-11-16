"""SOD-ROM: initial-state projection for state ROMs on the equal-energy test decay (S3, state dimension 8, static-release
   training). Energy-orthogonal projection q0 = (Phi^T W Phi)^-1 Phi^T W y0 against data covectors q0 = Psi^T y0;
   complex-mode truncation (left eigenvectors) for reference. Output: c17_out.txt."""
import numpy as np
from scipy.linalg import eigh
from systems import beam, state
from romlib2 import *
dt=1e-3; N=200000
for cname in ['dashpot','dashpot_heavy']:
    S=beam(dashpot=(0.2,1.0)); M,K,B=S['M'],S['K'],S['B']; C=S['C'][cname]; n=len(M); A,Bs=state(M,K,C,B)
    WE=np.block([[K,np.zeros((n,n))],[np.zeros((n,n)),M]])
    wL2,PL=eigh(K,M); kEE=np.where(np.sqrt(wL2)<np.pi/(4*dt))[0]
    x=PL[:,kEE]@((-1.0)**np.arange(len(kEE))/np.sqrt(wL2[kEE])); x/=abs(x).max(); y0=np.r_[x,np.zeros(n)]
    xs=np.linalg.solve(K,B@np.array([1,0.7])); xs/=abs(xs).max()
    Y=simulate_state(A,Bs,np.zeros((N,2)),dt,np.r_[xs,np.zeros(n)]); Psi,Phi,_=state_sod(Y,Y@A.T,8); del Y
    Xf=simulate_state(A,Bs,np.zeros((N,2)),dt,y0)[:,:n]
    T=Phi.T@WE; G=T@Phi; Ar=np.linalg.solve(G,T@A@Phi); Br=np.linalg.solve(G,T@Bs)
    for lab,q0 in [('energy-orthogonal',np.linalg.solve(G,T@y0)),('data covectors',Psi.T@y0)]:
        Q=simulate_state(Ar,Br,np.zeros((N,2)),dt,q0); print(cname,lab,f"{merr(Q@Phi[:n].T,Xf,M):.2e}")
    PsiC,PhiC=complex_modes(A,8); print(cname,'CM',f"{merr(rom_state(A,Bs,PsiC,PhiC,np.zeros((N,2)),dt,y0,n)[0],Xf,M):.2e}")
