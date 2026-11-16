"""SOD-ROM: state-space SOD ROM, Petrov-Galerkin with data covectors vs energy-weighted Galerkin on the smooth-mode subspace.
   S3 (c_d=0.2), basis from the free decay; tests: free decay (training IC) and random tip load. Shows why the PG ROM fails:
   data covectors are undetermined outside the data span and couple the input by 1e3-1e5."""
import numpy as np, sys; sys.path.insert(0, ".")
from systems import beam, state; from romlib2 import *
S=beam(dashpot=(0.2,1.0)); M,K,B=S['M'],S['K'],S['B']; C=S['C']['dashpot']; A,Bs=state(M,K,C,B); n=60; dt=1e-3; N=200000
xs=np.linalg.solve(K,B@np.array([1,0.7])); xs/=abs(xs).max()
Y=simulate_state(A,Bs,np.zeros((N,2)),dt,np.r_[xs,np.zeros(n)]); Yd=Y@A.T
Psi,Phi,w=state_sod(Y,Yd,8); print('sSOD freqs',np.round(w,3))
s=np.linalg.eigvals(A); s=s[np.argsort(abs(s))]; print('|s| slowest',np.round(abs(s[:8]),3))
print('Psi^T Phi', np.abs(Psi.T@Phi-np.eye(8)).max())
Ar=Psi.T@A@Phi; print('Ar eig',np.round(np.sort_complex(np.linalg.eigvals(Ar)),3))
print('A eig slowest', np.round(s[:8],3))
Xr,mx=rom_state(A,Bs,Psi[:,:6],Phi[:,:6],np.zeros((N,2)),dt,np.r_[xs,np.zeros(n)],n)
print('free decay self-test err', merr(Xr,Y[:,:n],M), mx)
q=Y@Psi; print('data q std',q.std(0)); Q=Xr  # 
print(np.abs(Y[0]-Phi@ (Psi.T@Y[0])).max(), np.abs(Y[0]).max())
W=np.block([[K,np.zeros((n,n))],[np.zeros((n,n)),M]])
def rom_gal(Phi,u,y0):
    T=Phi.T@W; Ar=np.linalg.solve(T@Phi,T@A@Phi); Br=np.linalg.solve(T@Phi,T@Bs); q0=None if y0 is None else np.linalg.solve(T@Phi,T@y0)
    return (simulate_state(Ar,Br,u,dt,q0)@Phi.T)[:,:n], np.max(np.linalg.eigvals(Ar).real)
rng=np.random.default_rng(1); from romlib import band_noise
u=band_noise(N,dt,250.0,2,rng)*np.r_[1,0]; Yt=simulate_state(A,Bs,u,dt)
for k in [2,4,6,8]:
    e1=merr(rom_state(A,Bs,Psi[:,:k],Phi[:,:k],u,dt,None,n)[0],Yt[:,:n],M); e2,mx=rom_gal(Phi[:,:k],u,None); e2=merr(e2,Yt[:,:n],M)
    f1=merr(rom_state(A,Bs,Psi[:,:k],Phi[:,:k],np.zeros((N,2)),dt,np.r_[xs,np.zeros(n)],n)[0],Y[:,:n],M); f2=merr(rom_gal(Phi[:,:k],np.zeros((N,2)),np.r_[xs,np.zeros(n)])[0],Y[:,:n],M)
    print(k,'rand: PG',f'{e1:.1e}','energy-Galerkin',f'{e2:.1e}',f'(maxRe {mx:.1e})','| free: PG',f'{f1:.1e}','EG',f'{f2:.1e}')
print('Psi^T Bs norms',np.linalg.norm(Psi[:,:8].T@Bs,axis=1))
