"""SOD-ROM core: exact ZOH simulation, excitation signals, LNM/POD/SOD bases, Galerkin ROMs, error metrics.
   SOD, POD, LNM, Galerkin and MAC delegate to the SOD Toolbox."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'SOD-toolbox', 'python'))
import sodtoolbox as _st
import numpy as np
from scipy.linalg import eigh, expm, solve
from scipy.signal import butter, sosfiltfilt
from systems import state
def zoh(A,Bs,dt):
    n=len(A); m=Bs.shape[1]; E=expm(np.block([[A,Bs],[np.zeros((m,n+m))]])*dt); return E[:n,:n],E[:n,n:]
def simulate(M,K,C,B,u,dt,y0=None):
    """exact for piecewise-constant input; returns X (N x n), V (N x n)"""
    A,Bs=state(M,K,C,B); Ad,Bd=zoh(A,Bs,dt); n=len(M); N=len(u)
    Y=np.zeros((N,2*n)); y=np.zeros(2*n) if y0 is None else y0.copy(); BdU=u@Bd.T
    for k in range(N): Y[k]=y; y=Ad@y+BdU[k]
    return Y[:,:n],Y[:,n:]
def band_noise(N,dt,wc,nin,rng):
    sos=butter(6,wc/(np.pi/dt),output='sos'); u=sosfiltfilt(sos,rng.standard_normal((N,nin)),axis=0); return u/u.std(0)
def chirp(N,dt,w0,w1,nin,which=0):
    t=np.arange(N)*dt; T=t[-1]; ph=w0*t+(w1-w0)*t**2/(2*T); u=np.zeros((N,nin)); u[:,which]=np.sin(ph); return u
def lnm(M,K,r): return _st.lnm(M,K,r)
def pod(X,r): return _st.pod(X,r,tol=None)[0]
def sod(X,V,r,tol=1e-10):
    """smooth modes (unit norm) and SOD frequencies (rad/s), smoothest first"""
    return _st.sod_modes(X,V,r,tol)
def galerkin(M,K,C,B,Phi): return _st.galerkin(M,K,C,B,Phi)
def rom_response(M,K,C,B,Phi,u,dt,y0=None):
    Mr,Kr,Cr,Br=galerkin(M,K,C,B,Phi)
    q0=None
    if y0 is not None:
        n=len(M); P=np.linalg.pinv(Phi); q0=np.r_[P@y0[:n],P@y0[n:]]
    Q,Qd=simulate(Mr,Kr,Cr,Br,u,dt,q0); return Q@Phi.T
def relerr(Xr,X): return np.linalg.norm(Xr-X)/np.linalg.norm(X)
def frf(M,K,C,B,out,w,inp=0):
    return np.array([solve(-wi**2*M+1j*wi*C+K,B[:,inp])[out] for wi in w])
def frf_rom(M,K,C,B,Phi,out,w,inp=0):
    Mr,Kr,Cr,Br=galerkin(M,K,C,B,Phi); return np.array([Phi[out]@solve(-wi**2*Mr+1j*wi*Cr+Kr,Br[:,inp]) for wi in w])
def mac(a,b): return _st.mac(a,b)
