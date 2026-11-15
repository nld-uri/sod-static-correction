"""SOD-NNM ROM study: LNM, POD and SOD Galerkin ROMs of the COST F3 nonlinear cantilever (cubic tip spring).
   Training: (i) damped decay (Rayleigh zeta=0.01, modes 1-2) from the static deflection under a 30 N tip load,
   1.5 s, sweeping energy down through the backbone; (ii) conservative oscillation from the same state, 0.3 s (one energy).
   Tests (conservative): free vibration from static deflections under tip loads 3, 10, 20, 30 N, 0.3 s;
   metrics: tip frequency error vs full model, displacement error over the first 0.1 s.
   Integrator for all models: average-acceleration Newmark, dt = 2e-5 s (nlsim.newmark)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'SOD-toolbox', 'python'))
import sodtoolbox as _st
import numpy as np, pickle, time, sys
from scipy.linalg import eigh
from costbeam import costbeam
from nlsim import *
def sod(X, V, r, tol=1e-10): return _st.sod_modes(X, V, r, tol)
def pod(X,r): w,U=eigh(X.T@X); return U[:,::-1][:,:r]
S=costbeam(); Sd=costbeam(zeta=0.01); n=S['n']; pt=np.zeros(n); pt[S['tip']]=1; dt=2e-5; sub=5
w2,PL=eigh(S['K'],S['M'])
x30=static_deflection(S,30.0)
_,Xd,Vd=newmark(Sd['M'],Sd['K'],Sd['C'],pt,S['k3'],x30,np.zeros(n),1.5,dt); Xd,Vd=Xd[::sub],Vd[::sub]
_,Xc,Vc=newmark(S['M'],S['K'],S['C'],pt,S['k3'],x30,np.zeros(n),0.3,dt); Xc,Vc=Xc[::sub],Vc[::sub]
fs_d=sod(Xd,Vd,4)[1]/2/np.pi; fs_c=sod(Xc,Vc,4)[1]/2/np.pi
print("SOD frequencies (Hz), decay data:",np.round(fs_d,2),"  conservative 30 N:",np.round(fs_c,2),"  LNM:",np.round(np.sqrt(w2[:4])/2/np.pi,2))
bases={'LNM':lambda r: PL[:,:r],'SOD (decay)':lambda r: sod(Xd,Vd,r)[0],'POD (decay)':lambda r: pod(Xd,r),
       'SOD (30 N)':lambda r: sod(Xc,Vc,r)[0],'POD (30 N)':lambda r: pod(Xc,r)}
if __name__=='__main__':
    tests={}
    for P in [3.0,10.0,20.0,30.0]:
        x0=static_deflection(S,P); t,X,V=newmark(S['M'],S['K'],S['C'],pt,S['k3'],x0,np.zeros(n),0.3,dt)
        tests[P]=dict(x0=x0,t=t,X=X,f=dom_freq(t,X[:,S['tip']]),H=energy(S,X[:1],V[:1])[0])
        print(f"test P={P:g} N: H={tests[P]['H']:.2e} J, f_full={tests[P]['f']:.3f} Hz",flush=True)
    res={}
    for bn,fb in bases.items():
        for r in [1,2,3,4]:
            Phi=fb(r); Mr=Phi.T@S['M']@Phi; Kr=Phi.T@S['K']@Phi; Cr=np.zeros((r,r)); prt=Phi[S['tip']]; Pi=np.linalg.pinv(Phi)
            row={}
            for P,tt in tests.items():
                t,Q,_=newmark(Mr,Kr,Cr,prt,S['k3'],Pi@tt['x0'],np.zeros(r),0.3,dt); Xr=Q@Phi.T
                m=t<=0.1
                row[P]=dict(ferr=dom_freq(t,Xr[:,S['tip']])/tt['f']-1,err=np.linalg.norm(Xr[m]-tt['X'][m])/np.linalg.norm(tt['X'][m]))
            res[(bn,r)]=row
            print(f"{bn:12s} r={r}: "+"  ".join(f"P={P:g}: df/f={row[P]['ferr']:+.1e} e={row[P]['err']:.1e}" for P in tests),flush=True)
    pickle.dump(dict(res=res,tests={P:dict(f=v['f'],H=v['H']) for P,v in tests.items()},fs_d=fs_d,fs_c=fs_c),open('n1_results.pkl','wb'))
