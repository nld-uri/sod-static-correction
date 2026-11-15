"""SOD-ROM: why SOD bases differ from LNMs. Mass-weighted MAC of SOD smooth modes vs LNMs, and SOD frequencies
   vs natural frequencies, for each training signal. Also the modal correlation of the training response
   (max |corr| between modal coordinates eta_i, eta_j, i<j<=6), which SOD-theory links to SOD mode mixing."""
import numpy as np, pickle
from scipy.linalg import eigh
from systems import shear_frame, beam
from romlib import *
rng=np.random.default_rng(11)
out={}
for S,p in [(shear_frame(),dict(dt=0.01,T=300,wc=23.0,w0=1.0)),(beam(),dict(dt=1e-3,T=200,wc=250.0,w0=2.0)),(beam(dashpot=(0.2,1.0)),dict(dt=1e-3,T=200,wc=250.0,w0=2.0))]:
    M,K,B=S['M'],S['K'],S['B']; n=len(M); N=int(p['T']/p['dt']); dt=p['dt']; nin=B.shape[1]
    w2,P=eigh(K,M); w=np.sqrt(w2)
    F0=(M@np.ones(n)) if nin==1 else (B@np.array([1.0,0.7])); xs=np.linalg.solve(K,F0); xs/=abs(xs).max()
    C=list(S['C'].values())[0]
    train={'random':(band_noise(N,dt,p['wc'],nin,rng)*np.r_[1,np.zeros(nin-1)],None),
           'free decay':(np.zeros((N,nin)),np.r_[xs,np.zeros(n)]),'swept sine':(chirp(N,dt,p['w0'],p['wc'],nin,0),None)}
    for tr,(u,y0) in train.items():
        X,V=simulate(M,K,C,B,u,dt,y0); Phi,ws=sod(X,V,8)
        Mac=np.array([[ (Phi[:,i]@M@P[:,j])**2/((Phi[:,i]@M@Phi[:,i])*(P[:,j]@M@P[:,j])) for j in range(10)] for i in range(Phi.shape[1])])
        eta=X@M@P[:,:6]; Cc=np.corrcoef(eta.T); cmax=np.max(abs(Cc-np.eye(6)))
        out[(S['name'],tr)]=dict(MAC=Mac,ws=ws,w=w[:10],cmax=cmax)
        best=Mac.argmax(1); print(f"{S['name']:30s} {tr:11s} best LNM: "+" ".join(f"{best[i]+1}({Mac[i,best[i]]:.2f})" for i in range(len(best)))+
              "  | w_SOD: "+" ".join(f"{ws[i]:.1f}" for i in range(len(best)))+f" | max modal corr {cmax:.2f}")
pickle.dump(out,open('c3_mac.pkl','wb'))
