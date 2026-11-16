"""SOD-ROM benchmark systems (nondimensional). Each returns M, K, C dictionaries for damping cases and an input map.
   S1  10-story shear frame, base excitation (force -M 1 xg''), tapered story stiffness.
   S2  Euler-Bernoulli cantilever, Ne Hermite elements (w, theta per node), consistent mass; tip/mid force inputs.
   S3  S2 + discrete viscous dashpot from the tip translation to ground (non-proportional damping).
   Damping cases: 'light' Rayleigh (zeta=0.02 at modes 1 and 4), 'heavy' Rayleigh (zeta=0.20 at modes 1 and 4),
   'dashpot' (S3 only: light Rayleigh + tip dashpot), 'dashpot_heavy' (larger dashpot)."""
import numpy as np
from scipy.linalg import eigh
def rayleigh(M,K,zeta,modes=(0,3)):
    w=np.sqrt(eigh(K,M,eigvals_only=True)); w1,w2=w[modes[0]],w[modes[1]]
    a,b=np.linalg.solve([[1/(2*w1),w1/2],[1/(2*w2),w2/2]],[zeta,zeta]); return a*M+b*K
def shear_frame(n=10):
    m=np.ones(n); k=np.linspace(1.0,0.55,n)*(2*np.pi)**2*n   # story 1 stiffest; scaled so omega_1 ~ O(1..2)
    M=np.diag(m); K=np.zeros((n,n))
    for i in range(n):
        K[i,i]+=k[i]
        if i+1<n: K[i,i]+=k[i+1]; K[i,i+1]-=k[i+1]; K[i+1,i]-=k[i+1]
    B=-M@np.ones((n,1))                                      # base acceleration input
    return dict(name='shear frame',M=M,K=K,B=B,inputs=['base'],out=n-1,k=k,
                C={'light':rayleigh(M,K,0.02),'heavy':rayleigh(M,K,0.20)})
def beam(Ne=30,L=1.0,EI=1.0,rhoA=1.0,dashpot=None):
    h=L/Ne; n=2*(Ne+1)
    ke=EI/h**3*np.array([[12,6*h,-12,6*h],[6*h,4*h*h,-6*h,2*h*h],[-12,-6*h,12,-6*h],[6*h,2*h*h,-6*h,4*h*h]])
    me=rhoA*h/420*np.array([[156,22*h,54,-13*h],[22*h,4*h*h,13*h,-3*h*h],[54,13*h,156,-22*h],[-13*h,-3*h*h,-22*h,4*h*h]])
    K=np.zeros((n,n)); M=np.zeros((n,n))
    for e in range(Ne):
        i=slice(2*e,2*e+4); K[i,i]+=ke; M[i,i]+=me
    K=K[2:,2:]; M=M[2:,2:]; n-=2                             # clamp node 0
    tip=n-2; mid=2*(Ne//2)-2                                 # translational DOFs
    B=np.zeros((n,2)); B[tip,0]=1; B[mid,1]=1
    Cl=rayleigh(M,K,0.02); C={'light':Cl,'heavy':rayleigh(M,K,0.20)}
    out=dict(name='cantilever beam',M=M,K=K,B=B,inputs=['tip','mid'],out=tip,tip=tip,mid=mid,Ne=Ne,C=C)
    if dashpot is not None:
        D=np.zeros((n,n)); D[tip,tip]=1.0
        out['name']='cantilever beam + tip dashpot'; out['C']={'dashpot':Cl+dashpot[0]*D,'dashpot_heavy':Cl+dashpot[1]*D}
        out['cd']=dashpot
    return out
def state(M,K,C,B):
    n=len(M); Mi=np.linalg.inv(M)
    A=np.block([[np.zeros((n,n)),np.eye(n)],[-Mi@K,-Mi@C]]); Bs=np.vstack([np.zeros_like(B),Mi@B]); return A,Bs
def modal_damping(M,K,C):
    A,_=state(M,K,C,np.zeros((len(M),1))); s=np.linalg.eigvals(A); s=s[s.imag>0]; s=s[np.argsort(abs(s))]
    return abs(s),-s.real/abs(s)
if __name__=="__main__":
    S1=shear_frame(); S2=beam(); 
    for S in [S1,S2]:
        w=np.sqrt(eigh(S['K'],S['M'],eigvals_only=True)); print(S['name'],'n=',len(S['M']),'omega 1..6:',np.round(w[:6],3),'omega_max=%.3g'%w[-1])
        for c,Cm in S['C'].items():
            wn,z=modal_damping(S['M'],S['K'],Cm); print('   ',c,'zeta 1..6:',np.round(z[:6],4))
    # choose tip dashpot coefficients: scan the first-mode damping ratio
    for cd in [0.05,0.1,0.2,0.5,1,2,5]:
        S3=beam(dashpot=(cd,cd)); wn,z=modal_damping(S3['M'],S3['K'],S3['C']['dashpot']); print('cd=%g zeta1..4'%cd,np.round(z[:4],4))
