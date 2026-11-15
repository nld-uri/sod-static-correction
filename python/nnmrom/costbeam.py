"""COST F3 nonlinear cantilever benchmark (Peeters et al. 2009, Sec. 4.4, Table 1): L=0.7 m, b=h=0.014 m,
   E=2.05e11 Pa, rho=7800 kg/m^3, 20 Euler-Bernoulli elements, cubic spring k3=6e9 N/m^3 from the tip to ground."""
import numpy as np
from scipy.linalg import eigh
def costbeam(Ne=20,zeta=0.0):
    L=0.7; b=h=0.014; E=2.05e11; rho=7800.0; A=b*h; I=b*h**3/12; EI=E*I; rA=rho*A; le=L/Ne; n=2*(Ne+1)
    ke=EI/le**3*np.array([[12,6*le,-12,6*le],[6*le,4*le*le,-6*le,2*le*le],[-12,-6*le,12,-6*le],[6*le,2*le*le,-6*le,4*le*le]])
    me=rA*le/420*np.array([[156,22*le,54,-13*le],[22*le,4*le*le,13*le,-3*le*le],[54,13*le,156,-22*le],[-13*le,-3*le*le,-22*le,4*le*le]])
    K=np.zeros((n,n)); M=np.zeros((n,n))
    for e in range(Ne): i=slice(2*e,2*e+4); K[i,i]+=ke; M[i,i]+=me
    K=K[2:,2:]; M=M[2:,2:]; n-=2; tip=n-2
    w=np.sqrt(eigh(K,M,eigvals_only=True))
    C=np.zeros((n,n))
    if zeta>0:
        a,bb=np.linalg.solve([[1/(2*w[0]),w[0]/2],[1/(2*w[1]),w[1]/2]],[zeta,zeta]); C=a*M+bb*K
    return dict(M=M,K=K,C=C,tip=tip,k3=6e9,n=n,w=w,L=L)
if __name__=="__main__":
    S=costbeam(); print("n=",S['n']," f_1..4 (Hz):",np.round(S['w'][:4]/2/np.pi,2)," f_max=%.3g Hz"%(S['w'][-1]/2/np.pi))
