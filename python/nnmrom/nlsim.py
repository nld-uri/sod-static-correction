"""Simulation helpers for the COST beam: full model and Galerkin ROMs with the cubic tip spring."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
def make_rhs(M,K,C,phi_tip,k3,Phi=None):
    """phi_tip: row selecting the tip translation in the (reduced) coordinates."""
    Mi=np.linalg.inv(M); A21=-Mi@K; A22=-Mi@C; g=Mi@phi_tip; n=len(M)
    def f(t,y):
        xt=phi_tip@y[:n]; return np.r_[y[n:],A21@y[:n]+A22@y[n:]-g*(k3*xt**3)]
    def jac(t,y):
        xt=phi_tip@y[:n]; return np.block([[np.zeros((n,n)),np.eye(n)],[A21-3*k3*xt**2*np.outer(g,phi_tip),A22]])
    return f,jac
def static_deflection(S,P):
    e=np.zeros(S['n']); e[S['tip']]=1
    x=np.linalg.solve(S['K'],P*e)
    x=fsolve(lambda x: S['K']@x+S['k3']*x[S['tip']]**3*e-P*e,x,fprime=lambda x: S['K']+3*S['k3']*x[S['tip']]**2*np.outer(e,e),xtol=1e-14)
    return x
def full_run(S,y0,T,dt,C=None,rtol=1e-9):
    e=np.zeros(S['n']); e[S['tip']]=1; f,jac=make_rhs(S['M'],S['K'],S['C'] if C is None else C,e,S['k3'])
    t=np.arange(0,T,dt); s=solve_ivp(f,[0,T],y0,method='Radau',jac=jac,rtol=rtol,atol=1e-12,t_eval=t)
    return t,s.y[:S['n']].T,s.y[S['n']:].T,s
def rom_run(S,Phi,y0,T,dt,C=None,rtol=1e-9,method='Radau'):
    Mr=Phi.T@S['M']@Phi; Kr=Phi.T@S['K']@Phi; Cr=Phi.T@(S['C'] if C is None else C)@Phi; pt=Phi[S['tip']]
    f,jac=make_rhs(Mr,Kr,Cr,pt,S['k3']); r=Phi.shape[1]; P=np.linalg.pinv(Phi)
    q0=np.r_[P@y0[:S['n']],P@y0[S['n']:]]; t=np.arange(0,T,dt)
    s=solve_ivp(f,[0,T],q0,method=method,jac=jac if method in ('Radau','BDF','LSODA') else None,rtol=rtol,atol=1e-12,t_eval=t)
    return t,s.y[:r].T@Phi.T,s
def dom_freq(t,x):
    """frequency from upward zero crossings of a (mean-free) signal, linear interpolation"""
    x=x-x.mean(); i=np.where((x[:-1]<0)&(x[1:]>=0))[0]; tc=t[i]-x[i]*(t[i+1]-t[i])/(x[i+1]-x[i])
    return (len(tc)-1)/(tc[-1]-tc[0]) if len(tc)>2 else np.nan
def energy(S,X,V): return 0.5*np.einsum('ij,jk,ik->i',V,S['M'],V)+0.5*np.einsum('ij,jk,ik->i',X,S['K'],X)+S['k3']/4*X[:,S['tip']]**4
def newmark(M,K,C,pt,k3,x0,v0,T,dt,f_ext=None,tol=1e-12,maxit=20):
    """Average-acceleration Newmark (gamma=1/2, beta=1/4) with Newton iterations; cubic spring k3 (pt^T x)^3 along pt.
       Unconditionally stable; period error ~ (w dt)^2/12. Returns t, X, V."""
    n=len(M); N=int(round(T/dt))+1; X=np.zeros((N,n)); V=np.zeros((N,n)); x=x0.copy(); v=v0.copy()
    fn=lambda x: k3*(pt@x)**3*pt
    a=np.linalg.solve(M,-C@v-K@x-fn(x)); X[0]=x; V[0]=v
    c0=4/dt**2; c1=2/dt; Keff0=K+c0*M+c1*C; P=np.outer(pt,pt)
    for k in range(1,N):
        xp=x+dt*v+dt**2/4*a                         # predictor
        rhs_const=M@(c0*x+4/dt*v+a)+C@(c1*x+v)
        xn=xp.copy()
        for it in range(maxit):
            R=Keff0@xn+fn(xn)-rhs_const-(0 if f_ext is None else f_ext(k*dt))
            J=Keff0+3*k3*(pt@xn)**2*P; dx=np.linalg.solve(J,-R); xn+=dx
            if np.linalg.norm(dx)<=tol*(1+np.linalg.norm(xn)): break
        an=c0*(xn-x)-4/dt*v-a; v=v+dt/2*(a+an); x=xn; a=an; X[k]=x; V[k]=v
    return np.arange(N)*dt,X,V
