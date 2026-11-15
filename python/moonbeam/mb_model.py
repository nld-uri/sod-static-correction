"""Two-well (magneto-elastic, 'Moon') cantilever: Euler-Bernoulli beam (Hermite elements, nondimensional EI = rhoA = L = 1)
   with a tip force from a two-well potential, f(x) = -k1 x + k3 x^3 (x = tip deflection), under harmonic base
   acceleration a_g(t) = A cos(Omega t). Equations: M xdd + C xd + K x + f(x_tip) e_tip = -M r a_g, r = 1 on translations.
   With k1 above the beam's tip stiffness (3 EI/L^3) the straight state is unstable and two buckled equilibria +-x* appear.
   Damping: Rayleigh, zeta at the first two modes of the beam linearized at a buckled equilibrium.
   Also: average-acceleration Newmark with Newton iterations for full models and Galerkin ROMs."""
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import fsolve
def beam(Ne=10, k1=6.0, k3=300.0, zeta=0.02):
    h = 1.0 / Ne; n = 2 * (Ne + 1)
    ke = 1 / h**3 * np.array([[12, 6*h, -12, 6*h], [6*h, 4*h*h, -6*h, 2*h*h], [-12, -6*h, 12, -6*h], [6*h, 2*h*h, -6*h, 4*h*h]])
    me = h / 420 * np.array([[156, 22*h, 54, -13*h], [22*h, 4*h*h, 13*h, -3*h*h], [54, 13*h, 156, -22*h], [-13*h, -3*h*h, -22*h, 4*h*h]])
    K = np.zeros((n, n)); M = np.zeros((n, n))
    for e in range(Ne): i = slice(2*e, 2*e+4); K[i, i] += ke; M[i, i] += me
    K, M = K[2:, 2:], M[2:, 2:]; n -= 2; tip = n - 2; et = np.zeros(n); et[tip] = 1.0
    r = np.zeros(n); r[0::2] = 1.0                                     # translations
    S = dict(M=M, K=K, n=n, tip=tip, et=et, r=r, k1=k1, k3=k3, Ne=Ne)
    xs = equilibrium(S, +1); S['xs'] = xs
    Kw = K + (-k1 + 3 * k3 * xs[tip]**2) * np.outer(et, et); w2, Pw = eigh(Kw, M); S['Kw'], S['ww'], S['Pw'] = Kw, np.sqrt(w2), Pw
    a, b = np.linalg.solve([[1 / (2 * S['ww'][0]), S['ww'][0] / 2], [1 / (2 * S['ww'][1]), S['ww'][1] / 2]], [zeta, zeta])
    S['C'] = a * M + b * K; S['zeta'] = zeta
    w02, P0 = eigh(K, M); S['w0'], S['P0'] = np.sqrt(w02), P0           # clamped-beam LNMs (no magnets)
    return S
def fnl(S, xt): return -S['k1'] * xt + S['k3'] * xt**3
def dfnl(S, xt): return -S['k1'] + 3 * S['k3'] * xt**2
def equilibrium(S, sgn):
    x0 = np.linalg.solve(S['K'], S['et']) * 0.1 * sgn
    g = lambda x: S['K'] @ x + fnl(S, x[S['tip']]) * S['et']
    J = lambda x: S['K'] + dfnl(S, x[S['tip']]) * np.outer(S['et'], S['et'])
    return fsolve(g, x0 * 3, fprime=J, xtol=1e-13)
def newmark(M, K, C, pt, S, x0, v0, T, dt, A, Om, Mr, tsave=None, tol=1e-11, maxit=20):
    """average-acceleration Newmark, Newton; force -Mr a_g with a_g = A cos(Om t); nonlinear tip force along pt
       (pt = row of the tip displacement in the model's coordinates). Returns times and states at tsave indices."""
    n = len(M); N = int(round(T / dt)); x, v = x0.copy(), v0.copy(); P = np.outer(pt, pt)
    a = np.linalg.solve(M, -C @ v - K @ x - fnl(S, pt @ x) * pt - Mr * A * np.cos(0.0))
    c0 = 4 / dt**2; Keff0 = K + c0 * M + 2 / dt * C
    save = set(tsave) if tsave is not None else None; X = []; V = []
    for k in range(1, N + 1):
        t = k * dt; rhs = M @ (c0 * x + 4 / dt * v + a) + C @ (2 / dt * x + v) - Mr * A * np.cos(Om * t)
        xn = x + dt * v + dt**2 / 4 * a
        for it in range(maxit):
            xt = pt @ xn; R = Keff0 @ xn + fnl(S, xt) * pt - rhs
            dx = np.linalg.solve(Keff0 + dfnl(S, xt) * P, -R); xn += dx
            if np.abs(dx).max() <= tol * (1 + np.abs(xn).max()): break
        an = c0 * (xn - x) - 4 / dt * v - a; v = v + dt / 2 * (a + an); x = xn; a = an
        if save is not None and k in save: X.append(x.copy()); V.append(v.copy())
    return np.array(X), np.array(V), x, v
if __name__ == "__main__":
    S = beam(); print("n =", S['n'], " x* tip =", S['xs'][S['tip']], " well frequencies", np.round(S['ww'][:4], 3), " clamped", np.round(S['w0'][:4], 3))
