"""SOD-ROM figure data: tip displacement of nonlinear Galerkin ROMs of the COST F3 cantilever against the full model,
   for two held-out plucked tests (static deflection under 20 N at the tip; 60 N at mid-span, node 18), released from
   rest, conservative, Newmark dt = 2e-5 s over 0.3 s. Bases at r = 2 and 4: LNM; MTA (r-1 LNMs + static tip-load
   vector); SOD of the static-release decay (30 N tip load, as n4); SOD of the equal-energy decay (LNMs 1-4, as n5).
   Output: n6_timehist.npz (every 5th sample)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'SOD-toolbox', 'python'))
import sodtoolbox as _st
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import fsolve
from costbeam import costbeam
from nlsim import newmark, static_deflection
S = costbeam(); Sd = costbeam(zeta=0.01); n = S['n']; tip = S['tip']; k3 = S['k3']; pt = np.zeros(n); pt[tip] = 1.0
w2, PL = eigh(S['K'], S['M']); dt = 2e-5; sub = 5
def sod(X, V, r, tol=1e-10): return _st.sod_modes(X, V, r, tol)[0]
def mta(r):
    s = np.linalg.solve(S['K'], pt); Q = np.c_[PL[:, :r - 1], s]; L = np.linalg.cholesky(Q.T @ S['M'] @ Q); return Q @ np.linalg.inv(L).T
x30 = static_deflection(S, 30.0); H30 = 0.5 * x30 @ S['K'] @ x30 + k3 / 4 * x30[tip]**4
_, Xd, Vd = newmark(Sd['M'], Sd['K'], Sd['C'], pt, k3, x30, np.zeros(n), 1.5, dt); Xs, Vs = Xd[::sub], Vd[::sub]
xee = sum(PL[:, k] * np.sqrt(2 * H30) / np.sqrt(w2[k]) for k in range(4))
_, Xd, Vd = newmark(Sd['M'], Sd['K'], Sd['C'], pt, k3, xee, np.zeros(n), 1.5, dt); Xe, Ve = Xd[::sub], Vd[::sub]; del Xd, Vd
bases = {'LNM': lambda r: PL[:, :r], 'MTA': mta, 'SOD_SR': lambda r: sod(Xs, Vs, r), 'SOD_EE': lambda r: sod(Xe, Ve, r)}
out = {}
for lab, P, node in [('tip20', 20.0, tip), ('mid60', 60.0, 18)]:
    e = np.zeros(n); e[node] = P
    x0 = fsolve(lambda x: S['K'] @ x + k3 * x[tip]**3 * pt - e, np.linalg.solve(S['K'], e), xtol=1e-14)
    t, X, V = newmark(S['M'], S['K'], S['C'], pt, k3, x0, np.zeros(n), 0.3, dt); out[f'{lab}_full'] = X[::sub, tip]
    for bn, fb in bases.items():
        for r in [2, 4]:
            Phi = fb(r); Mr, Kr = Phi.T @ S['M'] @ Phi, Phi.T @ S['K'] @ Phi; Pm = np.linalg.solve(Mr, Phi.T @ S['M'])
            _, Q, _ = newmark(Mr, Kr, np.zeros((r, r)), Phi[tip], k3, Pm @ x0, np.zeros(r), 0.3, dt); Xr = Q @ Phi.T
            D = Xr - X; err = np.sqrt(np.sum((D @ S['M']) * D) / np.sum((X @ S['M']) * X))
            out[f'{lab}_{bn}_r{r}'] = Xr[::sub, tip]; print(f"{lab} {bn} r={r}: mass-weighted error over 0.3 s {err:.2e}", flush=True)
np.savez('n6_timehist.npz', dt=dt * sub, **out)
