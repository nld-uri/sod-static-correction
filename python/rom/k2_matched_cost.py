"""SOD-ROM, cost section: cost of one subspace at MATCHED accuracy.
   (a) From k1_results_<k>.pkl: RK45 evaluations needed to reach a target displacement error (1e-6, 1e-8),
       log-log interpolation in the work-precision sweep (per-coordinate atol_i = rtol * max|q_i|).
   (b) New runs, cantilever (S2, light): Radau work-precision for the r = 8 SOD subspace in the raw, Morth and Ritz
       bases, and for the full model in physical and full-rank LNM coordinates (the paper's 11.5 s vs 5.6 s claim).
   Same chirp, record and settings as k1/c2. Usage: OPENBLAS_NUM_THREADS=1 python3 k2_matched_cost.py"""
import numpy as np, pickle, time
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, cholesky
from systems import beam
from romlib import simulate, band_noise, galerkin, lnm, sod

def at_err(wp, target):
    """nfev at a target error, log-log interpolation between bracketing sweep points (nan if not bracketed)"""
    wp = sorted(wp, key=lambda q: q[1])
    for (r1, n1, e1), (r2, n2, e2) in zip(wp[:-1], wp[1:]):
        if (e1 - target) * (e2 - target) <= 0 and e1 != e2:
            f = (np.log(target) - np.log(e1)) / (np.log(e2) - np.log(e1)); return np.exp(np.log(n1) + f * (np.log(n2) - np.log(n1)))
    return np.nan

print("(a) RK45 evaluations at matched displacement error (from k1 work-precision sweeps)")
for k in (0, 1):
    R = pickle.load(open(f'k1_results_{k}.pkl', 'rb'))
    print(f"   {'frame' if k == 0 else 'cantilever'}")
    for (sn, bn), d in R.items():
        print(f"      {sn:14s} {bn:9s} nfev@1e-6 {at_err(d['wp'], 1e-6):9.0f}   nfev@1e-8 {at_err(d['wp'], 1e-8):9.0f}"
              f"   | c2 settings: {d['rk_nfev']:7d} (err {d['rk_err']:.1e})")

# (b) Radau work-precision on the cantilever
S = beam(); M, K, B = S['M'], S['K'], S['B']; n = len(M); C = S['C']['light']; b = B[:, 0]
rng = np.random.default_rng(5); wc = 250.0; dt = 1e-3; Ntr = 200000
w1 = np.sqrt(eigh(K, M, eigvals_only=True)[0]); T = 10 * 2 * np.pi / w1
u = lambda t, w0=0.5 * w1: np.sin(w0 * t + (wc - w0) * t ** 2 / (2 * T))
X, V = simulate(M, K, C, B, band_noise(Ntr, dt, wc, 2, rng) * np.r_[1, 0], dt)
P = sod(X, V, n)[0][:, :8]; Mr = P.T @ M @ P
bases = {'SOD r=8 raw': P, 'SOD r=8 Morth': P @ np.linalg.inv(cholesky(Mr, lower=False)),
         'SOD r=8 Ritz': P @ eigh(P.T @ K @ P, Mr)[1], 'full physical': np.eye(n), 'full LNM': lnm(M, K, n)}
TE = np.linspace(0, T, 2001)
def rhs(Phi):
    Mr, Kr, Cr, br = galerkin(M, K, C, b[:, None], Phi); m = len(Mr); Mi = np.linalg.inv(Mr)
    A21 = -Mi @ Kr; A22 = -Mi @ Cr; bb = (Mi @ br)[:, 0]; Ar = np.block([[np.zeros((m, m)), np.eye(m)], [A21, A22]])
    return (lambda t, y: np.r_[y[m:], A21 @ y[:m] + A22 @ y[m:] + bb * u(t)]), (lambda t, y: Ar), m
refs = {}
for grp, rb in (('SOD r=8', 'SOD r=8 Ritz'), ('full', 'full LNM')):
    f, jac, m = rhs(bases[rb])
    s = solve_ivp(f, [0, T], np.zeros(2 * m), method='Radau', jac=jac, rtol=1e-11, atol=1e-14, t_eval=TE)
    refs[grp] = s.y[:m].T @ bases[rb].T
print("\n(b) Radau, cantilever: work-precision (per-coordinate atol_i = rtol*max|q_i|) and the c2 settings")
out = {}
for bn, Phi in bases.items():
    grp = 'full' if bn.startswith('full') else 'SOD r=8'; Xref = refs[grp]; f, jac, m = rhs(Phi)
    Qs = np.linalg.lstsq(Phi, Xref.T, rcond=None)[0]; sc = np.r_[abs(Qs).max(1), abs(np.gradient(Qs, TE, axis=1)).max(1)]
    wp = []
    for rt in (1e-4, 1e-6, 1e-8):
        t0 = time.perf_counter(); s = solve_ivp(f, [0, T], np.zeros(2 * m), method='Radau', jac=jac, rtol=rt, atol=rt * sc, t_eval=TE)
        wp.append((rt, s.nfev, np.linalg.norm(s.y[:m].T @ Phi.T - Xref) / np.linalg.norm(Xref), time.perf_counter() - t0, s.njev, s.nlu))
    t0 = time.perf_counter(); s = solve_ivp(f, [0, T], np.zeros(2 * m), method='Radau', jac=jac, rtol=1e-6, atol=1e-9, t_eval=TE)
    tc = time.perf_counter() - t0; ec = np.linalg.norm(s.y[:m].T @ Phi.T - Xref) / np.linalg.norm(Xref)
    out[bn] = dict(wp=wp, c2=(s.nfev, tc, ec, s.nlu))
    print(f"   {bn:14s} c2 settings: nfev {s.nfev:7d} LU {s.nlu:6d} {tc:6.2f}s err {ec:.1e} | WP " +
          "  ".join(f"{q[1]:7d}/{q[5]:5d}LU {q[3]:5.2f}s {q[2]:.1e}" for q in wp) +
          f" | nfev@1e-6 {at_err([q[:3] for q in wp], 1e-6):8.0f} @1e-8 {at_err([q[:3] for q in wp], 1e-8):8.0f}", flush=True)
pickle.dump(out, open('k2_results.pkl', 'wb'))
