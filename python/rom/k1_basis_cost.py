"""SOD-ROM, cost section: does the coordinate system of one SOD subspace change integration cost?
   Same models, training record (default_rng(5), random load at the training input), chirp load and integrators as
   c2_timing.py. For each SOD subspace span(SOD_r) (r = 4, 8, and full rank where available) five bases of the SAME
   subspace are compared:
     raw    smooth modes, unit Euclidean norm (as in the paper and the toolbox)
     Mnorm  smooth modes scaled to unit modal mass, phi^T M phi = 1 (directions unchanged)
     Eorth  Euclidean-orthonormal basis from the SVD of the smooth modes
     Morth  M-orthonormal basis (Cholesky of the reduced mass)
     Ritz   eigenvectors of the reduced pencil (K_r, M_r), M-normalized: decouples M_r and K_r (and C_r, Rayleigh)
   All five give the same reduced eigenvalues (similarity); reported: cond(M_r), departure from normality of the
   first-order matrix A_r, omega_max, central-difference steps, RK45 at the c2 settings (rtol 1e-6, atol 1e-9),
   Radau at the c2 settings, and a work-precision sweep of RK45 with per-coordinate atol_i = rtol * max|q_i|.
   Error: relative Euclidean error of the reconstructed displacements against a DOP853 reference (rtol 1e-12) of the
   same subspace. Usage: OPENBLAS_NUM_THREADS=1 python3 k1_basis_cost.py <0|1>  (0: S1 frame, 1: S2 beam)"""
import numpy as np, time, pickle, sys
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, svd, cholesky, eigvals
from systems import shear_frame, beam
from romlib import simulate, band_noise, galerkin, lnm, pod, sod

k = int(sys.argv[1])
S, wc, dt, Ttr = [(shear_frame(), 23.0, 0.01, 300), (beam(), 250.0, 1e-3, 200)][k]
rng = np.random.default_rng(5)
M, K, B = S['M'], S['K'], S['B']; n = len(M); C = S['C']['light']; b = B[:, 0]
w1 = np.sqrt(eigh(K, M, eigvals_only=True)[0]); T = (20 if n < 20 else 10) * 2 * np.pi / w1
u = lambda t, w0=0.5 * w1, w1_=wc, T=T: np.sin(w0 * t + (w1_ - w0) * t ** 2 / (2 * T))
Ntr = int(Ttr / dt)
X, V = simulate(M, K, C, B, band_noise(Ntr, dt, wc, B.shape[1], rng) * np.r_[1, np.zeros(B.shape[1] - 1)], dt)
Ps_all, _ = sod(X, V, n); rank = Ps_all.shape[1]; Pp_all = pod(X, n)
TE = np.linspace(0, T, 2001)

def variants(P):
    Mr = P.T @ M @ P; out = {'raw': P, 'Mnorm': P / np.sqrt(np.diag(Mr))}
    out['Eorth'] = svd(P, full_matrices=False)[0]
    out['Morth'] = P @ np.linalg.inv(cholesky(Mr, lower=False))
    Kr = P.T @ K @ P; out['Ritz'] = P @ eigh(Kr, Mr)[1]
    return out

def rhs(Phi):
    Mr, Kr, Cr, br = galerkin(M, K, C, b[:, None], Phi); br = br[:, 0]; m = len(Mr)
    Mi = np.linalg.inv(Mr); A21 = -Mi @ Kr; A22 = -Mi @ Cr; bb = Mi @ br
    Ar = np.block([[np.zeros((m, m)), np.eye(m)], [A21, A22]])
    f = lambda t, y: np.r_[y[m:], A21 @ y[:m] + A22 @ y[m:] + bb * u(t)]
    return f, (lambda t, y: Ar), Mr, Kr, Cr, Ar, m

def henrici(A):
    lam = eigvals(A); F = np.linalg.norm(A, 'fro'); return np.sqrt(max(F ** 2 - np.sum(abs(lam) ** 2), 0)) / F

def cd_steps(Mr, Kr):
    wmax = np.sqrt(eigh(Kr, Mr, eigvals_only=True).max()); return int(T / (0.9 * 2 / wmax)) + 1, wmax

def best(fun, reps):
    t = np.inf
    for _ in range(reps):
        t0 = time.perf_counter(); o = fun(); t = min(t, time.perf_counter() - t0)
    return t, o

subspaces = [(f'SOD r={r}', Ps_all[:, :r]) for r in (4, 8) if r <= rank]
if rank == n: subspaces.append(('SOD full-rank', Ps_all))
print(f"== {S['name']} (n={n}, SOD data rank {rank}), T={T:.1f}, OPENBLAS threads=1", flush=True)
res = {}
for sname, P in subspaces:
    V5 = variants(P)
    if sname == 'SOD r=8' and rank == 8: V5['POD'] = Pp_all[:, :8]           # same subspace at the data rank
    if sname == 'SOD full-rank': V5['identity'] = np.eye(n); V5['LNM'] = lnm(M, K, n)
    f, jac, *_ , m = rhs(V5['Ritz'])
    ref = solve_ivp(f, [0, T], np.zeros(2 * m), method='DOP853', rtol=1e-12, atol=1e-14, t_eval=TE)
    Xref = ref.y[:m].T @ V5['Ritz'].T
    print(f"\n-- {sname}: reference DOP853 nfev {ref.nfev}")
    print(f"{'basis':9s} {'cond Mr':>9} {'dep.norm':>8} {'w_max':>8} {'CD':>7} | {'RK45 nfev':>9} {'s':>6} {'err':>8} | "
          f"{'Radau nfev':>10} {'s':>6} | WP: rtol -> nfev, err (atol_i = rtol*max|q_i|)")
    for bn, Phi in V5.items():
        f, jac, Mr, Kr, Cr, Ar, m = rhs(Phi); ncd, wmax = cd_steps(Mr, Kr)
        heavy = wmax > 1e3
        trk, so = best(lambda: solve_ivp(f, [0, T], np.zeros(2 * m), method='RK45', rtol=1e-6, atol=1e-9, t_eval=TE),
                       1 if (heavy or m > 20) else 3)
        erk = np.linalg.norm(so.y[:m].T @ Phi.T - Xref) / np.linalg.norm(Xref) if so.success else np.nan
        trd, sd = best(lambda: solve_ivp(f, [0, T], np.zeros(2 * m), method='Radau', jac=jac, rtol=1e-6, atol=1e-9), 1)
        # per-coordinate scale from the reference of this basis
        Qs = np.linalg.lstsq(Phi, Xref.T, rcond=None)[0]                     # exact: Xref lies in span(Phi)
        sc = np.r_[abs(Qs).max(1), abs(np.gradient(Qs, TE, axis=1)).max(1)]
        wp = []
        for rt in (1e-4, 1e-6, 1e-8):
            s2 = solve_ivp(f, [0, T], np.zeros(2 * m), method='RK45', rtol=rt, atol=rt * sc, t_eval=TE)
            wp.append((rt, s2.nfev, np.linalg.norm(s2.y[:m].T @ Phi.T - Xref) / np.linalg.norm(Xref)))
        res[(sname, bn)] = dict(condM=np.linalg.cond(Mr), dep=henrici(Ar), wmax=wmax, cd=ncd, rk_nfev=so.nfev, rk_s=trk,
                                rk_err=erk, radau_nfev=sd.nfev, radau_s=trd, wp=wp)
        print(f"{bn:9s} {np.linalg.cond(Mr):9.2e} {henrici(Ar):8.2e} {wmax:8.3g} {ncd:7d} | {so.nfev:9d} {trk:6.2f} {erk:8.1e} | "
              f"{sd.nfev:10d} {trd:6.2f} | " + "  ".join(f"{q[1]:7d} {q[2]:7.1e}" for q in wp), flush=True)
pickle.dump(res, open(f'k1_results_{k}.pkl', 'wb'))
