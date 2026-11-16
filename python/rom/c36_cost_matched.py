"""SOD-ROM Sec. 7 (revised): integration cost at matched accuracy, every model in Ritz coordinates of its subspace.
   Same models, training record (default_rng(5), random load at the training input, as c2_timing.py) and chirp load
   (0.5 w1 -> w_c over 20 (S1) or 10 (S2) periods of mode 1) as c2. Light Rayleigh damping.
   Models: full (physical and LNM coordinates); LNM, MTA (both load vectors), POD and SOD subspaces at r = 4 and 8,
   each in Ritz coordinates (eigenvectors of the reduced pencil (K_r, M_r), M-normalized: M_r = I, K_r and the Rayleigh
   C_r diagonal); for contrast SOD r = 8 in the raw smooth-mode basis (unit Euclidean columns).
   Cost at matched accuracy: for each model and integrator (RK45, Radau with analytic Jacobian) the tolerance grid
   rtol = 1e-3 .. 1e-9 (half decades) x {scalar atol = rtol * physical scale, per-coordinate atol_i = rtol*max|q_i|}
   is swept; the model is charged the run with the fewest function evaluations whose displacement error
   (relative Euclidean, 2001 samples) against a tight reference of the SAME model is at most the target (1e-6, 1e-8).
   That run is then timed, best of 3. References: DOP853 rtol 1e-12 (ROMs); Radau rtol 1e-11 in LNM coordinates (full).
   The full cantilever with RK45 is stability-limited (largest eigenvalue -9.3e5): one run at rtol 1e-3, scalar atol.
   Central-difference steps at 0.9 of 2/w_max are reported as in c2.
   Usage: OPENBLAS_NUM_THREADS=1 python3 c36_cost_matched.py <0|1>  -> c36_out_<k>.txt (stdout), c36_results_<k>.pkl"""
import numpy as np, time, pickle, sys
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, cholesky
from systems import shear_frame, beam
from romlib import simulate, band_noise, galerkin, lnm, pod, sod
from romlib2 import mta

k = int(sys.argv[1])
S, wc, dt, Ttr = [(shear_frame(), 23.0, 0.01, 300), (beam(), 250.0, 1e-3, 200)][k]
rng = np.random.default_rng(5); M, K, B = S['M'], S['K'], S['B']; n = len(M); C = S['C']['light']; b = B[:, 0]
w1 = np.sqrt(eigh(K, M, eigvals_only=True)[0]); T = (20 if n < 20 else 10) * 2 * np.pi / w1
u = lambda t, w0=0.5 * w1: np.sin(w0 * t + (wc - w0) * t ** 2 / (2 * T))
X, V = simulate(M, K, C, B, band_noise(int(Ttr / dt), dt, wc, B.shape[1], rng) * np.r_[1, np.zeros(B.shape[1] - 1)], dt)
Ps = sod(X, V, n)[0]; Pp = pod(X, n); rank = Ps.shape[1]
TE = np.linspace(0, T, 2001); RT = [10 ** (-e / 2) for e in range(6, 19)]
def ritz(P): return P @ eigh(P.T @ K @ P, P.T @ M @ P)[1]

models = {'full, physical': np.eye(n), 'full, LNM coordinates': lnm(M, K, n)}
for r in (4, 8):
    models[f'LNM r={r}'] = lnm(M, K, r)
    models[f'MTA r={r}'] = ritz(mta(M, K, B, r))
    models[f'POD r={r}'] = ritz(Pp[:, :r])
    if r <= rank: models[f'SOD r={r}'] = ritz(Ps[:, :r])
if rank >= 8: models['SOD r=8, raw smooth modes'] = Ps[:, :8]

def rhs(Phi):
    Mr, Kr, Cr, br = galerkin(M, K, C, b[:, None], Phi); m = len(Mr); Mi = np.linalg.inv(Mr)
    A21 = -Mi @ Kr; A22 = -Mi @ Cr; bb = (Mi @ br)[:, 0]; Ar = np.block([[np.zeros((m, m)), np.eye(m)], [A21, A22]])
    return (lambda t, y: np.r_[y[m:], A21 @ y[:m] + A22 @ y[m:] + bb * u(t)]), (lambda t, y: Ar), m, Mr, Kr, Ar

def run(f, jac, m, meth, rt, at):
    kw = dict(jac=jac) if meth == 'Radau' else {}
    t0 = time.perf_counter(); s = solve_ivp(f, [0, T], np.zeros(2 * m), method=meth, rtol=rt, atol=at, t_eval=TE, **kw)
    return s, time.perf_counter() - t0

refs, res = {}, {}
print(f"== {S['name']} (n={n}, SOD data rank {rank}), T = {T:.1f}; cost to reach displacement error 1e-6 | 1e-8", flush=True)
print(f"{'model':28s} {'w_max':>9} {'rho(A_r)':>9} {'CD':>7} | {'RK45 nfev':>10} {'s':>7} | {'Radau nfev':>10} {'LU':>6} {'s':>7}"
      f" || {'RK45 nfev':>10} {'s':>7} | {'Radau nfev':>10} {'LU':>6} {'s':>7}", flush=True)
for name, Phi in models.items():
    f, jac, m, Mr, Kr, Ar = rhs(Phi)
    wmax = np.sqrt(eigh(Kr, Mr, eigvals_only=True).max()); rho = abs(np.linalg.eigvals(Ar)).max()
    ncd = int(T / (0.9 * 2 / wmax)) + 1
    grp = 'full' if name.startswith('full') else name.split(',')[0]
    if grp not in refs:
        Pr = lnm(M, K, n) if grp == 'full' else ritz(Phi); fr, jr, mr, *_ = rhs(Pr)
        s = (solve_ivp(fr, [0, T], np.zeros(2 * mr), method='Radau', jac=jr, rtol=1e-11, atol=1e-14, t_eval=TE) if grp == 'full'
             else solve_ivp(fr, [0, T], np.zeros(2 * mr), method='DOP853', rtol=1e-12, atol=1e-15, t_eval=TE))
        refs[grp] = s.y[:mr].T @ Pr.T
    Xref = refs[grp]; xs = abs(Xref).max(); vs = abs(np.gradient(Xref, TE, axis=0)).max()
    Qs = np.linalg.lstsq(Phi, Xref.T, rcond=None)[0]; sc = np.r_[abs(Qs).max(1), abs(np.gradient(Qs, TE, axis=1)).max(1)]
    err = lambda s: np.linalg.norm(s.y[:m].T @ Phi.T - Xref) / np.linalg.norm(Xref) if s.success else np.inf
    out = dict(wmax=wmax, rho=rho, cd=ncd, m=m)
    for meth in ('RK45', 'Radau'):
        if meth == 'RK45' and name.startswith('full') and wmax > 1e3:          # stability-limited: one run
            s, ts = run(f, jac, m, 'RK45', 1e-3, 1e-3 * np.r_[np.full(m, xs), np.full(m, vs)])
            pts = [(s.nfev, err(s), 0, 'scalar', 1e-3, ts)]
        else:
            pts = []
            for rt in RT:
                for strat, at in (('scalar', rt * np.r_[np.full(m, xs), np.full(m, vs)]), ('coord', rt * sc)):
                    s, ts = run(f, jac, m, meth, rt, at); pts.append((s.nfev, err(s), getattr(s, 'nlu', 0), strat, rt, ts))
        for tgt in (1e-6, 1e-8):
            ok = [q for q in pts if q[1] <= tgt]
            if not ok: out[(meth, tgt)] = None; continue
            q = min(ok, key=lambda q: q[0]); strat, rt = q[3], q[4]
            at = rt * (np.r_[np.full(m, xs), np.full(m, vs)] if strat == 'scalar' else sc)
            tb = q[5] if (meth == 'RK45' and name.startswith('full') and wmax > 1e3) else min(run(f, jac, m, meth, rt, at)[1] for _ in range(3))
            out[(meth, tgt)] = dict(nfev=q[0], err=q[1], nlu=q[2], strat=strat, rtol=rt, time=tb)
        out[(meth, 'pts')] = pts
    res[name] = out
    def cell(meth, tgt):
        d = out.get((meth, tgt))
        if d is None: return f"{'--':>10} {'':>7}" + (f" {'':>6}" if meth == 'Radau' else '')
        return f"{d['nfev']:10d} {d['time']:7.2f}" + (f" {d['nlu']:6d}" if meth == 'Radau' else '')
    print(f"{name:28s} {wmax:9.3g} {rho:9.3g} {ncd:7d} | {cell('RK45', 1e-6)} | {cell('Radau', 1e-6)} || "
          f"{cell('RK45', 1e-8)} | {cell('Radau', 1e-8)}", flush=True)
    pickle.dump(res, open(f'c36_results_{k}.pkl', 'wb'))
