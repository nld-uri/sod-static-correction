"""SOD of a forced record: every smooth mode is a load deflection at its own SOD frequency, up to three named terms.
   For M xdd + C xd + K x = B u (a local nonlinear force enters as one more input), averaging x^T times the equation
   over the record and using <xdd x^T> = [xd x^T]_0^T / T - Sigma_v gives, for every SOD pair (mu_i, psi_i),
       (K - mu_i M) phi_i =  B <u x^T> psi_i            (load)
                           - C <xd x^T> psi_i            (damping)
                           - M [xd x^T]_0^T psi_i / T    (record ends)
                           + sampling: the difference between the sampled <xdd x^T> + Sigma_v and [xd x^T]_0^T / T
                           + M (Sigma_v - mu_i Sigma_x) psi_i (rank: zero when SOD uses the full data space),
   phi_i = Sigma_x psi_i. The undamped, full-rank form with the load term is Farooq and Feeny (2008), Eq. (8), who set
   the load term to zero for white noise. Here each term is mapped through (K - mu_i M)^-1 and its share of phi_i is
   reported (mass norm), with the angle between phi_i and the load deflection d_i = (K - mu_i M)^-1 B c_i
   (c_i = <u x^T> psi_i, a scalar for one input). Records: the c18 random training records (same seeds and draw
   order as c18_ensemble.py, 20 realizations, cases 0-5), the 30 N static-release decay of the COST F3 beam (S4) and
   the cross-well decay of the two-well beam (S5). SOD: data-space rank tolerance 1e-10, as in the paper.
   Usage: python3 i1_forced_identity.py [case ...]   (cases 0-5, s4, s5; default all) -> i1_out.txt, i1_results.pkl"""
import os, sys, pickle, numpy as np
from scipy.linalg import eigh
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('rom', 'nnmrom', 'moonbeam'): sys.path.insert(0, os.path.join(HERE, '..', d))
NDIR = 8
def sod_pairs(X, V, tol=1e-10):
    w, U = eigh(X.T @ X); U = U[:, w > tol * w.max()]; Xp, Vp = X @ U, V @ U; N = len(X)
    mu, Pz = eigh(Vp.T @ Vp / N, Xp.T @ Xp / N); return mu, U @ Pz                   # psi in full coordinates
def budget(X, V, A, U, M, K, C, B, dt, nd=NDIR):
    """per-direction shares of the four terms in phi_i, closure, angle to the load deflection"""
    N = len(X); Sx, Sv = X.T @ X / N, V.T @ V / N; Lxu = U.T @ X / N; Dvx = V.T @ X / N; Eax = A.T @ X / N
    mu, Psi = sod_pairs(X, V); Mn = lambda v: np.sqrt(abs(v @ M @ v)); rows = []
    for i in range(min(nd, len(mu))):
        p = Psi[:, i]; phi = Sx @ p; G = K - mu[i] * M
        ends_true = -M @ ((np.outer(V[-1], X[-1]) - np.outer(V[0], X[0])) / ((N - 1) * dt) @ p)    # -M [xd x^T]_0^T psi / T
        terms = {'load': B @ (Lxu @ p), 'damp': -C @ (Dvx @ p), 'ends': ends_true, 'disc': -M @ ((Eax + Sv) @ p) - ends_true,
                 'rank': M @ ((Sv - mu[i] * Sx) @ p)}
        parts = {k: np.linalg.solve(G, v) for k, v in terms.items()}
        clos = Mn(phi - sum(parts.values())) / Mn(phi)
        d = parts['load']; c = abs(phi @ M @ d) / (Mn(phi) * Mn(d)) if Mn(d) > 0 else 0.0
        rows.append(dict(i=i + 1, w=np.sqrt(mu[i]), ang=np.degrees(np.arccos(min(1.0, c))), closure=clos,
                         **{k: Mn(v) / Mn(phi) for k, v in parts.items()}))
    return rows, len(mu)
def show(tag, rows, rank, out):
    out.append(f"{tag}: data rank {rank}")
    out.append("   i  SOD freq    angle(phi, load defl.) deg   share in phi: load    damping   ends      sampling  rank      closure")
    for r in rows:
        out.append(f"  {r['i']:2d} {r['w']:10.3f}   {r['ang']:12.2e}                 {r['load']:9.2e} {r['damp']:9.2e} {r['ends']:9.2e} {r['disc']:9.2e} {r['rank']:9.2e} {r['closure']:8.1e}")
def linear_case(ci, out, res):
    from systems import shear_frame, beam, state
    from romlib import band_noise
    from romlib2 import simulate_state
    CASES = [('S1', 'light'), ('S1', 'heavy'), ('S2', 'light'), ('S2', 'heavy'), ('S3', 'dashpot'), ('S3', 'dashpot_heavy')]
    sysname, cname = CASES[ci]
    S = shear_frame() if sysname == 'S1' else (beam() if sysname == 'S2' else beam(dashpot=(0.2, 1.0)))
    p = dict(dt=0.01, T=300, wc=23.0) if sysname == 'S1' else dict(dt=1e-3, T=200, wc=250.0)
    M, K, B = S['M'], S['K'], S['B']; C = S['C'][cname]; n = len(M); nin = B.shape[1]; dt = p['dt']; N = int(p['T'] / dt)
    Ast, Bs = state(M, K, C, B); e_in = np.r_[1, np.zeros(nin - 1)]; rng = np.random.default_rng(100 + ci); allr = []
    for rr in range(20):
        u = band_noise(N, dt, p['wc'], nin, rng) * e_in
        band_noise(N, dt, p['wc'], nin, rng)                                         # test draws of c18, kept in order
        if nin > 1: band_noise(N, dt, p['wc'], nin, rng)
        Y = simulate_state(Ast, Bs, u, dt); Yd = Y @ Ast.T + u @ Bs.T
        rows, rank = budget(Y[:, :n], Y[:, n:], Yd[:, n:], u, M, K, C, B, dt); allr.append(rows)
        if rr == 0: show(f"{sysname} {cname}, realization 1", rows, rank, out)
    res[(sysname, cname)] = allr
    nd = min(len(r) for r in allr)
    out.append(f"{sysname} {cname}, 20 realizations: median over realizations per direction")
    out.append("   i  SOD freq    angle deg    load      damping   ends      sampling  rank")
    for i in range(nd):
        g = lambda k: np.median([r[i][k] for r in allr])
        out.append(f"  {i + 1:2d} {g('w'):10.3f}   {g('ang'):9.2e}   {g('load'):9.2e} {g('damp'):9.2e} {g('ends'):9.2e} {g('disc'):9.2e} {g('rank'):9.2e}")
def s4(out, res):
    from costbeam import costbeam
    from nlsim import newmark, static_deflection
    S = costbeam(); Sd = costbeam(zeta=0.01); n = S['n']; tip = S['tip']; k3 = S['k3']; pt = np.zeros(n); pt[tip] = 1.0
    x30 = static_deflection(S, 30.0); _, X, V = newmark(Sd['M'], Sd['K'], Sd['C'], pt, k3, x30, np.zeros(n), 1.5, 2e-5)
    X, V = X[::5], V[::5]; u = -k3 * X[:, tip:tip + 1]**3                              # cubic tip force as the input
    A = np.linalg.solve(S['M'], (-Sd['C'] @ V.T - S['K'] @ X.T + np.outer(pt, u[:, 0]))).T
    rows, rank = budget(X, V, A, u, S['M'], S['K'], Sd['C'], pt[:, None], 1e-4, nd=4); show("S4, 30 N static-release decay (zeta 0.01)", rows, rank, out); res['S4'] = rows
def s5(out, res):
    from mb_model import beam as mbeam, newmark as mnewmark, fnl
    S = mbeam(); M, K, C, n, tip, et = S['M'], S['K'], S['C'], S['n'], S['tip'], S['et']; dt, T = 0.005, 60.0; N = int(round(T / dt))
    X, V, _, _ = mnewmark(M, K, C, et, S, S['xs'].copy(), -1.0 * S['r'], T, dt, 0.0, 1.0, M @ S['r'], tsave=range(1, N + 1))
    u = -fnl(S, X[:, tip])[:, None]                                                  # magnet force as the input
    A = np.linalg.solve(M, (-C @ V.T - K @ X.T + np.outer(et, u[:, 0]))).T
    rows, rank = budget(X, V, A, u, M, K, C, et[:, None], dt, nd=4); show("S5, cross-well decay (zeta 0.02)", rows, rank, out); res['S5'] = rows
if __name__ == '__main__':
    want = sys.argv[1:] or ['0', '1', '2', '3', '4', '5', 's4', 's5']; out = []; res = {}
    for w in want:
        if w == 's4': s4(out, res)
        elif w == 's5': s5(out, res)
        else: linear_case(int(w), out, res)
        print('\n'.join(out), flush=True); out.clear()
    pickle.dump(res, open(os.path.join(HERE, 'i1_results.pkl' if not sys.argv[1:] else f"i1_results_{'_'.join(want)}.pkl"), 'wb'))
