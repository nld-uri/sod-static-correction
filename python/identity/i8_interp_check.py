"""Does an SOD basis of a forced record satisfy the hypotheses of the interpolation theorem, and does it interpolate?
   Beattie and Gugercin (2009), Theorem 1(a), one-sided (Galerkin) case of their Example 2: if K(s)^-1 b lies in Ran(V)
   and both K(s) = s^2 M + s C + K and V^T K(s) V are nonsingular, the reduced transfer function matches the full one
   at s along b. For the undamped system at s = i w, K(s) = K - w^2 M and K(s)^-1 b = d(w).
   For the i2 records (same seeds and draw order, 20 realizations) and r = 1..8 this script reports, over the SOD
   frequencies w_i, i <= r:
     gap   : min_k |theta_k - w_i^2| / w_i^2, theta_k the eigenvalues of the reduced pencil (V^T K V, V^T M V).
             gap = 0 means V^T (K - w_i^2 M) V is singular (hypothesis fails). Also the same gap to the full model's
             eigenvalues (w_i equal to a natural frequency).
     e_und : || d(w_i) - V (V^T (K - w_i^2 M) V)^-1 V^T b ||_M / || d(w_i) ||_M     (undamped system)
     e_dmp : the same with K - w_i^2 M + i w_i C                                      (damped system)
   for V = SOD (smooth modes), D (model-built d(w_1..r); must interpolate the undamped system to rounding: positive
   control) and LNM (no load response in the basis; must NOT interpolate: failure control).
   Usage: python3 i8_interp_check.py <case 0..5> -> i8_out_<case>.txt (stdout), i8_case<case>.pkl"""
import os, sys, pickle, numpy as np
from scipy.linalg import eigh
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'rom'))
from systems import shear_frame, beam, state
from romlib import band_noise
from romlib2 import simulate_state, lnm
from i1_forced_identity import sod_pairs
ci = int(sys.argv[1]); NR = 20; RMAX = 8
CASES = [('S1', 'light'), ('S1', 'heavy'), ('S2', 'light'), ('S2', 'heavy'), ('S3', 'dashpot'), ('S3', 'dashpot_heavy')]
sysname, cname = CASES[ci]
S = shear_frame() if sysname == 'S1' else (beam() if sysname == 'S2' else beam(dashpot=(0.2, 1.0)))
p = dict(dt=0.01, T=300, wc=23.0) if sysname == 'S1' else dict(dt=1e-3, T=200, wc=250.0)
M, K, B = S['M'], S['K'], S['B']; C = S['C'][cname]; n = len(M); nin = B.shape[1]; dt = p['dt']; N = int(p['T'] / dt)
Ast, Bs = state(M, K, C, B); e_in = np.r_[1, np.zeros(nin - 1)]; b = B @ e_in
def unit(Q): return Q / np.sqrt(np.einsum('ij,ij->j', Q, M @ Q))
def morth(Q):
    Q = unit(Q); g, V = eigh(Q.T @ M @ Q)
    return None if g[0] <= 1e-12 * g[-1] else Q @ V / np.sqrt(g)
def mnorm(x): return np.sqrt(np.real(np.conj(x) @ (M @ x)))
PL = lnm(M, K, n); w2full = eigh(K, M, eigvals_only=True)
rng = np.random.default_rng(100 + ci); out = {}
for rr in range(NR):
    u_tr = band_noise(N, dt, p['wc'], nin, rng) * e_in
    band_noise(N, dt, p['wc'], nin, rng)                       # keep the i2 draw order (test loads, not used here)
    if nin > 1: band_noise(N, dt, p['wc'], nin, rng)
    Y = simulate_state(Ast, Bs, u_tr, dt); X, V = Y[:, :n], Y[:, n:]
    mu, Psi = sod_pairs(X, V); Sx = X.T @ X / N; Phi = unit(Sx @ Psi); w = np.sqrt(mu)
    D = unit(np.c_[tuple(np.linalg.solve(K - wi**2 * M, b) for wi in w)])
    for r in range(1, RMAX + 1):
        bases = {'SOD': morth(Phi[:, :r]), 'D': morth(D[:, :r]), 'LNM': PL[:, :r]}
        for bn, Q in bases.items():
            if Q is None: out.setdefault((bn, r, 'dep'), []).append(1); continue
            th = eigh(Q.T @ K @ Q, Q.T @ M @ Q, eigvals_only=True); g, gf, eu, ed = [], [], [], []
            for wi in w[:r]:
                g.append(np.min(np.abs(th - wi**2)) / wi**2); gf.append(np.min(np.abs(w2full - wi**2)) / wi**2)
                for store, Ks in ((eu, K - wi**2 * M), (ed, K - wi**2 * M + 1j * wi * C)):
                    d = np.linalg.solve(Ks, b); dr = Q @ np.linalg.solve(Q.T @ Ks @ Q, Q.T @ b)
                    store.append(mnorm(d - dr) / mnorm(d))
            for k, v in (('gap', min(g)), ('gapfull', min(gf)), ('e_und', max(eu)), ('e_dmp', max(ed))):
                out.setdefault((bn, r, k), []).append(v)
pickle.dump(dict(case=CASES[ci], out=out), open(os.path.join(HERE, f'i8_case{ci}.pkl'), 'wb'))
print(f"== {sysname} {cname}: {NR} realizations, r = 1..{RMAX}; each entry is over the SOD frequencies w_i, i <= r")
for k, lab, agg in (('gap', 'smallest relative gap of w_i^2 to the reduced pencil eigenvalues, min over realizations', np.min),
                    ('gapfull', 'smallest relative gap of w_i^2 to the full-model eigenvalues, min over realizations', np.min),
                    ('e_und', 'largest relative interpolation error, undamped system, median over realizations', np.median),
                    ('e_und', 'largest relative interpolation error, undamped system, max over realizations', np.max),
                    ('e_dmp', 'largest relative interpolation error, damped system, median over realizations', np.median),
                    ('e_dmp', 'largest relative interpolation error, damped system, max over realizations', np.max)):
    print(f" {lab}")
    for bn in ('SOD', 'D', 'LNM'):
        print(f"   {bn:4s} " + " ".join(f"{agg(out[(bn, r, k)]):8.1e}" if (bn, r, k) in out else "    --  " for r in range(1, RMAX + 1)))
dep = {k[:2]: len(v) for k, v in out.items() if k[2] == 'dep'}
if dep: print(" bases not formed (dependent vectors), count over realizations:", dep)
