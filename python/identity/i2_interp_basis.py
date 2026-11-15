"""Is an SOD basis of a forced record a model basis of load deflections at the SOD frequencies?
   i1_forced_identity.py shows that each smooth mode phi_i of a single-input record is close to the load deflection
   d(w_i) = (K - w_i^2 M)^-1 b at its own SOD frequency w_i = sqrt(mu_i): the undamped frequency response to the load,
   evaluated at w_i. A basis of such vectors is the one-sided rational Krylov (interpolatory) projection basis with shifts
   at the SOD frequencies. Here the model-built basis D_r = span{d(w_1), ..., d(w_r)}, with w_i taken from the training
   record, is compared with SOD, the LNMs and MTA with the training-load vector only (MTA-tip: r-1 LNMs + K^-1 b) on
   the c18 records (same seeds and draw order, 20 realizations) and tests: 'rand' (new random load at the training
   input) and, for the beams, 'mid' (random load at mid-span). Error and ROM as in c18 (merr, rom_config, every 5th
   sample). A set of vectors whose normalized Gram matrix has smallest eigenvalue below 1e-12 of its largest is
   reported as dependent and not used.
   Usage: python3 i2_interp_basis.py <case 0..5> -> i2_out_<case>.txt (stdout), i2_case<case>.pkl"""
import os, sys, pickle, time, numpy as np
from scipy.linalg import eigh
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'rom'))
from systems import shear_frame, beam, state
from romlib import band_noise
from romlib2 import simulate_state, lnm, rom_config, merr
from i1_forced_identity import sod_pairs
ci = int(sys.argv[1]); NR = 20; RMAX = 8; DEC = 5
CASES = [('S1', 'light'), ('S1', 'heavy'), ('S2', 'light'), ('S2', 'heavy'), ('S3', 'dashpot'), ('S3', 'dashpot_heavy')]
sysname, cname = CASES[ci]
S = shear_frame() if sysname == 'S1' else (beam() if sysname == 'S2' else beam(dashpot=(0.2, 1.0)))
p = dict(dt=0.01, T=300, wc=23.0) if sysname == 'S1' else dict(dt=1e-3, T=200, wc=250.0)
M, K, B = S['M'], S['K'], S['B']; C = S['C'][cname]; n = len(M); nin = B.shape[1]; dt = p['dt']; N = int(p['T'] / dt)
Ast, Bs = state(M, K, C, B); e_in = np.r_[1, np.zeros(nin - 1)]; b = B @ e_in
def morth(Q):
    Q = Q / np.sqrt(np.einsum('ij,ij->j', Q, M @ Q)); g, V = eigh(Q.T @ M @ Q)
    if g[0] <= 1e-12 * g[-1]: return None
    return Q @ V / np.sqrt(g)
def unit(Q): return Q / np.sqrt(np.einsum('ij,ij->j', Q, M @ Q))
PL = lnm(M, K, n); stat = np.linalg.solve(K, b)
rng = np.random.default_rng(100 + ci); res = {}; dep = {}; t0 = time.time()
for rr in range(NR):
    u_tr = band_noise(N, dt, p['wc'], nin, rng) * e_in
    tests = {'rand': band_noise(N, dt, p['wc'], nin, rng) * e_in}
    if nin > 1: tests['mid'] = band_noise(N, dt, p['wc'], nin, rng) * np.r_[0, 1]
    Y = simulate_state(Ast, Bs, u_tr, dt); X, V = Y[:, :n], Y[:, n:]
    mu, Psi = sod_pairs(X, V); Sx = X.T @ X / N; Phi = unit(Sx @ Psi)            # smooth modes, smoothest first
    w = np.sqrt(mu); D = unit(np.c_[tuple(np.linalg.solve(K - wi**2 * M, b) for wi in w)])
    Xt = {k: simulate_state(Ast, Bs, u, dt)[::DEC, :n] for k, u in tests.items()}
    for r in range(1, RMAX + 1):
        bases = {'LNM': PL[:, :r], 'MTA-tip': (morth(np.c_[PL[:, :r - 1], stat]) if r > 1 else unit(stat[:, None])),
                 'SOD': Phi[:, :r] if r <= Phi.shape[1] else None, 'D (load defl. at SOD freqs)': morth(D[:, :r]) if r <= D.shape[1] else None}
        for bn, Q in bases.items():
            if Q is None: dep[(bn, r)] = dep.get((bn, r), 0) + 1; continue
            for tk, u in tests.items():
                res.setdefault((bn, r, tk), []).append(merr(rom_config(M, K, C, B, Q, u, dt, None, dec=DEC), Xt[tk], M))
    print(f"{sysname} {cname}: realization {rr + 1}/{NR} ({time.time() - t0:.0f}s)", flush=True)
pickle.dump(dict(case=CASES[ci], res=res, dep=dep), open(os.path.join(HERE, f'i2_case{ci}.pkl'), 'wb'))
names = ['LNM', 'MTA-tip', 'SOD', 'D (load defl. at SOD freqs)']
for tk in tests:
    print(f"== {sysname} {cname}, test {tk}: median error over {NR} realizations, r = 1..{RMAX} (n = realizations used)")
    for bn in names:
        row = [f"{np.median(res[(bn, r, tk)]):8.1e}" if (bn, r, tk) in res else "    --  " for r in range(1, RMAX + 1)]
        cnt = [len(res.get((bn, r, tk), [])) for r in range(1, RMAX + 1)]
        print(f"   {bn:28s} " + " ".join(row) + f"   n = {cnt}")
    print("   median over realizations of D/SOD error ratio:", " ".join(
        f"{np.median(np.array(res[('D (load defl. at SOD freqs)', r, tk)]) / np.array(res[('SOD', r, tk)])):.2f}"
        if ('D (load defl. at SOD freqs)', r, tk) in res and len(res[('D (load defl. at SOD freqs)', r, tk)]) == len(res.get(('SOD', r, tk), [])) else "--"
        for r in range(1, RMAX + 1)))
if dep: print("bases not formed (dependent vectors or beyond the data rank), count over realizations:", dep)
