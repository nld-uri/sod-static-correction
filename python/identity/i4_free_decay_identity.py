"""Free-decay SOD directions: LNMs, plus responses to localized non-modal forces at the SOD frequencies.
   The record is the order-matched band-limited free decay of c18_ensemble.py (released from rest at
   x0 = sum_k a_k phi_k / omega_k, a_k = 1/(1 + (omega_k/omega_r)^4), scaled to max |x0| = 1), here for r = 8.
   In a free decay the only forces are internal. With Rayleigh damping alone (S1, S2) there is no localized force and
   i1's identity says each smooth mode is an LNM up to the damping and record-end terms. On S3 the tip dashpot is a
   localized force, f = -c_d xd_tip e_tip; taken as the input, each smooth mode is close to the dashpot-force response
   (K - mu_i M)^-1 e_tip at its own SOD frequency, up to the Rayleigh-damping, end, sampling and rank terms.
   Reported per direction: SOD frequency, nearest LNM and the angle to it (mass metric), the angle to the tip-force
   response at the SOD frequency, and (S3) the term shares of i1 with the dashpot force as the load.
   Usage: python3 i4_free_decay_identity.py -> i4_out.txt (stdout)"""
import os, sys, numpy as np
from scipy.linalg import eigh
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'rom'))
from systems import shear_frame, beam, state, rayleigh
from romlib2 import simulate_state
from i1_forced_identity import budget, sod_pairs
CASES = [('S1', 'light'), ('S1', 'heavy'), ('S2', 'light'), ('S2', 'heavy'), ('S3', 'dashpot'), ('S3', 'dashpot_heavy')]
R = 8
def ang(a, b, M): return np.degrees(np.arccos(min(1.0, abs(a @ M @ b) / np.sqrt((a @ M @ a) * (b @ M @ b)))))
for ci, (sysname, cname) in enumerate(CASES):
    S = shear_frame() if sysname == 'S1' else (beam() if sysname == 'S2' else beam(dashpot=(0.2, 1.0)))
    p = dict(dt=0.01, T=300) if sysname == 'S1' else dict(dt=1e-3, T=200)
    M, K, B = S['M'], S['K'], S['B']; C = S['C'][cname]; n = len(M); nin = B.shape[1]; dt = p['dt']; N = int(p['T'] / dt)
    Ast, Bs = state(M, K, C, B); wL2, PL = eigh(K, M); kEE = np.where(np.sqrt(wL2) < np.pi / (4 * dt))[0]; wk = np.sqrt(wL2[kEE])
    a = 1 / (1 + (wk / np.sqrt(wL2[R - 1]))**4); x0 = PL[:, kEE] @ (a / wk); x0 /= abs(x0).max()
    Y = simulate_state(Ast, Bs, np.zeros((N, nin)), dt, np.r_[x0, np.zeros(n)]); Yd = Y @ Ast.T
    X, V, A = Y[:, :n], Y[:, n:], Yd[:, n:]
    mu, Psi = sod_pairs(X, V); Phi = (X.T @ X / N) @ Psi
    print(f"== {sysname} {cname}: order-{R} band-limited free decay, data rank {len(mu)}")
    if sysname == 'S3':
        tip = S['tip']; et = np.zeros(n); et[tip] = 1.0; cd = S['cd'][0 if cname == 'dashpot' else 1]
        u = -cd * V[:, tip:tip + 1]; Cr = C - cd * np.outer(et, et)                  # dashpot force as the input
        rows, _ = budget(X, V, A, u, M, K, Cr, et[:, None], dt, nd=R)
    print("   i  SOD freq   nearest LNM (freq)   angle to it   angle to tip-force response" + ("   shares: load  Rayleigh  ends  sampling  rank" if sysname == 'S3' else ""))
    for i in range(min(R, len(mu))):
        ph = Phi[:, i]; angs = [ang(ph, PL[:, k], M) for k in range(n)]; k = int(np.argmin(angs))
        et_ = np.zeros(n); et_[S['tip'] if sysname != 'S1' else n - 1] = 1.0
        dang = ang(ph, np.linalg.solve(K - mu[i] * M, et_), M)
        line = f"  {i + 1:2d} {np.sqrt(mu[i]):9.3f}   LNM {k + 1:2d} ({np.sqrt(wL2[k]):8.3f})   {angs[k]:9.2e}     {dang:9.2e}"
        if sysname == 'S3':
            r_ = rows[i]; line += f"          {r_['load']:8.2e} {r_['damp']:8.2e} {r_['ends']:8.2e} {r_['disc']:8.2e} {r_['rank']:8.2e}"
        print(line, flush=True)
