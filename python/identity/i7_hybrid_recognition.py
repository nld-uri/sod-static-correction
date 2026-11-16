"""Hybrid of i5 and i6: configuration SOD basis, with its modal directions recognised through the state-SOD planes.
   i5: the spectral test misses the heavily damped first mode (zeta 0.60 at c_d = 1). i6: circularity recognises every
   in-band plane, but the state basis lacks the dashpot direction. Here each in-band plane recognised by circularity
   (sellib.pair_state on the state SOD of the same decay) marks as modal the configuration smooth mode closest (mass
   metric) to the plane's displacement span; RHB puts those first, by plane frequency, then the rest in SOD order.
   R0, baselines, decays, test loads and seeds as in i5. Original i5 header follows.
   Does recognising modes remove the low-order loss of a decay-trained SOD basis? (S3, tip dashpot.)
   Training: for each ROM order r = 2..8, the order-matched band-limited free decay of c18_ensemble.py (released from
   rest, a_k = 1/(1 + (omega_k/omega_r)^4)); configuration SOD with model velocities, rank tolerance 1e-10.
   Bases of order r from that one SOD:
     R0   the r smoothest smooth modes (plain SOD order);
     RHB  modes recognised by spectral consistency (sellib.select_config, band up to 1.2 omega_c) with peak frequency
          <= omega_c first, ordered by frequency, then the remaining coordinates in SOD order; first r taken.
   Baselines on the same test loads: LNM (r lowest), MTA-tip (r-1 LNMs + K^-1 e_tip: told where the dashpot is),
   CM (complex-mode truncation, state dimension 2r, as in c18).
   Tests (20 realizations, seed 500 + case): random load at the tip and random load at mid-span, band omega_c = 250,
   forced ROM response from rest; error as in c18 (merr, every 5th sample).
   Usage: python3 i7_hybrid_recognition.py <0|1> -> i7_out_<k>.txt (stdout), i7_case<k>.pkl"""
import os, sys, pickle, time, numpy as np
from scipy.linalg import eigh
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'rom'))
from systems import beam, state
from romlib import band_noise
from romlib2 import simulate_state, lnm, mta, complex_modes, rom_config, rom_state, merr
from sellib import gsod, pair_state, rom_order_config
from scipy.linalg import subspace_angles
Lc = None
ci = int(sys.argv[1]); cname = ['dashpot', 'dashpot_heavy'][ci]; NR = 20; ORD = range(2, 9); dt = 1e-3; N = 200000; wc = 250.0; wmax = 1.2 * wc
S = beam(dashpot=(0.2, 1.0)); M, K, B = S['M'], S['K'], S['B']; C = S['C'][cname]; n = len(M); A, Bs = state(M, K, C, B)
wL2, PL = eigh(K, M); wL = np.sqrt(wL2); kEE = np.where(wL < np.pi / (4 * dt))[0]; wk = wL[kEE]
bases = {}; info = []
for r in ORD:
    x0 = PL[:, kEE] @ (1 / (1 + (wk / wL[r - 1])**4) / wk); x0 /= abs(x0).max()
    Y = simulate_state(A, Bs, np.zeros((N, 2)), dt, np.r_[x0, np.zeros(n)]); Yd = Y @ A.T
    Psi, Phi, mu = gsod(Y[:, :n], Yd[:, :n], 1e-10); Q = Y[:, :n] @ Psi
    Ps_, Phs, mus = gsod(Y, Yd, 1e-10); planes = [p for p in pair_state(Y @ Ps_, dt, wmax, Phi=Phs) if p[2] <= wc]
    Lc = np.linalg.cholesky(M).T; ib = []; pk = np.zeros(Phi.shape[1]); angs = []
    for a_, b_, f_, c_ in sorted(planes, key=lambda p: p[2]):
        D2 = Lc @ Phs[:n, [a_, b_]]; U2 = np.linalg.svd(D2, full_matrices=False)[0]
        al = [np.degrees(np.arccos(min(1.0, np.linalg.norm(U2.T @ (Lc @ Phi[:, i])) / np.linalg.norm(Lc @ Phi[:, i])))) if i not in ib else 1e9 for i in range(Phi.shape[1])]
        i_ = int(np.argmin(al)); ib.append(i_); pk[i_] = f_; angs.append(round(al[i_], 2))
    od = rom_order_config(ib, Phi.shape[1]); un = lambda P: P / np.linalg.norm(P, axis=0)
    bases[('R0', r)] = un(Phi[:, :r]); bases[('RHB', r)] = un(Phi[:, od[:r]])
    bases[('LNM', r)] = lnm(M, K, r); bases[('MTA-tip', r)] = mta(M, K, B[:, :1], r)
    non = [i for i in range(r) if i not in ib]
    info.append(f"r={r}: rank {Phi.shape[1]}; SOD freqs of the first r {np.round(np.sqrt(mu[:r]), 1)}; recognised in-band modes at "
                f"{np.round(pk[ib], 1)} (angles to their planes, deg: {angs}); unrecognised among the first r: {[f'{np.sqrt(mu[i]):.1f}' for i in non]}; RHB takes {od[:r]}")
    print(info[-1], flush=True)
PsiC, PhiC = complex_modes(A, 2 * max(ORD))
rng = np.random.default_rng(500 + ci); res = {}; t0 = time.time()
for rr in range(NR):
    tests = {'tip': band_noise(N, dt, wc, 2, rng) * np.r_[1, 0], 'mid': band_noise(N, dt, wc, 2, rng) * np.r_[0, 1]}
    Xt = {k: simulate_state(A, Bs, v, dt)[::5, :n] for k, v in tests.items()}
    for r in ORD:
        for tk, v in tests.items():
            for bn in ('R0', 'RHB', 'LNM', 'MTA-tip'):
                res.setdefault((bn, r, tk), []).append(merr(rom_config(M, K, C, B, bases[(bn, r)], v, dt, dec=5), Xt[tk], M))
            Xr = rom_state(A, Bs, PsiC[:, :2 * r], PhiC[:, :2 * r], v, dt, None, n, W=None, dec=5)[0]
            res.setdefault(('CM', r, tk), []).append(merr(Xr, Xt[tk], M))
    print(f"S3 {cname}: realization {rr + 1}/{NR} ({time.time() - t0:.0f}s)", flush=True)
pickle.dump(dict(res=res, info=info), open(os.path.join(HERE, f'i7_case{ci}.pkl'), 'wb'))
for tk in ('tip', 'mid'):
    print(f"== S3 {cname}, random {tk} load: median error over {NR} realizations, r = 2..8")
    for bn in ('LNM', 'CM', 'MTA-tip', 'R0', 'RHB'):
        print(f"   {bn:8s} " + " ".join(f"{np.median(res[(bn, r, tk)]):8.1e}" for r in ORD))
    for bn in ('R0', 'RHB'):
        print(f"   median over realizations of {bn}/LNM: " + " ".join(f"{np.median(np.array(res[(bn, r, tk)]) / np.array(res[('LNM', r, tk)])):6.2f}" for r in ORD))
