"""Does recognising modes remove the low-order loss of a decay-trained SOD basis? (S3, tip dashpot.)
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
   Usage: python3 i5_decay_recognition.py <0|1>  (0: c_d = 0.2, 1: c_d = 1.0) -> i5_out_<k>.txt (stdout), i5_case<k>.pkl"""
import os, sys, pickle, time, numpy as np
from scipy.linalg import eigh
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'rom'))
from systems import beam, state
from romlib import band_noise
from romlib2 import simulate_state, lnm, mta, complex_modes, rom_config, rom_state, merr
from sellib import gsod, select_config, rom_order_config
ci = int(sys.argv[1]); cname = ['dashpot', 'dashpot_heavy'][ci]; NR = 20; ORD = range(2, 9); dt = 1e-3; N = 200000; wc = 250.0; wmax = 1.2 * wc
S = beam(dashpot=(0.2, 1.0)); M, K, B = S['M'], S['K'], S['B']; C = S['C'][cname]; n = len(M); A, Bs = state(M, K, C, B)
wL2, PL = eigh(K, M); wL = np.sqrt(wL2); kEE = np.where(wL < np.pi / (4 * dt))[0]; wk = wL[kEE]
bases = {}; info = []
for r in ORD:
    x0 = PL[:, kEE] @ (1 / (1 + (wk / wL[r - 1])**4) / wk); x0 /= abs(x0).max()
    Y = simulate_state(A, Bs, np.zeros((N, 2)), dt, np.r_[x0, np.zeros(n)]); Yd = Y @ A.T
    Psi, Phi, mu = gsod(Y[:, :n], Yd[:, :n], 1e-10); Q = Y[:, :n] @ Psi
    idx, pk, ws = select_config(Q, mu, dt, wmax, Phi=Phi); ib = [i for i in idx if pk[i] <= wc]
    od = rom_order_config(ib, Phi.shape[1]); un = lambda P: P / np.linalg.norm(P, axis=0)
    bases[('R0', r)] = un(Phi[:, :r]); bases[('RHB', r)] = un(Phi[:, od[:r]])
    bases[('LNM', r)] = lnm(M, K, r); bases[('MTA-tip', r)] = mta(M, K, B[:, :1], r)
    non = [i for i in range(r) if i not in ib]
    info.append(f"r={r}: rank {Phi.shape[1]}; SOD freqs of the first r {np.round(np.sqrt(mu[:r]), 1)}; recognised in-band modes at "
                f"{np.round(pk[ib], 1)}; unrecognised among the first r: {[f'{np.sqrt(mu[i]):.1f}' for i in non]}; RHB takes {od[:r]}")
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
pickle.dump(dict(res=res, info=info), open(os.path.join(HERE, f'i5_case{ci}.pkl'), 'wb'))
for tk in ('tip', 'mid'):
    print(f"== S3 {cname}, random {tk} load: median error over {NR} realizations, r = 2..8")
    for bn in ('LNM', 'CM', 'MTA-tip', 'R0', 'RHB'):
        print(f"   {bn:8s} " + " ".join(f"{np.median(res[(bn, r, tk)]):8.1e}" for r in ORD))
    for bn in ('R0', 'RHB'):
        print(f"   median over realizations of {bn}/LNM: " + " ".join(f"{np.median(np.array(res[(bn, r, tk)]) / np.array(res[('LNM', r, tk)])):6.2f}" for r in ORD))
