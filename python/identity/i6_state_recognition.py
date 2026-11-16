"""State-space version of i5: does the circularity test recognise the heavily damped planes that the spectral test misses?
   For each order r the same order-matched decay; state SOD (symmetric pencil on y = (x, xd), model derivatives);
   planes paired by sellib.pair_state (band up to 1.2 omega_c). State bases of dimension 2r, energy-Galerkin ROM:
     sR0   the 2r smoothest state smooth modes;   sRHB  recognised planes with frequency <= omega_c first, then SOD order.
   Same test loads (seed 500 + case) and baselines as i5, so the tables compare directly. The complex eigenvalues of
   the first modes are printed for reference. Original i5 header follows.
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
   Usage: python3 i6_state_recognition.py <0|1> -> i6_out_<k>.txt (stdout), i6_case<k>.pkl. R0/RHB below are sR0/sRHB."""
import os, sys, pickle, time, numpy as np
from scipy.linalg import eigh
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'rom'))
from systems import beam, state
from romlib import band_noise
from romlib2 import simulate_state, lnm, mta, complex_modes, rom_config, rom_state, merr
from sellib import gsod, pair_state, rom_order_state
ci = int(sys.argv[1]); cname = ['dashpot', 'dashpot_heavy'][ci]; NR = 20; ORD = range(2, 9); dt = 1e-3; N = 200000; wc = 250.0; wmax = 1.2 * wc
S = beam(dashpot=(0.2, 1.0)); M, K, B = S['M'], S['K'], S['B']; C = S['C'][cname]; n = len(M); A, Bs = state(M, K, C, B)
wL2, PL = eigh(K, M); wL = np.sqrt(wL2); kEE = np.where(wL < np.pi / (4 * dt))[0]; wk = wL[kEE]
bases = {}; info = []; WE = np.block([[K, np.zeros((n, n))], [np.zeros((n, n)), M]])
ev = np.linalg.eigvals(A); ev = ev[ev.imag > 0]; ev = ev[np.argsort(abs(ev))][:6]
print('complex modes |s|, zeta:', [(round(abs(s), 2), round(-s.real / abs(s), 3)) for s in ev], '; real eigenvalues:', int((abs(np.linalg.eigvals(A).imag) < 1e-9).sum()), flush=True)
for r in ORD:
    x0 = PL[:, kEE] @ (1 / (1 + (wk / wL[r - 1])**4) / wk); x0 /= abs(x0).max()
    Y = simulate_state(A, Bs, np.zeros((N, 2)), dt, np.r_[x0, np.zeros(n)]); Yd = Y @ A.T
    Psi, Phi, mu = gsod(Y, Yd, 1e-10); Q = Y @ Psi
    planes = pair_state(Q, dt, wmax, Phi=Phi); pb = [p for p in planes if p[2] <= wc]
    od = rom_order_state(pb, Phi.shape[1]); ok = Phi.shape[1] >= 2 * r
    bases[('R0', r)] = Phi[:, :2 * r] if ok else None; bases[('RHB', r)] = Phi[:, od[:2 * r]] if ok else None
    bases[('LNM', r)] = lnm(M, K, r); bases[('MTA-tip', r)] = mta(M, K, B[:, :1], r)
    info.append(f"r={r}: state rank {Phi.shape[1]}; SOD freqs of the first 2r {np.round(np.sqrt(mu[:2 * r]), 1)}; in-band planes (freq, one-sidedness) "
                f"{[(round(float(p[2]), 1), round(float(p[3]), 2)) for p in pb]}; sRHB takes {[int(i) for i in od[:2 * r]]}")
    print(info[-1], flush=True)
PsiC, PhiC = complex_modes(A, 2 * max(ORD))
rng = np.random.default_rng(500 + ci); res = {}; t0 = time.time()
for rr in range(NR):
    tests = {'tip': band_noise(N, dt, wc, 2, rng) * np.r_[1, 0], 'mid': band_noise(N, dt, wc, 2, rng) * np.r_[0, 1]}
    Xt = {k: simulate_state(A, Bs, v, dt)[::5, :n] for k, v in tests.items()}
    for r in ORD:
        for tk, v in tests.items():
            for bn in ('LNM', 'MTA-tip'):
                res.setdefault((bn, r, tk), []).append(merr(rom_config(M, K, C, B, bases[(bn, r)], v, dt, dec=5), Xt[tk], M))
            for bn in ('R0', 'RHB'):
                e = np.nan if bases[(bn, r)] is None else merr(rom_state(A, Bs, None, bases[(bn, r)], v, dt, None, n, W=WE, dec=5)[0], Xt[tk], M)
                res.setdefault((bn, r, tk), []).append(e)
            Xr = rom_state(A, Bs, PsiC[:, :2 * r], PhiC[:, :2 * r], v, dt, None, n, W=None, dec=5)[0]
            res.setdefault(('CM', r, tk), []).append(merr(Xr, Xt[tk], M))
    print(f"S3 {cname}: realization {rr + 1}/{NR} ({time.time() - t0:.0f}s)", flush=True)
pickle.dump(dict(res=res, info=info), open(os.path.join(HERE, f'i6_case{ci}.pkl'), 'wb'))
for tk in ('tip', 'mid'):
    print(f"== S3 {cname}, random {tk} load: median error over {NR} realizations, r = 2..8")
    for bn in ('LNM', 'CM', 'MTA-tip', 'R0', 'RHB'):
        print(f"   {bn:8s} " + " ".join(f"{np.median(res[(bn, r, tk)]):8.1e}" for r in ORD))
    for bn in ('R0', 'RHB'):
        print(f"   median over realizations of {bn}/LNM: " + " ".join(f"{np.median(np.array(res[(bn, r, tk)]) / np.array(res[('LNM', r, tk)])):6.2f}" for r in ORD))
