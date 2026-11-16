"""Referee round 4, issue 1: does the high-order gain of data-driven ROM bases survive measurement noise in the training
   record? S2 (light and heavy Rayleigh), random tip training load as c18 (band 250, 200 s, dt = 1e-3), 5 realizations
   (seed 31 + case). Training data: exact (model) kinematics, or all 60 displacement channels with white noise at 60 or
   40 dB of each channel's standard deviation, smoothed by Whittaker (p = 3, fc = 100 Hz, as c22) with velocities from
   the smoothed record. Bases: POD of x, SOD of (x, v) (rank tolerance 1e-10 of the largest POD value, as c18); LNM and
   MTA (tip + mid static vectors) for reference. Tests: new random tip load and random mid-span load.
   Usage: python3 c31_noisy_training.py <case 2|3> -> c31_case<i>.pkl, printed medians for r = 1..10."""
import sys, os, numpy as np, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kin'))
from systems import beam, state
from romlib import band_noise
from romlib2 import simulate_state, lnm, mta, pod, sod, rom_config, merr
from kinematics import diff_whittaker, _central
ci = int(sys.argv[1]); NR = 5; cname = {2: 'light', 3: 'heavy'}[ci]
S = beam(); dt = 1e-3; N = 200000; wc = 250.0; RMAX = 10; DEC = 5
M, K, B = S['M'], S['K'], S['B']; C = S['C'][cname]; n = len(M); A, Bs = state(M, K, C, B)
rng = np.random.default_rng(31 + ci); res = {}; t0 = time.time()
for rr in range(NR):
    u_tr = band_noise(N, dt, wc, 2, rng) * np.r_[1, 0]
    tests = {'rand': band_noise(N, dt, wc, 2, rng) * np.r_[1, 0], 'mid': band_noise(N, dt, wc, 2, rng) * np.r_[0, 1]}
    Xt = {k: simulate_state(A, Bs, u, dt)[::DEC, :n] for k, u in tests.items()}
    Y = simulate_state(A, Bs, u_tr, dt); X0, V0 = Y[:, :n], Y[:, n:]; del Y
    bases = {'LNM': lambda r: lnm(M, K, r), 'MTA': lambda r: mta(M, K, B, r)}
    for tag in ('exact', '60', '40'):
        if tag == 'exact': X, V = X0, V0
        else:
            x = X0 + X0.std(0) * 10**(-int(tag) / 20) * rng.standard_normal(X0.shape)
            X = np.column_stack([diff_whittaker(x[:, j], dt, 0, fc=100.0) for j in range(n)]); V = np.column_stack([_central(X[:, j], dt, 1) for j in range(n)]); del x
        Pp = pod(X, RMAX); Ps = sod(X, V, RMAX)[0]
        bases[f'POD {tag}'] = (lambda r, Pp=Pp: Pp[:, :r]); bases[f'SOD {tag}'] = (lambda r, Ps=Ps: Ps[:, :r])
    for bn, fb in bases.items():
        for r in range(1, RMAX + 1):
            Phi = fb(r)
            if Phi is None or Phi.shape[1] < r: continue
            for tk, u in tests.items():
                res.setdefault((bn, r, tk), []).append(merr(rom_config(M, K, C, B, Phi, u, dt, None, dec=DEC), Xt[tk], M))
    print(f"S2 {cname}: realization {rr + 1}/{NR} ({time.time() - t0:.0f}s)", flush=True)
pickle.dump(dict(case=('S2', cname), res=res, NR=NR), open(f'c31_case{ci}.pkl', 'wb'))
for tk in ('rand', 'mid'):
    print(f"== S2 {cname}, test {tk}: median error, r = 1..{RMAX} (-- : fewer than {NR} realizations reach the order)")
    for bn in bases:
        print(f"   {bn:10s} " + " ".join(f"{np.median(res[(bn, r, tk)]):8.1e}" if len(res.get((bn, r, tk), [])) == NR else "      --" for r in range(1, RMAX + 1)))
