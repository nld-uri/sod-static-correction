"""SOD-ROM story check: is the direction that displaces mode k at order r the tip static correction?
   Largest principal angle (M metric) between span(SOD_r) and span(LNM_1..r-1, K^-1 b_tip), r = 3..8;
   for contrast, with K^-1 b_mid. Same records as g1/g2. Usage: python3 g3_third_direction.py [NR]"""
import numpy as np, sys
from systems import beam, state
from romlib import band_noise
from romlib2 import simulate_state, lnm, sod
from sodtoolbox.rom import principal_angles
NR = int(sys.argv[1]) if len(sys.argv) > 1 else 10
S = beam(); M, K, B = S['M'], S['K'], S['B']; C = S['C']['light']; n = len(M); A, Bs = state(M, K, C, B)
dt = 1e-3; N = 200000; wc = 250.0; st = np.linalg.solve(K, B[:, 0]); sm = np.linalg.solve(K, B[:, 1])
rng = np.random.default_rng(102); R = {}
for rr in range(NR):
    u = band_noise(N, dt, wc, 2, rng) * np.r_[1, 0]; band_noise(N, dt, wc, 2, rng); band_noise(N, dt, wc, 2, rng)
    Y = simulate_state(A, Bs, u, dt); Ps = sod(Y[:, :n], Y[:, n:], 10)[0]
    for r in range(2, min(8, Ps.shape[1]) + 1):
        R.setdefault(r, []).append([principal_angles(Ps[:, :r], np.c_[lnm(M, K, r - 1), s], M).max() for s in (st, sm)]
                                   + [principal_angles(Ps[:, :r], lnm(M, K, r), M).max()])
import pickle; pickle.dump(R, open("g3_results.pkl", "wb"))
print(f"largest principal angle (deg, M metric), median [max] over {NR} realizations, S2 light, random tip training")
print("  r | SOD_r vs LNM_1..r-1 + K^-1 b_tip | vs LNM_1..r-1 + K^-1 b_mid | vs LNM_1..r")
for r, v in R.items():
    v = np.array(v); md = np.median(v, axis=0); mx = v.max(axis=0)
    print(f"{r:3d} | {md[0]:8.3f} [{mx[0]:7.3f}]              | {md[1]:8.3f} [{mx[1]:7.3f}]        | {md[2]:7.3f} [{mx[2]:7.3f}]")
