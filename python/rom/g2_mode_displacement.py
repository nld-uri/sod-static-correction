"""SOD-ROM story check: when the tip static direction enters SOD (rand) at r = 3..5, does it displace a linear mode?
   Same records as g1 (c18 case 2 seeds and draw order). For r = 1..8: M-norm residual of LNM k (k = 1..6) outside
   span(SOD_r), median over realizations; a mode counts as held if its residual < 0.1.
   Usage: python3 g2_mode_displacement.py [NR] -> stdout"""
import numpy as np, sys
from systems import beam, state
from romlib import band_noise
from romlib2 import simulate_state, lnm, sod
NR = int(sys.argv[1]) if len(sys.argv) > 1 else 10
S = beam(); M, K, B = S['M'], S['K'], S['B']; C = S['C']['light']; n = len(M); A, Bs = state(M, K, C, B)
dt = 1e-3; N = 200000; wc = 250.0; L = lnm(M, K, 8)
def mres(Phi, s):
    c = np.linalg.solve(Phi.T @ M @ Phi, Phi.T @ M @ s); d = s - Phi @ c; return np.sqrt(d @ M @ d / (s @ M @ s))
rng = np.random.default_rng(102); R = {}
for rr in range(NR):
    u = band_noise(N, dt, wc, 2, rng) * np.r_[1, 0]; band_noise(N, dt, wc, 2, rng); band_noise(N, dt, wc, 2, rng)
    Y = simulate_state(A, Bs, u, dt); Ps = sod(Y[:, :n], Y[:, n:], 10)[0]
    for r in range(1, min(8, Ps.shape[1]) + 1):
        R.setdefault(r, []).append([mres(Ps[:, :r], L[:, k]) for k in range(7)])
import pickle; pickle.dump(R, open("g2_results.pkl", "wb"))
print(f"median M-residual of LNM k outside span(SOD_r), S2 light, random tip training, {NR} realizations")
print("  r | " + " ".join(f"LNM{k+1:<5d}" for k in range(7)) + "| modes held (<0.1)")
for r, v in R.items():
    m = np.median(np.array(v), axis=0)
    print(f"{r:3d} | " + " ".join(f"{x:8.1e}" for x in m) + f" | {[k+1 for k in range(7) if m[k] < 0.1]}")
