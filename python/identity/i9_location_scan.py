"""Does the non-modal direction of a dashpot decay point at the dashpot? (S3, order-8 band-limited free decay.)
   The non-modal smooth mode (largest angle to its nearest LNM among the first 8) is compared with the force responses
   d_j(w) = (K - w^2 M)^-1 e_j at its own SOD frequency w, for a unit load e_j at EVERY degree of freedom j
   (translations and rotations of the 30 nodes). If the direction is the response to the dashpot force, the angle is
   smallest at the tip translation, where the dashpot acts. A modal direction (the second) is scanned as a control:
   near a resonance d_j is nearly the mode for any j, so its angles should not single out the tip.
   Both SOD implementations are used: 'plain' (i1/i4: POD-subspace cut at 1e-10 on unscaled channels) and 'scaled'
   (sellib.gsod, used for every ROM basis of i5-i7: unit-variance channels before the same cut).
   Angles in the mass metric, degrees. Usage: python3 i9_location_scan.py > i9_out.txt"""
import os, sys, numpy as np
from scipy.linalg import eigh
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'rom'))
from systems import beam, state
from romlib2 import simulate_state
from sellib import gsod
from i1_forced_identity import sod_pairs
R = 8; dt = 1e-3; N = 200000
def ang(a, b, M): return np.degrees(np.arccos(min(1.0, abs(a @ M @ b) / np.sqrt((a @ M @ a) * (b @ M @ b)))))
S = beam(dashpot=(0.2, 1.0)); M, K, B = S['M'], S['K'], S['B']; n = len(M); tip = S['tip']
wL2, PL = eigh(K, M); wL = np.sqrt(wL2); kEE = np.where(wL < np.pi / (4 * dt))[0]; wk = wL[kEE]
x0 = PL[:, kEE] @ (1 / (1 + (wk / wL[R - 1])**4) / wk); x0 /= abs(x0).max()
for cname, cd in (('dashpot', 0.2), ('dashpot_heavy', 1.0)):
    C = S['C'][cname]; A, Bs = state(M, K, C, B)
    Y = simulate_state(A, Bs, np.zeros((N, B.shape[1])), dt, np.r_[x0, np.zeros(n)]); X, V = Y[:, :n], Y[:, n:]
    for impl in ('plain', 'scaled'):
        if impl == 'plain': mu, Psi = sod_pairs(X, V); Phi = (X.T @ X / N) @ Psi
        else: Psi, Phi, mu = gsod(X, V, 1e-10)
        aL = [min(ang(Phi[:, i], PL[:, k], M) for k in range(n)) for i in range(R)]; inon = int(np.argmax(aL[:6]))
        print(f"== S3 c_d = {cd}, {impl} SOD, rank {len(mu)}: SOD frequencies {np.round(np.sqrt(mu[:R]), 2)}")
        for lab, i in (('non-modal direction', inon), ('control: modal direction 2', 1)):
            w = np.sqrt(mu[i]); D = np.linalg.solve(K - mu[i] * M, np.eye(n)); a = np.array([ang(Phi[:, i], D[:, j], M) for j in range(n)])
            o = np.argsort(a); kind = lambda j: ('translation' if j % 2 == 0 else 'rotation') + f" of node {j // 2 + 1}"
            print(f"   {lab}: i = {i + 1}, SOD frequency {w:.2f}, angle to nearest LNM {aL[i]:.2f}")
            print(f"      angle to d_j at the tip translation (DOF {tip}): {a[tip]:.3f}; rank of the tip translation among {n} DOFs: {int(np.where(o == tip)[0][0]) + 1}")
            print(f"      three smallest: " + "; ".join(f"{kind(j)} {a[j]:.3f}" for j in o[:3]) + f"; median over DOFs {np.median(a):.2f}; largest {a.max():.2f}")
            print(f"      translations of nodes 5, 10, 15, 20, 25, 28, 29, 30: " + " ".join(f"{a[2 * (k - 1)]:.2f}" for k in (5, 10, 15, 20, 25, 28, 29, 30)))
