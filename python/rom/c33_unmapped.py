"""Small checks behind numbers that no other script prints.
   (1) Static release puts 98--99% of the initial energy in mode 1 (Sec. 3, S1--S3 training decays).
   (2) Mid-span lies within 2% of a node of modes 3, 5, 7 of the 30-element cantilever (Sec. 4).
   (3) S4 (COST F3) linear frequencies: 23.66 Hz (mode 1), 148.3 Hz (mode 2). No random numbers."""
import numpy as np, sys
from scipy.linalg import eigh
from systems import shear_frame, beam
sys.path.insert(0, '../nnmrom'); from costbeam import costbeam
def mode_energy(M, K, x0):
    w2, P = eigh(K, M); eta = P.T @ M @ x0; E = 0.5 * w2 * eta**2; return E / E.sum()
S1, S2 = shear_frame(), beam()
e1 = mode_energy(S1['M'], S1['K'], np.linalg.solve(S1['K'], S1['M'] @ np.ones(len(S1['M']))))
f = np.zeros(len(S2['M'])); f[S2['tip']] = 1.0; f[S2['mid']] = 0.7
e2 = mode_energy(S2['M'], S2['K'], np.linalg.solve(S2['K'], f))
print(f"(1) static release, mode-1 energy share: S1 {e1[0]:.4f}, S2/S3 {e2[0]:.4f}")
w2, P = eigh(S2['K'], S2['M']); T = P[0::2]                                   # translational rows
for k in (2, 4, 6):
    print(f"(2) mode {k + 1}: |phi(mid)| / max|phi| = {abs(P[S2['mid'], k]) / abs(T[:, k]).max():.4f}")
S4 = costbeam(); print(f"(3) S4 linear frequencies (Hz): {S4['w'][0] / 2 / np.pi:.2f}, {S4['w'][1] / 2 / np.pi:.1f}")
