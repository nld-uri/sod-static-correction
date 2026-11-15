"""Two checks quoted in SOD-ROM Sec. 8 ("Bending out of the plane"), on the static-release decays (zeta = 0.01, 1.5 s):
   (i) configuration SOD of the decay: SOD frequency sqrt(mu_1) of the smoothest coordinate against its smoothed DCT
       peak (the spectral-consistency test), for 30-240 N;
   (ii) principal angle (mass metric) between the first configuration SOD direction and (K - w^2 M)^-1 e_tip, the
       static tip vector, LNM 1, with w the NNM1 plane frequency recognised by circularity (state SOD).
   Output: n13_out.txt."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rom'))
from scipy.linalg import eigh, subspace_angles
from costbeam import costbeam
from nlsim import newmark, static_deflection
from sellib import gsod, pair_state, dct_spectrum, peak_and_width
S = costbeam(); Sd = costbeam(zeta=0.01); n = S['n']; tip = S['tip']; k3 = S['k3']; pt = np.zeros(n); pt[tip] = 1.0
M, K, C = S['M'], S['K'], Sd['C']; ds = 1e-4; wl2, PL = eigh(K, M); L = np.linalg.cholesky(M).T; lines = []
def ang(a, b): return np.degrees(subspace_angles(L @ a[:, None], L @ b[:, None]).max())
for P in (30.0, 60.0, 120.0, 240.0):
    x0 = static_deflection(S, P); _, X, V = newmark(M, K, C, pt, k3, x0, np.zeros(n), 1.5, 2e-5); X, V = X[::5], V[::5]
    w, U = eigh(X.T @ X); U = U[:, w > 1e-10 * w.max()]; Xp = X @ U; Vp = V @ U; mu, Psi = eigh(Vp.T @ Vp, Xp.T @ Xp)
    Q = Xp @ Psi; Phi1 = U @ (Xp.T @ Xp) @ Psi[:, 0]; om, F = dct_spectrum(Q[:, :1], ds); pk, rel = peak_and_width(om, F)
    A = -np.linalg.solve(M, (K @ X.T + C @ V.T + k3 * np.outer(pt, X[:, tip]**3))).T
    Ps, Ph, ms = gsod(np.c_[X, V], np.c_[V, A], 1e-10); Qs = np.c_[X, V] @ Ps; wp = pair_state(Qs, ds, 2 * np.pi * 2000, Phi=Ph)[0][2]
    lines.append(f"P={P:.0f} N: smoothest config coordinate sqrt(mu) {np.sqrt(mu[0]):.1f} rad/s, peak {pk[0]:.1f}, separation {abs(pk[0] / np.sqrt(mu[0]) - 1):.3f}, "
                 f"half-power width {rel[0]:.3f}; first SOD direction vs (K - w^2 M)^-1 e_tip (w = {wp:.1f}) {ang(Phi1, np.linalg.solve(K - wp**2 * M, pt)):.2f} deg, "
                 f"vs static tip vector {ang(Phi1, np.linalg.solve(K, pt)):.2f} deg, vs LNM 1 {ang(Phi1, PL[:, 0]):.2f} deg")
    print(lines[-1], flush=True)
open('n13_out.txt', 'w').write("\n".join(lines) + "\n")
