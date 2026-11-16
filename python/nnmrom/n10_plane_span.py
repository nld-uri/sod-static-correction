"""Does the NNM1 plane already span what its companion adds? For the static-release decays of n9 (30-240 N):
   (i) the second displacement direction of the plane (singular values of its displacement parts, angle of the
   second direction to LNM 2 and to the static tip-load vector, mass metric); (ii) how much of the companion's
   displacement part lies outside the plane's displacement span (relative M-norm of the residual, angle to the span);
   (iii) where the configuration-SOD r = 2 and 3 subspaces lie relative to plane + companion (largest principal angle).
   Output: n10_out.txt."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rom'))
from scipy.linalg import eigh, subspace_angles
from costbeam import costbeam
from nlsim import newmark, static_deflection
from sellib import gsod, pair_state, dct_spectrum, peak_and_width
S = costbeam(); Sd = costbeam(zeta=0.01); n = S['n']; tip = S['tip']; k3 = S['k3']; pt = np.zeros(n); pt[tip] = 1.0
M, K, C = S['M'], S['K'], Sd['C']; dt = 2e-5; ds = 5 * dt; wl2, PL = eigh(K, M); st = np.linalg.solve(K, pt); L = np.linalg.cholesky(M).T
lines = []
def log(s): print(s, flush=True); lines.append(s)
def ang(A, B): return np.degrees(subspace_angles(L @ A, L @ B)).max()
def csod(X, V, tol=1e-10):
    w, U = eigh(X.T @ X); U = U[:, w > tol * w.max()]; Xp = X @ U; Vp = V @ U; mu, Psi = eigh(Vp.T @ Vp, Xp.T @ Xp); return U @ (Xp.T @ Xp) @ Psi
for P in (30.0, 60.0, 120.0, 240.0):
    x0 = static_deflection(S, P); _, X, V = newmark(M, K, C, pt, k3, x0, np.zeros(n), 1.5, dt); X, V = X[::5], V[::5]
    A = -np.linalg.solve(M, (K @ X.T + C @ V.T + k3 * np.outer(pt, X[:, tip]**3))).T
    Y, Yd = np.c_[X, V], np.c_[V, A]; Psi, Phi, mu = gsod(Y, Yd, 1e-10); Q = Y @ Psi
    om, Fq = dct_spectrum(Q, ds); pk, _ = peak_and_width(om, Fq); en = (Phi**2).sum(0) * Q.var(0)
    planes = pair_state(Q, ds, 2 * np.pi * 2000); a_, b_, fp, c = planes[0]
    D = Phi[:n, [a_, b_]]; Dm = L @ D; U, sv, _ = np.linalg.svd(Dm, full_matrices=False); U = np.linalg.solve(L, U)
    log(f"P={P:.0f} N: plane {fp:.1f} rad/s; displacement singular values ratio s2/s1 = {sv[1] / sv[0]:.2e}; "
        f"second direction: {ang(U[:, [1]], PL[:, [1]]):.1f} deg to LNM 2, {ang(U[:, [1]], st[:, None]):.1f} deg to static tip vector, "
        f"{ang(U[:, [1]], PL[:, [0]]):.1f} deg to LNM 1; first direction {ang(U[:, [0]], PL[:, [0]]):.2f} deg to LNM 1")
    used = {i for p in planes for i in p[:2]}; om2 = 2 * np.pi * np.fft.fftfreq(len(Q), ds)
    z = (Q[:, a_] - Q[:, a_].mean()) / Q[:, a_].std() + 1j * (Q[:, b_] - Q[:, b_].mean()) / Q[:, b_].std(); Fz = np.abs(np.fft.fft(z))**2
    side = om2 > 0 if Fz[om2 > 0].sum() >= Fz[om2 < 0].sum() else om2 < 0; w_s = abs(om2[side]); oo = np.argsort(w_s); cdf = np.cumsum(Fz[side][oo]) / Fz[side].sum()
    lo, hi = w_s[oo][np.searchsorted(cdf, 0.05)], w_s[oo][np.searchsorted(cdf, 0.95)]
    comp = sorted([i for i in range(Q.shape[1]) if i not in used and lo <= pk[i] <= hi], key=lambda i: -en[i])
    for i in comp[:1]:
        c_ = Phi[:n, i]; Pm = D @ np.linalg.lstsq(Dm, L @ c_, rcond=None)[0]; r_ = c_ - Pm
        log(f"   companion {i} (peak {pk[i]:.0f} rad/s, energy share {en[i] / en.sum():.1e}): residual outside plane span {np.sqrt(r_ @ M @ r_ / (c_ @ M @ c_)):.3f}, "
            f"angle to plane span {ang(c_[:, None], D):.1f} deg; residual direction {ang(r_[:, None], PL[:, [1]]):.1f} deg to LNM 2, {ang(r_[:, None], st[:, None]):.1f} deg to static vector")
    Cs = csod(X, V)
    for r in (2, 3):
        log(f"   config SOD r={r}: largest angle to plane span {ang(Cs[:, :r], D) if r == 2 else float('nan'):.1f} deg; to LNM 1-{r} {ang(Cs[:, :r], PL[:, :r]):.1f} deg; "
            f"SOD direction {r} vs LNM {r}: {ang(Cs[:, [r - 1]], PL[:, [r - 1]]):.1f} deg, vs static vector {ang(Cs[:, [r - 1]], st[:, None]):.1f} deg")
open('n10_out.txt', 'w').write("\n".join(lines) + "\n")
