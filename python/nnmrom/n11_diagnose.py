"""Two diagnostics for the frequency (rule-3) ordering on the COST beam static-release decays (as n9, model kinematics).
   A. Embedding dimension of NNM1: the state record (x, v) and the displacement record x are band-pass filtered (FFT
      mask) to the 5-95% power band of the NNM1 plane, and separately to 3x that band. Singular values of the filtered
      clouds in the energy metric (state: W = blkdiag(K, M); configuration: M), normalised by the largest. Prediction:
      configuration 1 + m, state 2 + 2m, for m bend directions.
   B. Why the mode-2 plane can fail circularity: all state-SOD coordinates with a peak within 10% of LNM 2; their SOD
      frequencies, energy shares, one-sidedness of every pair among them and with the mode-1 coordinates, the product
      rule (mu_a mu_b)^(1/4) against the LNM 2 frequency, and the largest principal angle (energy metric) of each pair's
      span to the LNM-2 state plane span{(phi2, 0), (0, phi2)}. Output: n11_out.txt."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rom'))
from scipy.linalg import eigh, subspace_angles
from costbeam import costbeam
from nlsim import newmark, static_deflection
from sellib import gsod, pair_state, dct_spectrum, peak_and_width, onesided
S = costbeam(); Sd = costbeam(zeta=0.01); n = S['n']; tip = S['tip']; k3 = S['k3']; pt = np.zeros(n); pt[tip] = 1.0
M, K, C = S['M'], S['K'], Sd['C']; dt = 2e-5; ds = 5 * dt; wl2, PL = eigh(K, M); wl = np.sqrt(wl2)
Lc = np.linalg.cholesky(M).T; Ls = np.linalg.cholesky(np.block([[K, np.zeros((n, n))], [np.zeros((n, n)), M]])).T
lines = []
def log(s): print(s, flush=True); lines.append(s)
def bandpass(Z, lo, hi):
    om = 2 * np.pi * np.fft.rfftfreq(len(Z), ds); F = np.fft.rfft(Z - Z.mean(0), axis=0); F[(om < lo) | (om > hi)] = 0
    return np.fft.irfft(F, n=len(Z), axis=0)
def sv(Z, L): s = np.linalg.svd(Z @ L.T, compute_uv=False); return s / s[0]
P2 = np.c_[np.r_[PL[:, 1], np.zeros(n)], np.r_[np.zeros(n), PL[:, 1]]]
for P in (30.0, 60.0, 120.0, 240.0):
    x0 = static_deflection(S, P); _, X, V = newmark(M, K, C, pt, k3, x0, np.zeros(n), 1.5, dt); X, V = X[::5], V[::5]
    A = -np.linalg.solve(M, (K @ X.T + C @ V.T + k3 * np.outer(pt, X[:, tip]**3))).T
    Y, Yd = np.c_[X, V], np.c_[V, A]; Psi, Phi, mu = gsod(Y, Yd, 1e-10); Q = Y @ Psi
    om, Fq = dct_spectrum(Q, ds); pk, _ = peak_and_width(om, Fq); en = (Phi**2).sum(0) * Q.var(0); om2 = 2 * np.pi * np.fft.fftfreq(len(Q), ds)
    planes = pair_state(Q, ds, 2 * np.pi * 2000); a_, b_, fp, c = planes[0]
    z = (Q[:, a_] - Q[:, a_].mean()) / Q[:, a_].std() + 1j * (Q[:, b_] - Q[:, b_].mean()) / Q[:, b_].std(); Fz = np.abs(np.fft.fft(z))**2
    side = om2 > 0 if Fz[om2 > 0].sum() >= Fz[om2 < 0].sum() else om2 < 0; w_s = abs(om2[side]); oo = np.argsort(w_s); cdf = np.cumsum(Fz[side][oo]) / Fz[side].sum()
    lo, hi = w_s[oo][np.searchsorted(cdf, 0.05)], w_s[oo][np.searchsorted(cdf, 0.95)]
    log(f"=== P={P:.0f} N: NNM1 plane {fp:.1f} rad/s, band {lo:.0f}-{hi:.0f} rad/s; recognised planes {[round(p[2]) for p in planes]}")
    for h in (1, 3):
        Yb = bandpass(Y, h * lo, h * hi); Xb = bandpass(X, h * lo, h * hi)
        log(f"A  band {h}x: state singular values {np.array2string(sv(Yb, Ls)[:6], precision=1, formatter={'float_kind': lambda v: f'{v:.1e}'})}; "
            f"configuration {np.array2string(sv(Xb, Lc)[:4], formatter={'float_kind': lambda v: f'{v:.1e}'})}")
    near = [i for i in range(Q.shape[1]) if abs(pk[i] / wl[1] - 1) < 0.1]
    log(f"B  LNM 2 at {wl[1]:.1f} rad/s; coordinates peaking within 10%: " + str([(i, round(pk[i], 1), round(float(np.sqrt(mu[i])), 1), float('%.1e' % (en[i] / en.sum()))) for i in near]))
    rec2 = [p for p in planes if abs(p[2] / wl[1] - 1) < 0.1]
    log(f"B  recognised plane near LNM 2: {[(p[0], p[1], round(p[2], 1), round(p[3], 3)) for p in rec2]}")
    cands = sorted(set(near) | {a_, b_})
    for x in range(len(cands)):
        for y in range(x + 1, len(cands)):
            i, j = cands[x], cands[y]; cs, fz = onesided(Q[:, i], Q[:, j], om2)
            ang = np.degrees(subspace_angles(Ls @ Phi[:, [i, j]], Ls @ P2)).max()
            log(f"B    pair ({i},{j}): one-sidedness {cs:.3f} at {fz:.1f} rad/s; (mu_a mu_b)^(1/4) = {(mu[i] * mu[j])**0.25:.1f}; "
                f"angle to LNM-2 plane {ang:.1f} deg; mu ratio {max(mu[i], mu[j]) / min(mu[i], mu[j]):.2f}")
    # spectra of the near-LNM-2 coordinates: power share in the NNM1 band and near LNM 2
    for i in near:
        f_ = np.abs(np.fft.rfft(Q[:, i] - Q[:, i].mean()))**2; w_ = 2 * np.pi * np.fft.rfftfreq(len(Q), ds)
        log(f"B    coordinate {i}: power share in NNM1 band {f_[(w_ >= lo) & (w_ <= hi)].sum() / f_.sum():.3f}, within 5% of LNM 2 {f_[abs(w_ / wl[1] - 1) < 0.05].sum() / f_.sum():.3f}")
open('n11_out.txt', 'w').write("\n".join(lines) + "\n")
