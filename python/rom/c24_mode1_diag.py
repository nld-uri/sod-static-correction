"""Diagnostic: why the mode-1 plane is sometimes not recognised from random-load records (S3, c_d = 0.2, model
   kinematics), run with the final sellib.py (smoothed spectra, circularity >= 0.85, loose frequency gate, one plane
   per frequency); all 8 records then recognise the mode-1 plane. For 8 fresh records (seed 24): coordinates whose DCT peak is within 15% of omega_d1 = 3.5 rad/s, their
   half-power widths, and the one-sidedness of every pair among them. Output: c24_out.txt."""
import numpy as np
from systems import beam, state
from romlib import band_noise
from romlib2 import simulate_state
from sellib import gsod, dct_spectrum, peak_and_width, onesided, pair_state
dt = 1e-3; N = 200000; wc = 250.0; rng = np.random.default_rng(24); lines = []
S = beam(dashpot=(0.2, 1.0)); M, K, B = S['M'], S['K'], S['B']; C = S['C']['dashpot']; n = len(M); A, Bs = state(M, K, C, B)
for rr in range(8):
    u = band_noise(N, dt, wc, 2, rng) * np.r_[1, 0]; Y = simulate_state(A, Bs, u, dt); Yd = Y @ A.T + u @ Bs.T
    Psi, Phi, mu = gsod(Y, Yd, 1e-10); Q = Y @ Psi; om, F = dct_spectrum(Q, dt); pk, rel = peak_and_width(om, F)
    near = [i for i in range(Q.shape[1]) if abs(pk[i] / 3.5 - 1) < 0.15]; om2 = 2 * np.pi * np.fft.fftfreq(N, dt)
    pr = [(a, b, round(onesided(Q[:, a], Q[:, b], om2)[0], 3)) for x, a in enumerate(near) for b in near[x + 1:]]
    pl = pair_state(Q, dt, 300.0, Phi=Phi)
    lines.append(f"record {rr}: coords near 3.5: {[(i, round(pk[i], 2), round(rel[i], 3), round(float(np.sqrt(mu[i])), 2)) for i in near]}; pairs one-sidedness {pr}; recognised planes {[round(p[2], 1) for p in pl]}")
    print(lines[-1], flush=True)
open('c24_out.txt', 'w').write("\n".join(lines) + "\n")
