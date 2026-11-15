"""Diagnostic: which accepted S2 items are spurious under the spectral-consistency rule, and does capping the
   half-power-width tolerance remove them? One record each (random tip load, band-limited decay), 60 and 40 dB,
   estimators exact, CD, WH, FUS (as c22). Run with the width cap off (wcap = inf), to show what the cap removes. Prints, per accepted item: SOD frequency, peak, |peak/f_SOD - 1|,
   half-power relative width, nearest LNM and its MAC. Seed 25. Output: c25_out.txt."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kin'))
from scipy.linalg import eigh
from systems import beam, state
from romlib import band_noise
from romlib2 import simulate_state
from sellib import gsod, select_config, dct_spectrum, peak_and_width, cd_correct
from kinematics import diff_whittaker, _central
from ksfast import ks_fast
dt = 1e-3; N = 200000; wc = 250.0; wmax = 1.2 * wc; rng = np.random.default_rng(25); lines = []
def log(s): print(s, flush=True); lines.append(s)
S = beam(); C = S['C']['light']; M, K, B = S['M'], S['K'], S['B']; n = len(M); A, Bs = state(M, K, C, B)
wL2, PL = eigh(K, M); wL = np.sqrt(wL2); kEE = np.where(wL < np.pi / (4 * dt))[0]
xbl = PL[:, kEE] @ (1 / (1 + (wL[kEE] / wc) ** 4) / wL[kEE]); xbl /= abs(xbl).max()
def sc(X, V, cd, lab):
    Psi, Phi, mu = gsod(X, V, 1e-10); Q = X @ Psi; idx, pk, ws = select_config(Q, mu, dt, wmax, cd=cd, Phi=Phi, wcap=np.inf)
    om, F = dct_spectrum(Q, dt); _, rel = peak_and_width(om, F)
    for i in idx:
        m = np.array([(Phi[:, i] @ M @ PL[:, k])**2 / ((Phi[:, i] @ M @ Phi[:, i]) * (PL[:, k] @ M @ PL[:, k])) for k in range(14)])
        kf = int(np.argmin(abs(wL[:14] / ws[i] - 1))); spur = abs(ws[i] / wL[kf] - 1) >= 0.05
        log(f"{lab:22s} fSOD {ws[i]:7.1f} peak {pk[i]:6.1f} dev {abs(pk[i] / ws[i] - 1):.3f} width {rel[i]:.3f} LNM{kf + 1} MAC {m[kf]:.2f}{'  SPURIOUS' if spur else ''}")
for rec in ('random', 'decay'):
    if rec == 'random': u = band_noise(N, dt, wc, 2, rng) * np.r_[1, 0]; y0 = None
    else: u = np.zeros((N, 2)); y0 = np.r_[xbl, np.zeros(n)]
    Y = simulate_state(A, Bs, u, dt, y0); Yd = Y @ A.T + u @ Bs.T; X0, V0, A0 = Y[:, :n], Y[:, n:], Yd[:, n:]
    sc(X0, V0, False, f"{rec} exact")
    for snr in (60, 40):
        x = X0 + X0.std(0) * 10**(-snr / 20) * rng.standard_normal(X0.shape); a = A0 + A0.std(0) * 10**(-snr / 20) * rng.standard_normal(A0.shape)
        sc(x[1:-1], (x[2:] - x[:-2]) / (2 * dt), True, f"{rec} {snr} CD")
        s = np.column_stack([diff_whittaker(x[:, j], dt, 0, fc=100.0) for j in range(n)]); sc(s, np.column_stack([_central(s[:, j], dt, 1) for j in range(n)]), False, f"{rec} {snr} WH"); del s
        Xf, Vf = np.empty_like(x), np.empty_like(x)
        for j in range(n):
            q = ks_fast(dt, x=x[:20000, j], a=a[:20000, j])['q']; r = ks_fast(dt, x=x[:, j], a=a[:, j], q=q, fit=False); Xf[:, j], Vf[:, j] = r['x'], r['v']
        sc(Xf, Vf, False, f"{rec} {snr} FUS")
open('c25_out.txt', 'w').write("\n".join(lines) + "\n")
