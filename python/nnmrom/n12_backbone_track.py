"""SOD-ROM, Fig. fig_c_track: how well SOD follows the NNM1 backbone of the COST F3 beam (S4).
   (a) ROM backbones on the frequency-energy plot: harmonic balance (Kh = 8) on Galerkin ROMs from LNM, MTA and SOD of the
       30 N static-release decay (as n4), r = 2 and 4; curves (H, f) kept outside internal-resonance tongues (largest |X_k|/|X_1|, k >= 5, below 0.02).
   (b) Tracking from the record alone: sliding windows (0.12 s, hop 0.03 s) along damped static-release decays (zeta =
       0.01, 1.5 s) from 30 and 120 N, model kinematics. Per window: configuration SOD frequency of the smoothest
       coordinate, sqrt(mu_1)/2pi; state-SOD NNM1 plane recognised by circularity (sellib.pair_state with the energy-
       gated rescue; the slowest recognised plane whose product-rule frequency agrees with its spectral peak, no reference
       used), frequency from the product rule (mu_a mu_b)^(1/4)/2pi
       and from the mean rotation rate of z = q_a + i q_b (unwrapped phase slope); window energy = mean full-model energy H(t).
       Reference: full-model backbone (n3 + n9a), main branch, interpolated at the window energy.
   Output: n12_results.pkl, n12_out.txt."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'SOD-toolbox', 'python'))
import sodtoolbox as _st
import sys, os, numpy as np, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rom'))
from scipy.linalg import eigh
from costbeam import costbeam
from nlsim import newmark, static_deflection, energy
from n3_backbone_hb import HBC, full_backbone
from sellib import gsod, pair_state
S = costbeam(); Sd = costbeam(zeta=0.01); n = S['n']; tip = S['tip']; k3 = S['k3']; pt = np.zeros(n); pt[tip] = 1.0
M, K, C = S['M'], S['K'], Sd['C']; dt = 2e-5; sub = 5; ds = dt * sub; wl2, PL = eigh(K, M); lines = []
def log(s): print(s, flush=True); lines.append(s)
F = pickle.load(open('n3_backbone_full_Kh8.pkl', 'rb')); E = pickle.load(open('n9a_backbone_ext.pkl', 'rb'))
Hf = np.r_[F['low']['H'], F['H'], E['H']]; ff = np.r_[F['low']['f'], F['f'], E['f']]; h5 = np.r_[F['low']['h5'], F['h5'], np.zeros(len(E['H']))]
mm = h5 < 0.02; o = np.argsort(Hf[mm]); Hm, fm = Hf[mm][o], ff[mm][o]
def fref(H): return np.interp(np.log(H), np.log(Hm), fm)
def sodb(X, V, r, tol=1e-10): return _st.sod_modes(X, V, r, tol)[0]
def mta(r):
    s = np.linalg.solve(K, pt); Q = np.c_[PL[:, :r - 1], s]; L = np.linalg.cholesky(Q.T @ M @ Q); return Q @ np.linalg.inv(L).T
x30 = static_deflection(S, 30.0); _, X30, V30 = newmark(M, K, C, pt, k3, x30, np.zeros(n), 1.5, dt); X30, V30 = X30[::sub], V30[::sub]
out = dict(rom={}, track={}, Hm=Hm, fm=fm)
for bn, B in [('LNM', lambda r: PL[:, :r]), ('MTA', mta), ('SOD', lambda r: sodb(X30, V30, r))]:
    for r in (2, 4):
        P = B(r); Mr, Kr = P.T @ M @ P, P.T @ K @ P; ptr = P[tip]; wr2, Pr = eigh(Kr, Mr); phi1 = Pr[:, 0] / (ptr @ Pr[:, 0])
        hb = HBC(Mr, Kr, ptr, k3, Kh=8); low, high = full_backbone(hb, phi1, np.sqrt(wr2[0]), Hmax=0.06); pts = low + high
        H = np.array([hb.energy(z) for z in pts]); f = np.array([z[-1] for z in pts]) / 2 / np.pi; hh = np.array([hb.harm(z)[2:].max() for z in pts])          # outside every k >= 5 tongue (5:1 and 9:1)
        k = hh < 0.02; oo = np.argsort(H[k]); out['rom'][(bn, r)] = (H[k][oo], f[k][oo])
        g = (H[k] > 6e-4) & (H[k] < 5.5e-2) & ~((H[k] > 3.5e-3) & (H[k] < 4.5e-3))
        log(f"ROM {bn} r={r}: deviation at H = 0.005, 0.02, 0.05 J: " + " ".join(f"{np.interp(np.log(h), np.log(H[k][oo]), f[k][oo]) / fref(h) - 1:+.1e}" for h in (0.005, 0.02, 0.05)) + f"; H max {H[k].max():.3f}")
        log(f"ROM {bn} r={r}: median |f/f_full - 1| = {np.median(abs(f[k][g] / fref(H[k][g]) - 1)):.1e} over {g.sum()} backbone points")
Tw, hop = 0.12, 0.03; nw, nh = int(Tw / ds), int(hop / ds)
for P in (30.0, 120.0):
    x0 = static_deflection(S, P); _, X, V = newmark(M, K, C, pt, k3, x0, np.zeros(n), 1.5, dt); X, V = X[::sub], V[::sub]
    A = -np.linalg.solve(M, (K @ X.T + C @ V.T + k3 * np.outer(pt, X[:, tip]**3))).T; Hs = energy(S, X, V)
    rows = []
    for i0 in range(0, len(X) - nw, nh):
        sl = slice(i0, i0 + nw); Xw, Vw, Aw = X[sl], V[sl], A[sl]; Hw = Hs[sl].mean()
        w, U = eigh(Xw.T @ Xw); U = U[:, w > 1e-10 * w.max()]; Xp = Xw @ U; Vp = Vw @ U; muc = eigh(Vp.T @ Vp, Xp.T @ Xp, eigvals_only=True)[0]
        Y, Yd = np.c_[Xw, Vw], np.c_[Vw, Aw]; Psi, Phi, mus = gsod(Y, Yd, 1e-10); Q = Y @ Psi
        pl = pair_state(Q, ds, 2 * np.pi * 2000, Phi=Phi); en = (Phi**2).sum(0) * Q.var(0)
        # data only: NNM1 is the slowest recognised plane whose product-rule frequency agrees with its one-sided peak
        # (within 1.5 FFT bins of the window, or 15%); short windows otherwise admit pairs of boundary coordinates
        binw = 2 * np.pi / (nw * ds)
        pl = sorted([p for p in pl if abs((mus[p[0]] * mus[p[1]])**0.25 - p[2]) < max(0.15 * p[2], 1.5 * binw)], key=lambda p: p[2])
        fs = (mus[pl[0][0]] * mus[pl[0][1]])**0.25 / 2 / np.pi if pl else np.nan
        fz = np.nan
        if pl:
            a_, b_ = pl[0][:2]; z = (Q[:, a_] - Q[:, a_].mean()) / Q[:, a_].std() + 1j * (Q[:, b_] - Q[:, b_].mean()) / Q[:, b_].std()
            ph = np.unwrap(np.angle(z)); fz = abs(ph[-1] - ph[0]) / ((len(z) - 1) * ds) / 2 / np.pi     # mean rotation rate of the plane
        rows.append((Hw, np.sqrt(muc) / 2 / np.pi, fs, fref(Hw) if Hw <= Hm.max() else np.nan, fz))
    R = np.array(rows); out['track'][P] = R; g = ~np.isnan(R[:, 3]) & ~((R[:, 0] > 3.5e-3) & (R[:, 0] < 4.5e-3))
    ec = abs(R[g, 1] / R[g, 3] - 1); es = abs(R[g, 2] / R[g, 3] - 1); ez = abs(R[g, 4] / R[g, 3] - 1)
    log(f"track P={P:.0f} N: {len(R)} windows, H {R[:, 0].min():.1e}-{R[:, 0].max():.1e} J; vs backbone ({g.sum()} windows): "
        f"configuration SOD median {np.median(ec):.1e} (max {ec.max():.1e}); state-SOD plane median {np.nanmedian(es):.1e} (max {np.nanmax(es):.1e}), plane found in {np.isfinite(R[:, 2]).sum()}/{len(R)}; plane rotation rate median {np.nanmedian(ez):.1e} (max {np.nanmax(ez):.1e})")
pickle.dump(out, open('n12_results.pkl', 'wb')); open('n12_out.txt', 'w').write("\n".join(lines) + "\n")
