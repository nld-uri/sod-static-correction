"""SOD-ROM, nonlinear benchmark (suggestion of D. Chelidze): do state-SOD coordinates that oscillate at the frequency
   of a circular (modal-plane) pair, but are not part of it, span the out-of-plane curvature of the NNM?
   Record: the damped decay of n4_rom_backbones.py (COST F3 beam, zeta = 0.01 Rayleigh on modes 1-2, released from the
   30 N static tip deflection, 1.5 s, Newmark dt = 2e-5 s, every 5th sample kept). Accelerations from the equations
   of motion (model kinematics). State SOD of (y, ydot) = ((x, v), (v, a)); planes recognised by circularity
   (sellib.pair_state); 'companions' of plane 1 = unpaired coordinates whose smoothed DCT peak lies inside the band
   swept by plane 1 (5th to 95th percentile of the one-sided power of z = q_a + i q_b: the NNM frequency drifts with
   energy during the decay), ranked by their response energy.
   Configuration basis 'plane+companions': the displacement parts of the plane-1 pair (principal directions with
   singular value > 1e-3 of the largest), then the companions' displacement parts, M-orthonormalized in that order.
   Its NNM1 backbone is computed by harmonic balance (Kh = 8) as in n4 and compared, at equal r, with LNM, MTA and
   configuration SOD of the same decay (n4_results.pkl). Output: n7_out.txt, n7_results.pkl."""
import sys, os, numpy as np, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rom'))
from scipy.linalg import eigh
from costbeam import costbeam
from nlsim import newmark, static_deflection
from n3_backbone_hb import HBC, full_backbone
from sellib import gsod, pair_state, dct_spectrum, peak_and_width
S = costbeam(); Sd = costbeam(zeta=0.01); n = S['n']; tip = S['tip']; k3 = S['k3']; pt = np.zeros(n); pt[tip] = 1.0
M, K, C = S['M'], S['K'], Sd['C']; dt = 2e-5; sub = 5; ds = dt * sub; lines = []
def log(s): print(s, flush=True); lines.append(s)
x30 = static_deflection(S, 30.0)
_, X, V = newmark(Sd['M'], Sd['K'], Sd['C'], pt, k3, x30, np.zeros(n), 1.5, dt); X, V = X[::sub], V[::sub]
A = -np.linalg.solve(M, (K @ X.T + C @ V.T + k3 * np.outer(pt, X[:, tip]**3))).T          # model accelerations
Y, Yd = np.c_[X, V], np.c_[V, A]
Psi, Phi, mu = gsod(Y, Yd, 1e-10); Q = Y @ Psi
om, F = dct_spectrum(Q, ds); pk, rel = peak_and_width(om, F)
planes = pair_state(Q, ds, 2 * np.pi * 400)
log(f"recognised planes (rad/s, one-sidedness): {[(round(p[2], 1), round(p[3], 3)) for p in planes]}")
p1 = planes[0]; used = {i for p in planes for i in p[:2]}
en = (Phi**2).sum(0) * Q.var(0)
a_, b_ = p1[:2]; om2 = 2 * np.pi * np.fft.fftfreq(len(Q), ds)
z = (Q[:, a_] - Q[:, a_].mean()) / Q[:, a_].std() + 1j * (Q[:, b_] - Q[:, b_].mean()) / Q[:, b_].std(); Fz = np.abs(np.fft.fft(z))**2
side = om2 > 0 if Fz[om2 > 0].sum() >= Fz[om2 < 0].sum() else om2 < 0; w_s = abs(om2[side]); o_ = np.argsort(w_s); cdf = np.cumsum(Fz[side][o_]) / Fz[side].sum()
f_lo, f_hi = w_s[o_][np.searchsorted(cdf, 0.05)], w_s[o_][np.searchsorted(cdf, 0.95)]
log(f"plane 1 band (5-95% of power): {f_lo:.1f}-{f_hi:.1f} rad/s")
comp = sorted([i for i in range(Q.shape[1]) if i not in used and f_lo <= pk[i] <= f_hi], key=lambda i: -en[i])
log(f"plane 1 at {p1[2]:.1f} rad/s; companions (index, peak rad/s, SOD freq, energy fraction): "
    f"{[(i, round(pk[i], 1), round(float(np.sqrt(mu[i])), 1), float('%.2e' % (en[i] / en.sum()))) for i in comp[:8]]}")
# configuration basis
D = Phi[:n, list(p1[:2])]; U, sv, _ = np.linalg.svd(D, full_matrices=False); cols = [U[:, j] for j in range(len(sv)) if sv[j] > 1e-3 * sv[0]]
cols += [Phi[:n, i] for i in comp]
B = []
for c in cols:
    for b in B: c = c - (b @ M @ c) * b
    nn = np.sqrt(c @ M @ c)
    if nn > 1e-6 * np.sqrt(cols[0] @ M @ cols[0]): B.append(c / nn)
B = np.array(B).T; log(f"plane+companions basis: {B.shape[1]} M-orthonormal directions ({len(cols) - len(comp)} from the plane)")
R4 = pickle.load(open('n4_results.pkl', 'rb')); Hg, fg, tm = R4['Hgrid'], R4['fgrid'], R4['tongue_mask']; out = {}
for r in range(1, min(4, B.shape[1]) + 1):
    Pb = B[:, :r]; Mr, Kr = Pb.T @ M @ Pb, Pb.T @ K @ Pb; ptr = Pb[tip]; wr2, Pr = eigh(Kr, Mr); phi1 = Pr[:, 0] / (ptr @ Pr[:, 0])
    hb = HBC(Mr, Kr, ptr, k3, Kh=8)
    try:
        low, high = full_backbone(hb, phi1, np.sqrt(wr2[0])); pts = low + high
        H = np.array([hb.energy(z) for z in pts]); f = np.array([z[-1] for z in pts]) / 2 / np.pi; hh = np.array([hb.harm(z)[2] for z in pts])
        mm = hh < 0.02; oo = np.argsort(H[mm]); fr = np.interp(np.log(Hg), np.log(H[mm][oo]), f[mm][oo]); e = abs(fr / fg - 1)
    except Exception as ex:
        log(f"r={r}: backbone failed: {ex}"); e = np.full(len(Hg), np.nan)
    out[r] = e
    ref = "  ".join(f"{bn} {np.nanmedian(R4['back'][(bn, r)][~tm]):.1e}" for bn in ('LNM', 'MTA', 'SOD (decay)', 'POD (decay)') if (bn, r) in R4['back'])
    log(f"r={r}: plane+companions backbone error median {np.nanmedian(e[~tm]):.1e} (max {np.nanmax(e[~tm]):.1e}) | n4 at same r: {ref}")
pickle.dump(dict(back=out, planes=planes, comp=comp), open('n7_results.pkl', 'wb')); open('n7_out.txt', 'w').write("\n".join(lines) + "\n")
# where does the companion point? principal angles of span(plane + companion) to span(LNM 1, K^-1 e_tip), the
# subspace of the MTA basis at r = 2, and to span(LNM 1, LNM 2)
from scipy.linalg import subspace_angles
wl2, PLs = eigh(K, M); st = np.linalg.solve(K, pt)
L = np.linalg.cholesky(M).T
for lab, Bref in [('LNM1 + static tip vector (MTA, r=2)', np.c_[PLs[:, 0], st]), ('LNM 1-2', PLs[:, :2])]:
    if B.shape[1] >= 2:
        ang = np.degrees(subspace_angles(L @ B[:, :2], L @ Bref)); log(f"principal angles (deg, mass metric) of plane+companion to {lab}: {np.round(ang, 2)}")
open('n7_out.txt', 'w').write("\n".join(lines) + "\n")
