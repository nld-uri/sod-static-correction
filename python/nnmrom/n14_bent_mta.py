"""Does an inertia-bent static vector account for SOD's backbone advantage on the COST F3 beam (S4)?
   MTA (n4_rom_backbones.py) is r-1 LNMs + the static tip vector K^-1 e_tip. Here the static vector is replaced by the
   tip deflection under a harmonic tip force, d(w) = (K - w^2 M)^-1 e_tip ("BMTA"), at two frequencies:
     w_sod  = sqrt(mu_1), the first SOD frequency of the training decay (no fit to the SOD direction);
     w_star = the frequency at which d(w) is closest (mass metric) to the first SOD direction.
   Three further model-built families test whether SOD's higher-order directions are bent vectors too: HARM (LNM 1 and
   d at 1, 3 and 5 times w_sod, the harmonics of a cubic force), BAND (LNM 1 and d at 160 rad/s, w_sod and the 30 N
   NNM frequency 2 pi 36.343, spanning the frequencies the decay sweeps) and LDR (Wilson's load-dependent Ritz vectors).
   D: d at the first r SOD frequencies of the decay, the basis i1_forced_identity.py predicts SOD to be (added for v5).
   The principal angles between the SOD r=4 subspace and MTA r=4 are printed as well.
   Training decay, SOD, backbone reference and error metric are those of n4_rom_backbones.py; MTA and SOD are recomputed
   here as a control: their medians must equal n4_out.txt (MTA r=4 1.4e-06, SOD r=4 5.2e-07).
   Output: n14_out.txt, n14_results.pkl."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'SOD-toolbox', 'python'))
import sodtoolbox as _st
import numpy as np, pickle
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from costbeam import costbeam
from nlsim import newmark, static_deflection
from n3_backbone_hb import HBC, full_backbone
S = costbeam(); Sd = costbeam(zeta=0.01); n = S['n']; tip = S['tip']; k3 = S['k3']; pt = np.zeros(n); pt[tip] = 1.0
M, K = S['M'], S['K']; w2, PL = eigh(K, M); dt = 2e-5; sub = 5
def mnorm(Q):                                       # M-orthonormal basis of span(Q); refuses a numerically dependent set
    Q = Q / np.sqrt(np.einsum('ij,ij->j', Q, M @ Q)); g, V = eigh(Q.T @ M @ Q)   # unit M-norm columns first
    if g[0] <= 1e-12 * g[-1]: raise np.linalg.LinAlgError(f"vectors dependent: Gram eigenvalue ratio {g[0] / g[-1]:.1e}")
    return Q @ V / np.sqrt(g)
def ang(a, b): return np.degrees(np.arccos(min(1.0, abs(a @ M @ b) / np.sqrt((a @ M @ a) * (b @ M @ b)))))
def d(w): return np.linalg.solve(K - w**2 * M, pt)
def ldr(r):                                         # load-dependent Ritz vectors (Wilson et al. 1982): K x1 = e, K x_i = M x_(i-1)
    V = [np.linalg.solve(K, pt)]
    for _ in range(r - 1): V.append(np.linalg.solve(K, M @ V[-1]))
    return np.c_[tuple(V)]
x30 = static_deflection(S, 30.0)
_, Xd, Vd = newmark(Sd['M'], Sd['K'], Sd['C'], pt, k3, x30, np.zeros(n), 1.5, dt); Xd, Vd = Xd[::sub], Vd[::sub]
s1 = _st.sod_modes(Xd, Vd, 1, 1e-10)[0][:, 0]
# first SOD frequency: generalized eigenproblem of the decay, as in n13_checks.py
w, U = eigh(Xd.T @ Xd); U = U[:, w > 1e-10 * w.max()]; Xp, Vp = Xd @ U, Vd @ U; mu = eigh(Vp.T @ Vp, Xp.T @ Xp, eigvals_only=True)
w_sod = np.sqrt(mu[0]); w1 = np.sqrt(w2[0]); w2nd = np.sqrt(w2[1])
W = np.linspace(0, 0.98 * w2nd, 2000); W = W[np.abs(W - w1) > 2.0]; A = np.array([ang(s1, d(x)) for x in W])
i = A.argmin(); w_star = minimize_scalar(lambda x: ang(s1, d(x)), bounds=(W[max(i - 1, 0)], W[min(i + 1, len(W) - 1)]), method='bounded', options={'xatol': 1e-6}).x
lines = [f"LNM 1 frequency {w1:.2f} rad/s; first SOD frequency of the 30 N decay {w_sod:.2f} rad/s",
         f"angle(first SOD direction, K^-1 e_tip) = {ang(s1, d(0.0)):.3f} deg; angle to LNM 1 = {ang(s1, PL[:, 0]):.3f} deg",
         f"angle to d(w_sod)  = {ang(s1, d(w_sod)):.4f} deg",
         f"smallest angle {ang(s1, d(w_star)):.2e} deg at w* = {w_star:.3f} rad/s ({w_star / w_sod:.4f} w_sod)"]
from scipy.linalg import subspace_angles
Lc = np.linalg.cholesky(M).T; P4 = _st.sod_modes(Xd, Vd, 4, 1e-10)[0]
lines.append("principal angles (deg), SOD r=4 vs MTA r=4: " + " ".join(f"{x:.3g}" for x in np.degrees(subspace_angles(Lc @ P4, Lc @ np.c_[PL[:, :3], d(0.0)]))))
lines.append("LNM k outside the SOD r=4 span (relative M-norm): " + " ".join(f"{k + 1}: {np.linalg.norm((y := Lc @ PL[:, k]) - (Q := np.linalg.qr(Lc @ P4)[0]) @ (Q.T @ y)) / np.linalg.norm(y):.1e}" for k in range(5)))
for x in lines: print(x, flush=True)
bases = {'MTA': lambda r: mnorm(np.c_[PL[:, :r - 1], d(0.0)]),
         'BMTA(w_sod)': lambda r: mnorm(np.c_[PL[:, :r - 1], d(w_sod)]),
         'BMTA(w*)': lambda r: mnorm(np.c_[PL[:, :r - 1], d(w_star)]),
         'HARM: LNM1, d(w), d(3w), d(5w)': lambda r: mnorm(np.c_[PL[:, :1], d(w_sod), d(3 * w_sod), d(5 * w_sod)][:, :r]),
         'BAND: LNM1, d(160), d(w), d(228)': lambda r: mnorm(np.c_[PL[:, :1], d(160.0), d(w_sod), d(2 * np.pi * 36.343)][:, :r]),
         'LDR (Wilson)': lambda r: mnorm(ldr(r)),
         'D: d at the SOD freqs 1..r': lambda r: mnorm(np.c_[tuple(d(np.sqrt(mu[k])) for k in range(r))]),
         'SOD (decay)': lambda r: _st.sod_modes(Xd, Vd, r, 1e-10)[0]}
F = pickle.load(open('n3_backbone_full_Kh8.pkl', 'rb'))
Hf = np.r_[F['low']['H'], F['H']]; ff = np.r_[F['low']['f'], F['f']]; h5 = np.r_[F['low']['h5'], F['h5']]
main = h5 < 0.02; o = np.argsort(Hf[main]); Hm, fm = Hf[main][o], ff[main][o]
Hgrid = np.geomspace(6e-4, 2.5e-2, 25); fgrid = np.interp(np.log(Hgrid), np.log(Hm), fm)
tongue = (Hgrid > 3.5e-3) & (Hgrid < 4.5e-3)
out = dict(w_sod=w_sod, w_star=w_star, W=W, A=A, back={})
for bn, fb in bases.items():
    for r in [1, 2, 3, 4]:
        if bn == 'MTA' and r < 2: continue
        try: Phi = fb(r)
        except np.linalg.LinAlgError as ex:
            line = f"{bn:12s} r={r}: basis not formed ({ex})"; lines.append(line); print(line, flush=True); continue
        Mr, Kr = Phi.T @ M @ Phi, Phi.T @ K @ Phi; ptr = Phi[tip]
        wr2, Pr = eigh(Kr, Mr); phi1 = Pr[:, 0] / (ptr @ Pr[:, 0]); hb = HBC(Mr, Kr, ptr, k3, Kh=8)
        try:
            low, high = full_backbone(hb, phi1, np.sqrt(wr2[0]))
            pts = low + high; H = np.array([hb.energy(z) for z in pts]); f = np.array([z[-1] for z in pts]) / 2 / np.pi
            hh = np.array([hb.harm(z)[2] for z in pts]); mm = hh < 0.02; oo = np.argsort(H[mm])
            fr = np.interp(np.log(Hgrid), np.log(H[mm][oo]), f[mm][oo]); e = abs(fr / fgrid - 1)
        except Exception as ex:
            print(bn, r, 'backbone failed:', ex); e = np.full(len(Hgrid), np.nan)
        out['back'][(bn, r)] = e
        line = f"{bn:12s} r={r}: backbone error median {np.nanmedian(e[~tongue]):.1e}, max {np.nanmax(e[~tongue]):.1e}"
        lines.append(line); print(line, flush=True)
pickle.dump(out, open('n14_results.pkl', 'wb'))
