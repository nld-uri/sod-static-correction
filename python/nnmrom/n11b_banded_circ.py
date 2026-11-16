"""Banded circularity: one-sidedness of z = q_a + i q_b computed only over |omega| within +-30% of the pair's
   one-sided peak, so that power a member carries at another mode's frequency (here the NNM1 bend, which points
   toward LNM 2) does not dilute the test. All candidate pairs that pass the frequency gate of sellib.pair_state, for the
   four static-release decays (30-240 N): broadband and banded one-sidedness; pairs on a true plane are marked by the
   largest principal angle (energy metric) of their span to an LNM plane span{(phi_k,0),(0,phi_k)}, k = 1..6, < 15 deg.
   Candidates restricted to peaks below 3000 rad/s (modes 1-3). Pairs peaking at zero frequency keep c1 = c0.
   Also compares acceptance rules: c0 >= 0.85; c1 >= 0.85; c0 >= t0 and c1 >= 0.85.
   Output: n11b_out.txt."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rom'))
from scipy.linalg import eigh, subspace_angles
from costbeam import costbeam
from nlsim import newmark, static_deflection
from sellib import gsod, dct_spectrum, peak_and_width
S = costbeam(); Sd = costbeam(zeta=0.01); n = S['n']; tip = S['tip']; k3 = S['k3']; pt = np.zeros(n); pt[tip] = 1.0
M, K, C = S['M'], S['K'], Sd['C']; dt = 2e-5; ds = 5 * dt; wl2, PL = eigh(K, M)
Ls = np.linalg.cholesky(np.block([[K, np.zeros((n, n))], [np.zeros((n, n)), M]])).T; lines = []
def log(s): print(s, flush=True); lines.append(s)
Pk = [np.c_[np.r_[PL[:, k], np.zeros(n)], np.r_[np.zeros(n), PL[:, k]]] for k in range(6)]
allm, allo = [], []
for P in (30.0, 60.0, 120.0, 240.0):
    x0 = static_deflection(S, P); _, X, V = newmark(M, K, C, pt, k3, x0, np.zeros(n), 1.5, dt); X, V = X[::5], V[::5]
    A = -np.linalg.solve(M, (K @ X.T + C @ V.T + k3 * np.outer(pt, X[:, tip]**3))).T
    Y, Yd = np.c_[X, V], np.c_[V, A]; Psi, Phi, mu = gsod(Y, Yd, 1e-10); Q = Y @ Psi
    om, Fq = dct_spectrum(Q, ds); pk, rel = peak_and_width(om, Fq); en = (Phi**2).sum(0) * Q.var(0); en = en / en.sum(); om2 = 2 * np.pi * np.fft.fftfreq(len(Q), ds)
    cand = [i for i in range(Q.shape[1]) if pk[i] < 3000.0]          # modes 1-3, where the LNM-plane label is reliable
    Qn = (Q - Q.mean(0)) / Q.std(0); FQ = np.fft.fft(Qn, axis=0); pos, neg = om2 > 0, om2 < 0
    for x, a in enumerate(cand):
        for b in cand[x + 1:]:
            if abs(pk[a] / pk[b] - 1) >= 4 * max(0.05, rel[a], rel[b]): continue
            Fz = np.abs(FQ[:, a] + 1j * FQ[:, b])**2; Pp, Pm = Fz[pos].sum(), Fz[neg].sum(); side = pos if Pp >= Pm else neg
            c0 = max(Pp, Pm) / (Pp + Pm); fpk = abs(om2[side][np.argmax(Fz[side])])
            c1 = c0
            if fpk > 0: bd = abs(abs(om2) / fpk - 1) < 0.3; Pp2, Pm2 = Fz[pos & bd].sum(), Fz[neg & bd].sum(); c1 = max(Pp2, Pm2) / (Pp2 + Pm2)
            ang = min(np.degrees(subspace_angles(Ls @ Phi[:, [a, b]], Ls @ Pk[k])).max() for k in range(6))
            modal = ang < 15; (allm if modal else allo).append((c0, c1, en[a] + en[b]))
            if modal or c1 > 0.8: log(f"P={P:.0f} pair ({a},{b}) peak {fpk:.0f}: broadband {c0:.3f}, banded {c1:.3f}, angle to nearest LNM plane {ang:.1f} deg{' MODAL' if modal else ''}")
allm, allo = np.array(allm), np.array(allo)
log(f"modal pairs: broadband {allm[:, 0].min():.3f}-{allm[:, 0].max():.3f}, banded {allm[:, 1].min():.3f}-{allm[:, 1].max():.3f} ({len(allm)})")
for t0 in (0.7, 0.75, 0.8):
    log(f"rule c0>={t0} & c1>=0.85: modal accepted {((allm[:, 0] >= t0) & (allm[:, 1] >= 0.85)).sum()}/{len(allm)}, other accepted {((allo[:, 0] >= t0) & (allo[:, 1] >= 0.85)).sum()}/{len(allo)}")
for es in (1e-3, 1e-2):
    ok = lambda R: ((R[:, 0] >= 0.85) | ((R[:, 0] >= 0.75) & (R[:, 1] >= 0.95) & (R[:, 2] >= es))).sum()
    log(f"rescue rule (c0>=0.85, or c0>=0.75 & c1>=0.95 & energy share>={es:g}): modal {ok(allm)}/{len(allm)}, other {ok(allo)}/{len(allo)}")
log(f"rule c0>=0.85: modal {(allm[:, 0] >= 0.85).sum()}/{len(allm)}, other {(allo[:, 0] >= 0.85).sum()}; rule c1>=0.85: modal {(allm[:, 1] >= 0.85).sum()}, other {(allo[:, 1] >= 0.85).sum()}")
log(f"other pairs: broadband {allo[:, 0].min():.3f}-{allo[:, 0].max():.3f}, banded {allo[:, 1].min():.3f}-{allo[:, 1].max():.3f} ({len(allo)}); banded > 0.85: {(allo[:, 1] > 0.85).sum()}")
open('n11b_out.txt', 'w').write("\n".join(lines) + "\n")
