"""Rule 3 under strong nonlinearity (suggestion of D. Chelidze): does ordering by frequency -- each recognised modal
   plane followed by its companions -- beat SOD order when the NNM bends further out of the plane of LNM 1?
   COST F3 beam, damped decays (zeta = 0.01 Rayleigh on modes 1-2, 1.5 s, Newmark dt = 2e-5 s, every 5th sample)
   released from the static deflection under a tip load P = 30, 60, 120, 240 N. Model kinematics (accelerations from
   the equations of motion). Bases at equal r:
     LNM, MTA (r-1 LNMs + static tip-load vector);
     SOD: configuration SOD of (x, v) (as n4);
     SS:  state SOD of (y, ydot), displacement parts in SOD order (control: same decomposition, no reordering);
     RC:  rule 3 -- recognised planes (sellib.pair_state) in frequency order, each followed by its companions
          (unpaired coordinates whose DCT peak lies in the plane's 5-95% power band, then in its 3x and 5x bands;
          energy order inside a band), then the remaining state-SOD coordinates in SOD order. Displacement parts,
          M-orthonormalised in that order; directions below 1e-6 of the first (relative M-norm) are dropped;
     RCd: as RC, but a companion enters only if more than 10% of its M-norm lies outside the span already chosen
          (n10_plane_span.py: the companion lies 0.5-4.5 deg from the plane's displacement span);
     RC1: as RCd, but each plane contributes only its dominant displacement direction (largest singular value);
     RC1f: as RC1, with companions from the fundamental band only (harmonic bands of a higher plane can hold another
          mode); harmonic-band candidates return to the rest, in SOD order;
     RCc: rule 3 on configuration SOD -- modes recognised by spectral consistency (sellib.select_config), each followed
          by the coordinates peaking in its 5-95% power band and its 3x and 5x bands, then the rest in SOD order.
   Metrics: (i) NNM1 backbone frequency error by harmonic balance (Kh = 8) against the full model (n3 + n9a, up to
   0.059 J, where a 9:1 internal resonance with mode 3 begins), median over a log grid outside the 5:1 tongue;
   (ii) held-out conservative releases (0.1 s) from static tip loads 45, 90, 180 N and mid-span loads 90, 270 N,
   mass-weighted displacement error; (iii) held-out conservative motion (0.1 s) started on the full-model NNM1 at
   H = 0.01, 0.025, 0.05 J (harmonic-balance state, zero velocity). Usage: python3 n9_strong_companions.py [band]  (any value: pair_state with the energy-gated banded rescue; none: broadband only)
   Output: n9_out[_bandX].txt, n9_results[_bandX].pkl."""
import sys, os, numpy as np, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rom'))
from scipy.linalg import eigh, subspace_angles
from scipy.optimize import fsolve
from costbeam import costbeam
from nlsim import newmark, static_deflection
from n3_backbone_hb import HBC, full_backbone
from sellib import gsod, pair_state, dct_spectrum, peak_and_width, select_config
BAND = float(sys.argv[1]) if len(sys.argv) > 1 else None; TAG = '' if BAND is None else f'_band{BAND:g}'
S = costbeam(); Sd = costbeam(zeta=0.01); n = S['n']; tip = S['tip']; k3 = S['k3']; pt = np.zeros(n); pt[tip] = 1.0
M, K, C = S['M'], S['K'], Sd['C']; dt = 2e-5; sub = 5; ds = dt * sub; lines = []
wl2, PL = eigh(K, M); st = np.linalg.solve(K, pt); L = np.linalg.cholesky(M).T
def log(s): print(s, flush=True); lines.append(s)
kept = []
def morth(cols, rel=1e-6):
    B = []; kept.clear()
    for c in cols:
        c = c.copy(); nn0 = np.sqrt(c @ M @ c)
        for _ in range(2):
            for b in B: c = c - (b @ M @ c) * b
        nn = np.sqrt(c @ M @ c)
        if nn > rel * np.sqrt(cols[0] @ M @ cols[0]) and nn > 1e-9 * nn0: B.append(c / nn); kept.append(True)
        else: kept.append(False)
    return np.array(B).T
def morth_gate(cols, labs, gate=0.1):
    """as morth, but a companion is accepted only if more than `gate` of its M-norm lies outside the span chosen so far"""
    B = []; acc = []
    for c, l in zip(cols, labs):
        c = c.copy(); nn0 = np.sqrt(c @ M @ c)
        for _ in range(2):
            for b in B: c = c - (b @ M @ c) * b
        nn = np.sqrt(c @ M @ c); thr = gate if l.startswith('comp') else 1e-9
        if nn > thr * nn0 and nn > 1e-6 * np.sqrt(cols[0] @ M @ cols[0]): B.append(c / nn); acc.append(l)
    return np.array(B).T, acc
def csod(X, V, tol=1e-10):
    Sx = X.T @ X; w, U = eigh(Sx); U = U[:, w > tol * w.max()]; Xp = X @ U; Vp = V @ U; mu, Psi = eigh(Vp.T @ Vp, Xp.T @ Xp)
    return U @ (Xp.T @ Xp) @ Psi, U @ Psi, mu
def band(q, d):
    om = 2 * np.pi * np.fft.rfftfreq(len(q), d); Fq = np.abs(np.fft.rfft(q - q.mean()))**2; c = np.cumsum(Fq) / Fq.sum()
    return om[np.searchsorted(c, 0.05)], om[np.searchsorted(c, 0.95)]
def mta(r): return morth([PL[:, j] for j in range(r - 1)] + [st / np.sqrt(st @ M @ st)])
# reference backbone
F = pickle.load(open('n3_backbone_full_Kh8.pkl', 'rb')); E = pickle.load(open('n9a_backbone_ext.pkl', 'rb'))
Hf = np.r_[F['low']['H'], F['H'], E['H']]; ff = np.r_[F['low']['f'], F['f'], E['f']]; h5 = np.r_[F['low']['h5'], F['h5'], np.zeros(len(E['H']))]
mm = h5 < 0.02; o = np.argsort(Hf[mm]); Hm, fm = Hf[mm][o], ff[mm][o]
Hg = np.geomspace(6e-4, 5.5e-2, 30); fg = np.interp(np.log(Hg), np.log(Hm), fm); tongue = (Hg > 3.5e-3) & (Hg < 4.5e-3)
def backbone(B):
    Mr, Kr = B.T @ M @ B, B.T @ K @ B; ptr = B[tip]; wr2, Pr = eigh(Kr, Mr); phi1 = Pr[:, 0] / (ptr @ Pr[:, 0])
    hb = HBC(Mr, Kr, ptr, k3, Kh=8)
    try:
        low, high = full_backbone(hb, phi1, np.sqrt(wr2[0]), Hmax=0.06); pts = low + high
        H = np.array([hb.energy(z) for z in pts]); f = np.array([z[-1] for z in pts]) / 2 / np.pi
        hh = np.array([hb.harm(z)[1:].max() if len(hb.harm(z)) > 1 else 0 for z in pts]); hh = np.array([hb.harm(z)[2] for z in pts])
        k = hh < 0.02; oo = np.argsort(H[k]); e = abs(np.interp(np.log(Hg), np.log(H[k][oo]), f[k][oo]) / fg - 1)
        e[(Hg < H[k].min()) | (Hg > H[k].max())] = np.nan; return e
    except Exception as ex:
        return np.full(len(Hg), np.nan)
# held-out tests
tests = {}
for lab, P, node in [('tip 45', 45.0, tip), ('tip 90', 90.0, tip), ('tip 180', 180.0, tip), ('mid 90', 90.0, 18), ('mid 270', 270.0, 18)]:
    e = np.zeros(n); e[node] = P
    x0 = fsolve(lambda x: K @ x + k3 * x[tip]**3 * pt - e, np.linalg.solve(K, e), xtol=1e-14)
    t, X, V = newmark(M, K, np.zeros((n, n)), pt, k3, x0, np.zeros(n), 0.1, dt); tests[lab] = (x0, X)
Fl = F['low']; Zall = np.r_[Fl['Z'], F['Z'], E['Z']]; Hall = np.r_[Fl['H'], F['H'], E['H']]; h5all = np.r_[Fl['h5'], F['h5'], np.zeros(len(E['H']))]
for Ht in (0.01, 0.025, 0.05):
    ok = np.where(h5all < 0.02)[0]; i = ok[np.argmin(abs(np.log(Hall[ok] / Ht)))]; x0 = Zall[i][:-1].reshape(8, n).sum(0)
    t, X, V = newmark(M, K, np.zeros((n, n)), pt, k3, x0, np.zeros(n), 0.1, dt); tests[f'NNM {Hall[i]:.3f} J'] = (x0, X)
def tderr(B):
    r = B.shape[1]; Mr, Kr = B.T @ M @ B, B.T @ K @ B; Pm = np.linalg.solve(Mr, B.T @ M); out = {}
    for lab, (x0, X) in tests.items():
        _, Q, _ = newmark(Mr, Kr, np.zeros((r, r)), B[tip], k3, Pm @ x0, np.zeros(r), 0.1, dt); D = Q @ B.T - X
        out[lab] = np.sqrt(np.sum((D @ M) * D) / np.sum((X @ M) * X))
    return out
res = {}
for P in (30.0, 60.0, 120.0, 240.0):
    x0 = static_deflection(S, P); H0 = 0.5 * x0 @ K @ x0 + k3 / 4 * x0[tip]**4
    _, X, V = newmark(M, K, C, pt, k3, x0, np.zeros(n), 1.5, dt); X, V = X[::sub], V[::sub]
    A = -np.linalg.solve(M, (K @ X.T + C @ V.T + k3 * np.outer(pt, X[:, tip]**3))).T
    Y, Yd = np.c_[X, V], np.c_[V, A]; Psi, Phi, mu = gsod(Y, Yd, 1e-10); Q = Y @ Psi
    om, Fq = dct_spectrum(Q, ds); pk, _ = peak_and_width(om, Fq); en = (Phi**2).sum(0) * Q.var(0)
    planes = pair_state(Q, ds, 2 * np.pi * 2000, Phi=Phi if BAND else None); used = {i for p in planes for i in p[:2]}; un = [i for i in range(Q.shape[1]) if i not in used]
    om2 = 2 * np.pi * np.fft.fftfreq(len(Q), ds); cols = []; comps = {}; labels = []; cols1 = []; labels1 = []; cols1f = []; labels1f = []
    for a_, b_, fp, c in planes:
        z = (Q[:, a_] - Q[:, a_].mean()) / Q[:, a_].std() + 1j * (Q[:, b_] - Q[:, b_].mean()) / Q[:, b_].std(); Fz = np.abs(np.fft.fft(z))**2
        side = om2 > 0 if Fz[om2 > 0].sum() >= Fz[om2 < 0].sum() else om2 < 0; w_s = abs(om2[side]); oo = np.argsort(w_s); cdf = np.cumsum(Fz[side][oo]) / Fz[side].sum()
        lo, hi = w_s[oo][np.searchsorted(cdf, 0.05)], w_s[oo][np.searchsorted(cdf, 0.95)]
        U, sv, _ = np.linalg.svd(Phi[:n, [a_, b_]], full_matrices=False); nd = [j for j in range(2) if sv[j] > 1e-3 * sv[0]]
        cols += [U[:, j] for j in nd]; labels += [f'plane {fp:.0f}'] * len(nd); cols1 += [U[:, 0]]; labels1 += [f'plane {fp:.0f}']
        cp = []
        for h in (1, 3, 5):
            ch = sorted([i for i in un if h * lo <= pk[i] <= h * hi and i not in cp and all(i not in v for v in comps.values())], key=lambda i: -en[i]); cp += ch
        comps[round(fp, 1)] = cp; cols += [Phi[:n, i] for i in cp]; labels += [f'comp {i} ({pk[i]:.0f})' for i in cp]
        cols1 += [Phi[:n, i] for i in cp]; labels1 += [f'comp {i} ({pk[i]:.0f})' for i in cp]
        cf = [i for i in cp if lo <= pk[i] <= hi]; cols1f += [U[:, 0]] + [Phi[:n, i] for i in cf]; labels1f += [f'plane {fp:.0f}'] + [f'comp {i} ({pk[i]:.0f})' for i in cf]
        log(f"P={P:.0f} N (H0={H0:.2e} J): plane {fp:.1f} rad/s (one-sidedness {c:.3f}), band {lo:.0f}-{hi:.0f}; companions "
            f"{[(i, round(pk[i], 1), round(float(np.sqrt(max(mu[i], 0))), 1), float('%.1e' % (en[i] / en.sum()))) for i in cp]}")
    rest = [i for i in range(Q.shape[1]) if i not in used and all(i not in v for v in comps.values())]
    RC = morth(cols + [Phi[:n, i] for i in rest]); log(f"P={P:.0f}: RC columns {[l for l, k in zip(labels + [f'rest {i}' for i in rest], kept) if k][:6]}")
    RCd, accd = morth_gate(cols + [Phi[:n, i] for i in rest], labels + [f'rest {i}' for i in rest]); log(f"P={P:.0f}: RCd columns {accd[:6]}")
    RC1, acc1 = morth_gate(cols1 + [Phi[:n, i] for i in rest], labels1 + [f'rest {i}' for i in rest]); log(f"P={P:.0f}: RC1 columns {acc1[:6]}")
    restf = [i for i in range(Q.shape[1]) if i not in used and not any(l.startswith(f'comp {i} ') for l in labels1f)]
    RC1f, acc1f = morth_gate(cols1f + [Phi[:n, i] for i in restf], labels1f + [f'rest {i}' for i in restf]); log(f"P={P:.0f}: RC1f columns {acc1f[:6]}")
    SS = morth([Phi[:n, i] for i in range(Phi.shape[1])]); Cs, Pc, muc = csod(X, V)
    SOD = morth([Cs[:, j] for j in range(Cs.shape[1])])
    Qc = X @ Pc; idc, pkc, _ = select_config(Qc, muc, ds, 2 * np.pi * 2000, Phi=Cs); omc, Fc = dct_spectrum(Qc, ds); pkc, _ = peak_and_width(omc, Fc)
    enc = (Cs**2).sum(0) * Qc.var(0); ordc = []; labc = []
    for i in idc:
        lo, hi = band(Qc[:, i], ds); ordc.append(i); labc.append(f'mode {pkc[i]:.0f}')
        for h in (1, 3, 5):
            ch = sorted([j for j in range(Qc.shape[1]) if j not in idc and j not in ordc and h * lo <= pkc[j] <= h * hi], key=lambda j: -enc[j]); ordc += ch; labc += [f'comp {j} ({pkc[j]:.0f})' for j in ch]
    restc = [j for j in range(Qc.shape[1]) if j not in ordc]; RCc = morth([Cs[:, j] for j in ordc + restc])
    log(f"P={P:.0f}: RCc columns {[l for l, k in zip(labc + [f'rest {j}' for j in restc], kept) if k][:6]}")
    log(f"P={P:.0f}: unpaired energy shares (top 6) {[(i, round(pk[i], 1), float('%.1e' % (en[i] / en.sum()))) for i in sorted(un, key=lambda i: -en[i])[:6]]}")
    bases = {'LNM': lambda r: morth([PL[:, j] for j in range(r)]), 'MTA': mta, 'SOD': lambda r: SOD[:, :r], 'SS': lambda r: SS[:, :r], 'RC': lambda r: RC[:, :r], 'RCd': lambda r: RCd[:, :r], 'RC1': lambda r: RC1[:, :r], 'RC1f': lambda r: RC1f[:, :r], 'RCc': lambda r: RCc[:, :r]}
    for r in (1, 2, 3, 4):
        row = []
        for bn, fb in bases.items():
            if bn == 'MTA' and r < 2: continue
            B = fb(r); e = backbone(B); td = tderr(B) if r > 1 else {}
            res[(P, bn, r)] = dict(back=e, td=td)
            row.append(f"{bn} {np.nanmedian(e[~tongue]):.1e}" + (f"/static {np.median([v for k, v in td.items() if 'NNM' not in k]):.1e}/NNM {np.median([v for k, v in td.items() if 'NNM' in k]):.1e}" if td else ""))
        log(f"P={P:.0f} r={r}: backbone median | held-out static releases / NNM states, medians: " + "  ".join(row))
    for bn in ('SOD', 'SS', 'RC', 'RCd'):
        B = bases[bn](2); log(f"P={P:.0f} {bn} r=2 principal angles (deg) to LNM 1-2: {np.round(np.degrees(subspace_angles(L @ B, L @ PL[:, :2])), 2)}, to MTA: {np.round(np.degrees(subspace_angles(L @ B, L @ mta(2))), 2)}")
pickle.dump(dict(res=res, Hg=Hg, fg=fg, tongue=tongue), open(f'n9_results{TAG}.pkl', 'wb')); open(f'n9_out{TAG}.txt', 'w').write("\n".join(lines) + "\n")
