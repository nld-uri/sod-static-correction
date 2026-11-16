"""Which circularity rule? Broadband one-sidedness c0 of z = q_a + i q_b against banded c1 (power within +-30% of the
   pair's one-sided peak; pairs whose peak is the zero-frequency bin get c1 = c0). S3 (c_d = 0.2), 3 records each of
   the band-limited decay and the random tip load, exact kinematics and estimated ones at 40 dB (WH, KS, FUS, TIK, as
   c22). Every candidate pair inside the frequency gate of sellib.pair_state is labelled modal if the largest principal
   angle (energy metric) of its span to a complex-mode plane 1-14 is below 15 deg. Rules compared by the pairs they
   accept (before greedy assignment): broadband c0 >= 0.85; banded c1 >= 0.85; combined c0 >= t0 with c1 >= 0.85;
   and a rescue rule: c0 >= 0.85, or c0 >= 0.75 with c1 >= 0.95 and a pair energy share (||phi||^2 var(q), both
   members, relative to all coordinates) of at least 1e-3 or 1e-2.
   Seed 26. Output: c26_out.txt."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kin'))
from scipy.linalg import subspace_angles, eigh
from systems import beam, state
from romlib import band_noise
from romlib2 import simulate_state, complex_modes
from sellib import gsod, dct_spectrum, peak_and_width
from kinematics import diff_whittaker, int_tikhonov, _central
from ksfast import ks_fast
dt = 1e-3; N = 200000; wc = 250.0; rng = np.random.default_rng(26); lines = []
def log(s): print(s, flush=True); lines.append(s)
S = beam(dashpot=(0.2, 1.0)); M, K, B = S['M'], S['K'], S['B']; C = S['C']['dashpot']; n = len(M); A, Bs = state(M, K, C, B)
wL2, PL = eigh(K, M); wL = np.sqrt(wL2); kEE = np.where(wL < np.pi / (4 * dt))[0]
xbl = PL[:, kEE] @ (1 / (1 + (wL[kEE] / wc) ** 4) / wL[kEE]); xbl /= abs(xbl).max()
Lw = np.linalg.cholesky(np.block([[K, np.zeros((n, n))], [np.zeros((n, n)), M]])).T; _, PhiC = complex_modes(A, 28)
om2 = 2 * np.pi * np.fft.fftfreq(N, dt); pos, neg = om2 > 0, om2 < 0
def pairs(Yz, Yd):
    Psi, Phi, mu = gsod(Yz, Yd, 1e-10); Q = Yz @ Psi; om, F = dct_spectrum(Q, dt); pk, rel = peak_and_width(om, F)
    cand = [i for i in range(Q.shape[1]) if pk[i] < 1.2 * wc]; en = (Phi**2).sum(0) * Q.var(0); en = en / en.sum(); Qn = (Q[:, cand] - Q[:, cand].mean(0)) / Q[:, cand].std(0); FQ = np.fft.fft(Qn, axis=0); out = []
    for x, a in enumerate(cand):
        for y in range(x + 1, len(cand)):
            b = cand[y]
            if abs(pk[a] / pk[b] - 1) >= 4 * max(0.05, rel[a], rel[b]): continue
            Fz = np.abs(FQ[:, x] + 1j * FQ[:, y])**2; Pp, Pm = Fz[pos].sum(), Fz[neg].sum(); side = pos if Pp >= Pm else neg
            c0 = max(Pp, Pm) / (Pp + Pm); f = abs(om2[side][np.argmax(Fz[side])]); c1 = c0
            if f > 0:
                bd = abs(abs(om2) / f - 1) < 0.3; P2, M2 = Fz[pos & bd].sum(), Fz[neg & bd].sum(); c1 = max(P2, M2) / (P2 + M2)
            ang = min(np.degrees(subspace_angles(Lw @ Phi[:, [a, b]], Lw @ PhiC[:, 2 * k:2 * k + 2])).max() for k in range(14))
            out.append((c0, c1, ang < 15, f, en[a] + en[b]))
    return out
res = {}
for rec in ('decay', 'random'):
    for rr in range(3):
        if rec == 'random': u = band_noise(N, dt, wc, 2, rng) * np.r_[1, 0]; y0 = None
        else: u = np.zeros((N, 2)); y0 = np.r_[xbl, np.zeros(n)]
        Y = simulate_state(A, Bs, u, dt, y0); Yd = Y @ A.T + u @ Bs.T; X0, V0, A0 = Y[:, :n], Y[:, n:], Yd[:, n:]
        res.setdefault((rec, 'exact'), []).extend(pairs(Y, Yd))
        x = X0 + X0.std(0) * 1e-2 * rng.standard_normal(X0.shape); a = A0 + A0.std(0) * 1e-2 * rng.standard_normal(A0.shape)
        s = np.column_stack([diff_whittaker(x[:, j], dt, 0, fc=100.0) for j in range(n)]); v = np.column_stack([_central(s[:, j], dt, 1) for j in range(n)]); ac = np.column_stack([_central(s[:, j], dt, 2) for j in range(n)])
        res.setdefault((rec, 'WH'), []).extend(pairs(np.c_[s, v], np.c_[v, ac])); del s, v, ac
        Xk, Vk, Ak = np.empty_like(x), np.empty_like(x), np.empty_like(x)
        for j in range(n):
            q = ks_fast(dt, x=x[:20000, j])['q']; r = ks_fast(dt, x=x[:, j], q=q, fit=False); Xk[:, j], Vk[:, j], Ak[:, j] = r['x'], r['v'], r['a']
        res.setdefault((rec, 'KS'), []).extend(pairs(np.c_[Xk, Vk], np.c_[Vk, Ak]))
        for j in range(n):
            Xk[:, j] = int_tikhonov(a[:, j], dt, 2, fc=0.1); Vk[:, j] = int_tikhonov(a[:, j], dt, 1, fc=0.1); Ak[:, j] = diff_whittaker(a[:, j], dt, 0, fc=100.0)
        res.setdefault((rec, 'TIK'), []).extend(pairs(np.c_[Xk, Vk], np.c_[Vk, Ak]))
        for j in range(n):
            q = ks_fast(dt, x=x[:20000, j], a=a[:20000, j])['q']; r = ks_fast(dt, x=x[:, j], a=a[:, j], q=q, fit=False); Xk[:, j], Vk[:, j], Ak[:, j] = r['x'], r['v'], r['a']
        res.setdefault((rec, 'FUS'), []).extend(pairs(np.c_[Xk, Vk], np.c_[Vk, Ak]))
        print(rec, rr, flush=True)
rules = {'broadband c0>=0.85': lambda c0, c1, e: c0 >= 0.85, 'banded c1>=0.85': lambda c0, c1, e: c1 >= 0.85}
for t0 in (0.75, 0.8): rules[f'c0>={t0} & c1>=0.85'] = (lambda t: (lambda c0, c1, e: c0 >= t and c1 >= 0.85))(t0)
for es in (1e-3, 1e-2): rules[f'rescue e>={es:g}'] = (lambda q: (lambda c0, c1, e: c0 >= 0.85 or (c0 >= 0.75 and c1 >= 0.95 and e >= q)))(es)
for key, L in res.items():
    L = np.array(L, float); md = L[:, 2] > 0
    log(f"{key[0]:6s} {key[1]:5s}: modal pairs {md.sum():3.0f} (c0 {L[md, 0].min() if md.any() else np.nan:.3f}-{L[md, 0].max() if md.any() else np.nan:.3f}), other {(~md).sum():3.0f}; accepted modal/other: "
        + "; ".join(f"{k} {sum(f(*r) for r in L[md][:, [0, 1, 4]])}/{sum(f(*r) for r in L[~md][:, [0, 1, 4]])}" for k, f in rules.items()))
open('c26_out.txt', 'w').write("\n".join(lines) + "\n")
