"""SOD-ROM / trend note: pair-wise spectral consistency for state-space SOD (suggestion of D. Chelidze).
   A complex mode gives two smooth coordinates spanning its modal plane. Both are real combinations of the complex
   modal coordinate (its real and imaginary parts), so both oscillate at the damped frequency omega_d of that plane,
   although their SOD values split around |s|. Rule P: walk up the smoothest coordinates; pair coordinate a with the
   next coordinate b if their DCT periodogram peaks agree within 5%, and keep the pair if the common peak lies within
   10% of the pair frequency (mu_a mu_b)^(1/4) (|s| vs omega_d differ by sqrt(1 - zeta^2)). Otherwise reject a and
   move on. Kept coordinates always come in pairs, so the state basis keeps whole planes.
   Rule P2 (circularity): read the pair as the real and imaginary parts of one complex coordinate; a modal plane
   rotates one way, so its spectrum is one-sided (see rule_circ).
   Truth: a coordinate is modal if its energy-norm projection onto one modal plane (complex-mode truncation) is >= 0.9;
   a kept pair is genuine if both members project onto the same plane.
   Records: S3 (c_d = 0.2 and 1), random tip load (spurious directions expected) and the static-release decay
   (a check for false rejections), 200 s, dt = 1e-3, seed 20, 10 realizations for random records; rank tolerance
   1e-10 and 1e-12. ROMs: energy Galerkin at state dimension 4, 8, 12; tests: new random tip load, random mid-span load.
   Output: c20_out.txt, c20_results.pkl."""
import numpy as np, pickle
from scipy.linalg import eigh
from scipy.fft import dct
from systems import beam, state
from romlib import band_noise
from romlib2 import simulate_state, rom_state, merr, complex_modes
dt = 1e-3; N = 200000; wc = 250.0; NR = 10; rng = np.random.default_rng(20)
def gsod(Z, D, tol):
    s = Z.std(0); s[s == 0] = 1; Zs, Ds = Z / s, D / s
    Sx = Zs.T @ Zs / len(Z); Sd = Ds.T @ Ds / len(Z); w, U = eigh(Sx); U = U[:, w > tol * w.max()]
    mu, P = eigh(U.T @ Sd @ U, U.T @ Sx @ U); Psi = (U @ P) / s[:, None]; Phi = s[:, None] * (U @ (U.T @ Sx @ U @ P))
    return Psi, Phi, mu
def peaks(Q):
    F = dct(Q - Q.mean(0), type=2, axis=0, norm='ortho')**2; om = np.pi * np.arange(len(Q)) / (len(Q) * dt)
    return om[np.argmax(F, axis=0)]
def rule_pairs(mu, pk, kmax, ptol=0.05, gtol=0.10):
    """P1: both members peak at the same frequency, near the pair frequency"""
    keep = []; i = 0
    while i < kmax - 1:
        a, b = i, i + 1; g = (mu[a] * mu[b]) ** 0.25; p = 0.5 * (pk[a] + pk[b])
        if abs(pk[a] / pk[b] - 1) < ptol and abs(p / g - 1) < gtol: keep += [a, b]; i += 2
        else: i += 1
    return np.array(keep, int)
def rule_circ(Q, mu, kmax, ctol=0.9, gtol=0.25):
    """P2: the pair, read as the real and imaginary parts of one complex coordinate z = q_a + i q_b, must rotate one
       way: a modal plane gives a one-sided spectrum of z. SOD coordinates are uncorrelated with unit variance, so a
       genuine plane is a round cloud and z is nearly circular. Also the one-sided spectral peak lies within 25%
       of the pair frequency."""
    keep = []; i = 0; om = 2 * np.pi * np.fft.fftfreq(len(Q), dt)
    while i < kmax - 1:
        a, b = i, i + 1; qa = (Q[:, a] - Q[:, a].mean()) / Q[:, a].std(); qb = (Q[:, b] - Q[:, b].mean()) / Q[:, b].std()
        F = np.abs(np.fft.fft(qa + 1j * qb))**2; Pp, Pm = F[om > 0].sum(), F[om < 0].sum(); c = max(Pp, Pm) / (Pp + Pm)
        side = om > 0 if Pp >= Pm else om < 0; cen = abs(om[side][np.argmax(F[side])]); g = (mu[a] * mu[b]) ** 0.25
        if c > ctol and abs(cen / g - 1) < gtol: keep += [a, b]; i += 2
        else: i += 1
    return np.array(keep, int)
lines = []; out = {}
def log(s): print(s, flush=True); lines.append(s)
for cname, cd in [('dashpot', 0.2), ('dashpot_heavy', 1.0)]:
    S = beam(dashpot=(0.2, 1.0)); M, K, B = S['M'], S['K'], S['B']; C = S['C'][cname]; n = len(M); A, Bs = state(M, K, C, B)
    WE = np.block([[K, np.zeros((n, n))], [np.zeros((n, n)), M]]); Lw = np.linalg.cholesky(WE).T
    PsiC, PhiC = complex_modes(A, 24); planes = [np.linalg.qr(Lw @ PhiC[:, 2 * i:2 * i + 2])[0] for i in range(12)]
    xs = np.linalg.solve(K, B @ np.array([1.0, 0.7])); xs /= abs(xs).max()
    def plane_of(phi):
        v = Lw @ phi; v = v / np.linalg.norm(v); pr = [np.linalg.norm(P.T @ v) for P in planes]; j = int(np.argmax(pr)); return j if pr[j] >= 0.9 else -1
    for rr in range(NR):
        u = band_noise(N, dt, wc, 2, rng) * np.r_[1, 0]
        tests = {'rand': band_noise(N, dt, wc, 2, rng) * np.r_[1, 0], 'mid': band_noise(N, dt, wc, 2, rng) * np.r_[0, 1]}
        Xt = {k: simulate_state(A, Bs, v, dt)[::5, :n] for k, v in tests.items()}
        recs = {'random': (u, None)}
        if rr == 0: recs['SR decay'] = (np.zeros((N, 2)), np.r_[xs, np.zeros(n)])
        for rec, (uu, y0) in recs.items():
            Y = simulate_state(A, Bs, uu, dt, y0); D = Y @ A.T + uu @ Bs.T
            for tol in (1e-10, 1e-12):
                Psi, Phi, mu = gsod(Y, D, tol); Q = Y @ Psi; pk = peaks(Q); kmax = min(12, Phi.shape[1])
                pl = np.array([plane_of(Phi[:, i]) for i in range(kmax)])
                sel = {'none': np.arange(kmax), 'P1 pair peaks': rule_pairs(mu, pk, kmax), 'P2 circular': rule_circ(Q, mu, kmax)}
                for rn, idx in sel.items():
                    o = out.setdefault((cd, rec, tol, rn), dict(n_modal=0, n_spur=0, kept_modal=0, kept_spur=0, pairs_ok=0, pairs=0, err={}))
                    if rn == 'none':
                        o['n_modal'] += int((pl >= 0).sum()); o['n_spur'] += int((pl < 0).sum())
                    else:
                        o['n_modal'] += int((pl >= 0).sum()); o['n_spur'] += int((pl < 0).sum())
                        o['kept_modal'] += int((pl[idx] >= 0).sum()); o['kept_spur'] += int((pl[idx] < 0).sum())
                        for a, b in zip(idx[0::2], idx[1::2]): o['pairs'] += 1; o['pairs_ok'] += int(pl[a] >= 0 and pl[a] == pl[b])
                    for k2 in (4, 8, 12):
                        if len(idx) < k2: continue
                        for tk, v in tests.items():
                            o['err'].setdefault((k2, tk), []).append(merr(rom_state(A, Bs, None, Phi[:, idx[:k2]], v, dt, None, n, W=WE, dec=5)[0], Xt[tk], M))
        log(f"c_d={cd}: realization {rr + 1}/{NR}")
    for key, o in out.items():
        if key[0] != cd: continue
        e = " ".join(f"{k2}/{tk} {np.median(o['err'][(k2, tk)]):.1e}({len(o['err'][(k2, tk)])})" for k2 in (4, 8, 12) for tk in ('rand', 'mid') if (k2, tk) in o['err'])
        extra = "" if key[3] == 'none' else f" kept modal {o['kept_modal']}, kept non-modal {o['kept_spur']}, genuine pairs {o['pairs_ok']}/{o['pairs']} |"
        log(f"c_d={cd} {key[1]:9s} tol={key[2]:.0e} {key[3]:14s}: smoothest 12 hold {o['n_modal']} modal, {o['n_spur']} non-modal |{extra} {e}")
pickle.dump(out, open('c20_results.pkl', 'wb')); open('c20_out.txt', 'w').write("\n".join(lines) + "\n")
