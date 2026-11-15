"""Trend note / SOD-ROM: rules for rejecting spurious smooth directions, tested where they occur (band-limited random
   response of the cantilever, SOD-ROM Sec. rank) and checked for false rejections on the trend example (t31).
   Rules (applied to the smooth coordinates q = Psi^T x of one record, Psi^T Sigma_x Psi = I):
     R1  two halves: each half returns a smooth mode with MAC > 0.9 and frequency within 5% (SOD-ROM, c8).
     R2  lag diagonality: for uncorrelated sources, Psi^T Sigma_x(tau) Psi is diagonal at every lag tau. Coordinate i is
         kept if max_{tau, j != i} |corr(q_i(t), q_j(t+tau))| (symmetric part) < eps_d. Lags: 1/16, 1/8, 1/4, 1/2 of
         the period 2 pi / omega_i of the coordinate itself, so the test probes each coordinate on its own time scale.
     R3  spectral consistency: the DCT-II periodogram peak of q_i lies within 5% of its SOD frequency omega_i
         (a narrowband coordinate has its mean-square frequency at its spectral peak; a mixture does not).
   Truth (S2): genuine if best mass-weighted MAC with an LNM >= 0.9 and SOD frequency within 5% of that LNM.
   Counts use the ten smoothest coordinates. State SOD (S3) is not tested: pair members split around |s|, so R3
   would need pair-wise frequencies, and dropping one member of a pair breaks the state basis.
   Records: random tip load, 200 s, dt = 1e-3, seed 19, 10 realizations; rank tolerance 1e-10, 1e-12, 1e-14;
   S2 also at 60 dB SNR (central differences). Output: c19_out.txt, c19_results.pkl."""
import numpy as np, sys, pickle
from scipy.linalg import eigh
from systems import beam, state
from romlib import band_noise
from romlib2 import simulate_state, rom_config, rom_state, merr, lnm, complex_modes
dt = 1e-3; N = 200000; wc = 250.0; NR = 10; rng = np.random.default_rng(19)
sy = lambda Z: (Z + Z.T) / 2
def gsod(Z, D, tol):
    s = Z.std(0); s[s == 0] = 1; Zs, Ds = Z / s, D / s
    Sx = Zs.T @ Zs / len(Z); Sd = Ds.T @ Ds / len(Z); w, U = eigh(Sx); U = U[:, w > tol * w.max()]
    mu, P = eigh(U.T @ Sd @ U, U.T @ Sx @ U); Psi = (U @ P) / s[:, None]; Phi = s[:, None] * (U @ (U.T @ Sx @ U @ P))
    return Psi, Phi, np.sqrt(np.clip(mu, 0, None))
def mac(a, b): return (a @ b)**2 / ((a @ a) * (b @ b))
def rule_halves(Z, D, tol, full, ftol=0.05, mtol=0.9):
    Psi, Phi, w = full; h = len(Z) // 2; H = [gsod(Z[:h], D[:h], tol), gsod(Z[h:], D[h:], tol)]; keep = []
    for i in range(Phi.shape[1]):
        ok = True
        for (_, Ph, wh) in H:
            m = [mac(Phi[:, i], Ph[:, j]) for j in range(Ph.shape[1])]; j = int(np.argmax(m))
            if m[j] < mtol or abs(wh[j] / max(w[i], 1e-12) - 1) > ftol: ok = False; break
        keep.append(ok)
    return np.array(keep)
def lagdiag(Q, w, dts):
    """index d_i = max over the coordinate's own lags and j != i of the symmetric lagged correlation"""
    Q = (Q - Q.mean(0)) / Q.std(0); n_ = Q.shape[1]; d = np.zeros(n_)
    for i in range(n_):
        for f in (1 / 16, 1 / 8, 1 / 4, 1 / 2):
            L = max(1, int(round(f * 2 * np.pi / max(w[i], 1e-9) / dts)))
            if L >= len(Q) // 4: continue
            c = (Q[:-L].T @ Q[L:, i] + Q[L:].T @ Q[:-L, i]) / (2 * (len(Q) - L)); c[i] = 0; d[i] = max(d[i], abs(c).max())
    return d
from scipy.fft import dct
def specpeak(Q, w, dts):
    """peak of the DCT-II periodogram (no wrap-around for non-periodic records); index k <-> omega = k pi / T"""
    F = dct(Q - Q.mean(0), type=2, axis=0, norm='ortho')**2; om = np.pi * np.arange(len(Q)) / (len(Q) * dts)
    return om[np.argmax(F, axis=0)]
out = {}; lines = []
def log(s): print(s, flush=True); lines.append(s)
for tag, S, cname in [('S2', beam(), 'light')]:
    M, K, B = S['M'], S['K'], S['B']; C = S['C'][cname]; n = len(M); A, Bs = state(M, K, C, B)
    WE = np.block([[K, np.zeros((n, n))], [np.zeros((n, n)), M]]); Lw = np.linalg.cholesky(WE).T
    wL2, PL = eigh(K, M); wL = np.sqrt(wL2)
    if tag == 'S3':
        PsiC, PhiC = complex_modes(A, 24); lam = np.linalg.eigvals(A); lam = lam[lam.imag > 1e-9]; sabs = np.sort(abs(lam))[:12]
        planes = [np.linalg.qr(Lw @ PhiC[:, 2 * i:2 * i + 2])[0] for i in range(12)]
    for rr in range(NR):
        u = band_noise(N, dt, wc, 2, rng) * np.r_[1, 0]; Y = simulate_state(A, Bs, u, dt)
        tests = {'rand': band_noise(N, dt, wc, 2, rng) * np.r_[1, 0], 'mid': band_noise(N, dt, wc, 2, rng) * np.r_[0, 1]}
        Xt = {k: simulate_state(A, Bs, v, dt)[::5, :n] for k, v in tests.items()}
        snrs = [None, 60] if tag == 'S2' else [None]
        for snr in snrs:
            if tag == 'S2':
                X = Y[:, :n]
                if snr is None: Z, D = X[1:-1], Y[1:-1, n:]
                else:
                    Xm = X + X.std(0) * 10**(-snr / 20) * rng.standard_normal(X.shape); Z, D = Xm[1:-1], (Xm[2:] - Xm[:-2]) / (2 * dt)
            else: Z, D = Y, Y @ A.T + u @ Bs.T
            for tol in (1e-10, 1e-12, 1e-14):
                full = gsod(Z, D, tol); Psi, Phi, w = full; Q = Z @ Psi; k = Phi.shape[1]
                if tag == 'S2':
                    truth = []
                    for i in range(k):
                        x = Phi[:, i]; m = [(x @ M @ PL[:, j])**2 / ((x @ M @ x) * (PL[:, j] @ M @ PL[:, j])) for j in range(14)]
                        j = int(np.argmax(m)); truth.append(m[j] >= 0.9 and abs(w[i] / wL[j] - 1) < 0.05)
                else:
                    truth = []
                    for i in range(k):
                        v = Lw @ Phi[:, i]; v = v / np.linalg.norm(v); pr = [np.linalg.norm(P.T @ v) for P in planes]; j = int(np.argmax(pr))
                        truth.append(pr[j] >= 0.95 and abs(w[i] / sabs[j] - 1) < 0.25)
                truth = np.array(truth)
                d = lagdiag(Q, w, dt); pk = specpeak(Q, w, dt)
                rules = {'none': np.ones(k, bool), 'R1 halves': rule_halves(Z, D, tol, full),
                         'R2 lag-diag 0.3': d < 0.3,
                         'R3 spectral': abs(pk / np.maximum(w, 1e-9) - 1) < 0.05}
                for rn, keep in rules.items():
                    key = (tag, snr, tol, rn); o = out.setdefault(key, dict(spur=0, spur_rej=0, gen=0, gen_rej=0, err={}))
                    tt, kp = truth[:10], keep[:10]                                     # the ten smoothest: those a ROM can use
                    o['spur'] += int((~tt).sum()); o['spur_rej'] += int((~tt & ~kp).sum())
                    o['gen'] += int(tt.sum()); o['gen_rej'] += int((tt & ~kp).sum())
                    idx = np.where(keep)[0]
                    for r in (2, 4, 6):
                        kk = r if tag == 'S2' else 2 * r
                        if len(idx) < kk: continue
                        Ph = Phi[:, idx[:kk]]
                        for tk, v in tests.items():
                            if tag == 'S2': Xr = rom_config(M, K, C, B, Ph / np.linalg.norm(Ph, axis=0), v, dt, dec=5)
                            else: Xr = rom_state(A, Bs, None, Ph, v, dt, None, n, W=WE, dec=5)[0]
                            o['err'].setdefault((r, tk), []).append(merr(Xr, Xt[tk], M))
        log(f"{tag}: realization {rr + 1}/{NR}")
    for key, o in out.items():
        if key[0] != tag: continue
        e = " ".join(f"r={r} {tk} {np.median(o['err'][(r, tk)]):.1e}({len(o['err'][(r, tk)])})" for r in (2, 4, 6) for tk in ('rand', 'mid') if (r, tk) in o['err'])
        log(f"{tag} snr={key[1]} tol={key[2]:.0e} {key[3]:16s}: spurious rejected {o['spur_rej']}/{o['spur']}, genuine rejected {o['gen_rej']}/{o['gen']} | {e}")
pickle.dump(out, open('c19_results.pkl', 'wb')); open('c19_out.txt', 'w').write("\n".join(lines))
