"""SOD-ROM, mechanism test for free-decay training. Hypothesis: in a transient the modal-plane cross-correlation eps_ij
   does not vanish with record length, so smooth modes carry a small component delta of planes above the ROM order;
   Galerkin turns that leak into a stiffness error ~ delta^2 omega_j^2, which dominates the ROM error.
   For S2 light and S3 c_d = 0.2, 1 and several training ICs, report: (i) max normalized inter-plane correlation in the
   training record; (ii) energy-norm leakage of the 8-dim state-SOD subspace (and 4-dim config SOD) outside planes 1-4;
   (iii) eigenvalue error of the state ROM; (iv) ROM errors (mid, freeEE tests).
   ICs: SR = static tip release (as c5); EE = equal energy, modes 1-9; RE = equal record energy, modes 1-9;
   EE4 = equal energy, modes 1-4 only (band-limited); HAM = equal energy times a hammer-like roll-off 1/(1+(w/w_c)^4),
   w_c = omega_4 (soft tip)."""
import numpy as np
from scipy.linalg import eigh
from systems import beam, state
from romlib import band_noise
from romlib2 import *
dt = 1e-3; N = 200000; wc = 250.0; r = 4
for tag, S, cname in [('S2', beam(), 'light'), ('S3', beam(dashpot=(0.2, 1.0)), 'dashpot'), ('S3', beam(dashpot=(0.2, 1.0)), 'dashpot_heavy')]:
    rng = np.random.default_rng(2026)
    M, K, B = S['M'], S['K'], S['B']; C = S['C'][cname]; n = len(M); A, Bs = state(M, K, C, B)
    WE = np.block([[K, np.zeros((n, n))], [np.zeros((n, n)), M]]); Lw = np.linalg.cholesky(WE).T
    wL2, PL = eigh(K, M); w = np.sqrt(wL2); kEE = np.where(w < np.pi / (4 * dt))[0]; m = len(kEE)
    PsiC, PhiC = complex_modes(A, 2 * n)                     # all planes (real form), slowest first
    nP = PhiC.shape[1]; Qc = np.linalg.qr(Lw @ PhiC[:, :2 * r])[0]          # energy-orthonormal basis of planes 1..r
    s_true = np.linalg.eigvals(A); s_true = np.sort_complex(s_true[np.argsort(abs(s_true))][:2 * r])
    lam = np.linalg.eigvals(A); lam = lam[np.argsort(abs(lam))]; groups = []; c0 = 0; seen = 0
    for l in lam:                                            # column groups of PhiC: complex pair -> 2 columns, real -> 1
        if l.imag < -1e-12: continue
        k = 2 if abs(l.imag) > 1e-12 else 1; groups.append(list(range(c0, c0 + k))); c0 += k
    R = []
    for k in kEE:
        Yk = simulate_state(A, Bs, np.zeros((N, 2)), dt, np.r_[PL[:, k] / w[k], np.zeros(n)])
        R.append(np.mean(np.sum((Yk @ Lw.T) ** 2, 1)) / 2); del Yk
    R = np.array(R) / R[0]
    def ic(wts):
        x = PL[:, kEE] @ (wts / w[kEE]); return np.r_[x / abs(x).max(), np.zeros(n)]
    ICs = {'SR': np.r_[np.linalg.solve(K, B[:, 0]) / abs(np.linalg.solve(K, B[:, 0])).max(), np.zeros(n)],
           'EE': ic(np.ones(m)), 'RE': ic(1 / np.sqrt(R)), 'EE4': ic((np.arange(m) < r).astype(float)),
           'HAM': ic(1 / (1 + (w[kEE] / w[r - 1]) ** 4))}
    u_mid = band_noise(N, dt, wc, 2, rng) * np.r_[0, 1]
    xte = ic((-1.0) ** np.arange(m))
    tests = {'mid': (u_mid, None), 'freeEE': (np.zeros((N, 2)), xte)}
    Xt = {tk: simulate_state(A, Bs, u, dt, y0)[:, :n] for tk, (u, y0) in tests.items()}
    print(f"== {tag} {cname}", flush=True)
    Wl = np.linalg.solve(PhiC, np.eye(2 * n))                # plane coordinates of a state: Wl @ y
    for icn, y0 in ICs.items():
        Y = simulate_state(A, Bs, np.zeros((N, 2)), dt, y0); Yd = Y @ A.T
        Z = Y @ Wl.T; G = Z.T @ Z; d = np.sqrt(np.diag(G)); Cn = G / np.outer(d, d)
        eps = max(abs(Cn[2 * i:2 * i + 2, grp]).max() for i in range(r) for grp in groups[:m] if grp[0] != 2 * i)
        Psi, Phi, _ = state_sod(Y, Yd, 2 * r); Phs = sod(Y[:, :n], Y[:, n:], r)[0]; del Y, Yd, Z
        Qs = np.linalg.qr(Lw @ Phi)[0]; leak = np.linalg.norm(Qs - Qc @ (Qc.T @ Qs), 2)
        Qx = np.linalg.qr(Lw @ np.r_[Phs, np.zeros((n, r))])[0]; Qx2 = np.linalg.qr(Lw @ np.r_[np.zeros((n, r)), Phs])[0]
        leakc = max(np.linalg.norm(Q - Qc @ (Qc.T @ Q), 2) for Q in (Qx, Qx2))
        T = Phi.T @ WE; Ar = np.linalg.solve(T @ Phi, T @ A @ Phi); sr = np.linalg.eigvals(Ar); sr = np.sort_complex(sr)
        ds = max(np.min(abs(sr - st)) / abs(st) for st in s_true)
        es = [merr(rom_state(A, Bs, Psi, Phi, u, dt, y0t, n, W=WE)[0], Xt[tk], M) for tk, (u, y0t) in tests.items()]
        ec = [merr(rom_config(M, K, C, B, Phs, u, dt, y0t), Xt[tk], M) for tk, (u, y0t) in tests.items()]
        print(f"  {icn:4s}: eps {eps:.2e} | leak sSOD {leak:.2e} SOD {leakc:.2e} | max rel eig err sROM {ds:.2e} | "
              f"sSOD mid {es[0]:.2e} freeEE {es[1]:.2e} | SOD mid {ec[0]:.2e} freeEE {ec[1]:.2e}", flush=True)
    print("  CM: " + " ".join(f"{tk} {merr(rom_state(A, Bs, PsiC[:, :2*r], PhiC[:, :2*r], u, dt, y0t, n)[0], Xt[tk], M):.2e}" for tk, (u, y0t) in tests.items()), flush=True)
