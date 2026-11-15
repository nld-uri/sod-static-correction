"""SOD-ROM, IC weighting test: equal energy at t = 0 (EE) versus equal record-integrated energy (RE).
   RE: each LNM k is released alone from phi_k/omega_k; its energy integrated over the record, R_k, is computed with the
   full damped model; the training IC uses amplitude 1/sqrt(R_k) per mode, so every mode carries the same energy over the
   record (for proportional damping this is amplitude ~ sqrt(zeta_k omega_k)/omega_k). Same signs as EE (+1 train, alternating test).
   Checks on S2 light and S3 c_d = 0.2, 1: smooth-mode/plane alignment of state SOD and ROM errors at state dimension 8.
   Test loads: random mid-span (seed 2026, as c11) and EE alternating-sign free decay."""
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
    wL2, PL = eigh(K, M); kEE = np.where(np.sqrt(wL2) < np.pi / (4 * dt))[0]
    R = []
    for k in kEE:                                      # record energy of each LNM released alone
        Yk = simulate_state(A, Bs, np.zeros((N, 2)), dt, np.r_[PL[:, k] / np.sqrt(wL2[k]), np.zeros(n)])
        R.append(np.mean(np.sum((Yk @ Lw.T) ** 2, 1)) / 2); del Yk
    R = np.array(R)
    def ic(sg, wts):
        x = PL[:, kEE] @ (sg * wts / np.sqrt(wL2[kEE])); return x / abs(x).max()
    PsiC, PhiC = complex_modes(A, 2 * r)
    u_mid = band_noise(N, dt, wc, 2, rng) * np.r_[0, 1]
    xte = ic((-1.0) ** np.arange(len(kEE)), np.ones(len(kEE)))
    tests = {'mid': (u_mid, None), 'freeEE': (np.zeros((N, 2)), np.r_[xte, np.zeros(n)])}
    Xt = {tk: simulate_state(A, Bs, u, dt, y0)[:, :n] for tk, (u, y0) in tests.items()}
    print(f"== {tag} {cname}: record energy ratio R_k/R_1 =", np.round(R / R[0], 3), flush=True)
    for icn, wts in [('EE', np.ones(len(kEE))), ('RE', 1 / np.sqrt(R / R[0]))]:
        Y = simulate_state(A, Bs, np.zeros((N, 2)), dt, np.r_[ic(np.ones(len(kEE)), wts), np.zeros(n)]); Yd = Y @ A.T
        Psi, Phi, w = state_sod(Y, Yd, 2 * r); Phs = sod(Y[:, :n], Y[:, n:], r)[0]; del Y, Yd
        Phw = Lw @ Phi; pr = []
        for j in range(Phi.shape[1]):
            v = Phw[:, j] / np.linalg.norm(Phw[:, j])
            pr.append(max(np.linalg.norm(np.linalg.qr(Lw @ PhiC[:, 2 * i:2 * i + 2])[0].T @ v) for i in range(r)))
        out = []
        for tk, (u, y0) in tests.items():
            e_s = merr(rom_state(A, Bs, Psi, Phi, u, dt, y0, n, W=WE)[0], Xt[tk], M)
            e_c = merr(rom_config(M, K, C, B, Phs, u, dt, y0), Xt[tk], M)
            out.append(f"{tk}: sSOD {e_s:.2e} SOD {e_c:.2e}")
        print(f"  {icn}: sSOD freqs {np.round(w, 1)}; best plane projection {np.round(pr, 2)}; " + "; ".join(out), flush=True)
    for tk, (u, y0) in tests.items():
        print(f"  CM {tk}: {merr(rom_state(A, Bs, PsiC, PhiC, u, dt, y0, n)[0], Xt[tk], M):.2e}", flush=True)
