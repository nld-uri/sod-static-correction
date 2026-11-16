"""SOD-ROM: state-space SOD on S3 (non-proportional damping). For each training record (free decay, random, swept sine)
   and c_d = 0.2, 1: smooth modes 1..10 are assigned to the slowest complex-mode planes of A by the largest projection;
   reports, per plane, the pair geometric-mean frequency (mu_a mu_b)^(1/4) against |s_i|, and the largest principal angle
   between the pair's span and the modal plane, both in the energy metric W = blockdiag(K, M). Configuration SOD MAC with the undamped LNMs is shown for contrast.
   Also the largest correlation between modal-plane coordinates of the training record."""
import numpy as np
from scipy.linalg import subspace_angles, eigh
from systems import beam, state
from romlib import band_noise, chirp
from romlib2 import simulate_state, state_sod, sod, complex_modes
dt = 1e-3; N = 200000; wc = 250.0
for cname, cd in [('dashpot', 0.2), ('dashpot_heavy', 1.0)]:
    S = beam(dashpot=(0.2, 1.0)); M, K, B = S['M'], S['K'], S['B']; C = S['C'][cname]; n = len(M); A, Bs = state(M, K, C, B)
    s = np.linalg.eigvals(A); s = s[s.imag > 0]; s = s[np.argsort(abs(s))][:5]
    PsiC, PhiC = complex_modes(A, 10)
    rng = np.random.default_rng(11); xs = np.linalg.solve(K, B @ np.array([1.0, 0.7])); xs /= abs(xs).max()
    recs = {'free decay': (np.zeros((N, 2)), np.r_[xs, np.zeros(n)]), 'random': (band_noise(N, dt, wc, 2, rng) * np.r_[1, 0], None),
            'swept sine': (chirp(N, dt, 2.0, wc, 2, 0), None)}
    print(f"== S3, c_d = {cd}: |s_1..5| = {np.round(abs(s), 2)}, zeta_1..5 = {np.round(-s.real / abs(s), 3)}")
    for tr, (u, y0) in recs.items():
        Y = simulate_state(A, Bs, u, dt, y0); Yd = Y @ A.T + u @ Bs.T
        Psi, Phi, w = state_sod(Y, Yd, 10)
        xi = Y @ PsiC                                      # modal-plane coordinates of the slowest 5 planes
        Cc = np.corrcoef(xi.T); blk = np.kron(np.eye(5), np.ones((2, 2))); cmax = np.max(abs(Cc * (1 - blk)))
        out = []; Lw = np.linalg.cholesky(np.block([[K, np.zeros((n, n))], [np.zeros((n, n)), M]])).T   # energy metric
        Phw = Lw @ Phi
        for i in range(5):
            P = Lw @ PhiC[:, 2 * i:2 * i + 2]; Q = np.linalg.qr(P)[0]
            proj = [np.linalg.norm(Q.T @ (Phw[:, j] / np.linalg.norm(Phw[:, j]))) for j in range(Phw.shape[1])]
            jj = np.argsort(proj)[-2:]; ang = np.degrees(subspace_angles(Phw[:, jj], P).max())
            gm = np.sqrt(np.sqrt(np.prod(w[jj]**2))); out.append((sorted(jj + 1), gm / abs(s[i]) - 1, ang))
        Pc, wcfg = sod(Y[:, :n], Y[:, n:], 6); L = eigh(K, M)[1]
        mac = [max((Pc[:, i] @ M @ L[:, j])**2 / ((Pc[:, i] @ M @ Pc[:, i]) * (L[:, j] @ M @ L[:, j])) for j in range(10)) for i in range(6)]
        print(f"  {tr:10s} max inter-plane corr {cmax:.2f} | " + " | ".join(f"plane {i+1}: modes {jj}, freq err {e:+.1e}, angle {a:.1f} deg" for i, (jj, e, a) in enumerate(out)))
        print(f"  {'':10s} configuration SOD best MAC with LNMs, modes 1-6: {np.round(mac, 2)}")
