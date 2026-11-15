"""SOD-ROM figures: ROM trajectories against the full model (one realization, seed 2026).
   Fig. c_traj: (a,b) S2 light Rayleigh, configuration ROMs with r = 4 (state dimension 8): tip displacement under a random
   mid-span load (transfer test) and in the equal-energy free decay test; (c,d) S3 c_d = 1, state dimension 8: same tests,
   with complex-mode truncation, MTA, and state SOD trained on three free decays: static release (tip 1, mid 0.7, as c5),
   equal energy over modes 1-9 (as c10), and equal energy rolled off above omega_4 (order-matched, as c15 at r = 4).
   Bottom row: running error |x_tip,ROM - x_tip,full| / RMS(x_tip,full). Data saved to c11_timehist.npz."""
import numpy as np, sys
from scipy.linalg import eigh
from systems import beam, state
from romlib import band_noise
from romlib2 import *
dt = 1e-3; N = 200000; wc = 250.0; rng = np.random.default_rng(2026); out = {}
for tag, S, cname in [('S2', beam(), 'light'), ('S3', beam(dashpot=(0.2, 1.0)), 'dashpot_heavy')]:
    M, K, B = S['M'], S['K'], S['B']; C = S['C'][cname]; n = len(M); A, Bs = state(M, K, C, B); tip, mid = S['tip'], S['mid']
    WE = np.block([[K, np.zeros((n, n))], [np.zeros((n, n)), M]])
    wL2, PL = eigh(K, M); kEE = np.where(np.sqrt(wL2) < np.pi / (4 * dt))[0]
    ee = lambda sg: (lambda x: x / abs(x).max())(PL[:, kEE] @ (sg / np.sqrt(wL2[kEE])))
    xtr, xte = ee(np.ones(len(kEE))), ee((-1.0) ** np.arange(len(kEE)))
    xbl = ee(1 / (1 + (np.sqrt(wL2[kEE]) / np.sqrt(wL2[3])) ** 4))
    xsr = np.linalg.solve(K, B @ np.array([1.0, 0.7])); xsr /= abs(xsr).max()
    Yf = simulate_state(A, Bs, np.zeros((N, 2)), dt, np.r_[xtr, np.zeros(n)]); Ydf = Yf @ A.T
    u_tr = band_noise(N, dt, wc, 2, rng) * np.r_[1, 0]; Yr = simulate_state(A, Bs, u_tr, dt); Xr, Vr = Yr[:, :n], Yr[:, n:]
    tests = {'mid': (band_noise(N, dt, wc, 2, rng) * np.r_[0, 1], None), 'freeEE': (np.zeros((N, 2)), np.r_[xte, np.zeros(n)])}
    r = 4
    for tk, (u, y0) in tests.items():
        Xf = simulate_state(A, Bs, u, dt, y0)[:, :n]; out[f'{tag}_{tk}_full'] = Xf[:, tip]
        rows = {'LNM': ('c', lnm(M, K, r)), 'MTA': ('c', mta(M, K, B, r)), 'SOD (free EE)': ('c', sod(Yf[:, :n], Yf[:, n:], r)[0]),
                'SOD (random)': ('c', sod(Xr, Vr, r)[0])}
        if tag == 'S3':
            PsiC, PhiC = complex_modes(A, 2 * r); rows = {'CM': ('sp', (PsiC, PhiC)), 'MTA': rows['MTA']}
            for lab, y0t in [('sSOD (SR)', np.r_[xsr, np.zeros(n)]), ('sSOD (free EE)', np.r_[xtr, np.zeros(n)]), ('sSOD (BL)', np.r_[xbl, np.zeros(n)])]:
                Yt = simulate_state(A, Bs, np.zeros((N, 2)), dt, y0t); Psi, Phi, _ = state_sod(Yt, Yt @ A.T, 2 * r); del Yt
                rows[lab] = ('s', (Psi, Phi))
        for bn, (kind, Bas) in rows.items():
            if kind == 'c': Xm = rom_config(M, K, C, B, Bas, u, dt, y0)
            elif kind == 's': Xm = rom_state(A, Bs, *Bas, u, dt, y0, n, W=WE)[0]
            else: Xm = rom_state(A, Bs, *Bas, u, dt, y0, n, W=None)[0]
            out[f'{tag}_{tk}_{bn}'] = Xm[:, tip]
            print(f"{tag} {tk} {bn}: mass-weighted error {merr(Xm, Xf, M):.2e}, tip RMS error/RMS {np.sqrt(np.mean((Xm[:,tip]-Xf[:,tip])**2))/np.sqrt(np.mean(Xf[:,tip]**2)):.2e}", flush=True)
np.savez('c11_timehist.npz', dt=dt, **{k.replace(' ', '_').replace('(', '').replace(')', ''): v[::2] for k, v in out.items()})
