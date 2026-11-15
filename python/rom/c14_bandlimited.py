"""SOD-ROM (c14): band-limited ("soft-tip") equal-energy free decay as training signal. One realization, same seed and
   tests as c10_equal_energy.py, so values compare directly with c10_case<i>.pkl.
   Training IC: x0 = sum_k a_k phi_k / omega_k with a_k = 1 / (1 + (omega_k / omega_c)^4), omega_c = the random-load
   cutoff (23 for S1, 250 for S2/S3), over the LNMs with omega_k < pi/(4 dt); released from rest; signs +1.
   Modes inside the load band are excited at nearly equal energy; modes above it roll off as (omega_c/omega_k)^8 in
   energy, as a softer hammer tip limits the excited band. Bases '(freeHAM)'. Only these bases plus LNM/CM are rerun.
   Usage: python3 c14_bandlimited.py <case 0..5> -> c14_case<i>.pkl"""
import numpy as np, sys, pickle, time
from systems import shear_frame, beam, state
from romlib import band_noise, chirp
from romlib2 import *
from scipy.linalg import eigh
ci = int(sys.argv[1]); NR = 1
CASES = [('S1', 'light'), ('S1', 'heavy'), ('S2', 'light'), ('S2', 'heavy'), ('S3', 'dashpot'), ('S3', 'dashpot_heavy')]
sysname, cname = CASES[ci]
S = shear_frame() if sysname == 'S1' else (beam() if sysname == 'S2' else beam(dashpot=(0.2, 1.0)))
p = dict(dt=0.01, T=300, wc=23.0, w0=1.0) if sysname == 'S1' else dict(dt=1e-3, T=200, wc=250.0, w0=2.0)
M, K, B = S['M'], S['K'], S['B']; C = S['C'][cname]; n = len(M); nin = B.shape[1]; dt = p['dt']; N = int(p['T'] / dt)
A, Bs = state(M, K, C, B); nonprop = sysname == 'S3'
WE = np.block([[K, np.zeros((n, n))], [np.zeros((n, n)), M]])   # energy weighting for state Galerkin
RMAX = 10
e_in = np.r_[1, np.zeros(nin - 1)]                                     # training input: tip (beam) / base (frame)
F_train = (M @ np.ones(n)) if nin == 1 else (B @ np.array([1.0, 0.7]))
F_test = np.eye(n)[:, 3] if nin == 1 else (B @ np.array([0.3, 1.0]))   # frame: point load at floor 4
xs_tr = np.linalg.solve(K, F_train); xs_tr /= abs(xs_tr).max()
xs_te = np.linalg.solve(K, F_test); xs_te /= abs(xs_te).max()
rng = np.random.default_rng(100 + ci)
DEC = 5                                   # errors evaluated on every 5th sample (Nyquist 5x above the load band)

def derivs(u, y0=None):
    Y = simulate_state(A, Bs, u, dt, y0); Yd = Y @ A.T + u @ Bs.T; return Y, Yd   # exact state derivative

def bases_from(Y, Yd, tag):
    X, V = Y[:, :n], Y[:, n:]; out = {}
    Pp = pod(X, RMAX); Ps = sod(X, V, RMAX)[0]                       # computed once, nested by order
    out[f'POD ({tag})'] = ('c', lambda r, Pp=Pp: Pp[:, :r]); out[f'SOD ({tag})'] = ('c', lambda r, Ps=Ps: Ps[:, :r])
    if nonprop:
        Psi, Phi, _ = state_sod(Y, Yd, 2 * RMAX)
        out[f'sSOD ({tag})'] = ('s', lambda k, Psi=Psi, Phi=Phi: (Psi[:, :k], Phi[:, :k]))
    return out

# deterministic training records and tests
fixed = {}
wL2, PL = eigh(K, M); kEE = np.where(np.sqrt(wL2) < np.pi / (4 * dt))[0]
def ee(signs):
    x = PL[:, kEE] @ (signs / np.sqrt(wL2[kEE])); return x / abs(x).max()
xee_tr = ee(np.ones(len(kEE))); xee_te = ee((-1.0) ** np.arange(len(kEE)))
aH = 1 / (1 + (np.sqrt(wL2[kEE]) / p['wc']) ** 4); xh = PL[:, kEE] @ (aH / np.sqrt(wL2[kEE])); xh /= abs(xh).max()
print('modal energy weights a_k^2:', np.round(aH ** 2, 4), flush=True)
Ye, Yde = derivs(np.zeros((N, nin)), np.r_[xh, np.zeros(n)]); fixed.update(bases_from(Ye, Yde, 'freeHAM')); del Ye, Yde
fixed['LNM'] = ('c', lambda r: lnm(M, K, r))
if nonprop:
    PsiC, PhiC = complex_modes(A, 2 * RMAX); fixed['CM'] = ('s', lambda k: (PsiC[:, :k], PhiC[:, :k]))
y0_te = np.r_[xs_te, np.zeros(n)]; X_free = simulate_state(A, Bs, np.zeros((N, nin)), dt, y0_te)[::DEC, :n]
y0_ee = np.r_[xee_te, np.zeros(n)]; X_ee = simulate_state(A, Bs, np.zeros((N, nin)), dt, y0_ee)[::DEC, :n]
res = {}; ranks = {}; unstable = {}
t0 = time.time()
for rr in range(NR):
    u_tr = band_noise(N, dt, p['wc'], nin, rng) * e_in
    rb = {}                                    # random-trained bases not rerun; u_tr drawn to keep the RNG sequence
    tests = {'rand': (band_noise(N, dt, p['wc'], nin, rng) * e_in, None)}
    if nin > 1: tests['mid'] = (band_noise(N, dt, p['wc'], nin, rng) * np.r_[0, 1], None)
    Xt = {k: simulate_state(A, Bs, u, dt)[::DEC, :n] for k, (u, y0) in tests.items()}
    tests['free'] = (np.zeros((N, nin)), y0_te); Xt['free'] = X_free
    tests['freeEE'] = (np.zeros((N, nin)), y0_ee); Xt['freeEE'] = X_ee
    bases = dict(fixed); bases.update(rb)
    for bn, (kind, fb) in bases.items():
        for r in range(1, RMAX + 1):
            if kind == 'c':
                Phi = fb(r)
                if Phi is None: continue
                order = Phi.shape[1]
                if order < r: continue                                     # beyond the data rank: not plotted at a false order
                for tk, (u, y0) in tests.items():
                    e = merr(rom_config(M, K, C, B, Phi, u, dt, y0, dec=DEC), Xt[tk], M)
                    res.setdefault((bn, r, tk), []).append(e)
            else:
                Psi, Phi = fb(2 * r)
                if Phi.shape[1] < 2 * r: continue
                for tk, (u, y0) in tests.items():
                    Xr, mx = rom_state(A, Bs, Psi, Phi, u, dt, y0, n, W=None if bn == 'CM' else WE, dec=DEC)
                    res.setdefault((bn, r, tk), []).append(merr(Xr, Xt[tk], M)); unstable.setdefault((bn, r), []).append(mx)
        if rr == 0:
            ranks[bn] = max([r for (b, r, t) in res if b == bn], default=0)
    print(f"{sysname} {cname}: realization {rr+1}/{NR} done ({time.time()-t0:.0f}s)", flush=True)
pickle.dump(dict(case=CASES[ci], res=res, ranks=ranks, unstable=unstable, NR=NR, tests=list(tests)), open(f'c14_case{ci}.pkl', 'wb'))
# summary: median error per basis and order for each test
for tk in tests:
    print(f"== {sysname} {cname}, test {tk}: median error, r = 1..{RMAX}")
    for bn in bases:
        row = [np.median(res[(bn, r, tk)]) if (bn, r, tk) in res else np.nan for r in range(1, RMAX + 1)]
        print(f"   {bn:16s} " + " ".join(f"{x:8.1e}" for x in row))
if unstable: print("max Re eig of state ROMs (max over realizations):", {k: f"{np.max(v):.1e}" for k, v in unstable.items() if np.max(v) > 0})
