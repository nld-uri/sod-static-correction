"""SOD-ROM, story check (union-paper logic): is the high-order gain of data bases the static-correction direction?
   Same case, seeds and draw order as c18_ensemble.py case 2 (S2, light Rayleigh; default_rng(102): per realization
   the training load, the tip test load, the mid-span test load). For each realization and order r:
     rho_tip(r), rho_mid(r): M-norm residual of the static vectors K^-1 b_tip, K^-1 b_mid outside span(basis_r),
       relative to the static vector itself;
     for SOD (rand), POD (rand), LNM, and the order-matched band-limited decay basis SOD (freeBL).
   At r = 8: principal angles (M metric) of SOD (rand) to MTA (tip + mid), MTA-tip and LNM.
   Usage: python3 g1_static_containment.py [NR]  -> g1_out.txt (stdout), g1_results.pkl"""
import numpy as np, sys, pickle, time
from systems import beam, state
from romlib import band_noise
from romlib2 import simulate_state, lnm, mta, pod, sod
from sodtoolbox.rom import principal_angles
from scipy.linalg import eigh, cho_factor, cho_solve

NR = int(sys.argv[1]) if len(sys.argv) > 1 else 20
S = beam(); M, K, B = S['M'], S['K'], S['B']; C = S['C']['light']; n = len(M)
dt = 1e-3; N = 200000; wc = 250.0; RMAX = 10
A, Bs = state(M, K, C, B)
s_tip = np.linalg.solve(K, B[:, 0]); s_mid = np.linalg.solve(K, B[:, 1])

def mres(Phi, s):
    """relative M-norm residual of s outside span(Phi)"""
    G = Phi.T @ M @ Phi; c = np.linalg.solve(G, Phi.T @ M @ s); d = s - Phi @ c
    return np.sqrt(d @ M @ d / (s @ M @ s))

# order-matched band-limited decays (deterministic; as c18 'freeBL')
wL2, PL = eigh(K, M); kEE = np.where(np.sqrt(wL2) < np.pi / (4 * dt))[0]; wk = np.sqrt(wL2[kEE])
def bl(a):
    x = PL[:, kEE] @ (a / wk); return x / abs(x).max()
BLb = {}
for r in range(1, RMAX + 1):
    Y = simulate_state(A, Bs, np.zeros((N, 2)), dt, np.r_[bl(1 / (1 + (wk / np.sqrt(wL2[r - 1])) ** 4)), np.zeros(n)])
    BLb[r] = sod(Y[:, :n], Y[:, n:], r)[0]
lnmres = {r: (mres(lnm(M, K, r), s_tip), mres(lnm(M, K, r), s_mid)) for r in range(1, RMAX + 1)}
blres = {r: (mres(BLb[r], s_tip), mres(BLb[r], s_mid)) for r in range(1, RMAX + 1) if BLb[r].shape[1] == r}

rng = np.random.default_rng(102); out = dict(sod={}, pod={}, ang={}, rank=[])
t0 = time.time()
for rr in range(NR):
    u_tr = band_noise(N, dt, wc, 2, rng) * np.r_[1, 0]
    band_noise(N, dt, wc, 2, rng); band_noise(N, dt, wc, 2, rng)          # tip and mid test loads (keep the stream)
    Y = simulate_state(A, Bs, u_tr, dt); X, V = Y[:, :n], Y[:, n:]
    Ps = sod(X, V, RMAX)[0]; Pp = pod(X, RMAX); del Y, X, V
    out['rank'].append(Ps.shape[1])
    for r in range(1, Ps.shape[1] + 1):
        out['sod'].setdefault(r, []).append((mres(Ps[:, :r], s_tip), mres(Ps[:, :r], s_mid)))
        out['pod'].setdefault(r, []).append((mres(Pp[:, :r], s_tip), mres(Pp[:, :r], s_mid)))
    if Ps.shape[1] >= 8:
        out['ang'].setdefault('MTA', []).append(principal_angles(Ps[:, :8], mta(M, K, B, 8), M))
        out['ang'].setdefault('MTA-tip', []).append(principal_angles(Ps[:, :8], mta(M, K, B[:, :1], 8), M))
        out['ang'].setdefault('LNM', []).append(principal_angles(Ps[:, :8], lnm(M, K, 8), M))
    print(f"realization {rr + 1}/{NR}, data rank {Ps.shape[1]} ({time.time() - t0:.0f}s)", flush=True)

pickle.dump(dict(out=out, lnm=lnmres, bl=blres, NR=NR), open('g1_results.pkl', 'wb'))
print(f"\n== S2 light: residual of K^-1 b outside span(basis_r), M norm, relative; median over {NR} realizations")
print(f"{'r':>3} | {'SOD tip':>9} {'SOD mid':>9} | {'POD tip':>9} {'POD mid':>9} | {'LNM tip':>9} {'LNM mid':>9} | {'BL tip':>9} {'BL mid':>9} | n")
for r in range(1, RMAX + 1):
    s = np.array(out['sod'].get(r, [(np.nan, np.nan)])); p = np.array(out['pod'].get(r, [(np.nan, np.nan)]))
    b = blres.get(r, (np.nan, np.nan))
    print(f"{r:3d} | {np.median(s[:, 0]):9.2e} {np.median(s[:, 1]):9.2e} | {np.median(p[:, 0]):9.2e} {np.median(p[:, 1]):9.2e} | "
          f"{lnmres[r][0]:9.2e} {lnmres[r][1]:9.2e} | {b[0]:9.2e} {b[1]:9.2e} | {len(out['sod'].get(r, []))}")
print("\n== principal angles (deg, M metric) between SOD (rand, r=8) and r=8 model bases; median over realizations, sorted")
for k, v in out['ang'].items():
    v = np.sort(np.array(v), axis=1); print(f"   {k:8s} " + " ".join(f"{x:7.3f}" for x in np.median(v, axis=0)) + f"   (max of largest {v[:, -1].max():.3f})")
print("data ranks:", out['rank'])
