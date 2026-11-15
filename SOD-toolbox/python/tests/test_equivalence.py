"""Equivalence of the toolbox with the paper libraries it replaces (run from this folder: python3 test_equivalence.py).
Each check compares the toolbox call with the original function on the same record. Vectors are compared up to sign
(and up to scale where the conventions differ); the worst relative deviation is printed against a tolerance."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, '..', '..', '..', 'python')
sys.path[:0] = [os.path.join(HERE, '..'), os.path.join(PAPER, 'rom'), os.path.join(PAPER, 'resid'), PAPER]
import sodtoolbox as st
import romlib2, sellib, rlib
from systems import beam, state

rows = []
def check(name, dev, tol=1e-8):
    rows.append((name, dev, tol, dev <= tol)); print(f"{'PASS' if dev <= tol else 'FAIL'}  {name:46s} {dev:.1e}  (tol {tol:.0e})")

def colsdev(A, B, scale=False):
    """max over columns of the deviation after fixing sign (and scale, if requested)"""
    d = 0.0
    for a, b in zip(A.T, B.T):
        c = (a @ b) / (b @ b) if scale else np.sign(a @ b)
        d = max(d, np.linalg.norm(a - c * b) / np.linalg.norm(a))
    return d

def reldev(a, b): return np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.maximum(np.abs(np.asarray(b)), 1e-300))

# ---------- record: S2 (cantilever, 30 elements) light Rayleigh, band-limited random tip force ----------
S = beam(); M, K, C, B = S['M'], S['K'], S['C']['light'], S['B'][:, :1]; n = len(M)
rng = np.random.default_rng(1); dt = 1e-3; N = 20000
from scipy.signal import butter, sosfiltfilt
u = sosfiltfilt(butter(6, 200.0 / (np.pi / dt), output='sos'), rng.standard_normal((N, 1)), axis=0)
A, Bs = state(M, K, C, B)
Y = romlib2.simulate_state(A, Bs, u, dt); X, V = Y[:, :n], Y[:, n:]; Yd = Y @ A.T + u @ Bs.T

# ---------- simulation and model bases ----------
check('simulate_state', np.abs(st.simulate_state(A, Bs, u, dt) - Y).max() / np.abs(Y).max(), 1e-13)
check('state_matrices', max(np.abs(st.state_matrices(M, K, C, B)[0] - A).max(), np.abs(st.state_matrices(M, K, C, B)[1] - Bs).max()), 0)
check('mta (r = 6)', np.abs(st.mta(M, K, B, 6) - romlib2.mta(M, K, B, 6)).max(), 0)
P1, F1 = st.complex_modes(A, 8); P0, F0 = romlib2.complex_modes(A, 8)
check('complex_modes', max(np.abs(P1 - P0).max(), np.abs(F1 - F0).max()), 0)

# ---------- configuration SOD ----------
Phi0, w0 = romlib2.sod(X, V, 8); Phi1, w1 = st.sod_modes(X, V, 8)
check('sod_modes vs romlib2.sod: modes', colsdev(Phi1, Phi0)); check('sod_modes vs romlib2.sod: freq', reldev(w1, w0))
Phi0, w0 = rlib.sod(X, V, 8); check('sod_modes vs rlib.sod: modes', colsdev(Phi1, Phi0))
U0 = romlib2.pod(X, 8); U1, _ = st.pod(X, 8); check('pod', colsdev(U1, U0))

# sellib.gsod: unit-RMS scaling, rank restriction, Psi and Phi in the original units
Psi0, Phi0, mu0 = sellib.gsod(X, V, 1e-10); R = st.sod(X, V, scale=True)
# gsod forms the full covariances and then projects; the toolbox projects the data first. On this record the data
# rank is 9 of 60 (cond(X^T X) ~ 1e20), so the two round-off paths differ at ~1e-7 near the rank cut.
check('sod(scale=True) vs gsod: mu', reldev(R.mu, mu0), 1e-6); check('sod(scale=True) vs gsod: Psi', colsdev(R.Psi, Psi0), 1e-5)
check('sod(scale=True) vs gsod: Phi', colsdev(R.Phi, Phi0), 1e-5)

# ---------- state SOD ----------
Psi0, Phi0, w0 = romlib2.state_sod(Y, Yd, 20); R = st.state_sod(Y, Yd, k=20)
check('state_sod vs romlib2: Psi', colsdev(R.Psi, Psi0)); check('state_sod vs romlib2: Phi', colsdev(R.Phi, Phi0))
check('state_sod vs romlib2: freq', reldev(R.freq, w0))

# ---------- recognition and damping (identical code paths on identical input) ----------
Rg = st.sod(Y, Yd, scale=True); Psi0, Phi0, mu0 = sellib.gsod(Y, Yd, 1e-10)
Q1, Q0 = Y @ Rg.Psi, Y @ Psi0
sgn = np.sign(np.sum(Q1 * Q0, 0)); Q1 = Q1 * sgn
p1 = st.pair_state(Q1, dt, 300.0, Phi=Rg.Phi * sgn); p0 = sellib.pair_state(Q0, dt, 300.0, Phi=Phi0)
check('pair_state: same planes', 0.0 if [x[:2] for x in p1] == [x[:2] for x in p0] else 1.0, 0)
c1 = st.select_config(Q1[:, :20], Rg.mu[:20], dt, 300.0); c0 = sellib.select_config(Q0[:, :20], mu0[:20], dt, 300.0)
check('select_config: same selection', 0.0 if list(c1[0]) == list(c0[0]) else 1.0, 0)

# free decay for the damping fit (reference: c29 sod_damping logic, inlined)
y0 = np.r_[np.linalg.solve(K, B[:, 0]), np.zeros(n)]
Yf = romlib2.simulate_state(A, Bs, np.zeros((10 * N, 1)), dt, y0); Yfd = Yf @ A.T   # 200 s: the record has decayed
Rf = st.sod(Yf, Yfd, scale=True); D = st.decay_damping(Rf, Yfd, dt, 300.0)
Psi0, Phi0, _ = sellib.gsod(Yf, Yfd, 1e-10); Q, Qd = Yf @ Psi0, Yfd @ Psi0; ref = []
for a, b, f, c in sellib.pair_state(Q, dt, 300.0, Phi=Phi0):
    q, qd = Q[:, [a, b]], Qd[:, [a, b]]; L = (qd.T @ q) @ np.linalg.inv(q.T @ q)
    sig = np.trace(L) / 2; wn = np.sqrt(max(np.linalg.det(L), 0)); ref.append(complex(sig, np.sqrt(max(wn ** 2 - sig ** 2, 0))))
check('decay_damping: plane eigenvalues', reldev([p[0] for p in D['plane']], ref) if len(ref) == len(D['plane']) else 1.0, 1e-6)
# the identity is exact for the chord derivative; with exact qdot and sampled sums its error scales as (|s| dt)^2 / zeta
# (oscillating integrand of size |s|, mean zeta |s|), so it is checked on the planes with (|s| dt)^2 / zeta < 0.1
ident = np.array(D['identity']); ok = np.array([(abs(p[0]) * dt) ** 2 / p[1] < 0.1 for p in D['plane']])
check(f'damping identity, {ok.sum()} well-sampled planes', reldev(ident[ok, 1], ident[ok, 0]), 1e-2)

# ---------- residual bases ----------
Aacc = Yd[:, n:]
for kind in ('a', 'k', 'x'):
    B0 = rlib.basis(X, V, Aacc, M, K, 2, 4, kind); B1 = st.residual_basis(X, V, Aacc, M, K, 2, 4, kind)
    check(f'residual_basis kind {kind}', colsdev(B1, B0))

# ---------- variants: trend SOD (centred, gradient or step-h central, lagged) and windowed SOD ----------
import importlib.util
sp = importlib.util.spec_from_file_location('t12', os.path.join(PAPER, 'trend', 't12_trend.py')); t12 = importlib.util.module_from_spec(sp)
sys.argv = ['t12']; sp.loader.exec_module(t12)
Xt = t12.S.T @ t12.A.T + 1e-3 * np.random.default_rng(5).standard_normal((t12.N, 6))
for lag, h in ((0, 1), (0, 3), (7, 3)):
    lam0, P0, _ = t12.sod(Xt, lag, h)
    R = st.sod(Xt, dt=t12.dt, deriv='gradient' if h == 1 else 'central', h=h, lag=lag, center=True, tol=None)
    good = np.isfinite(lam0) & (lam0 > 0)
    check(f'trend sod lag={lag} h={h}: SOV', reldev(R.sov[good], lam0[good]), 1e-7)
    check(f'trend sod lag={lag} h={h}: Psi (to scale)', colsdev(R.Psi[:, good], P0[:, good], scale=True), 1e-7)

Xw = X[:3001]; w, psi, m = st.sod_window(Xw[:, ::6], dt, boundary=True)
xc = Xw[1:-1, ::6]; vd = (Xw[2:, ::6] - Xw[:-2, ::6]) / (2 * dt); Nw = len(xc); T = (Nw - 1) * dt
from scipy.linalg import eigh
Bd = (np.outer(xc[-1], vd[-1]) - np.outer(xc[0], vd[0])) / T; mus, Ps = eigh(vd.T @ vd / Nw - (Bd + Bd.T) / 2, xc.T @ xc / Nw)
check('sod_window (boundary) vs t14 windowed', max(abs(w - np.sqrt(mus[0])) / w, colsdev(psi[:, None], Ps[:, :1])))

nf = sum(not r[3] for r in rows); print(f"\n{len(rows) - nf}/{len(rows)} checks passed")
sys.exit(1 if nf else 0)
