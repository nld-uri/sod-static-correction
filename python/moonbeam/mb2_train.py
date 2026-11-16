"""Training record and bases for the two-well beam ROMs.
   Training: cross-well free decay of the full model, started at the buckled equilibrium +x* with the velocity of a base
   velocity step, v0 = -V0 r (V0 from argv, default 1.0). Exact kinematics (Newmark states), dt = 0.005, T = 60.
   Bases at r = 1..4: clamped LNMs (LNM0), LNMs linearized in a well (LNMw), MTA (r-1 clamped LNMs + K^-1 e_tip),
   POD and SOD of the training record (second moments about zero, so the buckled offsets stay in the span).
   Output: mb2_bases.pkl (bases, smooth frequencies, record summary)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'SOD-toolbox', 'python'))
import sodtoolbox as _st
import numpy as np, pickle, sys
from scipy.linalg import eigh
from mb_model import beam, newmark
V0 = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
S = beam(); M, K, C, n, tip = S['M'], S['K'], S['C'], S['n'], S['tip']
dt, T = 0.005, 60.0; N = int(round(T / dt))
X, V, _, _ = newmark(M, K, C, S['et'], S, S['xs'].copy(), -V0 * S['r'], T, dt, 0.0, 1.0, M @ S['r'], tsave=range(1, N + 1))
y = X[:, tip]; cross = np.sum(np.diff(np.sign(y)) != 0)
print(f"V0 = {V0}: tip range [{y.min():+.3f}, {y.max():+.3f}], {cross} zero crossings, last at t = "
      f"{dt * (np.nonzero(np.diff(np.sign(y)))[0].max() + 1) if cross else 0:.1f}")
def mnorm(Q): return Q / np.sqrt(np.einsum('ij,ij->j', Q, M @ Q))
def pod(X, r): w, U = eigh(X.T @ X); return U[:, ::-1][:, :r], w[::-1]
def sod(X, V, r):
    """smooth modes (unit norm, data subspace; Sigma_x is rank deficient) and all SOD frequencies"""
    R = _st.sod(X, V); return R.modes_unit()[:, :r], np.sqrt(R.mu)
_, P0 = eigh(K, M); _, Pw = eigh(S['Kw'], M); stat = np.linalg.solve(K, S['et'])
Up, sv = pod(X, n); Ps, ws = sod(X, V, n); print('data rank', len(ws))
print("POD energy fractions", np.round(sv[:5] / sv.sum(), 5))
print("SOD frequencies", np.round(ws[:5], 3), " (well LNM", np.round(S['ww'][:4], 3), ")")

B = {}
for r in range(1, 5):
    B[('LNM0', r)] = P0[:, :r]; B[('LNMw', r)] = Pw[:, :r]
    B[('MTA', r)] = mnorm(np.c_[P0[:, :r - 1], stat]) if r > 1 else mnorm(stat[:, None])
    B[('POD', r)] = Up[:, :r]; B[('SOD', r)] = Ps[:, :r]
def mac(a, b): return (a @ M @ b)**2 / ((a @ M @ a) * (b @ M @ b))
for k in range(3):
    print(f"direction {k + 1}: MAC(SOD, LNMw) = {mac(Ps[:, k], Pw[:, k]):.4f}, MAC(POD, LNMw) = {mac(Up[:, k], Pw[:, k]):.4f}, "
          f"MAC(SOD1, static) = {mac(Ps[:, 0], stat):.4f}" if k == 0 else
          f"direction {k + 1}: MAC(SOD, LNMw) = {mac(Ps[:, k], Pw[:, k]):.4f}, MAC(POD, LNMw) = {mac(Up[:, k], Pw[:, k]):.4f}")
pickle.dump(dict(V0=V0, B=B, ws=ws, sv=sv, ytip=y[::10], dt=dt * 10), open('mb2_bases.pkl', 'wb'))
