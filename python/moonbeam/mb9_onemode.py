"""One-mode ROMs of the two-well beam: with phi scaled to unit tip displacement, the ROM is
   m q'' + c q' + (k - k1) q + k3 q^3 = -Gamma m a_g, so its buckled equilibrium is q* = sqrt((k1 - k)/k3) and its
   well frequency is sqrt(2 (k1 - k) / m). Prints both against the full model (x* and omega_w) for each basis,
   and the angle (mass metric) between the first SOD direction and the static vector K^-1 e_tip."""
import pickle, numpy as np
from mb_model import beam
S = beam(); B = pickle.load(open('mb2_bases.pkl', 'rb'))['B']; M, K = S['M'], S['K']; k1, k3 = S['k1'], S['k3']
xs, ww = S['xs'][S['tip']], S['ww'][0]
print(f"full model: x* = {xs:.5f}, omega_w = {ww:.4f}")
for n in ['SOD', 'POD', 'MTA', 'LNMw', 'LNM0']:
    p = B[(n, 1)][:, 0]; p = p / p[S['tip']]; m, k = p @ M @ p, p @ K @ p; q = np.sqrt((k1 - k) / k3); w = np.sqrt(2 * (k1 - k) / m)
    print(f"{n:5s} q*/x* - 1 = {q / xs - 1:+.2e}   omega_w(ROM)/omega_w - 1 = {w / ww - 1:+.2e}   Gamma = {p @ M @ S['r'] / m:.4f}")
p, q = B[('SOD', 1)][:, 0], np.linalg.solve(K, S['et'])
print(f"angle(SOD 1, K^-1 e_tip) = {np.degrees(np.arccos(abs(p @ M @ q) / np.sqrt((p @ M @ p) * (q @ M @ q)))):.2f} deg")
