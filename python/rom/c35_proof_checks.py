"""Numerical checks of the proofs in the appendix of SOD-ROM.
   (1) Damped SDOF decay: I_v = omega a^2/(4 zeta), omega^2 I_x / I_v = 1 + 4 zeta^2 cos^2 phi - 2 zeta sin 2 phi.
   (2) First-order mode rotation coefficients c_ij of the pencil G_v chi = mu G_x chi.
   (3) Proposition 1: pair product |s|^4, and equal values only for G_i proportional to I."""
import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh, eig
w, z, a = 2.3, 0.07, 1.3
for phi in (0.0, 0.7, 2.1):
    wd = w * np.sqrt(1 - z * z); x0, v0 = a * np.cos(phi), -w * a * np.sin(phi)
    x = lambda t: np.exp(-z * w * t) * (x0 * np.cos(wd * t) + (v0 + z * w * x0) / wd * np.sin(wd * t))
    v = lambda t: (x(t + 1e-6) - x(t - 1e-6)) / 2e-6
    Ix = quad(lambda t: x(t)**2, 0, 400, limit=2000)[0]; Iv = quad(lambda t: v(t)**2, 0, 400, limit=2000)[0]
    print(f"(1) phi={phi}: Iv/(w a^2/4z) - 1 = {Iv / (w * a * a / 4 / z) - 1:+.1e};  ratio formula error = "
          f"{w * w * Ix / Iv - (1 + 4 * z * z * np.cos(phi)**2 - 2 * z * np.sin(2 * phi)):+.1e}")
rng = np.random.default_rng(35); n = 4; eps = 1e-4
gx = np.diag(rng.uniform(1, 2, n)); gv = np.diag(rng.uniform(1, 2, n) * np.array([1, 4, 9, 16.]))
E = rng.standard_normal((n, n)); E = E + E.T; np.fill_diagonal(E, 0); F = rng.standard_normal((n, n)); F = F + F.T; np.fill_diagonal(F, 0)
Gx = gx + eps * E; Gv = gv + eps * F                                     # off-diagonal (correlation) perturbation only
mu, X = eigh(Gv, Gx); mu0 = np.diag(gv) / np.diag(gx); o = np.argsort(mu0)
for i in o[:2]:
    k = np.argmin(abs(mu - mu0[i])); chi = X[:, k] / X[i, k]
    c = np.array([(Gv[j, i] - mu0[i] * Gx[j, i]) / ((mu0[i] - mu0[j]) * Gx[j, j]) if j != i else 0 for j in range(n)])
    print(f"(2) mode {i}: max |chi_j - c_ij| / eps = {np.max(np.abs(np.delete(chi - c, i))) / eps:.1e} (O(eps) expected); "
          f"|mu - mu0| / eps^2 = {abs(mu[k] - mu0[i]) / eps**2:.1e} (O(1): second order)")
s = -0.3 + 2.0j; L = np.array([[s.real, s.imag], [-s.imag, s.real]])
for G in (np.array([[2.0, 0.3], [0.3, 0.7]]), 1.7 * np.eye(2)):
    m = eig(L @ G @ L.T, G, right=False).real; print(f"(3) G = {G.tolist()}: mu = {np.round(m, 4)}, product/|s|^4 - 1 = {m.prod() / abs(s)**4 - 1:+.1e}")
