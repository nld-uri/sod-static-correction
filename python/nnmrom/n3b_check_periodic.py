"""Check of the harmonic-balance backbone (n3_backbone_hb.py, Kh=8): integrate the full COST beam over one period
   from x0 = sum_k X_k, v0 = 0 (Radau, rtol 1e-10) and report the periodicity defect and the HB frequency."""
import numpy as np, pickle
from scipy.integrate import solve_ivp
from costbeam import costbeam
from nlsim import make_rhs
S = costbeam(); n = S['n']; pt = np.zeros(n); pt[S['tip']] = 1
F = pickle.load(open('n3_backbone_full_Kh8.pkl', 'rb')); Kh = F['Kh']
f, jac = make_rhs(S['M'], S['K'], S['C'], pt, S['k3'])
for lab, Z, H, h5 in [('low', F['low']['Z'], F['low']['H'], F['low']['h5']), ('high', F['Z'], F['H'], F['h5'])]:
    for Ht in [1e-3, 3e-3, 1e-2, 2e-2]:
        ok = h5 < 0.02
        if not ok.any(): continue
        i = np.where(ok)[0][np.argmin(abs(np.log(H[ok] / Ht)))]
        if abs(np.log(H[i] / Ht)) > 0.5: continue
        z = Z[i]; X = z[:-1].reshape(Kh, n); x0 = X.sum(0); T = 2 * np.pi / z[-1]
        s = solve_ivp(f, [0, T], np.r_[x0, np.zeros(n)], method='Radau', jac=jac, rtol=1e-10, atol=1e-14)
        d = np.linalg.norm(s.y[:n, -1] - x0) / np.linalg.norm(x0)
        print(f"{lab} branch H={H[i]:.2e} J f={z[-1]/2/np.pi:.4f} Hz: periodicity defect |x(T)-x0|/|x0| = {d:.1e}", flush=True)
