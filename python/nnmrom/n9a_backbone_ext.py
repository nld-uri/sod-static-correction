"""Extension of the full-model NNM1 backbone (n3_backbone_hb.py, Kh = 8) above H = 0.03 J, by pseudo-arclength
   continuation from the last high-branch point of n3_backbone_full_Kh8.pkl. The main backbone runs into a 9:1 internal
   resonance with mode 3 (415 Hz) at 46.7 Hz, H = 0.059 J, where the continuation follows the tongue; natural-parameter
   continuation in the tip amplitude did not pass it (it jumped to a branch without fundamental). The reference
   backbone therefore ends at the onset of the tongue: points kept while the largest harmonic k >= 5 stays below 2%
   of the fundamental. Output: n9a_backbone_ext.pkl (H, f, hr, Z)."""
import numpy as np, pickle
from scipy.linalg import eigh
from costbeam import costbeam
from n3_backbone_hb import HBC, branch_from
S = costbeam(); n = S['n']; pt = np.zeros(n); pt[S['tip']] = 1.0; w1 = np.sqrt(eigh(S['K'], S['M'], eigvals_only=True)[0])
hb = HBC(S['M'], S['K'], pt, S['k3'], Kh=8)
F = pickle.load(open('n3_backbone_full_Kh8.pkl', 'rb')); z0 = F['Z'][np.argmax(F['H'])]; a0 = hb.tip_amp(z0)
pts = branch_from(hb, z0, a0, 1e-7, 0.07, 0.02, 1e-2, w1, dsmax=0.01, maxsteps=40000)
H = np.array([hb.energy(z) for z in pts]); f = np.array([z[-1] for z in pts]) / 2 / np.pi; hr = np.array([hb.harm(z)[1:] for z in pts])
m = np.cumprod(hr[:, 1:].max(1) < 0.02).astype(bool)
print(f"kept {m.sum()} of {len(pts)} points: H {H[m].min():.3g}..{H[m].max():.3g} J, f {f[m].min():.2f}..{f[m].max():.2f} Hz")
pickle.dump(dict(H=H[m], f=f[m], hr=hr[m], Z=np.array(pts)[m]), open('n9a_backbone_ext.pkl', 'wb'))
