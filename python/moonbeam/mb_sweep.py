"""Stroboscopic bifurcation diagrams of the two-well beam (mb_model.py) and its Galerkin ROMs under harmonic base
   acceleration A cos(Omega t), Omega swept up and down over [0.2, 1.4] omega_w (121 values), the state carried from one
   frequency to the next. Per frequency: 100 forcing periods of transient, then 100 stroboscopic samples of the tip
   displacement (phase cos = 1) and the tip extremes and RMS over those 100 periods; 100 steps per period (average-acceleration Newmark).
   Usage (module): sweep(model, A, direction) with model = dict(M, K, C, pt, Mr, x0, v0, Phi or None)."""
import numpy as np
from mb_model import beam, newmark
OMS = np.linspace(0.2, 1.4, 121)
def sweep(S, mdl, A, direction=+1, ntr=100, nrec=100, spp=100):
    oms = OMS if direction > 0 else OMS[::-1]; x, v = mdl['x0'].copy(), mdl['v0'].copy(); out = []; ext = []
    for f in oms:
        Om = f * S['ww'][0]; T = 2 * np.pi / Om; dt = T / spp; idx = range(spp * ntr + 1, spp * (ntr + nrec) + 1)
        X, V, x, v = newmark(mdl['M'], mdl['K'], mdl['C'], mdl['pt'], S, x, v, T * (ntr + nrec), dt, A, Om, mdl['Mr'], tsave=idx)
        y = X @ mdl['pt']; out.append(y[spp - 1::spp]); ext.append((y.min(), y.max(), np.sqrt(np.mean(y**2))))
    return oms, np.array(out), np.array(ext)
def full_model(S):
    return dict(M=S['M'], K=S['K'], C=S['C'], pt=S['et'], Mr=S['M'] @ S['r'], x0=S['xs'].copy(), v0=np.zeros(S['n']))
def rom_model(S, Phi):
    M, K, C = S['M'], S['K'], S['C']; Mr_ = Phi.T @ M @ Phi
    q0 = np.linalg.solve(Mr_, Phi.T @ M @ S['xs'])
    return dict(M=Mr_, K=Phi.T @ K @ Phi, C=Phi.T @ C @ Phi, pt=Phi[S['tip']].copy(), Mr=Phi.T @ M @ S['r'], x0=q0, v0=np.zeros(Phi.shape[1]))
