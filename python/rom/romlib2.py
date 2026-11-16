"""SOD-ROM, revision 2: fast exact simulation, model-based baselines (LNM, LNM + static correction (MTA),
   complex-mode truncation) and data-driven bases (POD, configuration SOD, state-space SOD), Galerkin and
   Petrov-Galerkin ROMs, mass-weighted error.
   State-space SOD: y = (x, xdot); symmetric pencil <ydot ydot^T> psi = mu <y y^T> psi; smooth covectors Psi,
   smooth modes Phi = Sigma_y Psi (so Psi^T Phi = I); ROM  q = Psi^T y,  qdot = Psi^T A Phi q + Psi^T Bs u.
   Since the SOD Toolbox release the code lives in sodtoolbox (core, rom); this module keeps the paper names."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'SOD-toolbox', 'python'))
import numpy as np
from scipy.linalg import eigh, expm, eig, solve, lu_factor, lu_solve   # re-exported for scripts using import *
from scipy.signal import lfilter
from systems import state
import sodtoolbox as _st
from sodtoolbox.rom import zoh, simulate_state, simulate, lnm, mta, complex_modes, rom_config, rom_state, merr
from sodtoolbox.core import data_rank_basis

def pod(X, r, tol=1e-10): return _st.pod(X, r, tol)[0]

def sod(X, V, r, tol=1e-10):
    """configuration SOD inside the POD data subspace; returns smooth modes (actual order <= r) and frequencies"""
    return _st.sod_modes(X, V, r, tol)

def state_sod(Y, Yd, k, tol=1e-10):
    """state-space SOD; returns covectors Psi, modes Phi = Sigma_y Psi (Psi^T Phi = I), frequencies sqrt(mu)"""
    R = _st.state_sod(Y, Yd, k=k, tol=tol); return R.Psi, R.Phi, np.sqrt(R.mu)

def _check_sim():
    rng = np.random.default_rng(0); A = rng.standard_normal((4, 4)) - 3 * np.eye(4); Bs = rng.standard_normal((4, 1))
    u = rng.standard_normal((50, 1)); y0 = rng.standard_normal(4); dt = 0.05
    Ad, Bd = zoh(A, Bs, dt); Y = np.zeros((50, 4)); y = y0.copy()
    for k in range(50): Y[k] = y; y = Ad @ y + Bd @ u[k]
    return np.abs(simulate_state(A, Bs, u, dt, y0) - Y).max()

if __name__ == "__main__":
    print("simulation check (max abs diff vs explicit ZOH loop):", _check_sim())
