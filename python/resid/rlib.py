"""Linear-core-plus-residual bases (SOD + R). Given a training record X, V, A (rows = time) and the order r1 of the
   linear core, fit qdd = q Ka + qd Ca on the r1 smoothest SOD coordinates (least squares), form the equation error
   e = A - (q Ka + qd Ca) Phi1^T in physical coordinates, and take its leading directions.
     R_a : POD of e itself (acceleration-level residual shapes; data only)
     R_k : POD of K^-1 M e (quasi-static displacement driven by the residual force; needs K, as any Galerkin ROM does)
     R_f : POD of e low-passed at fc (data only; removes the acceleration of fast modes, keeps low harmonics)
     R_x : POD of the displacement complement X - q Phi1^T (data only; the literal span Y2 = span Y1 + span R)
   The basis is [Phi1, R] M-orthonormalized. By the normal equations <e q^T> = <e qd^T> = 0.
   Since the SOD Toolbox release the code lives in sodtoolbox.residual; this module keeps the paper names."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'SOD-toolbox', 'python'))
import numpy as np
import sodtoolbox as _st
from sodtoolbox.rom import mortho
from sodtoolbox.residual import _lowpass as lowpass

def sod(X, V, r, tol=1e-10): return _st.sod_modes(X, V, r, tol)
def pod(Z, r):
    U, w = _st.pod(Z, r, tol=None); return U, w
def residual(X, V, A, r1): return _st.linear_core(X, V, A, r1)
def resid_dirs(X, V, A, M, K, r1, kind, dt=None, fc=None): return _st.residual_directions(X, V, A, M, K, r1, kind, dt, fc)
def basis(X, V, A, M, K, r1, r, kind, dt=None, fc=None): return _st.residual_basis(X, V, A, M, K, r1, r, kind, dt, fc)
def diagnostics(X, V, A, M, K, r1, refs, dt=None, fc=None): return _st.residual_diagnostics(X, V, A, M, K, r1, refs, dt, fc)
