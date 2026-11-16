"""Selection, pairing and ordering of SOD coordinates (SOD-ROM, trend note). Rules agreed with D. Chelidze:
   - modes and modal planes are identified, paired and ordered by their associated frequency or circularity, never
     by smoothness (the SOD value);
   - for model reduction the kinematics come from the model; estimated kinematics are for identification from
     measured records only (see c22_id_measured.py);
   - selection for identification (ID) and for the ROM basis are separate and reported separately.
   Frequencies are read from the DCT-II periodogram (no wrap-around for non-periodic records). The spectral tolerance
   is damping-aware: the larger of 5% and the relative half-power width of the coordinate's own peak.
   Since the SOD Toolbox release the code lives in sodtoolbox (core, recognize); this module keeps the names the
   paper scripts use."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'SOD-toolbox', 'python'))
import numpy as np
import sodtoolbox as _st
from sodtoolbox.recognize import (dct_spectrum, peak_and_width, cd_correct, select_config, onesided, pair_state,
                                  rom_order_config, rom_order_state)

def gsod(Z, D, tol=1e-10):
    """SOD pencil <D D^T> psi = mu <Z Z^T> psi on unit-variance channels, restricted to the data rank (tolerance tol).
       Returns covectors Psi (q = Z Psi), modes Phi (Z ~ q Phi^T), SOD values mu (ascending = smoothest first)."""
    R = _st.sod(Z, D, scale=True, tol=tol); return R.Psi, R.Phi, R.mu
