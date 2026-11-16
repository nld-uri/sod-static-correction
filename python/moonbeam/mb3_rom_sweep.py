"""Bifurcation diagrams of Galerkin ROMs of the two-well beam (bases from mb2_bases.pkl), same sweeps as mb1_full.py.
   Usage: python3 mb3_rom_sweep.py A BASIS [r ...]   (BASIS in LNM0, LNMw, MTA, POD, SOD; default r = 1 2 3 4)
   Output: mb3_A<A>_<BASIS>_r<r>.pkl, one per r."""
import numpy as np, pickle, sys, time
from mb_model import beam
from mb_sweep import sweep, rom_model
A = float(sys.argv[1]); name = sys.argv[2]; rs = [int(a) for a in sys.argv[3:]] or [1, 2, 3, 4]
S = beam(); B = pickle.load(open('mb2_bases.pkl', 'rb'))['B']
for r in rs:
    t0 = time.time(); mdl = rom_model(S, B[(name, r)]); res = {d: sweep(S, mdl, A, d) for d in (+1, -1)}
    pickle.dump(dict(A=A, name=name, r=r, res=res), open(f'mb3_A{A:g}_{name}_r{r}.pkl', 'wb'))
    print(f"{name} r={r}: {time.time() - t0:.0f}s", flush=True)
