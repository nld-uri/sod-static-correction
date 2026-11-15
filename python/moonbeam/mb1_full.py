"""Full-model bifurcation diagrams of the two-well beam, A from argv, up and down sweeps. Output: mb1_full_A<A>.pkl."""
import numpy as np, pickle, sys, time
from mb_model import beam
from mb_sweep import sweep, full_model
A = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
S = beam(); t0 = time.time(); res = {}
for d in (+1, -1):
    res[d] = sweep(S, full_model(S), A, d); print(f"direction {d}: {time.time() - t0:.0f}s", flush=True)
pickle.dump(dict(A=A, res=res), open(f'mb1_full_A{A:g}.pkl', 'wb'))
