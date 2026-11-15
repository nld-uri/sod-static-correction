"""Amplitude bifurcation diagrams of the two-well beam at fixed forcing frequency Omega = 0.85 omega_w (base acceleration
   A cos(Omega t)), A swept up and down over [0.01, 0.30] (291 values), the state carried from one amplitude to the next.
   Per amplitude: 150 forcing periods of transient, then 64 stroboscopic samples of the tip displacement (phase cos = 1) and
   the tip extremes and RMS over those 64 periods; 100 steps per period (average-acceleration Newmark).
   The probe (mb5b_fine.py) found period 1 -> 2 (A = 0.13) -> 4 (0.14) -> cross-well chaos (0.145) at this frequency.
   Usage: python3 mb6_amp_sweep.py full | <BASIS> [r ...]   (bases from mb2_bases.pkl; default r = 1 2 3 4)
   Output: mb6_full.pkl or mb6_<BASIS>_r<r>.pkl."""
import numpy as np, pickle, sys, time
from mb_model import beam, newmark
from mb_sweep import full_model, rom_model
FOM = 0.85; AS = np.linspace(0.01, 0.30, 291); NTR, NREC, SPP = 150, 64, 100
def amp_sweep(S, mdl, direction):
    As = AS if direction > 0 else AS[::-1]; x, v = mdl['x0'].copy(), mdl['v0'].copy(); out, ext = [], []
    Om = FOM * S['ww'][0]; T = 2 * np.pi / Om; dt = T / SPP; idx = range(SPP * NTR + 1, SPP * (NTR + NREC) + 1)
    for A in As:
        X, V, x, v = newmark(mdl['M'], mdl['K'], mdl['C'], mdl['pt'], S, x, v, T * (NTR + NREC), dt, A, Om, mdl['Mr'], tsave=idx)
        y = X @ mdl['pt']; out.append(y[SPP - 1::SPP]); ext.append((y.min(), y.max(), np.sqrt(np.mean(y**2))))
    return As, np.array(out), np.array(ext)
if __name__ == "__main__":
    S = beam(); name = sys.argv[1]
    if name == 'full':
        t0 = time.time(); res = {d: amp_sweep(S, full_model(S), d) for d in (+1, -1)}
        pickle.dump(dict(name='full', Om=FOM, res=res), open('mb6_full.pkl', 'wb')); print(f"full: {time.time() - t0:.0f}s")
    else:
        B = pickle.load(open('mb2_bases.pkl', 'rb'))['B']
        for r in [int(a) for a in sys.argv[2:]] or [1, 2, 3, 4]:
            t0 = time.time(); res = {d: amp_sweep(S, rom_model(S, B[(name, r)]), d) for d in (+1, -1)}
            pickle.dump(dict(name=name, r=r, Om=FOM, res=res), open(f'mb6_{name}_r{r}.pkl', 'wb'))
            print(f"{name} r={r}: {time.time() - t0:.0f}s", flush=True)
