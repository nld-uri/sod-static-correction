"""Zoom on the period-doubling cascade of mb6_amp_sweep.py (Omega = 0.85 omega_w): A ramped up from 0.10 to 0.12 in
   steps of 1e-3, then 0.120 to 0.145 in 251 steps of 1e-4, the state carried throughout. 300 transient periods per
   amplitude (slow transients near each doubling), then 64 strobes at phase 0 and 64 at phase pi. The phase-pi strobes, negated, give the strobes of the
   mirror orbit x -> -x(t + T/2), so a ROM that settles in the other well can be compared on the same branch. Up sweep only.
   Usage: python3 mb6z_cascade.py full | <BASIS> [r ...]. Output: mb6z_full.pkl or mb6z_<BASIS>_r<r>.pkl."""
import numpy as np, pickle, sys, time
from mb_model import beam, newmark
from mb_sweep import full_model, rom_model
FOM = 0.85; AZ = np.linspace(0.120, 0.145, 251); NTR, NREC, SPP = 300, 64, 100
def cascade(S, mdl):
    x, v = mdl['x0'].copy(), mdl['v0'].copy(); Om = FOM * S['ww'][0]; T = 2 * np.pi / Om; dt = T / SPP; out, ext, half = [], [], []
    for A in np.r_[np.linspace(0.01, 0.119, 110), AZ]:
        idx = range(SPP * NTR + 1, SPP * (NTR + NREC) + 1)
        X, V, x, v = newmark(mdl['M'], mdl['K'], mdl['C'], mdl['pt'], S, x, v, T * (NTR + NREC), dt, A, Om, mdl['Mr'], tsave=idx)
        if A >= AZ[0] - 1e-12:
            y = X @ mdl['pt']; out.append(y[SPP - 1::SPP]); half.append(y[SPP // 2 - 1::SPP]); ext.append((y.min(), y.max(), np.sqrt(np.mean(y**2))))
    return AZ, np.array(out), np.array(ext), np.array(half)
if __name__ == "__main__":
    S = beam(); name = sys.argv[1]
    if name == 'full':
        t0 = time.time(); pickle.dump(dict(name='full', res=cascade(S, full_model(S))), open('mb6z_full.pkl', 'wb')); print(f"full: {time.time() - t0:.0f}s")
    else:
        B = pickle.load(open('mb2_bases.pkl', 'rb'))['B']
        for r in [int(a) for a in sys.argv[2:]] or [1, 2, 3, 4]:
            t0 = time.time(); pickle.dump(dict(name=name, r=r, res=cascade(S, rom_model(S, B[(name, r)]))), open(f'mb6z_{name}_r{r}.pkl', 'wb'))
            print(f"{name} r={r}: {time.time() - t0:.0f}s", flush=True)
