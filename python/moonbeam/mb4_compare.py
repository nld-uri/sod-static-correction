"""Compare ROM bifurcation diagrams (mb3_*.pkl) with the full model (mb1_full_A<A>.pkl).
   Per forcing frequency and sweep direction:
     class  = (cross-well?, period k <= 8 or 'NP' = not periodic within 8), from 100 strobe samples rounded to 1e-3 x*
              and the orbit extremes over the 100 recorded periods;
     W1     = Wasserstein distance between ROM and full strobe sets, divided by x* (tip of the buckled equilibrium);
     amp    = |peak-to-peak ROM - peak-to-peak full| / x*.
   Usage: python3 mb4_compare.py A.   Prints, per basis and r: class agreement (%), median and 90th percentile W1/x*,
   median amp error; on periodic points only, median W1/x*; frequencies where cross-well motion starts (^) or ends (v)."""
import numpy as np, pickle, sys, glob
from scipy.stats import wasserstein_distance
A = float(sys.argv[1]); XS = 0.1
F = pickle.load(open(f'mb1_full_A{A:g}.pkl', 'rb'))['res']
def classify(y, e):
    k = len(np.unique(np.round(y / (1e-3 * XS)))); cw = bool(e[0] < 0 < e[1])
    return (cw, k if k <= 8 else 'NP')
def compare(R):
    agree, W, amp, Wp = [], [], [], []
    for d in (+1, -1):
        _, Yf, Ef = F[d]; _, Yr, Er = R[d]
        for yf, ef, yr, er in zip(Yf, Ef, Yr, Er):
            cf, cr = classify(yf, ef), classify(yr, er); agree.append(cf == cr)
            w = wasserstein_distance(yf, yr) / XS; W.append(w); amp.append(abs((er[1] - er[0]) - (ef[1] - ef[0])) / XS)
            if cf[1] != 'NP': Wp.append(w)
    return np.mean(agree) * 100, np.median(W), np.percentile(W, 90), np.median(amp), np.median(Wp) if Wp else np.nan
def transitions(R):
    """forcing frequencies where the sweep enters or leaves cross-well motion (midpoint of the two samples)"""
    out = {}
    for d in (+1, -1):
        oms, _, E = R[d]; cw = (E[:, 0] < 0) & (E[:, 1] > 0); i = np.nonzero(cw[1:] != cw[:-1])[0]
        out[d] = [f"{0.5 * (oms[j] + oms[j + 1]):.3f}{'v' if cw[j] else '^'}" for j in i]
    return out
if __name__ == "__main__":
    cls = [classify(y, e) for d in (+1, -1) for y, e in zip(F[d][1], F[d][2])]
    print(f"A = {A}: full model, {len(cls)} points: {sum(c[1] == 'NP' for c in cls)} not periodic, "
          f"{sum(c[0] for c in cls)} cross-well, {sum(c == (True, 1) for c in cls)} cross-well period-1")
    print("full-model cross-well transitions (up, down):", transitions(F))
    print(f"{'basis':6s} r  agree%  W1med   W1p90   amp_med  W1med(periodic)")
    for f in sorted(glob.glob(f'mb3_A{A:g}_*_r*.pkl')):
        D = pickle.load(open(f, 'rb')); a, wm, w9, am, wp = compare(D['res'])
        print(f"{D['name']:6s} {D['r']}  {a:5.1f}  {wm:.2e} {w9:.2e} {am:.2e} {wp:.2e}  {transitions(D['res'])}")
