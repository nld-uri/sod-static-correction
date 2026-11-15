"""Compare ROM amplitude bifurcation diagrams (mb6_<BASIS>_r<r>.pkl) with the full model (mb6_full.pkl).
   Per amplitude and sweep direction: class = (cross-well?, period k <= 16 or 'NP'), from 64 strobes rounded to 1e-3 x*
   and the orbit extremes; W1 = Wasserstein distance of the strobe sets / x*.
   Landmarks along each sweep: first A of period 2 and of period 4 (up sweep, before escape), escape to cross-well
   motion (up), return to in-well motion (down). Prints class agreement, median W1 on points in-well in both models and in the same well, median W1 on all points,
   the number of in-well points where the ROM settled in the other well, the median error of the in-well orbit
   extremes (mirror orbit x -> -x(t + T/2) when the ROM is in the other well), and the landmarks."""
import numpy as np, pickle, glob
from scipy.stats import wasserstein_distance
XS = 0.1
def classify(y, e):
    k = len(np.unique(np.round(y / (1e-3 * XS)))); return (bool(e[0] < 0 < e[1]), k if k <= 16 else 'NP')
def landmarks(R):
    As, Y, E = R[+1]; c = [classify(y, e) for y, e in zip(Y, E)]; L = {}
    esc = next((i for i, q in enumerate(c) if q[0]), None); L['escape'] = As[esc] if esc is not None else np.nan
    i0 = 0
    for k in (2, 4, 8):                                   # each doubling searched after the previous one
        i = next((i for i in range(i0, esc if esc is not None else len(c)) if c[i] == (False, k)), None)
        L[f'P{k}'] = As[i] if i is not None else np.nan; i0 = i if i is not None else i0
    As, Y, E = R[-1]; c = [classify(y, e) for y, e in zip(Y, E)]
    i = next((i for i, q in enumerate(c) if not q[0]), None); L['return'] = As[i] if i is not None else np.nan
    return L
def compare(F, R):
    ag, Wi, Wa, ww, Xe = [], [], [], 0, []
    for d in (+1, -1):
        for yf, ef, yr, er in zip(F[d][1], F[d][2], R[d][1], R[d][2]):
            cf, cr = classify(yf, ef), classify(yr, er); ag.append(cf == cr); w = wasserstein_distance(yf, yr) / XS; Wa.append(w)
            if not cf[0] and not cr[0]:
                same = np.sign(yf.mean()) == np.sign(yr.mean()); lo, hi = (er[0], er[1]) if same else (-er[1], -er[0])
                Xe.append(max(abs(lo - ef[0]), abs(hi - ef[1])) / XS)   # orbit extremes, mirror orbit if other well
                if same: Wi.append(w)
                else: ww += 1                             # settled in the other well: attractor choice, not accuracy
    return np.mean(ag) * 100, np.median(Wi), np.median(Wa), ww, np.median(Xe)
if __name__ == "__main__":
    F = pickle.load(open('mb6_full.pkl', 'rb'))['res']; Lf = landmarks(F)
    print("full:", {k: round(float(v), 4) for k, v in Lf.items()})
    print(f"{'basis':5s} r agree%  W1(in-well) W1(all)  other-well ext-err(in-well)  landmarks (P2, P4, P8, escape, return)")
    for f in sorted(glob.glob('mb6_*_r*.pkl')):
        D = pickle.load(open(f, 'rb')); a, wi, wa, ww, xe = compare(F, D['res']); L = landmarks(D['res'])
        print(f"{D['name']:5s} {D['r']} {a:5.1f}  {wi:.2e}   {wa:.2e}  {ww:3d}  {xe:.2e}  " + ' '.join(f"{L[k]:.3f}" for k in ('P2', 'P4', 'P8', 'escape', 'return')))
