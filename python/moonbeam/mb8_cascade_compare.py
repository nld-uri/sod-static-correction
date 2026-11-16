"""Period-doubling cascade at Omega = 0.85 omega_w (mb6z_cascade.py): first amplitude of period 2, 4, 8, 16 and of
   cross-well motion, for the full model and each ROM; ROMs settled in the other well are compared through the mirror
   orbit (strobes -y at phase pi). Also the ratio (A4 - A2)/(A8 - A4), to set against Feigenbaum's 4.669.
   Usage: python3 mb8_cascade_compare.py"""
import numpy as np, pickle, glob
XS = 0.1; TOL = 1e-3
def period(y):
    k = len(np.unique(np.round(y / (TOL * XS)))); return k if k <= 32 else 99
def branch(D):
    """strobes on the full model's branch (positive well); mirror if the ROM sits in the negative well"""
    As, Y, E, H = D; return As, np.array([-h if e[1] < 0 else y for y, e, h in zip(Y, E, H)]), E   # e[1] < 0: whole orbit in the negative well
def landmarks(D):
    As, Y, E = branch(D); cw = (E[:, 0] < 0) & (E[:, 1] > 0); esc = np.argmax(cw) if cw.any() else len(As)
    L = {'escape': As[esc] if cw.any() else np.nan}; i0 = 0
    for k in (2, 4, 8, 16):
        idx = [i for i in range(i0, esc) if period(Y[i]) == k]; L[f'P{k}'] = As[idx[0]] if idx else np.nan; i0 = idx[0] if idx else i0
    L['delta'] = (L['P4'] - L['P2']) / (L['P8'] - L['P4']) if np.isfinite(L['P8']) and L['P8'] > L['P4'] else np.nan
    return L
if __name__ == "__main__":
    F = pickle.load(open('mb6z_full.pkl', 'rb'))['res']; Lf = landmarks(F)
    keys = ('P2', 'P4', 'P8', 'P16', 'escape', 'delta')
    print(f"{'model':8s} " + ' '.join(f"{k:>7s}" for k in keys) + "   rel. error of P2, P4, escape")
    print(f"{'full':8s} " + ' '.join(f"{Lf[k]:7.4f}" for k in keys))
    for f in sorted(glob.glob('mb6z_*_r*.pkl')):
        D = pickle.load(open(f, 'rb')); L = landmarks(D['res'])
        print(f"{D['name'] + ' ' + str(D['r']):8s} " + ' '.join(f"{L[k]:7.4f}" for k in keys) + "   " +
              ' '.join(f"{(L[k] - Lf[k]) / Lf[k]:+.1e}" for k in ('P2', 'P4', 'escape')))
