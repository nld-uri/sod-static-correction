"""Does an inertia-corrected static vector explain SOD's low-order edge over MTA on the two-well beam?
   MTA uses K^-1 e_tip (clamped-beam stiffness K, nonlinear force at the tip). Here the static vector is replaced by
   the dynamic correction d(w) = (K - w^2 M)^-1 e_tip, the deflection under a harmonic tip force at frequency w.
   (1) angle (mass metric) between the first SOD direction and d(w) over w in [0, 6]; the w of the smallest angle;
   (2) one-mode ROM equilibrium and well frequency (as mb9_onemode.py) for d(omega_w) and d(w*);
   (3) bases DMTA (d(omega_w)) and DMTS (d(w*)) at r = 1, 2: r-1 clamped LNMs + d, mass-normalized; written to
       mb10_bases.pkl for the amplitude sweep (mb6_amp_sweep.py reads mb2_bases.pkl, so run_sweep() below calls it).
   Usage: python3 mb10_dynamic_correction.py            (angles, one-mode table, bases)
          python3 mb10_dynamic_correction.py sweep NAME r   (amplitude sweep -> mb6_NAME_r<r>.pkl)"""
import numpy as np, pickle, sys, time
from scipy.linalg import eigh
from mb_model import beam
S = beam(); M, K = S['M'], S['K']; k1, k3, tip, et = S['k1'], S['k3'], S['tip'], S['et']
B2 = pickle.load(open('mb2_bases.pkl', 'rb'))['B']; _, P0 = eigh(K, M)
def ang(p, q): return np.degrees(np.arccos(min(1.0, abs(p @ M @ q) / np.sqrt((p @ M @ p) * (q @ M @ q)))))
def d(w): return np.linalg.solve(K - w**2 * M, et)
def mnorm(Q): return Q / np.sqrt(np.einsum('ij,ij->j', Q, M @ Q))
ww, xs = S['ww'][0], S['xs'][tip]
if len(sys.argv) == 1:
    s1 = B2[('SOD', 1)][:, 0]
    W = np.linspace(0, 6, 601); A = np.array([ang(s1, d(w)) for w in W]); ws = W[A.argmin()]
    print(f"clamped w1 = {S['w0'][0]:.4f}, well w_w = {ww:.4f}")
    print(f"angle(SOD1, K^-1 e)        = {ang(s1, d(0.0)):.3f} deg")
    print(f"angle(SOD1, d(omega_w))    = {ang(s1, d(ww)):.3f} deg")
    print(f"smallest angle {A.min():.3f} deg at w* = {ws:.3f} (= {ws / ww:.3f} omega_w)")
    for w in (1, 2, 3, 4, 5): print(f"   w = {w}: {ang(s1, d(w)):.3f} deg")
    B = {}
    for name, w in (('DMTA', ww), ('DMTS', ws)):
        for r in (1, 2): B[(name, r)] = mnorm(np.c_[P0[:, :r - 1], d(w)]) if r > 1 else mnorm(d(w)[:, None])
    print("one-mode ROMs (phi scaled to unit tip displacement):")
    for name, p in (('SOD', B2[('SOD', 1)][:, 0]), ('MTA', B2[('MTA', 1)][:, 0]), ('DMTA', B[('DMTA', 1)][:, 0]), ('DMTS', B[('DMTS', 1)][:, 0])):
        p = p / p[tip]; m, k = p @ M @ p, p @ K @ p; q = np.sqrt((k1 - k) / k3); w = np.sqrt(2 * (k1 - k) / m)
        print(f"  {name:5s} q*/x* - 1 = {q / xs - 1:+.2e}   omega_w(ROM)/omega_w - 1 = {w / ww - 1:+.2e}")
    pickle.dump(dict(B=B, ws=ws), open('mb10_bases.pkl', 'wb'))
else:
    import mb6_amp_sweep as m6
    from mb_sweep import rom_model
    name, r = sys.argv[2], int(sys.argv[3]); B = pickle.load(open('mb10_bases.pkl', 'rb'))['B']
    t0 = time.time(); res = {dd: m6.amp_sweep(S, rom_model(S, B[(name, r)]), dd) for dd in (+1, -1)}
    pickle.dump(dict(name=name, r=r, Om=m6.FOM, res=res), open(f'mb6_{name}_r{r}.pkl', 'wb'))
    print(f"{name} r={r}: {time.time() - t0:.0f}s", flush=True)
