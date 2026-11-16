"""SOD-ROM (n5): as n4_rom_backbones.py, but the training decay starts from rest at an equal-energy sum of LNMs 1-4
   (hammer-like), each with the potential energy of the 30 N static state (H30). Bases: SOD and POD of that decay.
   Original n4 doc: SOD-ROM, nonlinear section: NNM1 backbones of Galerkin ROMs of the COST F3 cantilever against the full-model backbone
   (n3_backbone_hb.py), and time-domain tests from initial conditions not used in training.
   Bases: LNM (r lowest), MTA (r-1 LNMs + static tip-load vector), SOD and POD of a damped decay (zeta=0.01, Rayleigh,
   modes 1-2) from the static deflection under a 30 N tip load (as in SOD-NNM, n1_rom.py).
   Backbone metric: |f_ROM(H)/f_full(H) - 1| on the main backbone (|X5|/|X1| < 0.02, i.e. outside the 5:1 tongue).
   Time-domain tests (conservative, Newmark dt = 2e-5 s, 0.3 s): static deflections under tip loads 3, 10, 20 N and
   mid-span loads 20, 60 N; the 30 N tip load (training state) is reported separately."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'SOD-toolbox', 'python'))
import sodtoolbox as _st
import numpy as np, pickle, sys
from scipy.linalg import eigh
from costbeam import costbeam
from nlsim import newmark, static_deflection, dom_freq, energy
from n3_backbone_hb import HBC, full_backbone
S = costbeam(); Sd = costbeam(zeta=0.01); n = S['n']; tip = S['tip']; k3 = S['k3']; pt = np.zeros(n); pt[tip] = 1.0
w2, PL = eigh(S['K'], S['M']); w1 = np.sqrt(w2[0]); dt = 2e-5; sub = 5
def sod(X, V, r, tol=1e-10): return _st.sod_modes(X, V, r, tol)[0]
def pod(X, r): w, U = eigh(X.T @ X); return U[:, ::-1][:, :r]
def mta(r):
    s = np.linalg.solve(S['K'], pt); Q = np.c_[PL[:, :r - 1], s]; L = np.linalg.cholesky(Q.T @ S['M'] @ Q); return Q @ np.linalg.inv(L).T
x30 = static_deflection(S, 30.0); H30 = 0.5 * x30 @ S['K'] @ x30 + k3 / 4 * x30[tip]**4
xee = sum(PL[:, k] * np.sqrt(2 * H30) / np.sqrt(w2[k]) for k in range(4))          # mass-normalized LNMs: 1/2 w^2 a^2 = H30 each
print(f"equal-energy IC: tip {xee[tip]*1e3:.2f} mm, linear energy per mode {H30:.2e} J, total H {0.5*xee@S['K']@xee + k3/4*xee[tip]**4:.2e} J", flush=True)
_, Xd, Vd = newmark(Sd['M'], Sd['K'], Sd['C'], pt, k3, xee, np.zeros(n), 1.5, dt); Xd, Vd = Xd[::sub], Vd[::sub]
bases = {'SOD (decay EE)': lambda r: sod(Xd, Vd, r), 'POD (decay EE)': lambda r: pod(Xd, r)}
# full-model backbone
F = pickle.load(open('n3_backbone_full_Kh8.pkl', 'rb'))
Hf = np.r_[F['low']['H'], F['H']]; ff = np.r_[F['low']['f'], F['f']]; h5 = np.r_[F['low']['h5'], F['h5']]
main = h5 < 0.02; o = np.argsort(Hf[main]); Hm, fm = Hf[main][o], ff[main][o]
Hgrid = np.geomspace(6e-4, 2.5e-2, 25); fgrid = np.interp(np.log(Hgrid), np.log(Hm), fm)
tongue = (Hgrid > 3.5e-3) & (Hgrid < 4.5e-3)                        # interpolation across the tongue gap is excluded
out = dict(Hgrid=Hgrid, fgrid=fgrid, tongue_mask=tongue, back={}, td={})
for bn, fb in bases.items():
    for r in [1, 2, 3, 4]:
        if bn == 'MTA' and r < 2: continue
        Phi = fb(r); Mr, Kr = Phi.T @ S['M'] @ Phi, Phi.T @ S['K'] @ Phi; ptr = Phi[tip]
        wr2, Pr = eigh(Kr, Mr); phi1 = Pr[:, 0] / (ptr @ Pr[:, 0])
        hb = HBC(Mr, Kr, ptr, k3, Kh=8)
        try:
            low, high = full_backbone(hb, phi1, np.sqrt(wr2[0]))
            pts = low + high; H = np.array([hb.energy(z) for z in pts]); f = np.array([z[-1] for z in pts]) / 2 / np.pi
            hh = np.array([hb.harm(z)[2] for z in pts]); mm = hh < 0.02; oo = np.argsort(H[mm])
            fr = np.interp(np.log(Hgrid), np.log(H[mm][oo]), f[mm][oo]); e = abs(fr / fgrid - 1)
        except Exception as ex:
            print(bn, r, 'backbone failed:', ex); e = np.full(len(Hgrid), np.nan)
        out['back'][(bn, r)] = e
        print(f"{bn:12s} r={r}: backbone error median {np.nanmedian(e[~tongue]):.1e}, max {np.nanmax(e[~tongue]):.1e}", flush=True)
# time-domain held-out tests
tests = {}
for lab, P, node in [('tip 3 N', 3.0, tip), ('tip 10 N', 10.0, tip), ('tip 20 N', 20.0, tip), ('mid 20 N', 20.0, 18), ('mid 60 N', 60.0, 18), ('tip 30 N (training)', 30.0, tip)]:
    e = np.zeros(n); e[node] = P
    x0 = np.linalg.solve(S['K'], e)
    from scipy.optimize import fsolve
    x0 = fsolve(lambda x: S['K'] @ x + k3 * x[tip]**3 * pt - e, x0, xtol=1e-14)
    t, X, V = newmark(S['M'], S['K'], S['C'], pt, k3, x0, np.zeros(n), 0.3, dt)
    tests[lab] = dict(x0=x0, X=X, f=dom_freq(t, X[:, tip]), H=energy(S, X[:1], V[:1])[0])
    print(f"test {lab}: H={tests[lab]['H']:.2e} J, f={tests[lab]['f']:.3f} Hz", flush=True)
for bn, fb in bases.items():
    for r in [2, 4]:
        Phi = fb(r); Mr, Kr = Phi.T @ S['M'] @ Phi, Phi.T @ S['K'] @ Phi; ptr = Phi[tip]; Pm = np.linalg.solve(Mr, Phi.T @ S['M'])
        row = {}
        for lab, tt in tests.items():
            t, Q, _ = newmark(Mr, Kr, np.zeros((r, r)), ptr, k3, Pm @ tt['x0'], np.zeros(r), 0.3, dt); Xr = Q @ Phi.T; m = t <= 0.1
            D = Xr[m] - tt['X'][m]; row[lab] = dict(ferr=dom_freq(t, Xr[:, tip]) / tt['f'] - 1,
                                                  err=np.sqrt(np.sum((D @ S['M']) * D) / np.sum((tt['X'][m] @ S['M']) * tt['X'][m])))
        out['td'][(bn, r)] = row
        print(f"{bn:12s} r={r}: " + "  ".join(f"{k}: df/f={v['ferr']:+.1e} e={v['err']:.1e}" for k, v in row.items()), flush=True)
out['tests'] = {k: dict(f=v['f'], H=v['H']) for k, v in tests.items()}
pickle.dump(out, open('n5_results.pkl', 'wb'))
