"""Fig. (SOD-ROM): which free decay to train on. One realization (c10_equal_energy.py, c14_bandlimited.py, same seed).
   (a) Initial modal energy of the training decays on the cantilever: static release (SR), equal energy over modes 1-9
       (EE), band-limited equal energy with omega_b = omega_4 (the order-matched IC for r = 4).
   (b) State-SOD subspace leakage outside planes 1-4 against the largest relative eigenvalue error of the 8-state energy
       Galerkin ROM (c13_leakage.py, all ICs and systems); dashed: 0.5 x leak^2.
   (c-f) order-matched band-limited decays from c15_matched.py (omega_b = omega_r).
   (c,d) Configuration SOD, new random tip load: S1 and S2 with heavy Rayleigh damping.
   (e,f) State SOD, random mid-span load: S3 with c_d = 0.2 and 1; complex-mode truncation (CM) for reference."""
import numpy as np, pickle, re, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.linalg import eigh
from systems import beam
import plotprefs as pp
pp.apply_preferences(sizes=(9, 8, 7)); c = pp.MATLAB
col = {'SR': c['blue'], 'EE': c['orange'], 'HAM': c['darkgreen'], 'RE': c['darkred'], 'EE4': c['plum'], 'ref': 'k'}
print("cvd:", pp.cvd_check([col[k] for k in ['SR', 'EE', 'HAM', 'RE', 'EE4']]))
fig, ax = plt.subplots(2, 3, figsize=(7.4, 4.9))
# (a) initial modal energies, beam
S = beam(); M, K, B = S['M'], S['K'], S['B']; w2, PL = eigh(K, M); w = np.sqrt(w2); k = np.arange(9)
xs = np.linalg.solve(K, B @ np.array([1.0, 0.7])); eta = PL[:, k].T @ M @ xs; Esr = 0.5 * w2[k] * eta**2
Eee = np.ones(9); Eham = (1 / (1 + (w[k] / w[3]) ** 4)) ** 2
a = ax[0, 0]; bw = 0.27
for j, (lab, E) in enumerate([('SR', Esr), ('EE', Eee), ('HAM', Eham)]):
    a.bar(k + 1 + (j - 1) * bw, E / E.max(), bw, color=col[lab], label={'SR': 'static release', 'EE': 'equal energy', 'HAM': r'band-limited, $\omega_{\mathrm b}=\omega_4$'}[lab])
a.set_yscale('log'); a.set_ylim(1e-9, 1e6); a.set_xlabel('LNM $k$'); a.set_ylabel('initial modal energy'); a.set_xticks(k + 1)
a.legend(frameon=False, fontsize=6.5, loc='upper right')
# (b) leakage law
a = ax[0, 1]; txt = open('c13_out.txt').read(); mk = {'S2 light': 'o', 'S3 dashpot': 's', 'S3 dashpot_heavy': '^'}
for blk in re.split(r'== ', txt)[1:]:
    name = blk.splitlines()[0].strip()
    for m in re.finditer(r'\s+(\w+)\s*: eps \S+ \| leak sSOD (\S+) SOD \S+ \| max rel eig err sROM (\S+)', blk):
        icn, lk, de = m.group(1), float(m.group(2)), float(m.group(3))
        a.loglog(lk, de, mk[name], color=col[icn], ms=4.5, mfc=col[icn] if name != 'S3 dashpot_heavy' else 'none')
xx = np.geomspace(1e-4, 1, 10); a.loglog(xx, 0.5 * xx**2, 'k--', lw=0.7)
a.set_xlabel('leakage outside planes 1--4'); a.set_ylabel('eigenvalue error, 8-state ROM'); a.set_xlim(1e-4, 1.5); a.set_ylim(1e-10, 2)
for nm, m_ in mk.items():
    a.plot([], [], m_, color='0.4', ms=4, mfc='0.4' if nm != 'S3 dashpot_heavy' else 'none',
           label={'S2 light': 'S2', 'S3 dashpot': r'S3, $c_{\mathrm d}=0.2$', 'S3 dashpot_heavy': r'S3, $c_{\mathrm d}=1$'}[nm])
a.legend(frameon=False, fontsize=6.5, loc='lower right')
# (c-f) errors against state dimension
def curve(a, ci, tk, bn_map, title):
    D10 = pickle.load(open(f'c10_case{ci}.pkl', 'rb'))['res']; D14 = pickle.load(open(f'c15_case{ci}.pkl', 'rb'))['res']
    for lab, (src, bn, ls, mk_) in bn_map.items():
        D = D10 if src == 10 else D14; rs = [r for r in range(1, 10 if ci in (0, 1) else 11) if (bn, r, tk) in D]   # S1: r = 10 is the full model
        a.semilogy(2 * np.array(rs), [D[(bn, r, tk)][0] for r in rs], ls=ls, marker=mk_, ms=3, lw=1.0, color=col[lab], label=lab_name[lab])
    a.set_title(title, fontsize=8); a.set_xlabel('state dimension $2r$'); a.set_ylim(1e-6, 1)
lab_name = {'SR': 'SOD, static release', 'EE': 'SOD, equal energy', 'HAM': r'SOD, band-limited, $\omega_{\mathrm b}=\omega_r$', 'ref': 'LNM / CM'}
cfg = lambda: {'ref': (10, 'LNM', '-', None), 'SR': (10, 'SOD (free)', '-', 'o'), 'EE': (10, 'SOD (freeEE)', '--', 's'), 'HAM': (14, 'SOD (freeBL)', ':', '^')}
st = lambda: {'ref': (10, 'CM', '-', None), 'SR': (10, 'sSOD (free)', '-', 'o'), 'EE': (10, 'sSOD (freeEE)', '--', 's'), 'HAM': (14, 'sSOD (freeBL)', ':', '^')}
curve(ax[0, 2], 1, 'rand', cfg(), 'S1, heavy Rayleigh, random load')
curve(ax[1, 0], 3, 'rand', cfg(), 'S2, heavy Rayleigh, random tip load')
curve(ax[1, 1], 4, 'mid', st(), r'S3, $c_{\mathrm d}=0.2$, random mid-span load')
curve(ax[1, 2], 5, 'mid', st(), r'S3, $c_{\mathrm d}=1$, random mid-span load')
for a_ in (ax[0, 2], ax[1, 0]): a_.set_ylabel('error')
h, l = ax[1, 0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=7, loc='lower center', ncol=4)
for kk, a_ in enumerate(ax.flat): a_.text(0.03, 0.9, f'({"abcdef"[kk]})', transform=a_.transAxes, fontsize=8)
pp.latexify(fig); fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig('../../figures/fig_c_ic.pdf')
