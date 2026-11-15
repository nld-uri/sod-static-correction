"""Fig. (SOD-ROM): COST F3 cantilever. (a) Frequency-energy plot of NNM1 from harmonic balance (n3_backbone_hb.py):
   main backbone, 5:1 tongue, and the zero-crossing frequencies of the plucked tests (n4). (b) Backbone frequency error of
   ROMs against energy, r = 2 (open) and r = 4 (filled) (n4_rom_backbones.py; band-limited decay: n5_equal_energy.py)."""
import numpy as np, pickle, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import plotprefs as pp
pp.apply_preferences(sizes=(10, 9, 8)); c = pp.MATLAB
F = pickle.load(open('n3_backbone_full_Kh8.pkl', 'rb')); R = pickle.load(open('n4_results.pkl', 'rb')); R5 = pickle.load(open('n5_results.pkl', 'rb'))
R['back'].update(R5['back'])
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
a = ax[0]
for H, f, h5 in [(F['low']['H'], F['low']['f'], F['low']['h5']), (F['H'], F['f'], F['h5'])]:
    m = h5 < 0.02; a.semilogx(H[m], f[m], 'k.', ms=1.5)
    a.semilogx(H[~m], f[~m], '.', color=c['gray'], ms=1.5)
a.plot([], [], 'k-', label='NNM1 backbone'); a.plot([], [], '-', color=c['gray'], label='5:1 tongue')
for k, v in R['tests'].items():
    a.plot(v['H'], v['f'], 'o' if 'tip' in k else 's', color=c['orange'] if 'training' not in k else c['darkred'], ms=4, mfc='none')
a.plot([], [], 'o', color=c['orange'], mfc='none', label='plucked tests (tip, mid)')
a.set_xlim(3e-4, 3.5e-2); a.set_ylim(23, 43); a.set_xlabel('energy $H$ (J)'); a.set_ylabel('frequency (Hz)'); a.legend(frameon=False, fontsize=7, loc='upper left')
b = ax[1]; Hg = R['Hgrid']; t = R['tongue_mask']
sty = {'LNM': (c['black'], 'o'), 'MTA': (c['darkred'], 's'), 'SOD (decay EE)': (c['blue'], '^'), 'POD (decay EE)': (c['orange'], 'v'), 'SOD (decay)': (c['cyan'], 'D')}
labn = {'LNM': 'LNM', 'MTA': 'MTA', 'SOD (decay EE)': 'SOD, band-limited decay', 'POD (decay EE)': 'POD, band-limited decay', 'SOD (decay)': 'SOD, static release'}
print("cvd:", pp.cvd_check([c['darkred'], c['blue'], c['orange'], c['cyan']]))
for bn, (col, mk) in sty.items():
    for r, fill in [(2, 'none'), (4, col)]:
        e = R['back'].get((bn, r))
        if e is None: continue
        e = np.where(t, np.nan, e)
        b.loglog(Hg, np.maximum(e, 1e-9), '-', color=col, marker=mk, ms=3, lw=0.9, mfc=fill, label=labn[bn] if r == 4 else None)
b.set_xlabel('energy $H$ (J)'); b.set_ylabel(r'$|f_{\mathrm{ROM}}/f_{\mathrm{full}}-1|$'); b.set_ylim(1e-8, 1e-1)
b.legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, columnspacing=0.8); b.text(0.03, 0.92, 'open: $r=2$, filled: $r=4$', transform=b.transAxes, fontsize=7)
for k, lab in enumerate('ab'): ax[k].text(-0.13, 1.02, f'({lab})', transform=ax[k].transAxes)
pp.latexify(fig); fig.tight_layout(); fig.savefig('../../figures/fig_c_nl.pdf')
