"""Fig. (SOD-ROM, revision 2): ROM error against state dimension 2r, median and interquartile range over 20 realizations.
   Columns: S1 heavy Rayleigh; S2 light Rayleigh; S3 c_d = 1 (non-proportional). Rows: new random load at the training
   location; transfer test (free decay from another static load for S1, random mid-span load for S2 and S3).
   Data: c18_case{1,2,5}.pkl from c18_ensemble.py (free-decay bases: order-matched band-limited decays)."""
import numpy as np, pickle, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import plotprefs as pp
pp.apply_preferences(sizes=(10, 9, 8)); c = pp.MATLAB
cols = [(1, 'S1, heavy Rayleigh', 'free'), (2, 'S2, light Rayleigh', 'mid'), (5, r'S3, $c_{\mathrm d}=1$', 'mid')]
sty = {'LNM': (c['black'] if 'black' in c else 'k', '-', 'o', 'none'), 'MTA': (c['darkred'], '--', 's', 'none'),
       'POD (rand)': (c['orange'], '-', 'v', 'full'), 'SOD (rand)': (c['blue'], '-', '^', 'full'),
       'SOD (freeBL)': (c['cyan'], ':', 'D', 'none'), 'CM': (c['plum'], '-.', 'x', 'none'),
       'sSOD (freeBL)': (c['darkgreen'], '-', 'P', 'full'), 'sSOD (rand)': (c['darkgreen'], '--', 'P', 'none')}
print("cvd:", pp.cvd_check([c['darkred'], c['orange'], c['blue'], c['cyan'], c['plum'], c['darkgreen']]))
fig, ax = plt.subplots(2, 3, figsize=(7.4, 5.0), sharex=True)
for j, (ci, title, tr) in enumerate(cols):
    D = pickle.load(open(f'c18_case{ci}.pkl', 'rb')); res = D['res']
    for i, tk in enumerate(['rand', tr]):
        a = ax[i, j]
        for bn, (col, ls, mk, fill) in sty.items():
            rs = [r for r in range(1, 10 if ci in (0, 1) else 11) if (bn, r, tk) in res and len(res[(bn, r, tk)]) == D["NR"]]   # all realizations; S1: r = n is the full model
            if not rs: continue
            q = np.array([np.percentile(res[(bn, r, tk)], [25, 50, 75]) for r in rs]); x = 2 * np.array(rs)
            a.semilogy(x, q[:, 1], ls=ls, color=col, marker=mk, ms=3.2, lw=1.0, mfc=col if fill == 'full' else 'none', label=bn.replace('freeBL', 'free, BL'))
            a.fill_between(x, q[:, 0], q[:, 2], color=col, alpha=0.15, lw=0)
        a.set_ylim(1e-7, 1)
        if i == 0: a.set_title(title, fontsize=9)
        if i == 1: a.set_xlabel('state dimension $2r$')
    ax[0, 0].set_ylabel('error, training load'); ax[1, 0].set_ylabel('error, transfer test')
for k, a in enumerate(ax.flat): a.text(0.04, 0.06, f'({"abcdef"[k]})', transform=a.transAxes)
h, l = [], []
for a in ax.flat:
    for hh, ll in zip(*a.get_legend_handles_labels()):
        if ll not in l: h.append(hh); l.append(ll)
fig.legend(h, l, frameon=False, fontsize=7.5, loc='lower center', ncol=4)
pp.latexify(fig); fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig('../../figures/fig_c_error.pdf')
