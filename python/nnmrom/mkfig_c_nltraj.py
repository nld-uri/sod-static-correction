"""Fig. (SOD-ROM): nonlinear COST F3 cantilever, held-out plucked tests (20 N at the tip; 60 N at mid-span), released
   from rest. Tip displacement of the full model and Galerkin ROMs (n6_timehist.npz from n6_timehist.py).
   Rows: trajectories at r = 2 over 0.2-0.3 s; tip error at r = 2 and r = 4 (0-0.3 s): RMS over a sliding 0.03 s window (about one period),
   normalized by the RMS full-model tip displacement."""
import sys; sys.path.insert(0, '../rom')
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import plotprefs as pp
pp.apply_preferences(sizes=(9, 8, 7)); c = pp.MATLAB
Z = np.load('n6_timehist.npz'); h = float(Z['dt'])
sty = {'LNM': (c['blue'], '-', 'LNM'), 'MTA': (c['darkred'], '--', 'MTA'), 'SOD_SR': (c['orange'], '-', 'SOD, static-release decay'),
       'SOD_EE': (c['darkgreen'], ':', 'SOD, equal-energy decay')}
print("cvd:", pp.cvd_check([v[0] for v in sty.values()]))
fig, ax = plt.subplots(3, 2, figsize=(7.4, 5.6), gridspec_kw=dict(height_ratios=[1.3, 0.8, 0.8]))
for j, (lab, title) in enumerate([('tip20', r'20\,N tip load'), ('mid60', r'60\,N mid-span load')]):
    full = Z[f'{lab}_full']; t = np.arange(len(full)) * h; rms = np.sqrt(np.mean(full**2)); m = t >= 0.2
    a = ax[0, j]; a.plot(t[m], full[m] / rms, 'k-', lw=1.6, label='full model')
    for bn, (cc, ls, ln) in sty.items():
        a.plot(t[m], Z[f'{lab}_{bn}_r2'][m] / rms, ls=ls, color=cc, lw=0.9, label=ln)
        for i, r in [(1, 2), (2, 4)]:
            e2 = (Z[f'{lab}_{bn}_r{r}'] - full) ** 2; L = int(round(0.03 / h)); kern = np.ones(L) / L
            ax[i, j].semilogy(t, np.sqrt(np.convolve(e2, kern, 'same') / np.convolve(np.ones_like(e2), kern, 'same')) / rms, ls=ls, color=cc, lw=1.0)
    a.set_title(f'{title}, $r=2$', fontsize=8); a.set_ylabel(r'$x_{\mathrm{tip}}/\mathrm{RMS}$')
    ax[1, j].set_ylabel('error, $r=2$'); ax[2, j].set_ylabel('error, $r=4$'); ax[2, j].set_xlabel('time $t$ (s)')
    for i in (1, 2): ax[i, j].set_ylim(1e-4, 1); ax[i, j].set_xlim(0, 0.3)
for kk, a in enumerate(ax.flat): a.text(0.01, 0.86, f'({"abcdef"[kk]})', transform=a.transAxes, fontsize=8)
h_, l_ = ax[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, fontsize=7, loc='lower center', ncol=5)
pp.latexify(fig); fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig('../../figures/fig_c_nltraj.pdf')
