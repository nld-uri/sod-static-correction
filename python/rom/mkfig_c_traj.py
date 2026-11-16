"""Fig. (SOD-ROM): ROM tip trajectories against the full model, state dimension 8. Data: c11_timehist.npz (c11_timehist.py).
   Columns: random mid-span load (window 100-102 s); equal-energy free decay (0-2 s). Rows: S2 light Rayleigh
   trajectories, S2 tip error, S3 c_{\mathrm d}=1 trajectories, S3 tip error (RMS of x_ROM - x_full over a sliding 0.1 s window, / RMS of x_full over the plotted window)."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import plotprefs as pp
pp.apply_preferences(sizes=(9, 8, 7)); c = pp.MATLAB
Z = np.load('c11_timehist.npz'); h = 2 * float(Z['dt'])
sty = {'LNM': (c['blue'], '-'), 'MTA': (c['darkred'], '--'), 'SOD_free_EE': (c['orange'], '-'), 'SOD_random': (c['gray'], ':'),
       'CM': (c['plum'], '-'), 'sSOD_SR': (c['cyan'], '-'), 'sSOD_free_EE': (c['orange'], '--'), 'sSOD_BL': (c['darkgreen'], '-.')}
lab = {'LNM': 'LNM', 'MTA': 'MTA', 'SOD_free_EE': 'SOD (free, equal energy)', 'SOD_random': 'SOD (random)', 'CM': 'complex modes',
       'sSOD_SR': 'state SOD (static release)', 'sSOD_free_EE': 'state SOD (equal energy)', 'sSOD_BL': 'state SOD (band-limited)'}
print("cvd:", pp.cvd_check([v[0] for k, v in sty.items() if k != 'sSOD_free_EE']))
fig, ax = plt.subplots(4, 2, figsize=(7.4, 7.0), sharex='col', gridspec_kw=dict(height_ratios=[1.4, 0.8, 1.4, 0.8]))
for j, (tk, t0, t1) in enumerate([('mid', 100, 102), ('freeEE', 0, 2)]):
    for i, tag in enumerate(['S2', 'S3']):
        full = Z[f'{tag}_{tk}_full']; t = np.arange(len(full)) * h; m = (t >= t0) & (t <= t1); rms = np.sqrt(np.mean(full[m]**2))
        a, e = ax[2 * i, j], ax[2 * i + 1, j]
        a.plot(t[m], full[m] / rms, 'k-', lw=1.6, label='full model')
        for bn, (col, ls) in sty.items():
            key = f'{tag}_{tk}_{bn}'
            if key not in Z.files: continue
            y = Z[key]; a.plot(t[m], y[m] / rms, ls=ls, color=col, lw=0.9, label=lab[bn])
            e2 = (y - full) ** 2; L = int(round(0.1 / h)); kern = np.ones(L)
            ew = np.sqrt(np.convolve(e2, kern, 'same') / np.convolve(np.ones_like(e2), kern, 'same')) / rms
            e.semilogy(t[m], ew[m], ls=ls, color=col, lw=1.0)
        a.set_ylabel(r'$x_{\mathrm{tip}}/\mathrm{RMS}$'); e.set_ylabel('error'); e.set_ylim(1e-5, 1)
        a.set_title({'S2': 'S2, light Rayleigh', 'S3': r'S3, $c_{\mathrm d}=1$'}[tag] + (', random mid-span load' if tk == 'mid' else ', equal-energy free decay'), fontsize=8)
    ax[3, j].set_xlabel('time $t$'); ax[3, j].set_xticks(np.arange(t0, t1 + 1e-9, 0.5))
for k, a in enumerate(ax.flat): a.text(0.01, 0.86, f'({"abcdefgh"[k]})', transform=a.transAxes, fontsize=8)
hh, ll = [], []
for a in ax.flat:
    for x, y in zip(*a.get_legend_handles_labels()):
        if y not in ll: hh.append(x); ll.append(y)
fig.legend(hh, ll, frameon=False, fontsize=7, loc='lower center', ncol=4)
pp.latexify(fig); fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig('../../figures/fig_c_traj.pdf')
