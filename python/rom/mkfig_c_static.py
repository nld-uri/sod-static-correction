"""Fig. (SOD-ROM, v3): what a random-load record adds to the SOD basis. S2, light Rayleigh, 20 realizations (c18 case-2 seeds).
   (a) M-norm residual of the static load vectors K^-1 b_tip (filled) and K^-1 b_mid (open) outside the span of the first r
       basis vectors: SOD (random tip load), LNM, SOD of the order-matched band-limited decay; median and interquartile range.
   (b) Largest principal angle (M metric) between span(SOD_r) and span(LNM_1..r), span(LNM_1..r-1, K^-1 b_tip) and
       span(LNM_1..r-1, K^-1 b_mid); median and range over the realizations.
   Data: g1_results.pkl (g1_static_containment.py), g3_results.pkl (g3_third_direction.py)."""
import numpy as np, pickle, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import plotprefs as pp
pp.apply_preferences(sizes=(10, 9, 8)); c = pp.MATLAB
print("cvd:", pp.cvd_check([c['blue'], c['cyan'], c['orange'], c['gray']]))
G1 = pickle.load(open('g1_results.pkl', 'rb')); G3 = pickle.load(open('g3_results.pkl', 'rb'))
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.5))
a = ax[0]
for j, (mk, mf, lab) in enumerate((('o', 'full', r'$\bm{K}^{-1}\bm{b}_{\mathrm{tip}}$'), ('o', 'none', r'$\bm{K}^{-1}\bm{b}_{\mathrm{mid}}$'))):
    rs = sorted(r for r, v in G1['out']['sod'].items() if len(v) == G1['NR'])
    q = np.array([np.percentile(np.array(G1['out']['sod'][r])[:, j], [25, 50, 75]) for r in rs])
    a.semilogy(rs, q[:, 1], '-' if j == 0 else '--', color=c['blue'], marker=mk, mfc=c['blue'] if mf == 'full' else 'none', ms=3.5, lw=1.0,
               )
    a.fill_between(rs, q[:, 0], q[:, 2], color=c['blue'], alpha=0.15, lw=0)
    rl = sorted(G1['lnm']); a.semilogy(rl, [G1['lnm'][r][j] for r in rl], '-' if j == 0 else '--', color='k', marker='s',
                                       mfc='k' if mf == 'full' else 'none', ms=3.0, lw=0.9)
    rb = sorted(G1['bl']); a.semilogy(rb, [G1['bl'][r][j] for r in rb], ':', color=c['cyan'], marker='D',
                                      mfc=c['cyan'] if mf == 'full' else 'none', ms=3.0, lw=1.0)
a.set_xlabel('basis order $r$'); a.set_ylabel('residual of static vector, $\\bm{M}$ norm'); a.set_xticks(range(1, 11))
a.set_ylim(3e-8, 0.2)
a = ax[1]
labs = [(0, c['orange'], '-', r'to $\mathrm{LNM}_{1..r-1}\cup\bm{K}^{-1}\bm{b}_{\mathrm{tip}}$'),
        (1, c['gray'], '--', r'to $\mathrm{LNM}_{1..r-1}\cup\bm{K}^{-1}\bm{b}_{\mathrm{mid}}$'),
        (2, c['blue'], '-', r'to $\mathrm{LNM}_{1..r}$')]
rs = sorted(r for r, v in G3.items() if len(v) == G1['NR'])
for j, col, ls, lab in labs:
    v = np.array([np.array(G3[r])[:, j] for r in rs])
    a.plot(rs, np.median(v, 1), ls, color=col, marker='o', ms=3.2, lw=1.0, label=lab)
    a.fill_between(rs, v.min(1), v.max(1), color=col, alpha=0.15, lw=0)
a.set_xlabel('basis order $r$'); a.set_ylabel(r'largest principal angle of $\mathrm{SOD}_r$ (deg)'); a.set_ylim(0, 90)
a.set_xticks(range(2, 9))
for k, a in enumerate(ax): a.text(0.03, 0.04 if k == 0 else 0.90, f'({"ab"[k]})', transform=a.transAxes)
from matplotlib.lines import Line2D
hp = [Line2D([], [], color=c['blue'], marker='o', ms=3.5, lw=1.0, label='SOD, random load'),
      Line2D([], [], color='k', marker='s', ms=3.0, lw=0.9, label='LNM'),
      Line2D([], [], color=c['cyan'], ls=':', marker='D', ms=3.0, lw=1.0, label='SOD, band-limited decay'),
      Line2D([], [], color='0.3', marker='o', ls='-', ms=3.5, lw=0.9, label=r'$\bm{K}^{-1}\bm{b}_{\mathrm{tip}}$ (filled, solid)'),
      Line2D([], [], color='0.3', marker='o', mfc='none', ls='--', ms=3.5, lw=0.9, label=r'$\bm{K}^{-1}\bm{b}_{\mathrm{mid}}$ (open, dashed)')]
ax[0].legend(handles=hp, frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
ax[1].legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)
pp.latexify(fig); fig.tight_layout(); fig.subplots_adjust(bottom=0.36); fig.savefig('../../figures/fig_c_static.pdf')
