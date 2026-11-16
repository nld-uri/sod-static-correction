"""Figure 2 of SOD-ROM v5: decay-trained SOD bases on the dashpot cantilever (S3), from i7_case0_thr5.pkl and i7_case1_thr5.pkl.
   Median ROM error over 20 test loads against order r, for the LNMs, MTA with the tip vector, the SOD basis in plain
   order and with the recognised modes first. Output: ../../figures/fig_v5_decay.pdf (vector, drawn at print width)."""
import os, sys, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'moonbeam'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotprefs as pp
pp.apply_preferences(); C = pp.MATLAB
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8})
sty = {'LNM': dict(color='black', ls='-', marker='o', label='LNM'),
       'MTA-tip': dict(color=C['blue'], ls='--', marker='s', label='MTA, tip vector'),
       'R0': dict(color=C['orange'], ls='-.', marker='^', label='SOD of the decay, plain order'),
       'RHB': dict(color=C['darkgreen'], ls='-', marker='D', label='SOD of the decay, modes first')}
fig, ax = plt.subplots(2, 2, figsize=(6.3, 4.7), sharex=True)
R = list(range(2, 9)); letters = 'abcd'
for j, (ci, cd) in enumerate([(0, '0.2'), (1, '1')]):
    res = pickle.load(open(os.path.join(HERE, f'i7_case{ci}_thr5.pkl'), 'rb'))['res']
    for i, (tk, name) in enumerate([('tip', 'tip load'), ('mid', 'mid-span load')]):
        a = ax[i, j]
        for bn, st in sty.items():
            a.semilogy(R, [np.median(res[(bn, r, tk)]) for r in R], ms=3.5, lw=1.1, mfc='none' if bn != 'RHB' else st['color'], **st)
        a.text(0.03, 0.06, f"({letters[2 * i + j]}) $c_{{\\mathrm{{d}}}}={cd}$, {name}", transform=a.transAxes)
        a.grid(True, which='major', lw=0.3, alpha=0.5)
        if i == 1: a.set_xlabel('ROM order $r$')
        if j == 0: a.set_ylabel('relative error')
h, l = ax[0, 0].get_legend_handles_labels()
pp.latexify(fig); fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.93))
fig.legend(h, [pp.tex_escape(x) for x in l], loc='upper center', ncol=2, frameon=False, handlelength=2.6, columnspacing=2.0)
out = os.path.join(HERE, '..', '..', 'figures', 'fig_v5_decay.pdf'); fig.savefig(out); print('wrote', out)
