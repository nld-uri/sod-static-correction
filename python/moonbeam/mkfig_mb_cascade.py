"""Figure: the period-doubling cascade of the two-well beam at Omega = 0.85 omega_w (mb6z_cascade.py), full model and
   ROMs at r = 1 (left) and r = 2 (right). Stroboscopic tip displacement / x* on the positive-well branch (a ROM that
   settled in the negative well is shown through its mirror orbit). Dotted lines: full-model P2, P4 and escape amplitudes.
   Usage: python3 mkfig_mb_cascade.py [basis ...]   (default SOD MTA POD LNM0 LNMw)"""
import numpy as np, pickle, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import plotprefs as pp
from mb8_cascade_compare import branch, landmarks
names = sys.argv[1:] or ['SOD', 'MTA', 'POD', 'LNM0', 'LNMw']; XS = 0.1
lbl = {'SOD': 'SOD', 'MTA': 'MTA', 'POD': 'POD', 'LNM0': 'LNM (clamped)', 'LNMw': 'LNM (in well)'}
pp.apply_preferences(sizes=(9, 8, 7)); c = pp.MATLAB
F = pickle.load(open('mb6z_full.pkl', 'rb'))['res']; Lf = landmarks(F)
fig, ax = plt.subplots(1 + len(names), 2, figsize=(7.4, 1.15 * (1 + len(names))), sharex=True, sharey=True)
for j, r in enumerate((1, 2)):
    rows = [('full model', F, c['darkred'])] + [(f'{lbl[n]}, $r={r}$', pickle.load(open(f'mb6z_{n}_r{r}.pkl', 'rb'))['res'], c['blue']) for n in names]
    for i, (lab, D, col) in enumerate(rows):
        a = ax[i, j]; As, Y, _ = branch(D); a.plot(np.repeat(As, Y.shape[1]), Y.ravel() / XS, '.', color=col, ms=0.7, rasterized=True)
        for k in ('P2', 'P4', 'escape'): a.axvline(Lf[k], color='k', lw=0.4, ls=':')
        a.set_ylim(1.1, 1.36); a.text(0.02, 0.08, lab, transform=a.transAxes, fontsize=7)
        if j == 0: a.set_ylabel(r'$x_{\mathrm{tip}}/x^*$')
    ax[-1, j].set_xlim(0.12, 0.145); ax[-1, j].set_xlabel('$A$')
pp.latexify(fig); fig.tight_layout(h_pad=0.2, w_pad=0.6)
fig.savefig('../../figures/fig_mb_cascade.pdf', dpi=300)
fig.savefig('/tmp/claude-0/-home-claude/5f080035-e77c-5060-ad01-0b8c439e28f8/scratchpad/fig_mb_cascade.png', dpi=110)
