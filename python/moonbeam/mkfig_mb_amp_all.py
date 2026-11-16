"""Figure: amplitude bifurcation diagrams of the two-well beam at Omega = 0.85 omega_w over the whole range swept,
   A = 0.01-0.30 (mb6_amp_sweep.py): full model (top row) and ROMs at r = 1 (left) and r = 2 (right).
   Stroboscopic tip displacement / x*; up sweep blue, down sweep orange. A ROM may settle on the mirror orbit in the
   other well after the jump near A = 0.075; that is a choice between coexisting attractors, not a model error.
   Usage: python3 mkfig_mb_amp_all.py [basis ...]   (default SOD MTA POD LNM0 LNMw)"""
import numpy as np, pickle, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import plotprefs as pp
names = sys.argv[1:] or ['SOD', 'MTA', 'POD', 'LNM0', 'LNMw']; XS = 0.1
lab = {'SOD': 'SOD', 'MTA': 'MTA', 'POD': 'POD', 'LNM0': 'LNM (clamped)', 'LNMw': 'LNM (in well)'}
pp.apply_preferences(sizes=(9, 8, 7)); c = pp.MATLAB
print("cvd:", pp.cvd_check([c['blue'], c['orange']]))
F = pickle.load(open('mb6_full.pkl', 'rb'))['res']
fig, ax = plt.subplots(1 + len(names), 2, figsize=(7.4, 1.25 * (1 + len(names))), sharex=True, sharey=True)
for j, r in enumerate((1, 2)):
    rows = [('full model', F)] + [(f'{lab[n]}, $r={r}$', pickle.load(open(f'mb6_{n}_r{r}.pkl', 'rb'))['res']) for n in names]
    for i, (t, R) in enumerate(rows):
        a = ax[i, j]
        for d, col in [(+1, c['blue']), (-1, c['orange'])]:
            As, Y, _ = R[d]; a.plot(np.repeat(As, Y.shape[1]), Y.ravel() / XS, '.', color=col, ms=0.5, rasterized=True)
        a.set_ylim(-2.2, 2.2); a.set_yticks([-2, 0, 2]); a.text(0.98, 0.84, t, transform=a.transAxes, fontsize=7, ha='right')
        if j == 0: a.set_ylabel(r'$x_{\mathrm{tip}}/x^*$')
    ax[-1, j].set_xlim(0.01, 0.30); ax[-1, j].set_xlabel('$A$')
pp.latexify(fig); fig.tight_layout(h_pad=0.2, w_pad=0.6)
fig.savefig('../../figures/fig_mb_amp.pdf', dpi=300)
fig.savefig('/tmp/claude-0/-home-claude/5f080035-e77c-5060-ad01-0b8c439e28f8/scratchpad/fig_mb_amp.png', dpi=110)
