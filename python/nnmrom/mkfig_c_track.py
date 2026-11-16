"""Fig. (SOD-ROM): how well SOD follows the NNM1 backbone of the COST F3 beam (n12_backbone_track.py).
   (a) Backbones of two-mode ROMs (LNM, MTA, SOD of the 30 N static-release decay) and of the four-mode SOD ROM on the
       full-model frequency-energy plot; inset: high-energy end. (b) Frequency read from sliding windows of the decays
       (30 N and 120 N releases): configuration SOD (smoothest coordinate) and the state-SOD NNM1 plane (product rule).
   (c) Relative frequency error of the window estimates against the full-model backbone."""
import numpy as np, pickle, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import plotprefs as pp
pp.apply_preferences(sizes=(9, 8, 7)); c = pp.MATLAB
D = pickle.load(open('n12_results.pkl', 'rb')); Hm, fm = D['Hm'], D['fm']
print("cvd:", pp.cvd_check([c['darkred'], c['blue'], c['orange']]))
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.9))
a = ax[0]; a.semilogx(Hm, fm, 'k-', lw=2.4, label='full model')
sty = {('LNM', 2): (c['gray'], '--', 'LNM, $r=2$'), ('MTA', 2): (c['darkred'], '-', 'MTA, $r=2$'), ('SOD', 2): (c['blue'], '-', 'SOD, $r=2$'), ('SOD', 4): (c['orange'], ':', 'SOD, $r=4$')}
for k, (col, ls, lab) in sty.items():
    H, f = D['rom'][k]; m = ~((H > 3.5e-3) & (H < 4.5e-3)); a.semilogx(np.where(m, H, np.nan), f, ls, color=col, lw=1.0, label=lab)
a.set_xlim(5e-4, 6e-2); a.set_ylim(24, 48); a.set_xlabel('energy $H$ (J)'); a.set_ylabel('frequency (Hz)'); a.legend(frameon=False, fontsize=6.5, loc='upper left')
ins = a.inset_axes([0.5, 0.07, 0.47, 0.45]); ins.plot(Hm, fm, 'k-', lw=2.4)
for k, (col, ls, lab) in sty.items(): H, f = D['rom'][k]; ins.plot(H, f, ls, color=col, lw=1.0)
ins.set_xlim(0.04, 0.055); ins.set_ylim(44.0, 46.6); ins.set_xticks([0.04, 0.05]); ins.set_xticklabels(['0.04', '0.05']); ins.tick_params(labelsize=5.5)
b = ax[1]; b.semilogx(Hm, fm, 'k-', lw=1.6)
for P, mk in [(30.0, 'o'), (120.0, 's')]:
    R = D['track'][P]; b.semilogx(R[:, 0], R[:, 1], mk, color=c['orange'], ms=3, mfc='none', label=rf'configuration SOD, {P:.0f}\,N')
    b.semilogx(R[:, 0], R[:, 2], mk, color=c['blue'], ms=3, label=rf'state-SOD plane, {P:.0f}\,N')
b.set_xlim(1e-4, 6e-2); b.set_ylim(22, 48); b.set_xlabel('energy $H$ (J)'); b.set_ylabel('frequency (Hz)'); b.legend(frameon=False, fontsize=6, loc='upper left')
d = ax[2]
for P, mk in [(30.0, 'o'), (120.0, 's')]:
    R = D['track'][P]; g = np.isfinite(R[:, 3]) & ~((R[:, 0] > 3.5e-3) & (R[:, 0] < 4.5e-3))
    d.loglog(R[g, 0], abs(R[g, 1] / R[g, 3] - 1), mk, color=c['orange'], ms=3, mfc='none'); d.loglog(R[g, 0], abs(R[g, 2] / R[g, 3] - 1), mk, color=c['blue'], ms=3)
d.set_xlim(1e-4, 6e-2); d.set_ylim(1e-5, 1e-1); d.set_xlabel('energy $H$ (J)'); d.set_ylabel(r'$|f_{\mathrm{SOD}}/f_{\mathrm{full}}-1|$')
for k, lab in enumerate('abc'): ax[k].text(-0.2, 1.02, f'({lab})', transform=ax[k].transAxes)
pp.latexify(fig); fig.tight_layout(); fig.savefig('../../figures/fig_c_track.pdf'); fig.savefig('/tmp/claude-0/-home-claude/5f080035-e77c-5060-ad01-0b8c439e28f8/scratchpad/fig_c_track.png', dpi=110)
