"""Table (SOD-ROM v3, Sec. 10): cost at matched accuracy, from c36_results_<k>.pkl (c36_cost_matched.py).
   One row per model: omega_max, spectral radius of A_r, central-difference steps, and for a displacement error of 1e-6
   the least RK45 evaluations and Radau evaluations / LU factorizations, with the best-of-3 wall time of that run.
   Writes ../../SOD-ROM/tables/tab_c_cost.tex (LaTeX tabular body). Usage: python3 mktab_c_cost.py [machine label]"""
import pickle, sys, os
lab = sys.argv[1] if len(sys.argv) > 1 else ''
def g(x, d=0): return f"{x:,.{d}f}".replace(',', '\\,')
def sci(x):
    if x < 1e4: return g(x, 1 if x < 100 else 0)
    e = len(str(int(x))) - 1; return f"${x / 10 ** e:.1f}\\times10^{{{e}}}$"
rows = []
for k, sysname in ((0, 'S1'), (1, 'S2')):
    R = pickle.load(open(f'c36_results_{k}.pkl', 'rb')); first = True
    for name, d in R.items():
        rk, ra = d.get(('RK45', 1e-6)), d.get(('Radau', 1e-6))
        cells = [sysname if first else '', name.replace('r=', '$r=$').replace('$r=$4', '$r=4$').replace('$r=$8', '$r=8$'),
                 sci(d['wmax']), sci(d['rho']), g(d['cd']),
                 g(rk['nfev']) if rk else '--', f"{rk['time']:.2f}" if rk else '--',
                 g(ra['nfev']) if ra else '--', g(ra['nlu']) if ra else '--', f"{ra['time']:.2f}" if ra else '--']
        rows.append(' & '.join(cells) + r'\\'); first = False
    rows.append(r'\midrule')
rows = rows[:-1]
os.makedirs('../../SOD-ROM/tables', exist_ok=True)
open('../../SOD-ROM/tables/tab_c_cost.tex', 'w').write(
    r"\begin{tabular}{llrrrrrrrr}" "\n\\toprule\n"
    r"system & model & $\omega_{\max}$ & $\rho(\bm{A}_r)$ & CD steps & RK45 evals & RK45 (s) & Radau evals & Radau LU & Radau (s)\\" "\n\\midrule\n"
    + '\n'.join(rows) + "\n\\bottomrule\n\\end{tabular}\n" + (f"% times: {lab}\n" if lab else ''))
print('\n'.join(rows))
