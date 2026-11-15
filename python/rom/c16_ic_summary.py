"""SOD-ROM: summary of free-decay training ICs (one realization; c10, c14, c15). Writes c16_ic_summary.txt.
   SR static release; EE equal energy over all modes below pi/(4 dt); BLc band-limited at omega_c; BLr order-matched."""
import pickle, numpy as np
names = ['S1 light', 'S1 heavy', 'S2 light', 'S2 heavy', 'S3 c_d=0.2', 'S3 c_d=1']
out = []
for ci in range(6):
    D = {10: pickle.load(open(f'c10_case{ci}.pkl', 'rb'))['res'], 14: pickle.load(open(f'c14_case{ci}.pkl', 'rb'))['res'],
         15: pickle.load(open(f'c15_case{ci}.pkl', 'rb'))['res']}
    pre = 'sSOD' if ci >= 4 else 'SOD'; ref = 'CM' if ci >= 4 else 'LNM'
    rows = [('ref ' + ref, 10, ref), ('SR', 10, f'{pre} (free)'), ('EE', 10, f'{pre} (freeEE)'), ('BLc', 14, f'{pre} (freeHAM)'), ('BLr', 15, f'{pre} (freeBL)')]
    if ci >= 4: rows += [('cfgSR', 10, 'SOD (free)'), ('cfgEE', 10, 'SOD (freeEE)')]
    for tk in ['rand', 'mid', 'free', 'freeEE']:
        if (ref, 1, tk) not in D[10]: continue
        out.append(f"== {names[ci]}, test {tk}, r = 1..10 (state dimension 2r)")
        for lab, src, bn in rows:
            v = [D[src].get((bn, r, tk), [np.nan])[0] for r in range(1, 11)]
            out.append(f"   {lab:7s} {bn:16s}" + " ".join(f"{x:8.1e}" for x in v))
open('c16_ic_summary.txt', 'w').write("\n".join(out)); print("\n".join(out))
