"""Summary of c18_ensemble.py results: median errors per basis and order, ratios to LNM and MTA, data ranks."""
import numpy as np, pickle, sys, glob
for f in sorted(glob.glob('c18_case*.pkl')):
    D = pickle.load(open(f, 'rb')); res = D['res']; NR = D['NR']
    bases = sorted({b for (b, r, t) in res}); tests = sorted({t for (b, r, t) in res})
    print(f"\n######## {D['case']}  (realizations {NR})")
    print("max order per basis:", {b: max(r for (bb, r, t) in res if bb == b) for b in bases})
    for t in tests:
        print(f"-- test {t}: median [IQR/median]; r = 1..10")
        for b in bases:
            row = []
            for r in range(1, 11):
                v = res.get((b, r, t))
                row.append(f"{np.median(v):7.1e}" if v is not None else "   --  ")
            print(f"   {b:13s} " + " ".join(row))
        for ref in ['LNM', 'MTA', 'CM']:
            if not any(b == ref for b in bases): continue
            print(f"   ratio of medians to {ref}:")
            for b in bases:
                if b == ref: continue
                row = []
                for r in range(1, 11):
                    v = res.get((b, r, t)); w = res.get((ref, r, t))
                    row.append(f"{np.median(v)/np.median(w):7.2f}" if v is not None and w is not None else "   --  ")
                print(f"      {b:13s} " + " ".join(row))
    if D.get('unstable'):
        bad = {k: max(v) for k, v in D['unstable'].items() if max(v) > 0}
        print("unstable state ROMs:", bad if bad else "none")
