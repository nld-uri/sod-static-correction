"""Referee round 3, issue 1: is the mid-span transfer test fair? MTA in c18 holds the static vectors of BOTH load
   locations, so it contains the mid-span test load's static response, while the data bases saw only the tip.
   Same cases, seeds and draw order as c18_ensemble.py (default_rng(100+i): training load, tip test load, mid-span test
   load per realization; 20 realizations), configuration ROMs, test = random mid-span load (and the tip test), r = 1..10:
     LNM; MTA (tip + mid static vectors, as c18); MTA-tip (tip static vector only, as the data saw);
     SOD (rand) and POD (rand) trained at the tip (as c18);
     SOD (rand2), POD (rand2): trained on independent random loads at the tip AND mid-span (drawn after the c18 draws,
       so the c18 loads are reproduced exactly).
   Usage: python3 c28_fair_transfer.py <case 2..5>  -> c28_case<i>.pkl, printed medians."""
import numpy as np, sys, pickle, time
from systems import beam, state
from romlib import band_noise
from romlib2 import simulate_state, lnm, mta, pod, sod, rom_config, merr
ci = int(sys.argv[1]); NR = 20
CASES = {2: ('S2', 'light'), 3: ('S2', 'heavy'), 4: ('S3', 'dashpot'), 5: ('S3', 'dashpot_heavy')}
sysname, cname = CASES[ci]
S = beam() if sysname == 'S2' else beam(dashpot=(0.2, 1.0)); dt = 1e-3; N = 200000; wc = 250.0; RMAX = 10; DEC = 5
M, K, B = S['M'], S['K'], S['B']; C = S['C'][cname]; n = len(M); A, Bs = state(M, K, C, B)
rng = np.random.default_rng(100 + ci); res = {}; t0 = time.time()
fixed = {'LNM': lambda r: lnm(M, K, r), 'MTA': lambda r: mta(M, K, B, r), 'MTA-tip': lambda r: mta(M, K, B[:, :1], r)}
for rr in range(NR):
    u_tr = band_noise(N, dt, wc, 2, rng) * np.r_[1, 0]
    tests = {'rand': band_noise(N, dt, wc, 2, rng) * np.r_[1, 0], 'mid': band_noise(N, dt, wc, 2, rng) * np.r_[0, 1]}
    u_tr2 = band_noise(N, dt, wc, 2, rng)                                   # independent loads at tip and mid-span
    Xt = {k: simulate_state(A, Bs, u, dt)[::DEC, :n] for k, u in tests.items()}
    bases = dict(fixed)
    for tag, u in (('rand', u_tr), ('rand2', u_tr2)):
        Y = simulate_state(A, Bs, u, dt); X, V = Y[:, :n], Y[:, n:]
        Pp = pod(X, RMAX); Ps = sod(X, V, RMAX)[0]; del Y, X, V
        bases[f'POD ({tag})'] = (lambda r, Pp=Pp: Pp[:, :r]); bases[f'SOD ({tag})'] = (lambda r, Ps=Ps: Ps[:, :r])
    for bn, fb in bases.items():
        for r in range(1, RMAX + 1):
            Phi = fb(r)
            if Phi is None or Phi.shape[1] < r: continue
            for tk, u in tests.items():
                res.setdefault((bn, r, tk), []).append(merr(rom_config(M, K, C, B, Phi, u, dt, None, dec=DEC), Xt[tk], M))
    print(f"{sysname} {cname}: realization {rr + 1}/{NR} ({time.time() - t0:.0f}s)", flush=True)
pickle.dump(dict(case=CASES[ci], res=res, NR=NR), open(f'c28_case{ci}.pkl', 'wb'))
for tk in ('rand', 'mid'):
    print(f"== {sysname} {cname}, test {tk}: median error (number of realizations reaching the order), r = 1..{RMAX}")
    for bn in bases:
        print(f"   {bn:12s} " + " ".join(f"{np.median(res[(bn, r, tk)]):8.1e}" if len(res.get((bn, r, tk), [])) == NR else "      --" for r in range(1, RMAX + 1)))
