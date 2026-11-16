"""SOD-ROM, model-reduction track: basis selection with kinematics taken from the model (exact velocities and state
   derivatives; no differencing, no noise). Three ROM bases from the same SOD of a training record:
     R0  smoothness order (the plain SOD basis);
     RID modes selected for identification (sellib.select_config / pair_state, band up to wmax = 1.2 omega_c),
         ordered by frequency;
     RH  RID first, then the remaining coordinates in smoothness order (keeps quasi-static directions);
     RHB as RH, but only recognised modes inside the load band (frequency <= omega_c) go first: modes above the
         band respond quasi-statically to the load, and the smoother direction that represents that response
         serves the ROM better than the mode itself;
     RF  frequency-interleaved (suggestion of D. Chelidze): every coordinate is placed by an associated frequency,
         the peak (plane) frequency for recognised modes and the SOD frequency sqrt(mu) for the others, so a
         quasi-static direction enters where its SOV puts it among the modes;
     RQ  every coordinate placed by the Rayleigh-quotient frequency of its displacement shape, sqrt(phi^T K phi /
         phi^T M phi) (model-based, admissible for model reduction): a quasi-static direction is placed by the
         stiffness it carries, not by the load spectrum it follows.
   S2 light Rayleigh (configuration SOD): random tip load, rank tolerance 1e-10 and 1e-12.
   S3 c_d = 0.2 (state SOD): random tip load and a band-limited equal-energy decay (omega_b = omega_c), tolerance 1e-10.
   10 realizations (seed 21) of the random training and test loads; ROM orders r = 2, 4, 6, 8 (state dimension 2r);
   tests: new random tip load and random mid-span load. wmax = 1.2 omega_c bounds the identified band.
   Output: c21_out.txt, c21_results.pkl."""
import numpy as np, pickle
from scipy.linalg import eigh
from systems import beam, state
from romlib import band_noise
from romlib2 import simulate_state, rom_config, rom_state, merr
from sellib import gsod, select_config, pair_state, rom_order_config, rom_order_state
dt = 1e-3; N = 200000; wc = 250.0; NR = 10; rng = np.random.default_rng(21); wmax = 1.2 * wc
lines = []; out = {}
def log(s): print(s, flush=True); lines.append(s)
for tag, S, cname in [('S2', beam(), 'light'), ('S3', beam(dashpot=(0.2, 1.0)), 'dashpot')]:
    M, K, B = S['M'], S['K'], S['B']; C = S['C'][cname]; n = len(M); A, Bs = state(M, K, C, B)
    WE = np.block([[K, np.zeros((n, n))], [np.zeros((n, n)), M]]); wL2, PL = eigh(K, M); wL = np.sqrt(wL2)
    kEE = np.where(wL < np.pi / (4 * dt))[0]
    xbl = PL[:, kEE] @ (1 / (1 + (wL[kEE] / wc) ** 4) / wL[kEE]); xbl /= abs(xbl).max()
    for rr in range(NR):
        u = band_noise(N, dt, wc, 2, rng) * np.r_[1, 0]
        tests = {'rand': band_noise(N, dt, wc, 2, rng) * np.r_[1, 0], 'mid': band_noise(N, dt, wc, 2, rng) * np.r_[0, 1]}
        Xt = {k: simulate_state(A, Bs, v, dt)[::5, :n] for k, v in tests.items()}
        recs = {'random': (u, None)}
        if tag == 'S3' and rr == 0: recs['BL decay'] = (np.zeros((N, 2)), np.r_[xbl, np.zeros(n)])
        for rec, (uu, y0) in recs.items():
            Y = simulate_state(A, Bs, uu, dt, y0); Yd = Y @ A.T + uu @ Bs.T          # exact state derivative from the model
            tols = (1e-10, 1e-12) if tag == 'S2' else (1e-10,)
            for tol in tols:
                if tag == 'S2':
                    Psi, Phi, mu = gsod(Y[:, :n], Yd[:, :n], tol); Q = Y[:, :n] @ Psi
                    idx, pk, ws = select_config(Q, mu, dt, wmax, Phi=Phi)
                    ib = [i for i in idx if pk[i] <= wc]
                    fkey = np.sqrt(np.clip(mu, 0, None)); fkey[idx] = pk[idx]
                    orders = {'R0': list(range(Phi.shape[1])), 'RID': list(idx), 'RH': rom_order_config(idx, Phi.shape[1]), 'RHB': rom_order_config(ib, Phi.shape[1]),
                              'RF': list(np.argsort(fkey, kind='stable'))}
                    rq = np.array([np.sqrt((Phi[:, i] @ K @ Phi[:, i]) / (Phi[:, i] @ M @ Phi[:, i])) for i in range(Phi.shape[1])])
                    orders['RQ'] = list(np.argsort(rq, kind='stable'))
                    info = f"ID modes at {np.round(pk[idx], 1)}"
                else:
                    Psi, Phi, mu = gsod(Y, Yd, tol); Q = Y @ Psi
                    planes = pair_state(Q, dt, wmax, Phi=Phi)
                    orders = {'R0': list(range(Phi.shape[1])), 'RID': [i for p in planes for i in p[:2]], 'RH': rom_order_state(planes, Phi.shape[1]),
                              'RHB': rom_order_state([p for p in planes if p[2] <= wc], Phi.shape[1])}
                    fkey = np.sqrt(np.clip(mu, 0, None)).copy()
                    for a_, b_, f, c in planes: fkey[a_] = f; fkey[b_] = f * (1 + 1e-9)        # keep the two members adjacent
                    orders['RF'] = list(np.argsort(fkey, kind='stable'))
                    rq = np.array([np.sqrt((Phi[:n, i] @ K @ Phi[:n, i]) / max(Phi[:n, i] @ M @ Phi[:n, i], 1e-300)) for i in range(Phi.shape[1])])
                    for a_, b_, f, c in planes: g = np.sqrt(rq[a_] * rq[b_]); rq[a_] = g; rq[b_] = g * (1 + 1e-9)   # planes stay together
                    orders['RQ'] = list(np.argsort(rq, kind='stable'))
                    info = f"ID planes at {np.round([p[2] for p in planes], 1)}"
                if rr == 0: log(f"{tag} {rec} tol={tol:.0e}: {info}")
                for rn, od in orders.items():
                    o = out.setdefault((tag, rec, tol, rn), {})
                    for r in (2, 4, 6, 8):
                        k2 = r if tag == 'S2' else 2 * r
                        if len(od) < k2: continue
                        Pb = Phi[:, od[:k2]]
                        for tk, v in tests.items():
                            if tag == 'S2': Xr = rom_config(M, K, C, B, Pb / np.linalg.norm(Pb, axis=0), v, dt, dec=5)
                            else: Xr = rom_state(A, Bs, None, Pb, v, dt, None, n, W=WE, dec=5)[0]
                            o.setdefault((r, tk), []).append(merr(Xr, Xt[tk], M))
        log(f"{tag}: realization {rr + 1}/{NR}")
    for key, o in out.items():
        if key[0] != tag: continue
        log(f"{tag} {key[1]:9s} tol={key[2]:.0e} {key[3]:4s}: " + " ".join(
            f"r={r} {tk} {np.median(o[(r, tk)]):.1e}({len(o[(r, tk)])})" for r in (2, 4, 6, 8) for tk in ('rand', 'mid') if (r, tk) in o))
pickle.dump(out, open('c21_results.pkl', 'wb')); open('c21_out.txt', 'w').write("\n".join(lines) + "\n")
