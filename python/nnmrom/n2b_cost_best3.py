"""S4 (COST F3) Newmark wall time, best of three, for SOD-ROM Sec. 9.
   Same models, test and step as n2_cost.py: conservative oscillation from the 30 N static
   deflection, 0.3 s, average-acceleration Newmark at dt = 2e-5 s.
   Usage: OPENBLAS_NUM_THREADS=1 python3 n2b_cost_best3.py > n2b_out.txt"""
import numpy as np, time, platform, scipy
from nlsim import *
from n1_rom import sod, pod, Xd, Vd, PL, S, pt, n
x0 = static_deflection(S, 30.0); T = 0.3; dt = 2e-5
models = {'full': np.eye(n)}
for r in (2, 4):
    models[f'LNM r={r}'] = PL[:, :r]; models[f'SOD r={r}'] = sod(Xd, Vd, r)[0]; models[f'POD r={r}'] = pod(Xd, r)
print(f"python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, {platform.machine()}")
for nm, Phi in models.items():
    Mr = Phi.T @ S['M'] @ Phi; Kr = Phi.T @ S['K'] @ Phi; prt = Phi[S['tip']]; r = Phi.shape[1]
    q0 = np.linalg.pinv(Phi) @ x0; ts = []
    for _ in range(3):
        t0 = time.perf_counter(); newmark(Mr, Kr, np.zeros((r, r)), prt, S['k3'], q0, np.zeros(r), T, dt)
        ts.append(time.perf_counter() - t0)
    print(f"{nm:10s} best of 3: {min(ts):.3f} s  (runs {', '.join(f'{x:.3f}' for x in ts)})", flush=True)
