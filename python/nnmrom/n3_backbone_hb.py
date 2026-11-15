"""COST F3 cantilever: conservative NNM1 backbone of the full FE model and of Galerkin ROMs by odd-harmonic balance
   with pseudo-arclength continuation (review item M8 of SOD-NNM, moved to SOD-ROM).
   x(t) = sum_k X_k cos(k w t), k = 1,3,...,2Kh-1;  (K - k^2 w^2 M) X_k + F_k = 0,  F = k3 (e_tip^T x)^3 e_tip,
   F_k by collocation (exact for the cubic with Nt > 4(2Kh-1)). Energy H = x0^T K x0 / 2 + k3 x_tip(0)^4 / 4, x0 = sum X_k.
   Continuation: secant predictor, Newton corrector on the augmented system with the arclength constraint."""
import numpy as np, pickle, sys
from scipy.linalg import eigh
from costbeam import costbeam

class HBC:
    def __init__(self, M, K, pt, k3, Kh=8):
        self.M, self.K, self.pt, self.k3, self.Kh, self.n = M, K, pt, k3, Kh, len(M)
        self.ks = np.arange(1, 2 * Kh, 2); Nt = 4 * (2 * Kh) + 8; self.Nt = Nt
        th = 2 * np.pi * np.arange(Nt) / Nt; self.C = np.cos(np.outer(th, self.ks))
    def RJ(self, z):
        n, Kh = self.n, self.Kh; X = z[:-1].reshape(Kh, n); w = z[-1]
        xt = self.C @ (X @ self.pt)                                  # tip displacement at collocation points
        Ft = (2 / self.Nt) * self.C.T @ (self.k3 * xt**3)           # harmonics of the scalar force
        R = np.einsum('ij,kj->ki', self.K, X) - (self.ks**2 * w**2)[:, None] * (X @ self.M.T) + np.outer(Ft, self.pt)
        dF = (2 / self.Nt) * np.einsum('tk,t,tl->kl', self.C, 3 * self.k3 * xt**2, self.C)   # dF_k / d xt_l (harmonic l)
        J = np.zeros((Kh * n, Kh * n + 1))
        P = np.outer(self.pt, self.pt)
        for i, k in enumerate(self.ks):
            J[i*n:(i+1)*n, i*n:(i+1)*n] += self.K - k**2 * w**2 * self.M
            for l in range(Kh): J[i*n:(i+1)*n, l*n:(l+1)*n] += dF[i, l] * P
            J[i*n:(i+1)*n, -1] = -2 * k**2 * w * (self.M @ X[i])
        return R.reshape(-1), J
    def energy(self, z):
        X = z[:-1].reshape(self.Kh, self.n); x0 = X.sum(0); return 0.5 * x0 @ self.K @ x0 + self.k3 / 4 * (self.pt @ x0)**4
    def tip_amp(self, z): return abs((z[:-1].reshape(self.Kh, self.n) @ self.pt).sum())
    def harm(self, z): X = z[:-1].reshape(self.Kh, self.n); return np.linalg.norm(X, axis=1) / np.linalg.norm(X[0])

def continue_backbone(hb, phi1, w1, Hmax, ds=0.002, maxsteps=20000, scale=None, dsmax=0.005):
    """start on the linear mode at small amplitude; arclength in scaled variables (X/scale, w/w1)"""
    n, Kh = hb.n, hb.Kh
    a0 = 1e-6 if scale is None else 1e-3 * scale
    z = np.zeros(Kh * n + 1); z[:n] = a0 * phi1; z[-1] = w1
    D = np.r_[np.full(Kh * n, 1.0 / (scale or 1.0)), 1.0 / w1]          # scaling of the arclength metric
    def corr(z, zp=None, t=None, ds=None, it=30):
        for _ in range(it):
            R, J = hb.RJ(z)
            if zp is None:   # fix tip fundamental amplitude for the first point
                g = z[:n] @ hb.pt - a0; Jg = np.zeros(len(z)); Jg[:n] = hb.pt
            else:
                g = (D * (z - zp)) @ t; Jg = D * t
            Ja = np.vstack([J, Jg]); Ra = np.r_[R, g]
            dz = np.linalg.lstsq(Ja, -Ra, rcond=None)[0]; z = z + dz
            if np.abs(dz).max() < 1e-12 * (1 + np.abs(z).max()): return z, True
        return z, np.abs(Ra).max() < 1e-8
    z, ok = corr(z); pts = [z.copy()]
    # second point: small amplitude step
    a1 = 1.5 * a0; z2 = z.copy(); z2[:-1] *= 1.5
    # correct z2 at fixed tip amplitude a1
    for _ in range(30):
        R, J = hb.RJ(z2); g = z2[:n] @ hb.pt - a1; Jg = np.zeros(len(z2)); Jg[:n] = hb.pt
        dz = np.linalg.lstsq(np.vstack([J, Jg]), -np.r_[R, g], rcond=None)[0]; z2 += dz
        if np.abs(dz).max() < 1e-12 * (1 + np.abs(z2).max()): break
    pts.append(z2.copy())
    for step in range(maxsteps):
        t = D * (pts[-1] - pts[-2]); t /= np.linalg.norm(t); zp = pts[-1] + ds * t / D
        zc, ok = corr(zp, zp=zp, t=t)
        if not ok:
            ds *= 0.5
            if ds < 1e-6: break
            continue
        pts.append(zc)
        ds = min(ds * 1.3, dsmax)
        if hb.energy(zc) > Hmax: break
    return pts

def correct_at_amp(hb, z, a):
    n = hb.n
    for _ in range(40):
        R, J = hb.RJ(z); g = z[:-1].reshape(hb.Kh, n).sum(0) @ hb.pt - a; Jg = np.zeros(len(z)); Jg[:-1] = np.tile(hb.pt, hb.Kh)
        dz = np.linalg.lstsq(np.vstack([J, Jg]), -np.r_[R, g], rcond=None)[0]; z = z + dz
        if np.abs(dz).max() < 1e-13 * (1 + np.abs(z).max()): return z, True
    return z, False

def branch_from(hb, z0, a0, da, Hmax, Hmin, scale, w1, dsmax=0.005, maxsteps=20000):
    """two points at tip amplitudes a0, a0+da, then arclength continuation until H leaves [Hmin, Hmax]"""
    D = np.r_[np.full(hb.Kh * hb.n, 1.0 / scale), 1.0 / w1]
    z1, ok1 = correct_at_amp(hb, z0, a0); z2, ok2 = correct_at_amp(hb, z1 * np.r_[np.full(len(z1) - 1, (a0 + da) / a0), 1.0], a0 + da)
    pts = [z1, z2]; ds = 0.002
    for _ in range(maxsteps):
        t = D * (pts[-1] - pts[-2]); t /= np.linalg.norm(t); zp = pts[-1] + ds * t / D; z = zp.copy(); ok = False
        for _ in range(30):
            R, J = hb.RJ(z); g = (D * (z - zp)) @ t
            dz = np.linalg.lstsq(np.vstack([J, D * t]), -np.r_[R, g], rcond=None)[0]; z = z + dz
            if np.abs(dz).max() < 1e-12 * (1 + np.abs(z).max()): ok = True; break
        if not ok:
            ds *= 0.5
            if ds < 1e-7: break
            continue
        pts.append(z); ds = min(ds * 1.3, dsmax)
        H = hb.energy(z)
        if H > Hmax or H < Hmin: break
    return pts

def full_backbone(hb, phi1, w1, Hmax=0.03, scale=1e-2):
    """low branch from the linear limit up to the 5:1 tongue; high branch from a coarse point above the tongue, both ways"""
    low = continue_backbone(hb, phi1, w1, Hmax=Hmax, scale=scale)
    coarse = continue_backbone(hb, phi1, w1, Hmax=Hmax, scale=scale, dsmax=0.2)
    zc = [z for z in coarse if hb.energy(z) > 8e-3 and hb.harm(z)[2] < 1e-2][0]; ac = (zc[:-1].reshape(hb.Kh, hb.n).sum(0) @ hb.pt)
    up = branch_from(hb, zc, ac, 1e-6, Hmax, 1e-9, scale, w1)
    down = branch_from(hb, zc, ac, -1e-6, Hmax, 1e-9, scale, w1, maxsteps=3000)
    return low, down[::-1] + up

if __name__ == "__main__":
    S = costbeam(); n = S['n']; pt = np.zeros(n); pt[S['tip']] = 1.0; k3 = S['k3']
    w2, P = eigh(S['K'], S['M']); phi1 = P[:, 0] / P[S['tip'], 0]; w1 = np.sqrt(w2[0])
    hb = HBC(S['M'], S['K'], pt, k3, Kh=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    low, high = full_backbone(hb, phi1, w1)
    pts = high
    Hl = np.array([hb.energy(z) for z in low]); fl = np.array([z[-1] for z in low]) / 2 / np.pi; h5l = np.array([hb.harm(z)[2] for z in low])
    print(f"low branch: {len(low)} points, f {fl.min():.3f}..{fl.max():.3f} Hz; max |X5|/|X1| {h5l.max():.2e}")
    H = np.array([hb.energy(z) for z in pts]); f = np.array([z[-1] for z in pts]) / 2 / np.pi; A = np.array([hb.tip_amp(z) for z in pts])
    h5 = np.array([hb.harm(z)[2] for z in pts])
    print(f"high branch: {len(pts)} points; H {H.min():.2e}..{H.max():.2e} J; f {f.min():.3f}..{f.max():.3f} Hz")
    for Ht in [6e-4, 2e-3, 5e-3, 1e-2, 1.5e-2]:
        i = np.argmin(abs(np.log(H / Ht))); print(f"  H~{H[i]:.2e}: f={f[i]:.4f} Hz, tip amp {A[i]*1e3:.3f} mm, |X5|/|X1|={h5[i]:.1e}")
    # tongue: non-monotone frequency or large 5th harmonic
    i5 = np.argmax(h5); print(f"max |X5|/|X1| = {h5[i5]:.2e} at f = {f[i5]:.3f} Hz, H = {H[i5]:.2e}")
    pickle.dump(dict(H=H, f=f, A=A, h5=h5, Kh=hb.Kh, Z=np.array(pts), low=dict(H=Hl, f=fl, h5=h5l, Z=np.array(low))), open(f'n3_backbone_full_Kh{hb.Kh}.pkl', 'wb'))
