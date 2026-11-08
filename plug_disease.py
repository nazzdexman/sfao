"""
ทดสอบความทั่วไปของกลไกโรคระบาด: PSO vs PSO+Disease, DE vs DE+Disease
(PSO: inertia ลดเชิงเส้น 0.9->0.4, c1=c2=2;  DE: DE/rand/1/bin, F=0.5, CR=0.9)
"""
import numpy as np
from baselines import _Counter, _bounds
from disease import Disease


def pso_x(func, dim, lb, ub, max_fe, seed=None, N=30, disease=False):
    rng = np.random.default_rng(seed); lb, ub = _bounds(lb, ub, dim)
    ev = _Counter(func, max_fe); vmax = 0.2 * (ub - lb)
    X = lb + rng.random((N, dim)) * (ub - lb); V = np.zeros_like(X)
    F = np.array([ev(x) for x in X]); P, PF = X.copy(), F.copy()
    dis = Disease(N) if disease else None
    T = max_fe // N; t = 0
    while not ev.done:
        w = 0.9 - 0.5 * min(t / T, 1); g = P[np.argmin(PF)]
        V = w * V + 2 * rng.random((N, dim)) * (P - X) + 2 * rng.random((N, dim)) * (g - X)
        V = np.clip(V, -vmax, vmax)
        if dis is not None:
            V *= (1 - dis.lethargy())[:, None]          # ตัวป่วยเคลื่อนที่ช้าลง
            skip = dis.skip(rng)
        else:
            skip = np.zeros(N, bool)
        X = np.clip(X + V, lb, ub)
        PF_old = PF.copy()
        for i in range(N):
            if skip[i]:
                continue
            f = ev(X[i])
            if f < PF[i]: P[i], PF[i] = X[i].copy(), f
        if dis is not None:
            for c in dis.update(P, PF, PF_old, rng):
                if ev.done: break
                X[c] = dis.replacement(P[c], lb, ub, rng); V[c] = 0
                P[c] = X[c].copy(); PF[c] = ev(X[c]); dis.reset(c)
        t += 1
    return ev.result()


def de_x(func, dim, lb, ub, max_fe, seed=None, N=50, Fw=0.5, CR=0.9, disease=False):
    rng = np.random.default_rng(seed); lb, ub = _bounds(lb, ub, dim)
    ev = _Counter(func, max_fe)
    X = lb + rng.random((N, dim)) * (ub - lb); F = np.array([ev(x) for x in X])
    dis = Disease(N) if disease else None
    while not ev.done:
        F_old = F.copy()
        skip = dis.skip(rng) if dis is not None else np.zeros(N, bool)
        leth = dis.lethargy() if dis is not None else np.zeros(N)
        for i in range(N):
            if skip[i]:
                continue
            r1, r2, r3 = rng.choice(np.delete(np.arange(N), i), 3, replace=False)
            v = X[r1] + Fw * (1 - leth[i]) * (X[r2] - X[r3])
            v = np.where(v < lb, (lb + X[i]) / 2, np.where(v > ub, (ub + X[i]) / 2, v))
            mask = rng.random(dim) < CR; mask[rng.integers(dim)] = True
            u = np.where(mask, v, X[i]); fu = ev(u)
            if fu <= F[i]: X[i], F[i] = u, fu
        if dis is not None:
            for c in dis.update(X, F, F_old, rng):
                if ev.done: break
                X[c] = dis.replacement(X[c], lb, ub, rng); F[c] = ev(X[c]); dis.reset(c)
    return ev.result()


PLUG = {
    "PSO": lambda *a, **k: pso_x(*a, **k, disease=False),
    "PSO+Dis": lambda *a, **k: pso_x(*a, **k, disease=True),
    "DE": lambda *a, **k: de_x(*a, **k, disease=False),
    "DE+Dis": lambda *a, **k: de_x(*a, **k, disease=True),
}
