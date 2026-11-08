"""
อัลกอริทึมคู่แข่งมาตรฐาน (ใช้งบ FE เท่ากันทุกตัว, minimization)
PSO, GWO, WOA, MFO, ABC  — คืนค่า (best_x, best_f, curve)
"""
import numpy as np


class _Counter:
    def __init__(self, func, max_fe):
        self.func, self.max_fe = func, max_fe
        self.fe, self.best_f, self.best_x, self.curve = 0, np.inf, None, []

    def __call__(self, x):
        if self.fe >= self.max_fe:
            return np.inf
        v = float(self.func(x))
        self.fe += 1
        if v < self.best_f:
            self.best_f, self.best_x = v, x.copy()
        self.curve.append(self.best_f)
        return v

    @property
    def done(self):
        return self.fe >= self.max_fe

    def result(self):
        return self.best_x, self.best_f, np.array(self.curve[:self.max_fe])


def _bounds(lb, ub, D):
    lb = np.full(D, lb, float) if np.isscalar(lb) else np.asarray(lb, float)
    ub = np.full(D, ub, float) if np.isscalar(ub) else np.asarray(ub, float)
    return lb, ub


def pso(func, dim, lb, ub, max_fe, seed=None, N=30):
    rng = np.random.default_rng(seed); lb, ub = _bounds(lb, ub, dim)
    ev = _Counter(func, max_fe); vmax = 0.2 * (ub - lb)
    X = lb + rng.random((N, dim)) * (ub - lb); V = np.zeros_like(X)
    F = np.array([ev(x) for x in X]); P, PF = X.copy(), F.copy()
    T = max_fe // N
    t = 0
    while not ev.done:
        w = 0.9 - 0.5 * min(t / T, 1); g = P[np.argmin(PF)]
        V = w * V + 2 * rng.random((N, dim)) * (P - X) + 2 * rng.random((N, dim)) * (g - X)
        V = np.clip(V, -vmax, vmax); X = np.clip(X + V, lb, ub)
        for i in range(N):
            f = ev(X[i])
            if f < PF[i]: P[i], PF[i] = X[i].copy(), f
        t += 1
    return ev.result()


def gwo(func, dim, lb, ub, max_fe, seed=None, N=30):
    rng = np.random.default_rng(seed); lb, ub = _bounds(lb, ub, dim)
    ev = _Counter(func, max_fe)
    X = lb + rng.random((N, dim)) * (ub - lb); F = np.array([ev(x) for x in X])
    T = max_fe // N; t = 0
    while not ev.done:
        idx = np.argsort(F); A_, B_, D_ = X[idx[0]], X[idx[1]], X[idx[2]]
        a = 2 - 2 * min(t / T, 1)
        newX = np.zeros_like(X)
        for leader in (A_, B_, D_):
            A = 2 * a * rng.random((N, dim)) - a; C = 2 * rng.random((N, dim))
            newX += leader - A * np.abs(C * leader - X)
        X = np.clip(newX / 3, lb, ub)
        F = np.array([ev(x) for x in X]); t += 1
    return ev.result()


def woa(func, dim, lb, ub, max_fe, seed=None, N=30):
    rng = np.random.default_rng(seed); lb, ub = _bounds(lb, ub, dim)
    ev = _Counter(func, max_fe)
    X = lb + rng.random((N, dim)) * (ub - lb); F = np.array([ev(x) for x in X])
    T = max_fe // N; t = 0
    while not ev.done:
        best = ev.best_x; a = 2 - 2 * min(t / T, 1)
        for i in range(N):
            A = 2 * a * rng.random() - a; C = 2 * rng.random(); p = rng.random(); l = rng.uniform(-1, 1)
            if p < 0.5:
                ref = best if abs(A) < 1 else X[rng.integers(N)]
                X[i] = ref - A * np.abs(C * ref - X[i])
            else:
                X[i] = np.abs(best - X[i]) * np.exp(l) * np.cos(2 * np.pi * l) + best
        X = np.clip(X, lb, ub); F = np.array([ev(x) for x in X]); t += 1
    return ev.result()


def mfo(func, dim, lb, ub, max_fe, seed=None, N=30):
    rng = np.random.default_rng(seed); lb, ub = _bounds(lb, ub, dim)
    ev = _Counter(func, max_fe)
    M = lb + rng.random((N, dim)) * (ub - lb); OM = np.array([ev(x) for x in M])
    idx = np.argsort(OM); Fl, OF = M[idx].copy(), OM[idx].copy()
    T = max_fe // N; t = 1
    while not ev.done:
        nf = int(round(N - t * (N - 1) / T)); nf = max(nf, 1)
        a = -1 - min(t / T, 1)
        for i in range(N):
            k = min(i, nf - 1)
            d = np.abs(Fl[k] - M[i]); l = (a - 1) * rng.random(dim) + 1
            M[i] = d * np.exp(l) * np.cos(2 * np.pi * l) + Fl[k]
        M = np.clip(M, lb, ub); OM = np.array([ev(x) for x in M])
        allx = np.vstack([Fl, M]); allf = np.concatenate([OF, OM])
        idx = np.argsort(allf)[:N]; Fl, OF = allx[idx].copy(), allf[idx].copy(); t += 1
    return ev.result()


def abc(func, dim, lb, ub, max_fe, seed=None, N=30):
    rng = np.random.default_rng(seed); lb, ub = _bounds(lb, ub, dim)
    ev = _Counter(func, max_fe); SN = N // 2; limit = SN * dim
    X = lb + rng.random((SN, dim)) * (ub - lb); F = np.array([ev(x) for x in X]); trial = np.zeros(SN)

    def fit(f):
        return 1 / (1 + f) if f >= 0 else 1 + abs(f)

    def try_neighbor(i):
        k = rng.integers(SN - 1); k += k >= i; j = rng.integers(dim)
        v = X[i].copy(); v[j] += rng.uniform(-1, 1) * (X[i, j] - X[k, j]); v = np.clip(v, lb, ub)
        fv = ev(v)
        if fv < F[i]: X[i], F[i], trial[i] = v, fv, 0
        else: trial[i] += 1

    while not ev.done:
        for i in range(SN): try_neighbor(i)
        fits = np.array([fit(f) for f in F]); p = fits / fits.sum()
        for _ in range(SN): try_neighbor(rng.choice(SN, p=p))
        s = int(np.argmax(trial))
        if trial[s] > limit:
            X[s] = lb + rng.random(dim) * (ub - lb); F[s] = ev(X[s]); trial[s] = 0
    return ev.result()


ALGORITHMS = {"PSO": pso, "GWO": gwo, "WOA": woa, "MFO": mfo, "ABC": abc}
