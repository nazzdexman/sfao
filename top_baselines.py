"""
คู่แข่งระดับท็อป (minimization, งบ FE เท่ากัน) — คืนค่า (best_x, best_f, curve)
  L-SHADE  : Tanabe & Fukunaga (2014), ผู้ชนะ CEC2014
  jSO      : Brest, Maučec & Bošković (2017), อันดับ 2 CEC2017
  CMA-ES   : Hansen, ใช้แพ็กเกจ cma แบบ IPOP (เพิ่มประชากรเมื่อรีสตาร์ต)
"""
import numpy as np
from baselines import _Counter, _bounds


def _lehmer(vals, w):
    return np.sum(w * vals ** 2) / max(np.sum(w * vals), 1e-300)


def _shade_core(func, dim, lb, ub, max_fe, seed, variant, np_cap=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    ev = _Counter(func, max_fe)
    if variant == "lshade":
        NP_init, H, arc_rate = int(round(18 * dim)), 6, 2.6
        MF, MCR = np.full(H, 0.5), np.full(H, 0.5)
    else:  # jSO
        NP_init, H, arc_rate = int(round(25 * np.log(dim) * np.sqrt(dim))), 5, 1.0
        MF, MCR = np.full(H, 0.3), np.full(H, 0.8)
    if np_cap:
        NP_init = min(NP_init, np_cap)
    NP_min = 4
    NP = NP_init
    X = lb + rng.random((NP, dim)) * (ub - lb)
    F = np.array([ev(x) for x in X])
    A = np.empty((0, dim))
    k = 0
    while not ev.done:
        prog = ev.fe / max_fe
        if variant == "lshade":
            p = 0.11
        else:
            p = 0.25 - 0.125 * prog
        r = rng.integers(H, size=NP)
        mcr, mf = MCR[r], MF[r]
        if variant == "jso":
            mcr = np.where(r == H - 1, 0.9, mcr); mf = np.where(r == H - 1, 0.9, mf)
        CR = np.where(mcr < 0, 0.0, np.clip(rng.normal(mcr, 0.1), 0, 1))
        if variant == "jso":
            CR = np.where(prog < 0.25, np.maximum(CR, 0.7), np.where(prog < 0.5, np.maximum(CR, 0.6), CR))
        Fs = np.empty(NP)
        for i in range(NP):
            f = -1
            while f <= 0:
                f = mf[i] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
            Fs[i] = min(f, 1.0)
        if variant == "jso":
            Fs = np.where((prog < 0.6) & (Fs > 0.7), 0.7, Fs)
        order = np.argsort(F)
        n_p = max(2, int(round(p * NP)))
        pool = np.vstack([X, A]) if len(A) else X
        U = np.empty_like(X)
        for i in range(NP):
            pb = X[order[rng.integers(n_p)]]
            r1 = rng.integers(NP)
            while r1 == i: r1 = rng.integers(NP)
            r2 = rng.integers(len(pool))
            while r2 == i or r2 == r1: r2 = rng.integers(len(pool))
            if variant == "jso":
                Fw = Fs[i] * (0.7 if prog < 0.2 else 0.8 if prog < 0.4 else 1.2)
                v = X[i] + Fw * (pb - X[i]) + Fs[i] * (X[r1] - pool[r2])
            else:
                v = X[i] + Fs[i] * (pb - X[i]) + Fs[i] * (X[r1] - pool[r2])
            v = np.where(v < lb, (lb + X[i]) / 2, v)
            v = np.where(v > ub, (ub + X[i]) / 2, v)
            mask = rng.random(dim) < CR[i]
            mask[rng.integers(dim)] = True
            U[i] = np.where(mask, v, X[i])
        FU = np.array([ev(u) for u in U])
        if ev.done and np.all(np.isinf(FU)):
            break
        better = FU < F
        succ = FU <= F
        if better.any():
            dF = F[better] - FU[better]
            w = dF / dF.sum()
            A = np.vstack([A, X[better]])
            sCR, sF = CR[better], Fs[better]
            if variant == "jso":
                pos = k if k != H - 1 else 0
            else:
                pos = k
            if MCR[pos] < 0 or sCR.max() == 0:
                MCR[pos] = -1
            else:
                new = _lehmer(sCR, w)
                MCR[pos] = new if variant == "lshade" else (new + MCR[pos]) / 2
            new = _lehmer(sF, w)
            MF[pos] = new if variant == "lshade" else (new + MF[pos]) / 2
            k = (k + 1) % (H if variant == "lshade" else H - 1)
        X[succ], F[succ] = U[succ], FU[succ]
        # ลดขนาดประชากรเชิงเส้น (LPSR)
        newNP = int(round((NP_min - NP_init) * ev.fe / max_fe + NP_init))
        if newNP < NP:
            keep = np.argsort(F)[:max(newNP, NP_min)]
            X, F = X[keep], F[keep]
            NP = len(X)
        amax = int(round(arc_rate * NP))
        if len(A) > amax:
            A = A[rng.choice(len(A), amax, replace=False)]
    return ev.result()


def lshade(func, dim, lb, ub, max_fe, seed=None, np_cap=None):
    return _shade_core(func, dim, lb, ub, max_fe, seed, "lshade", np_cap)


def jso(func, dim, lb, ub, max_fe, seed=None, np_cap=None):
    return _shade_core(func, dim, lb, ub, max_fe, seed, "jso", np_cap)


class _Budget(Exception):
    pass


def cmaes(func, dim, lb, ub, max_fe, seed=None):
    import cma
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    ev = _Counter(func, max_fe)

    def f(x):
        if ev.done:
            raise _Budget
        return ev(np.asarray(x))

    x0 = lb + rng.random(dim) * (ub - lb)
    opts = {"bounds": [list(lb), list(ub)], "verbose": -9, "seed": int(rng.integers(1, 2**31 - 1)),
            "maxfevals": max_fe, "tolfun": 1e-12, "tolx": 1e-12}
    try:
        cma.fmin2(f, x0, 0.3 * float(np.mean(ub - lb)), opts, restarts=20, incpopsize=2,
                  eval_initial_x=False)
    except _Budget:
        pass
    return ev.result()


TOP = {"L-SHADE": lshade, "jSO": jso, "CMA-ES": cmaes}
