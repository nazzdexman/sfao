"""
PSO รุ่นพัฒนา: CLPSO — Comprehensive Learning PSO
J. J. Liang, A. K. Qin, P. N. Suganthan, S. Baskar, "Comprehensive learning particle swarm optimizer
for global optimization of multimodal functions", IEEE Trans. Evol. Comput. 10(3):281-295, 2006.

ตามบทความต้นฉบับ:
  w ลดเชิงเส้น 0.9 -> 0.4 ; c = 1.49445 ; refreshing gap m = 7 ; Vmax = 0.2 * ช่วงค้นหา
  ความน่าจะเป็นการเรียนรู้ Pc_i = 0.05 + 0.45 * (exp(10(i-1)/(N-1)) - 1) / (exp(10) - 1)
  ตัวอย่าง (exemplar) ต่อมิติ: ถ้า rand < Pc_i ใช้ pbest ของผู้ชนะ tournament สองตัว (ไม่รวมตัวเอง) ไม่งั้นใช้ pbest ตัวเอง
  ถ้าทุกมิติเป็นของตัวเอง สุ่มหนึ่งมิติให้เรียนจากตัวอื่น ; สร้าง exemplar ใหม่เมื่อ pbest ไม่ดีขึ้นครบ m รุ่น
  ประเมินค่า (และนับ FE) เฉพาะเมื่ออนุภาคอยู่ในขอบเขต (ตามต้นฉบับ)
ขนาดประชากร: 10 สำหรับ D<=10, 40 สำหรับ D>10 (ตามการทดลองในบทความ)
"""
import numpy as np
from baselines import _Counter, _bounds


def clpso(func, dim, lb, ub, max_fe, seed=None, N=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    ev = _Counter(func, max_fe)
    N = N or (10 if dim <= 10 else 40)
    span = ub - lb
    vmax = 0.2 * span
    c, m = 1.49445, 7
    i_idx = np.arange(N)
    Pc = 0.05 + 0.45 * (np.exp(10 * i_idx / (N - 1)) - 1) / (np.exp(10) - 1)

    X = lb + rng.random((N, dim)) * span
    V = (rng.random((N, dim)) * 2 - 1) * vmax
    F = np.array([ev(x) for x in X])
    P, PF = X.copy(), F.copy()
    flag = np.zeros(N, int)
    max_gen = max(1, max_fe // N)

    def exemplar(i):
        ex = np.full(dim, i)
        learn = rng.random(dim) < Pc[i]
        for d in np.where(learn)[0]:
            a, b = rng.choice([j for j in range(N) if j != i], 2, replace=False)
            ex[d] = a if PF[a] < PF[b] else b
        if np.all(ex == i):
            d = rng.integers(dim)
            a, b = rng.choice([j for j in range(N) if j != i], 2, replace=False)
            ex[d] = a if PF[a] < PF[b] else b
        return ex

    EX = np.array([exemplar(i) for i in range(N)])
    gen, idle = 0, 0
    while not ev.done:
        w = 0.9 - 0.5 * min(gen, max_gen) / max_gen
        evaluated = 0
        for i in range(N):
            if ev.done:
                break
            if flag[i] >= m:
                EX[i] = exemplar(i); flag[i] = 0
            pb = P[EX[i], np.arange(dim)]
            V[i] = np.clip(w * V[i] + c * rng.random(dim) * (pb - X[i]), -vmax, vmax)
            X[i] = X[i] + V[i]
            if np.all(X[i] >= lb) and np.all(X[i] <= ub):
                fx = ev(X[i]); evaluated += 1
                if fx < PF[i]:
                    P[i], PF[i] = X[i].copy(), fx; flag[i] = 0
                else:
                    flag[i] += 1
            else:
                flag[i] += 1
        gen += 1
        idle = idle + 1 if evaluated == 0 else 0
        if idle > 50:                      # กันวนไม่รู้จบถ้าทุกตัวหลุดขอบ (ไม่เกิดในทางปฏิบัติ)
            X = np.clip(X, lb, ub); idle = 0
    return ev.result()


IMPROVED = {"CLPSO": clpso}
