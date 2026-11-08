"""ทดสอบความลำเอียงเชิงโครงสร้าง (structural bias) แบบ Kononova et al. (2015) / Vermetten et al. (2022):
รันบนฟังก์ชัน f0(x) = U(0,1) (ไม่มีข้อมูลเชิงพื้นที่) ; ตำแหน่งคำตอบสุดท้ายควรกระจายสม่ำเสมอในโดเมน
ถ้าอัลกอริทึมไม่ลำเอียง. D=10, โดเมน [0,1]^D, 10,000 FE, 100 รอบ ; ทดสอบ KS ต่อมิติ (ได้ 1000 ค่า) และ
ระยะเฉลี่ยจากจุดกึ่งกลาง (ค่าคาดหวังภายใต้ uniform ≈ 0.889 ที่ D=10)
"""
import json, os, sys, numpy as np
from multiprocessing import Pool
from scipy.stats import kstest
from sfao import SFAO
from baselines import ALGORITHMS
from top_baselines import TOP

D, FE, RUNS = 10, 10_000, 100


def f0_factory(seed):
    r = np.random.default_rng(10_000 + seed)
    return lambda x: float(r.random())


def job(a):
    name, s = a
    f = f0_factory(s)
    if name == "SFAO":
        x, _, _ = SFAO(use_cocoon=True, food_per_worm=100, spin_mode="adaptive").optimize(f, D, 0.0, 1.0, FE, seed=s)
    else:
        x, _, _ = {**ALGORITHMS, **TOP}[name](f, D, 0.0, 1.0, FE, seed=s)
    return name, s, list(map(float, x))


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    algos = ["SFAO", "L-SHADE", "jSO", "PSO", "GWO", "WOA"]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as p:
        res = p.map(job, [(a, s) for a in algos for s in range(RUNS)])
    out = {}
    U = np.random.default_rng(0).random((200000, D))
    ref = np.linalg.norm(U - 0.5, axis=1).mean()
    for a in algos:
        X = np.array([x for n, s, x in res if n == a])
        p = [kstest(X[:, d], "uniform").pvalue for d in range(D)]
        out[a] = {"ks_p_min": float(min(p)), "ks_reject_dims_bonf": int(sum(v < 0.05 / D for v in p)),
                  "mean_dist_center": float(np.linalg.norm(X - 0.5, axis=1).mean()), "ref_uniform": float(ref),
                  "frac_within_0.1_of_center": float((np.abs(X - 0.5) < 0.1).all(1).mean())}
        print(a, out[a], flush=True)
    json.dump({"settings": {"D": D, "FE": FE, "runs": RUNS}, "results": out,
               "positions": {a: [x for n, s, x in res if n == a] for a in algos}}, open("output/bias_test.json", "w"))
