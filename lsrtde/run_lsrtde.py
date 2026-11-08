"""
รัน L-SRTDE (โค้ด C++ ของผู้แต่ง: github.com/VladimirStanovov/L-SRTDE_CEC-2024, commit fa72910)
ผ่าน lsrtde_port/lsrtde17|22 (สร้างโดย make_driver.py ; ดู driver.diff) แล้วบันทึกเป็น jsonl รูปแบบเดียวกับ full_benchmark
  CEC2017: F1,F3-F30, D=10 (100,000 FE) และ D=30 (300,000 FE) ; CEC2022: F1-F12, D=10 (200,000 FE) ; seed 0-29
  error = best ภายในงบ - f* ; < 1e-8 ถือเป็น 0 (กฎเดียวกับ full_benchmark)
"""
import json, os, subprocess, sys
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
PORT = os.path.join(HERE, "build")
OUT = os.path.join(os.path.dirname(HERE), "results", "lsrtde_results.jsonl")
N_CURVE = 60


def job(a):
    suite, D, f, seed = a
    b = "17" if suite == "cec2017" else "22"
    fe = 10_000 * D if suite == "cec2017" else 200_000
    p = subprocess.run([os.path.join(PORT, "lsrtde" + b), str(f), str(D), str(fe), str(seed)],
                       capture_output=True, text=True, cwd=os.path.join(PORT, "run" + b), check=True)
    lines = {l.split()[0]: l.split()[1:] for l in p.stdout.strip().splitlines()}
    err = float(lines["FINAL"][0])
    steps = np.array(list(map(float, lines["CURVE"])))           # 1000 จุดที่ k*fe/1000
    idx = np.unique(np.geomspace(100, fe, N_CURVE).astype(int))   # จุดเดียวกับ full_benchmark
    pos = np.clip(np.ceil(idx / (fe / 1000)).astype(int) - 1, 0, 999)
    return {"suite": suite, "D": D, "f": f, "algo": "L-SRTDE", "seed": seed, "err": err,
            "nfe": int(lines["NFE"][0]), "curve_fe": idx.tolist(),
            "curve": [0.0 if v < 1e-8 else float(v) for v in steps[pos]],
            "curve_note": "from author's 1000-step record (value at the next record point)"}


if __name__ == "__main__":
    from cec2017 import FUNCS
    jobs = [("cec2017", D, f, s) for D in (10, 30) for f in FUNCS for s in range(30)] + \
           [("cec2022", 10, f, s) for f in range(1, 13) for s in range(30)]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); done.add((r["suite"], r["D"], r["f"], r["seed"]))
    jobs = [j for j in jobs if j not in done]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()) as pool, open(OUT, "a") as fo:
        for n, r in enumerate(pool.imap_unordered(job, jobs), 1):
            fo.write(json.dumps(r) + "\n"); fo.flush()
            if n % 200 == 0:
                print(n, "/", len(jobs), flush=True)
    print("DONE", flush=True)
