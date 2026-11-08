"""
ชุดทดลองเต็มสำหรับเปเปอร์ (ข้อ 3) — รันบนเครื่องแรง ๆ ได้เลย รันต่อจากเดิมได้ถ้าหยุดกลางคัน

ตัวอย่าง
  python full_benchmark.py --suite cec2022 --dims 10 20 --runs 30 --workers 16
  python full_benchmark.py --suite cec2017 --dims 10 30 50 --runs 30 --workers 32
  python full_benchmark.py --suite cec2022 --dims 10 --runs 30 --algos SFAO SFAO:nodisease SFAO:nomemory
  python full_benchmark.py --report            # สรุปผลจากไฟล์ที่มี (ตาราง, Wilcoxon, Friedman, กราฟ)

งบการประเมินตามกติกา: CEC2017 = 10,000 x D ; CEC2022 = 200,000 (D=10), 1,000,000 (D=20)
อัลกอริทึม: SFAO, L-SHADE, jSO, CMA-ES, PSO, PSO+Dis, DE, DE+Dis, ABC, GWO, WOA, MFO
           และ SFAO:no<กลไก> สำหรับ ablation (attract swing toxic crowd molt disease spin memory)
ต้องติดตั้ง: pip install numpy scipy matplotlib cma
CEC2022/CEC2017 ใช้โค้ดทางการ (cec2022_data, cec2017_data) — opfunu ให้ค่าไม่ตรงกับของทางการทั้งสองชุด
"""
import argparse, json, os, sys, time, warnings
import numpy as np
from multiprocessing import Pool

warnings.filterwarnings("ignore")
OUT = os.environ.get("SFAO_OUT", "full_results.jsonl")
N_CURVE = 60


def budget(suite, D):
    if suite == "cec2017":
        return 10_000 * D
    return {10: 200_000, 20: 1_000_000}.get(D, 10_000 * D)


def functions(suite):
    if suite == "cec2017":
        # โค้ด C ทางการผ่าน ctypes (cec2017.py) — 29 ฟังก์ชัน F1, F3–F30 (ตัด F2 ตามธรรมเนียม)
        from cec2017 import FUNCS
        return list(FUNCS)
    return list(range(1, 13))


def get_algo(name):
    from sfao import SFAO
    from top_baselines import TOP
    from baselines import ALGORITHMS
    from plug_disease import PLUG
    if name == "SFAO" or name.startswith("SFAO:"):
        kw = {}
        for part in name.split(":")[1:]:
            if part.startswith("no"):
                kw["use_" + part[2:]] = False
            elif part == "cocoon":
                kw["use_cocoon"] = True
                kw["food_per_worm"] = 100      # กฎความหนาแน่นการเลี้ยง: ฝูงเริ่ม <= งบ FE / 100
            elif part == "aspin":
                kw["spin_mode"] = "adaptive"
        return lambda f, D, lb, ub, fe, seed: SFAO(**kw).optimize(f, D, lb, ub, fe, seed=seed)
    for table in (TOP, PLUG, ALGORITHMS):
        if name in table:
            fn = table[name]
            return lambda f, D, lb, ub, fe, seed: fn(f, D, lb, ub, fe, seed=seed)
    from improved_baselines import IMPROVED
    if name in IMPROVED:
        fn = IMPROVED[name]
        return lambda f, D, lb, ub, fe, seed: fn(f, D, lb, ub, fe, seed=seed)
    from recent_baselines import RECENT          # ต้องติดตั้ง mealpy (pip install mealpy==3.0.3)
    if name in RECENT:
        fn = RECENT[name]
        return lambda f, D, lb, ub, fe, seed: fn(f, D, lb, ub, fe, seed=seed)
    raise ValueError(f"ไม่รู้จักอัลกอริทึม {name}")


def job(args):
    warnings.filterwarnings("ignore")
    suite, D, fid, algo, seed = args
    if suite == "cec2022":
        from cec2022 import CEC2022            # โค้ดทางการ (ตรวจแล้วตรงกับต้นฉบับ)
        func = CEC2022(fid, D)
    else:
        from cec2017 import CEC2017            # โค้ด C ทางการ (ตรวจแล้ว; opfunu ให้ค่าไม่ตรง)
        func = CEC2017(fid, D)
    fe = budget(suite, D)
    t = time.time()
    _, bf, c = get_algo(algo)(func, D, -100.0, 100.0, fe, seed)
    c = np.asarray(c)
    if len(c) < fe:
        c = np.concatenate([c, np.full(fe - len(c), c[-1])])
    idx = np.unique(np.geomspace(100, fe, N_CURVE).astype(int)) - 1
    err = lambda v: 0.0 if v < 1e-8 else float(v)
    return {"suite": suite, "D": D, "f": fid, "algo": algo, "seed": seed, "err": err(bf),
            "curve_fe": (idx + 1).tolist(), "curve": [err(v) for v in c[idx]], "sec": time.time() - t}


def safe_job(args):
    """เหมือน job แต่ถ้างานไหน error จะคืนข้อความ error แทน (งานอื่นรันต่อได้ ไม่ล้มทั้งชุด)"""
    import traceback
    try:
        return job(args)
    except Exception:
        return {"_error": True, "args": list(args), "trace": traceback.format_exc()}


def run(a):
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); done.add((r["suite"], r["D"], r["f"], r["algo"], r["seed"]))
    fl = a.funcs or functions(a.suite)
    jobs = [(a.suite, D, f, al, s) for s in range(a.runs) for D in a.dims for f in fl
            for al in a.algos if (a.suite, D, f, al, s) not in done]
    print(f"{len(jobs)} งานที่ต้องรัน ({len(done)} งานเสร็จแล้ว)", flush=True)
    t0 = time.time()
    errors = 0
    with Pool(a.workers) as pool, open(OUT, "a") as out:
        for n, r in enumerate(pool.imap_unordered(safe_job, jobs), 1):
            if r.get("_error"):
                errors += 1
                with open(OUT + ".errors.txt", "a") as ef:
                    ef.write(json.dumps(r["args"]) + "\n" + r["trace"] + "\n")
                print("ERROR in job", r["args"], "- see", OUT + ".errors.txt", flush=True)
                continue
            out.write(json.dumps(r) + "\n"); out.flush()
            if n % 20 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"{n}/{len(jobs)}  {el / 60:.1f} นาที  (เหลือราว {el / n * (len(jobs) - n) / 60:.0f} นาที)", flush=True)
    if errors:
        print(f"*** {errors} jobs failed - send {OUT}.errors.txt to the chat ***", flush=True)


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import ranksums, rankdata, friedmanchisquare
    rs = [json.loads(l) for l in open(OUT)]
    for suite, D in sorted({(r["suite"], r["D"]) for r in rs}):
        sub = [r for r in rs if r["suite"] == suite and r["D"] == D]
        algos = list(dict.fromkeys(r["algo"] for r in sub))
        fids = sorted({r["f"] for r in sub})
        E = {(al, f): np.array([r["err"] for r in sub if r["algo"] == al and r["f"] == f]) for al in algos for f in fids}
        print(f"\n######## {suite.upper()}  D={D}  runs={min(len(v) for v in E.values())} ########")
        print(f"{'F':5s}" + "".join(f"{al:>20s}" for al in algos))
        for f in fids:
            print(f"F{f:<4d}" + "".join(f"{E[al, f].mean():10.2e}±{E[al, f].std():7.1e}".rjust(20) for al in algos))
        ref = "SFAO" if "SFAO" in algos else algos[0]
        print(f"\nWilcoxon rank-sum α=0.05 ({ref} ชนะ/เสมอ/แพ้):")
        for al in algos:
            if al == ref: continue
            w = [0, 0, 0]
            for f in fids:
                x, y = E[ref, f], E[al, f]
                if len(x) == 0 or len(y) == 0: continue
                if np.allclose(x, y) or ranksums(x, y).pvalue >= 0.05: w[1] += 1
                elif np.median(x) < np.median(y) or (np.median(x) == np.median(y) and x.mean() < y.mean()): w[0] += 1
                else: w[2] += 1
            print(f"  vs {al:16s} {w[0]:3d} / {w[1]:3d} / {w[2]:3d}")
        M = np.array([[E[al, f].mean() for al in algos] for f in fids])
        rk = np.array([rankdata(row) for row in M]).mean(0)
        print("Friedman mean rank:", ", ".join(f"{algos[k]} {rk[k]:.2f}" for k in np.argsort(rk)))
        if len(algos) >= 3:
            print(f"Friedman test p-value = {friedmanchisquare(*M.T).pvalue:.3g}")
        cols = plt.cm.tab10.colors
        nc = 4; nr = int(np.ceil(len(fids) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.2 * nr), squeeze=False)
        for ax, f in zip(axes.flat, fids):
            for k, al in enumerate(algos):
                cs = [r for r in sub if r["algo"] == al and r["f"] == f]
                y = np.median([r["curve"] for r in cs], axis=0)
                ax.plot(cs[0]["curve_fe"], np.maximum(y, 1e-8), color=cols[k % 10],
                        lw=2.4 if al == "SFAO" else 1.3, label=al)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"F{f}", fontsize=10); ax.grid(alpha=.3)
        for ax in axes.flat[len(fids):]: ax.axis("off")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{suite.upper()} D={D}: median error", fontsize=12); fig.tight_layout()
        fig.savefig(f"conv_{suite}_D{D}.png", dpi=110); plt.close(fig)
        print(f"บันทึก conv_{suite}_D{D}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["cec2017", "cec2022"], default="cec2022")
    ap.add_argument("--dims", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--algos", nargs="+", default=["SFAO", "L-SHADE", "jSO", "CMA-ES"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--funcs", type=int, nargs="+", default=None, help="เลือกเฉพาะบางฟังก์ชัน")
    ap.add_argument("--out", default=None, help="ไฟล์ผลลัพธ์ (ค่าเริ่มต้น full_results.jsonl)")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.out:
        OUT = a.out
    report() if a.report else run(a)
