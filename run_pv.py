"""
หาค่าพารามิเตอร์เซลล์แสงอาทิตย์ (RTC France: SDM 5 พารามิเตอร์, DDM 7 พารามิเตอร์)
python run_pv.py [รอบ] [FE] [workers]          ค่าเริ่มต้น 30 รอบ, 50,000 FE, ทุกคอร์
python run_pv.py --report                      สรุปผล (ตาราง, Wilcoxon, Friedman, กราฟ)
ผลเขียนต่อท้าย pv_results.jsonl (รันต่อได้ถ้าหยุดกลางคัน)
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool

OUT = "pv_results.jsonl"
ALGOS = ["SFAO", "SFAO:cocoon", "SFAO:cocoon:aspin", "SFAO:cocoon:aspin:notoxic:nomolt", "L-SHADE", "jSO", "CMA-ES", "PSO", "GWO", "WOA"]
MODELS = ["SDM", "DDM"]


def job(args):
    from pv import scaled, BOUNDS
    from sfao import SFAO
    from top_baselines import TOP
    from baselines import ALGORITHMS
    algo, model, seed, fe = args
    f, D, lo, hi = scaled(model)
    t = time.time()
    if algo.startswith("SFAO"):
        from full_benchmark import get_algo          # อ่านชื่อแบบเดียวกับ CEC (cocoon, aspin, no<กลไก>, กฎความหนาแน่น)
        bx, bf, c = get_algo(algo)(f, D, 0.0, 1.0, fe, seed)
    elif algo in TOP:
        bx, bf, c = TOP[algo](f, D, 0.0, 1.0, fe, seed=seed)
    else:
        bx, bf, c = ALGORITHMS[algo](f, D, 0.0, 1.0, fe, seed=seed)
    c = np.asarray(c)
    if len(c) < fe: c = np.concatenate([c, np.full(fe - len(c), c[-1])])
    idx = np.unique(np.geomspace(50, fe, 50).astype(int)) - 1
    params = (lo + np.clip(np.asarray(bx), 0, 1) * (hi - lo)).tolist()
    return {"algo": algo, "model": model, "seed": seed, "fe": fe, "rmse": float(bf), "params": params,
            "curve_fe": (idx + 1).tolist(), "curve": c[idx].tolist(), "sec": time.time() - t}


def report():
    from scipy.stats import ranksums, rankdata
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from pv import BEST_KNOWN
    rs = [json.loads(l) for l in open(OUT)]
    algos = [a for a in ALGOS if any(r["algo"] == a for r in rs)]
    E = {(a, m): np.array([r["rmse"] for r in rs if r["algo"] == a and r["model"] == m]) for a in algos for m in MODELS}
    for m in MODELS:
        print(f"\n===== {m}  (ค่าดีที่สุดในวรรณกรรม RMSE ≈ {BEST_KNOWN[m]:.5e}) =====")
        print(f"{'algorithm':14s}{'best':>12s}{'mean':>12s}{'median':>12s}{'worst':>12s}{'std':>11s}  ถึงค่าดีสุด(±0.01%)")
        for a in algos:
            e = E[a, m]
            hit = int((e <= BEST_KNOWN[m] * 1.0001).sum())
            print(f"{a:14s}{e.min():12.5e}{e.mean():12.5e}{np.median(e):12.5e}{e.max():12.5e}{e.std():11.2e}  {hit}/{len(e)}")
        ref = next(a for a in ["SFAO:cocoon:aspin:notoxic:nomolt", "SFAO:cocoon:aspin", "SFAO:cocoon", "SFAO"] if a in algos)
        print(f"Wilcoxon ({ref} vs อื่น, + = {ref} ดีกว่า):")
        for a in algos:
            if a == ref: continue
            x, y = E[ref, m], E[a, m]
            s = "=" if (np.allclose(x, y) or ranksums(x, y).pvalue >= 0.05) else ("+" if np.median(x) < np.median(y) else "-")
            print(f"   vs {a:12s} {s}   (p={ranksums(x, y).pvalue:.2g})")
    rk = np.array([rankdata([E[a, m].mean() for a in algos]) for m in MODELS]).mean(0)
    print("\nFriedman mean rank (2 แบบจำลอง):", ", ".join(f"{algos[k]} {rk[k]:.2f}" for k in np.argsort(rk)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, m in zip(axes, MODELS):
        for a in algos:
            cs = [r for r in rs if r["algo"] == a and r["model"] == m]
            ax.plot(cs[0]["curve_fe"], np.median([r["curve"] for r in cs], axis=0), lw=2.4 if "SFAO" in a else 1.2, label=a)
        ax.axhline(BEST_KNOWN[m], color="k", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"RTC France {m}"); ax.grid(alpha=.3)
        ax.set_xlabel("function evaluations"); ax.set_ylabel("RMSE (median)")
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig("pv_convergence.png", dpi=120)
    print("บันทึก pv_convergence.png")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report(); sys.exit()
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    fe = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count()
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); done.add((r["algo"], r["model"], r["seed"]))
    jobs = [(a, m, s, fe) for s in range(runs) for m in MODELS for a in ALGOS if (a, m, s) not in done]
    print(f"{len(jobs)} งาน ({len(done)} เสร็จแล้ว), {workers} คอร์", flush=True)
    t0 = time.time()
    with Pool(workers) as pool, open(OUT, "a") as f:
        for n, r in enumerate(pool.imap_unordered(job, jobs), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            if n % 20 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"{n}/{len(jobs)}  {el/60:.1f} นาที (เหลือราว {el/n*(len(jobs)-n)/60:.0f} นาที)", flush=True)
    print("done")
