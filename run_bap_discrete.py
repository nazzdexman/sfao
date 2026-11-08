"""
BAP รอบ 2: อัลกอริทึมแบบจัดลำดับ — D-SFAO เทียบ GA (ลำดับ) และ ILS
python run_bap_discrete.py [รอบ] [FE] [workers]   (รันต่อจากที่ค้างได้)
ผลเขียนที่ bap_discrete.jsonl  (เทียบกับ bap_results.jsonl ของอัลกอริทึมต่อเนื่องได้: ชุดข้อมูล/งบ/seed เดียวกัน)
"""
import json, sys, time
import numpy as np
from multiprocessing import Pool

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MAX_FE = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
INSTANCES = ["bcn_36A_7", "bcn_36A_10", "bcn_36A_14", "bcn_36A_15", "bcn_36A_16",
             "bcn_36A_20", "bcn_24B_34", "bcn_24B_22", "bcn_24B_44", "bcn_24B_15"]
ALGOS = ["D-SFAO", "D-SFAO:nomate", "D-SFAO:nodisease", "GA", "ILS"]


def job(args):
    from bap import BAP
    from dsfao_bap import DSFAO, ga_perm, ils
    algo, inst, seed = args
    P = BAP(inst); t = time.time()
    if algo.startswith("D-SFAO"):
        kw = {}
        if algo.endswith("nomate"): kw["mating"] = False
        if algo.endswith("nodisease"): kw["use_disease"] = False
        o, p, v, _ = DSFAO(**kw).optimize(P, MAX_FE, seed)
    elif algo == "GA":
        o, p, v, _ = ga_perm(P, MAX_FE, seed)
    else:
        o, p, v, _ = ils(P, MAX_FE, seed)
    start, pos = P.decode_order(o, p); P.check(start, pos)
    return {"algo": algo, "inst": inst, "seed": seed, "makespan": P.makespan(start),
            "total_completion": int((start + P.h).sum()), "start": start.tolist(),
            "pos": pos.tolist(), "sec": time.time() - t}


if __name__ == "__main__":
    out_file = "bap_discrete.jsonl"; done = set()
    try:
        for l in open(out_file):
            r = json.loads(l); done.add((r["algo"], r["inst"], r["seed"]))
    except FileNotFoundError:
        pass
    jobs = [(a, i, s) for s in range(RUNS) for i in INSTANCES for a in ALGOS if (a, i, s) not in done]
    print(f"{len(jobs)} งาน", flush=True); t0 = time.time()
    with Pool(WORKERS) as pool, open(out_file, "a") as out:
        for n, r in enumerate(pool.imap_unordered(job, jobs), 1):
            out.write(json.dumps(r) + "\n"); out.flush()
            if n % 10 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"{n}/{len(jobs)}  {el/60:.1f} นาที (เหลือราว {el/n*(len(jobs)-n)/60:.0f} นาที)", flush=True)
    print("done")
