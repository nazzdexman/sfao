"""
BAP รอบ 3: SFAO ตัวใหม่ (cocoon + ชักใยปรับวง) บนท่าเรือบาร์เซโลนา 2024
ชุดข้อมูล/งบ/seed เดียวกับรอบก่อน: 10 instance x 10 รอบ x 10,000 FE (encoding แบบ random-key)
python run_bap_full.py [รอบ] [FE] [workers]      (รันต่อจากที่ค้างได้)
ผลเขียนต่อท้าย bap_results.jsonl — ไฟล์ในแพ็กนี้มีผลรอบก่อนอยู่แล้ว 500 บรรทัด
(SFAO, L-SHADE, jSO, CMA-ES, PSO) สคริปต์จะข้ามงานที่เสร็จแล้ว รันเฉพาะตัวใหม่

ตัวใหม่ 2 แบบ (กำหนดไว้ก่อนเห็นผล):
  SFAO:cocoon:aspin         ฝูงเริ่มต้น 18*D ตามค่าเริ่มต้น (BAP D~42-118 => ฝูงเริ่ม 750-2,100 ตัว)
  SFAO:cocoon:aspin:cap100  จำกัดฝูงเริ่มต้นไม่เกิน 100 ตัว = เพดานเดียวกับที่ให้ L-SHADE/jSO ในการทดลอง BAP
"""
import json, sys, time
import numpy as np
from multiprocessing import Pool

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MAX_FE = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
INSTANCES = ["bcn_36A_7", "bcn_36A_10", "bcn_36A_14", "bcn_36A_15", "bcn_36A_16",
             "bcn_36A_20", "bcn_24B_34", "bcn_24B_22", "bcn_24B_44", "bcn_24B_15"]
ALGOS = ["SFAO:cocoon:aspin", "SFAO:cocoon:aspin:cap100", "SFAO:cocoon:aspin:notoxic:nomolt"]
NP_CAP = 100


def job(args):
    from bap import BAP
    from sfao import SFAO
    algo, inst, seed = args
    P = BAP(inst); t = time.time()
    if algo.endswith("cap100"):
        bx, bf, _ = SFAO(use_cocoon=True, spin_mode="adaptive", cocoon_cap=NP_CAP).optimize(
            P.objective, P.dim, 0.0, 1.0, MAX_FE, seed=seed)
    else:                                   # ชื่อแบบ CEC: cocoon => กฎความหนาแน่น (ฝูงเริ่ม <= งบ/100 = 100)
        from full_benchmark import get_algo
        bx, bf, _ = get_algo(algo)(P.objective, P.dim, 0.0, 1.0, MAX_FE, seed)
    start, pos = P.decode(np.asarray(bx)); P.check(start, pos)
    return {"algo": algo, "inst": inst, "seed": seed, "makespan": P.makespan(start),
            "total_completion": int((start + P.h).sum()), "start": start.tolist(),
            "pos": pos.tolist(), "sec": time.time() - t}


if __name__ == "__main__":
    out_file = "bap_results.jsonl"; done = set()
    try:
        for l in open(out_file):
            r = json.loads(l); done.add((r["algo"], r["inst"], r["seed"]))
    except FileNotFoundError:
        pass
    jobs = [(a, i, s) for s in range(RUNS) for i in INSTANCES for a in ALGOS if (a, i, s) not in done]
    print(f"{len(jobs)} งาน ({len(done)} เสร็จแล้ว), {WORKERS} คอร์", flush=True); t0 = time.time()
    with Pool(WORKERS) as pool, open(out_file, "a") as out:
        for n, r in enumerate(pool.imap_unordered(job, jobs), 1):
            out.write(json.dumps(r) + "\n"); out.flush()
            if n % 10 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"{n}/{len(jobs)}  {el/60:.1f} นาที (เหลือราว {el/n*(len(jobs)-n)/60:.0f} นาที)", flush=True)
    print("done")
