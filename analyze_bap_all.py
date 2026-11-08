"""สรุปผล BAP ทุกตัว (random-key: bap_results.jsonl + แบบลำดับ: bap_discrete.jsonl ถ้ามี)
makespan ในตาราง = นับแบบเดียวกับไฟล์ BKS (เวลาเสร็จ = จอด + h - 1) จึงลบ 1 จากค่าในไฟล์ผล"""
import json, os
import numpy as np
from scipy.stats import ranksums, rankdata
from bap import BAP

rs = [json.loads(l) for l in open("bap_results.jsonl")]
if os.path.exists("bap_discrete.jsonl"):
    rs += [json.loads(l) for l in open("bap_discrete.jsonl")]
order = ["SFAO", "SFAO:cocoon:aspin", "SFAO:cocoon:aspin:cap100", "SFAO:cocoon:aspin:notoxic:nomolt", "D-SFAO", "D-SFAO:nomate",
         "D-SFAO:nodisease", "GA", "ILS", "L-SHADE", "jSO", "CMA-ES", "PSO"]
algos = [a for a in order if any(r["algo"] == a for r in rs)]
insts = list(dict.fromkeys(r["inst"] for r in rs))
M = {(a, i): np.array([r["makespan"] - 1 for r in rs if r["algo"] == a and r["inst"] == i]) for a in algos for i in insts}
short = {"SFAO:cocoon:aspin": "SFAO+c+as", "SFAO:cocoon:aspin:cap100": "SFAO+c+as100", "SFAO:cocoon:aspin:notoxic:nomolt": "SFAO-final",
         "D-SFAO:nomate": "D-SFAO-mate", "D-SFAO:nodisease": "D-SFAO-dis"}
nm = lambda a: short.get(a, a)

print(f"=== BAP บาร์เซโลนา 2024: makespan เฉลี่ย (ดีสุด), runs={min(len(v) for v in M.values())} ===")
print(f"{'instance':12s}{'n':>4s}{'BKS':>5s}" + "".join(f"{nm(a):>14s}" for a in algos))
gap = {a: [] for a in algos}
csv = ["instance,n,BKS," + ",".join(f"{a} mean,{a} best" for a in algos)]
for i in insts:
    P = BAP(i); row = f"{i:12s}{P.n:4d}{P.bks:5.0f}"
    for a in algos:
        m = M[a, i]; row += f"{m.mean():8.1f}({m.min():3d})".rjust(14)
        gap[a].append(100 * (m.mean() - P.bks) / P.bks)
    print(row)
    csv.append(f"{i},{P.n},{P.bks}," + ",".join(f"{M[a, i].mean():.2f},{M[a, i].min()}" for a in algos))
open("bap_table_all.csv", "w").write("\n".join(csv))

print("\nช่องว่างเฉลี่ยจาก BKS (%), เรียงจากดีไปแย่:")
for a in sorted(algos, key=lambda a: np.mean(gap[a])):
    print(f"  {a:26s} {np.mean(gap[a]):6.2f}%")

for ref in [a for a in ["SFAO:cocoon:aspin:notoxic:nomolt", "SFAO:cocoon:aspin:cap100"] if a in algos]:
    print(f"\nWilcoxon: {ref} ชนะ/เสมอ/แพ้ (จาก {len(insts)} instance)")
    for a in algos:
        if a == ref: continue
        w = [0, 0, 0]
        for i in insts:
            x, y = M[ref, i], M[a, i]
            if np.all(x == y) or ranksums(x, y).pvalue >= 0.05: w[1] += 1
            elif x.mean() < y.mean(): w[0] += 1
            else: w[2] += 1
        print(f"  vs {a:26s} {w[0]} / {w[1]} / {w[2]}")

rk = np.array([rankdata([M[a, i].mean() for a in algos]) for i in insts]).mean(0)
print("\nFriedman mean rank:", ", ".join(f"{algos[k]} {rk[k]:.2f}" for k in np.argsort(rk)))
for i in insts:
    P = BAP(i)
    for a in algos:
        if M[a, i].min() < P.bks:
            print(f"  ** {a} ได้ดีกว่า BKS ใน {i}: {M[a, i].min()} < {P.bks:.0f}")
print("บันทึก bap_table_all.csv")
