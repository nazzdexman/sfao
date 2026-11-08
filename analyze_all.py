"""สรุปรวมทุกชุด (CEC2022 D10, CEC2017 D10/D30) ด้วยกฎเดียวกับ analyze_cec.py / analyze_c17.py
   Wilcoxon rank-sum a=0.05 (เสมอถ้า allclose หรือ p>=0.05 ; ชนะ/แพ้ตัดสินด้วย median แล้ว mean) ; Friedman จากค่าเฉลี่ยต่อฟังก์ชัน
   ใช้:  python analyze_all.py            -> พิมพ์ผลและเขียน summary_all.json
"""
import json, collections, numpy as np
from scipy.stats import ranksums, rankdata, friedmanchisquare

SF = "SFAO:cocoon:aspin"
FILES = {"cec2022": ["cec30.jsonl", "swarm_pc.jsonl", "swarm_recent.jsonl", "clpso22.jsonl", "lsrtde_results.jsonl"],
         "cec2017": ["c17_full.jsonl", "c17_d30.jsonl", "c17_swarm.jsonl", "lsrtde_results.jsonl"]}
CHAMP = ["L-SRTDE", "jSO", "L-SHADE", "CMA-ES"]
ANIMAL = ["CLPSO", "ABC", "PSO", "GWO", "WOA", "MFO", "GJO", "COA", "SHO", "WaOA", "OOA"]


def load(suite, D):
    E = collections.defaultdict(dict)
    for fn in FILES[suite]:
        try:
            fh = open(fn)
        except FileNotFoundError:
            continue
        for l in fh:
            r = json.loads(l)
            if r.get("suite", "cec2022") != suite or r["D"] != D:
                continue
            a = "SFAO" if r["algo"] == SF else ("SFAO-proto" if r["algo"] == "SFAO" else r["algo"])
            e = r["err"]
            E[(a, r["f"])][r["seed"]] = 0.0 if e < 1e-8 else e
    return E


def wtl(x, y):
    if np.allclose(x, y) or ranksums(x, y).pvalue >= 0.05:
        return "T"
    if np.median(x) < np.median(y) or (np.median(x) == np.median(y) and x.mean() < y.mean()):
        return "W"
    return "L"


def analyse(suite, D, funcs):
    E = load(suite, D)
    algos = [a for a in ["SFAO"] + CHAMP + ANIMAL if all(len(E.get((a, f), {})) == 30 for f in funcs)]
    A = {(a, f): np.array([E[(a, f)][s] for s in range(30)]) for a in algos for f in funcs}
    M = {a: [A[(a, f)].mean() for f in funcs] for a in algos}
    res = {"algos": algos, "wtl": {}, "means": {a: dict(zip(map(str, funcs), M[a])) for a in algos}}
    for b in algos[1:]:
        c = collections.Counter(wtl(A[("SFAO", f)], A[(b, f)]) for f in funcs)
        res["wtl"][b] = f"{c['W']}/{c['T']}/{c['L']}"
    R = np.array([rankdata([M[a][i] for a in algos]) for i in range(len(funcs))]).mean(0)
    res["friedman"] = {algos[i]: round(float(R[i]), 2) for i in np.argsort(R)}
    G = [a for a in ["SFAO"] + ANIMAL if a in algos]
    RG = np.array([rankdata([M[a][i] for a in G]) for i in range(len(funcs))]).mean(0)
    res["friedman_animal"] = {G[i]: round(float(RG[i]), 2) for i in np.argsort(RG)}
    res["friedman_p"] = float(friedmanchisquare(*[M[a] for a in algos]).pvalue)
    return res


if __name__ == "__main__":
    from cec2017 import FUNCS
    out = {"CEC2022 D10": analyse("cec2022", 10, list(range(1, 13))),
           "CEC2017 D10": analyse("cec2017", 10, list(FUNCS)),
           "CEC2017 D30": analyse("cec2017", 30, list(FUNCS))}
    for k, r in out.items():
        print(f"\n===== {k}  ({len(r['algos'])} algorithms) =====")
        print("Friedman:", ", ".join(f"{a} {v}" for a, v in r["friedman"].items()))
        print("Animal group:", ", ".join(f"{a} {v}" for a, v in r["friedman_animal"].items()))
        print("SFAO W/T/L:", r["wtl"])
    json.dump(out, open("summary_all.json", "w"), indent=1)
