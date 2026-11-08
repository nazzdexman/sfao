"""สถิติเพิ่มเติมสำหรับต้นฉบับ IJIES (ทุกค่าคำนวณจากไฟล์ผลรายรอบ)
  1) Friedman + post-hoc Holm (SFAO เป็นตัวควบคุม; z-test ตาม Derrac et al. 2011) + Nemenyi CD
  2) W/T/L แยกประเภทฟังก์ชัน เทียบแชมป์ 4 ตัว
  3) Vargha-Delaney A12 ของ SFAO เทียบแชมป์ ต่อฟังก์ชัน
  4) อัตราสำเร็จ F2 + Wilson 95% CI
  5) Ablation (รุ่นที่ล็อกเทียบรุ่นตัดทีละกลไก) + ART ANOVA 2x2 (toxic x molt)
  6) PV: RMSE แบบ exact (Lambert W) ของพารามิเตอร์ที่ได้
ใช้: python paper_stats.py  -> paper/stats.json และรูปใน paper/
"""
import json, os, collections, math
import numpy as np
from scipy.stats import ranksums, rankdata, norm, studentized_range, f as fdist
from scipy.special import lambertw
import analyze_all as AA

OUT = "output"; os.makedirs(OUT, exist_ok=True)
SF = AA.SF
CH = ["L-SRTDE", "jSO", "L-SHADE", "CMA-ES"]
CLASSES = {"cec2022": {"Unimodal": [1], "Basic": [2, 3, 4, 5], "Hybrid": [6, 7, 8], "Composition": [9, 10, 11, 12]},
           "cec2017": {"Unimodal": [1, 3], "Simple multimodal": list(range(4, 11)),
                       "Hybrid": list(range(11, 21)), "Composition": list(range(21, 31))}}
SUITES = [("CEC2022 D10", "cec2022", 10), ("CEC2017 D10", "cec2017", 10), ("CEC2017 D30", "cec2017", 30)]


def arrays(suite, D, funcs):
    E = AA.load(suite, D)
    algos = [a for a in ["SFAO"] + AA.CHAMP + AA.ANIMAL
             if all(len(E.get((a, f), {})) == 30 for f in funcs)]
    return algos, {(a, f): np.array([E[(a, f)][s] for s in range(30)]) for a in algos for f in funcs}


def holm(algos, M, n):
    k = len(algos)
    R = np.array([rankdata(row) for row in M]).mean(0)
    se = math.sqrt(k * (k + 1) / (6 * n))
    c = algos.index("SFAO")
    rows = []
    for j, a in enumerate(algos):
        if j == c: continue
        z = (R[j] - R[c]) / se
        rows.append([a, float(R[j]), float(z), float(2 * norm.sf(abs(z)))])
    rows.sort(key=lambda r: r[3])
    m = len(rows); adj = 0
    for i, r in enumerate(rows):
        adj = max(adj, min(1, (m - i) * r[3])); r.append(adj)
    q = studentized_range.ppf(0.95, k, np.inf) / math.sqrt(2)
    return {"ranks": {a: round(float(R[i]), 2) for i, a in enumerate(algos)}, "CD": q * se,
            "holm": [{"algo": r[0], "rank": round(r[1], 2), "z": round(r[2], 3), "p": r[3], "p_holm": r[4],
                      "sig": r[4] < 0.05} for r in rows]}


def a12(x, y):   # P(SFAO < other) + 0.5 P(tie)  (minimisation: >0.5 favours SFAO)
    gt = (x[:, None] < y[None, :]).mean(); eq = (x[:, None] == y[None, :]).mean()
    return float(gt + 0.5 * eq)


def wilson(s, n, z=1.959964):
    p = s / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, c - h), min(1, c + h)]


def art_2x2(y, A, B):
    """Aligned Rank Transform (Wobbrock et al. 2011) สำหรับแผนแฟกทอเรียล 2x2 สมดุล"""
    y = np.asarray(y, float); A = np.asarray(A); B = np.asarray(B)
    g = y.mean(); cell = {(a, b): y[(A == a) & (B == b)].mean() for a in (0, 1) for b in (0, 1)}
    resid = y - np.array([cell[(a, b)] for a, b in zip(A, B)])
    mA = {a: y[A == a].mean() for a in (0, 1)}; mB = {b: y[B == b].mean() for b in (0, 1)}
    effA = np.array([mA[a] - g for a in A]); effB = np.array([mB[b] - g for b in B])
    effAB = np.array([cell[(a, b)] - mA[a] - mB[b] + g for a, b in zip(A, B)])
    out = {}
    for name, eff in (("A", effA), ("B", effB), ("AB", effAB)):
        r = rankdata(resid + eff)
        gr = r.mean(); cr = {(a, b): r[(A == a) & (B == b)].mean() for a in (0, 1) for b in (0, 1)}
        n = len(r)
        ssA = sum(((A == a).sum()) * (r[A == a].mean() - gr) ** 2 for a in (0, 1))
        ssB = sum(((B == b).sum()) * (r[B == b].mean() - gr) ** 2 for b in (0, 1))
        ssC = sum(((A == a) & (B == b)).sum() * (cr[(a, b)] - gr) ** 2 for a in (0, 1) for b in (0, 1))
        ssAB = ssC - ssA - ssB
        sse = sum(((r[(A == a) & (B == b)] - cr[(a, b)]) ** 2).sum() for a in (0, 1) for b in (0, 1))
        dfe = n - 4; ss = {"A": ssA, "B": ssB, "AB": ssAB}[name]
        F = ss / (sse / dfe) if sse > 0 else 0.0
        out[name] = {"F": float(F), "p": float(fdist.sf(F, 1, dfe)) if sse > 0 else 1.0,
                     "eta2p": float(ss / (ss + sse)) if ss + sse > 0 else 0.0}
    return out


def load_rows(files):
    rows = []
    for fn in files:
        for l in open(fn):
            rows.append(json.loads(l))
    return rows


def pv_exact(params, model):
    import pv
    if model != "SDM":
        return None
    Iph, I0, Rs, Rsh, n = params
    a = n * pv.Vt
    arg = Rs * I0 * Rsh / (a * (Rs + Rsh)) * np.exp(Rsh * (Rs * Iph + Rs * I0 + pv.V) / (a * (Rs + Rsh)))
    Ic = (Rsh * (Iph + I0) - pv.V) / (Rs + Rsh) - a / Rs * np.real(lambertw(arg))
    return float(np.sqrt(np.mean((Ic - pv.I) ** 2)))


if __name__ == "__main__":
    from cec2017 import FUNCS
    S = {}
    for name, suite, D in SUITES:
        funcs = list(range(1, 13)) if suite == "cec2022" else list(FUNCS)
        algos, A = arrays(suite, D, funcs)
        M = np.array([[A[(a, f)].mean() for a in algos] for f in funcs])
        res = holm(algos, M, len(funcs))
        cls = {}
        for b in CH:
            cls[b] = {}
            for cname, fl in CLASSES[suite].items():
                fl = [f for f in fl if f in funcs]
                c = collections.Counter(AA.wtl(A[("SFAO", f)], A[(b, f)]) for f in fl)
                cls[b][cname] = f"{c['W']}/{c['T']}/{c['L']}"
        eff = {b: {str(f): round(a12(A[("SFAO", f)], A[(b, f)]), 3) for f in funcs} for b in CH}
        effsum = {}
        for b in CH:
            v = np.array(list(eff[b].values()))
            effsum[b] = {"large_for_SFAO": int((v >= 0.714).sum()), "large_against": int((v <= 0.286).sum()),
                         "negligible": int((abs(v - 0.5) < 0.064).sum())}
        wtl_all = {b: AA.analyse(suite, D, funcs)["wtl"].get(b) for b in CH}
        S[name] = {"algos": algos, **res, "class_wtl": cls, "A12": eff, "A12_summary": effsum,
                   "means": {a: {str(f): float(A[(a, f)].mean()) for f in funcs} for a in algos},
                   "sd": {a: {str(f): float(A[(a, f)].std(ddof=1)) for f in funcs} for a in algos},
                   "median": {a: {str(f): float(np.median(A[(a, f)])) for f in funcs} for a in algos}}
        if suite == "cec2022":
            S[name]["F2_success"] = {a: {"s": int((A[(a, 2)] < 1e-3).sum()),
                                         "ci": wilson(int((A[(a, 2)] < 1e-3).sum()), 30)} for a in algos}

    # ---- ablation (CEC2022 D10) ----
    rows = load_rows(["cec30.jsonl", "abl_pc.jsonl", "abl_mac.jsonl", "cell_2x2_pc.jsonl"])
    E = collections.defaultdict(dict)
    for r in rows:
        if r.get("suite", "cec2022") == "cec2022" and r["D"] == 10:
            E[(r["algo"], r["f"])][r["seed"]] = 0.0 if r["err"] < 1e-8 else r["err"]
    G = lambda a, f: np.array([E[(a, f)][s] for s in range(30)])
    abl = {}
    for tag, a in [("M1 attraction", "noattract"), ("M2 head-swing operators", "noswing"),
                   ("M3 avoidance memory", "notoxic"), ("M4 crowding/depletion", "nocrowd"),
                   ("M5 molting event", "nomolt"), ("M6 disease/culling", "nodisease"),
                   ("M7 adaptive spinning", "nospin"), ("M8 direction memory", "nomemory")]:
        al = SF + ":" + a
        c = collections.Counter(AA.wtl(G(SF, f), G(al, f)) for f in range(1, 13))
        abl[tag] = {"full_better": c["W"], "nd": c["T"], "full_worse": c["L"],
                    "F2_success": int((G(al, 2) < 1e-3).sum())}
    c = collections.Counter(AA.wtl(G(SF, f), G("SFAO:aspin", f)) for f in range(1, 13))
    abl["M9 population reduction + density cap"] = {"full_better": c["W"], "nd": c["T"], "full_worse": c["L"],
                                                    "F2_success": int((G("SFAO:aspin", 2) < 1e-3).sum())}
    art = {}
    cells = {(1, 1): SF, (0, 1): SF + ":notoxic", (1, 0): SF + ":nomolt", (0, 0): SF + ":notoxic:nomolt"}
    for f in range(1, 13):
        y, Aa, Bb = [], [], []
        for (ta, mb), al in cells.items():
            v = G(al, f); y += list(v); Aa += [ta] * 30; Bb += [mb] * 30
        if np.ptp(y) == 0:
            art[f] = None; continue
        art[f] = art_2x2(y, Aa, Bb)
        art[f]["mean_with_toxic"] = float(np.mean([y[i] for i in range(120) if Aa[i] == 1]))
        art[f]["mean_without_toxic"] = float(np.mean([y[i] for i in range(120) if Aa[i] == 0]))
    S["ablation"] = abl; S["art_2x2"] = art
    S["cells_F2"] = {al: int((G(al, 2) < 1e-3).sum()) for al in cells.values()}

    # ---- PV exact RMSE ----
    pvr = [json.loads(l) for l in open("pv_results.jsonl")]
    pvs = {}
    for alg in ["SFAO:cocoon:aspin", "L-SHADE", "jSO", "CMA-ES", "PSO", "GWO", "WOA"]:
        for m in ["SDM", "DDM"]:
            rr = [r for r in pvr if r["algo"] == alg and r["model"] == m]
            v = np.array([r["rmse"] for r in rr]); b = rr[int(np.argmin(v))]
            pvs[f"{alg}|{m}"] = {"best": float(v.min()), "mean": float(v.mean()), "median": float(np.median(v)),
                                 "worst": float(v.max()), "sd": float(v.std(ddof=1)),
                                 "best_params": b["params"], "exact_rmse_best": pv_exact(b["params"], m)}
    S["pv"] = pvs
    json.dump(S, open(f"{OUT}/stats.json", "w"), indent=1)

    for name in [n for n, _, _ in SUITES]:
        r = S[name]
        print(f"\n== {name}: CD={r['CD']:.2f}  ranks:", dict(sorted(r["ranks"].items(), key=lambda x: x[1])))
        for h in r["holm"][:16]:
            print(f"   {h['algo']:8s} rank {h['rank']:5.2f} z {h['z']:6.2f} p {h['p']:.2e} holm {h['p_holm']:.2e} {'*' if h['sig'] else ''}")
        print("  class:", r["class_wtl"]); print("  A12:", r["A12_summary"])
        if "F2_success" in r: print("  F2:", {a: v["s"] for a, v in r["F2_success"].items()})
    print("\nablation:", json.dumps(abl, indent=0))
    for f, v in art.items():
        if v: print(f"F{f}: toxic p={v['A']['p']:.3g} eta={v['A']['eta2p']:.2f} | molt p={v['B']['p']:.3g} | AB p={v['AB']['p']:.3g}  mean with/without toxic {v['mean_with_toxic']:.3g}/{v['mean_without_toxic']:.3g}")
    print("F2 cells", S["cells_F2"])
    for k, v in pvs.items(): print(k, {kk: vv for kk, vv in v.items() if kk != "best_params"})
