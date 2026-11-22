"""Recompute the second-revision numbers (Section 6.8 and Table S16; summary of Table S10).

Usage (from follow_up/):
    python revision_r2_tables.py

Reads F2_WSL/run_summary.csv, SFAO_RDE26_CEC2022_F2_Linux_results.zip and
R21_full_budget_geometry/analysis_r21.json. The full Table S10 is produced by
R21_full_budget_geometry/code/analyze_r21.py applied to the unzipped M1 results.
"""
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

HERE = Path(__file__).resolve().parent
BUDGET = 200000


def wilson(k, n, z=1.959963984540054):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def rows(env):
    if env == "WSL2":
        return list(csv.DictReader(open(HERE / "F2_WSL" / "run_summary.csv")))
    with zipfile.ZipFile(HERE / "SFAO_RDE26_CEC2022_F2_Linux_results.zip") as z:
        return list(csv.DictReader(io.TextIOWrapper(z.open("results/run_summary.csv"))))


for env in ("WSL2", "Linux"):
    r = rows(env)
    res = {}
    print(f"== CEC2022 F2, {env}")
    for alg in ("SFAO", "RDE26-SOP"):
        rs = [t for t in r if t["algorithm"] == alg]
        assert sorted(int(t["seed"]) for t in rs) == list(range(30))
        e = np.array([float(t["raw_final_error"]) for t in rs])
        h = np.array([float(t["first_target_fe"]) for t in rs if t["success"] == "True"])
        k = len(h)
        lo, hi = wilson(k, 30)
        ert = (h.sum() + BUDGET * (30 - k)) / k
        res[alg] = (k, e)
        print(f"{alg:10s} success {k}/30 (Wilson {lo:.3f}-{hi:.3f})  mean {e.mean():.6g}  median {np.median(e):.6g}  "
              f"<1e-8 {int((e < 1e-8).sum())}  median FE {np.median(h):.1f} ({h.min():.0f}-{h.max():.0f})  "
              f"after 160000 FE {int((h >= 160000).sum())}  ERT {ert:.0f}")
    (ks, es), (kr, er) = res["SFAO"], res["RDE26-SOP"]
    print(f"Fisher exact p = {fisher_exact([[ks, 30 - ks], [kr, 30 - kr]]).pvalue:.3g}")
    for lab, f in (("raw", lambda x: x), ("zeroed <1e-8", lambda x: np.where(x < 1e-8, 0.0, x))):
        m = mannwhitneyu(f(es), f(er))
        print(f"Mann-Whitney ({lab}): p = {m.pvalue:.3g}, A12 for SFAO = {1 - m.statistic / 900:.2f}")

a = json.load(open(HERE / "R21_full_budget_geometry" / "analysis_r21.json"))
print("\n== Full-budget geometry experiment (Table S10)")
for fam, v in a["verdict"].items():
    print(f"{fam}: {v}")
for r in a["primary"]:
    print(f"  {r['family']:10s} {r['switch']}% {r['comparison']:22s} win {r['win_learned']:.2f}  Holm p {r['p_holm']:.3g}")
