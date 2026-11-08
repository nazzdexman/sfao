"""รูปสำหรับต้นฉบับ (สร้างจากไฟล์ผลรายรอบ) -> paper/fig*.png"""
import json, collections, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import analyze_all as AA

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.linewidth": 0.7, "savefig.dpi": 300})
SF = AA.SF
ALG = ["SFAO", "L-SRTDE", "jSO", "L-SHADE", "CMA-ES", "CLPSO"]
COL = {"SFAO": "#c0392b", "L-SRTDE": "#1f4e79", "jSO": "#2e86c1", "L-SHADE": "#7fb3d5", "CMA-ES": "#239b56",
       "CLPSO": "#9b9b9b"}
S = json.load(open("output/stats.json"))


def curves(suite, D, f):
    C = collections.defaultdict(list); FE = {}
    for fn in AA.FILES[suite]:
        try: fh = open(fn)
        except FileNotFoundError: continue
        for l in fh:
            r = json.loads(l)
            if r.get("suite", "cec2022") != suite or r["D"] != D or r["f"] != f: continue
            a = "SFAO" if r["algo"] == SF else r["algo"]
            if a in ALG and "curve" in r:
                C[a].append(r["curve"]); FE[a] = r["curve_fe"]
    return C, FE


def errs(suite, D, f):
    E = AA.load(suite, D)
    return {a: np.array([E[(a, f)][s] for s in range(30)]) for a in ALG if len(E.get((a, f), {})) == 30}


PANELS = [("cec2022", 10, 2, "CEC2022 F2 (D=10)"), ("cec2022", 10, 6, "CEC2022 F6 (D=10)"),
          ("cec2022", 10, 11, "CEC2022 F11 (D=10)"), ("cec2017", 30, 4, "CEC2017 F4 (D=30)"),
          ("cec2017", 30, 21, "CEC2017 F21 (D=30)"), ("cec2017", 30, 13, "CEC2017 F13 (D=30)")]

# ---- Fig. convergence ----
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.3))
for ax, (su, D, f, t) in zip(axes.flat, PANELS):
    C, FE = curves(su, D, f)
    for a in ALG:
        if a not in C: continue
        y = np.median(np.array(C[a]), axis=0)
        ax.plot(FE[a], np.maximum(y, 1e-8), color=COL[a], lw=1.8 if a == "SFAO" else 1.0, label=a)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(t, fontsize=8.5); ax.grid(alpha=.25, lw=.4)
    ax.tick_params(labelsize=7)
for ax in axes[1]: ax.set_xlabel("Function evaluations", fontsize=8)
for ax in axes[:, 0]: ax.set_ylabel("Median error", fontsize=8)
axes[0, 0].legend(fontsize=6.5, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig("output/fig_convergence.png"); plt.close(fig)

# ---- Fig. boxplots ----
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.3))
for ax, (su, D, f, t) in zip(axes.flat, PANELS):
    E = errs(su, D, f)
    data = [np.maximum(E[a], 1e-8) for a in ALG if a in E]
    bp = ax.boxplot(data, widths=.6, patch_artist=True, showfliers=True,
                    flierprops=dict(markersize=2, markeredgewidth=.4), medianprops=dict(color="k", lw=.8))
    for p, a in zip(bp["boxes"], [a for a in ALG if a in E]):
        p.set_facecolor(COL[a]); p.set_alpha(.75); p.set_linewidth(.5)
    ax.set_yscale("log"); ax.set_title(t, fontsize=8.5)
    ax.set_xticks(range(1, len(data) + 1)); ax.set_xticklabels([a for a in ALG if a in E], rotation=45, fontsize=6.5)
    ax.tick_params(axis="y", labelsize=7); ax.grid(axis="y", alpha=.25, lw=.4)
for ax in axes[:, 0]: ax.set_ylabel("Final error (30 runs)", fontsize=8)
fig.tight_layout(); fig.savefig("output/fig_boxplots.png"); plt.close(fig)

# ---- Fig. Friedman ranks ----
names = ["CEC2022 D10", "CEC2017 D10", "CEC2017 D30"]
order = sorted(S[names[0]]["ranks"], key=lambda a: np.mean([S[n]["ranks"][a] for n in names]))
fig, ax = plt.subplots(figsize=(7.0, 3.6))
mk = {"CEC2022 D10": ("o", "#c0392b"), "CEC2017 D10": ("s", "#1f4e79"), "CEC2017 D30": ("^", "#239b56")}
for j, n in enumerate(names):
    y = np.arange(len(order)) + (j - 1) * 0.22
    ax.scatter([S[n]["ranks"][a] for a in order], y, marker=mk[n][0], color=mk[n][1], s=16,
               label=f"{n} (CD = {S[n]['CD']:.2f})", zorder=3)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7.5)
for t in ax.get_yticklabels():
    if t.get_text() == "SFAO": t.set_fontweight("bold"); t.set_color("#c0392b")
ax.invert_yaxis(); ax.set_xlabel("Friedman mean rank (lower is better)", fontsize=8)
ax.set_xlim(0.5, 16.5); ax.grid(axis="x", alpha=.25, lw=.4); ax.tick_params(axis="x", labelsize=7)
ax.axhspan(order.index("SFAO") - .45, order.index("SFAO") + .45, color="#fadbd8", alpha=.5, zorder=0)
ax.legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig("output/fig_ranks.png"); plt.close(fig)

# ---- Fig. ablation success on F2 ----
abl = S["ablation"]
lab = ["Full SFAO"] + [k.split(" ", 1)[0] + " off" for k in abl]
val = [30] + [v["F2_success"] for v in abl.values()]
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.bar(range(len(val)), val, color=["#c0392b"] + ["#7f8c8d"] * (len(val) - 1))
ax.set_xticks(range(len(val))); ax.set_xticklabels(lab, rotation=55, fontsize=7, ha="right")
ax.set_ylabel("Runs with error < 1e-3 (of 30)", fontsize=7.5); ax.set_ylim(0, 32)
for i, v in enumerate(val): ax.text(i, v + .5, str(v), ha="center", fontsize=6.5)
ax.tick_params(axis="y", labelsize=7); ax.set_title("CEC2022 F2: success when one mechanism is removed", fontsize=7.5)
fig.tight_layout(); fig.savefig("output/fig_ablation_f2.png"); plt.close(fig)

# ---- Fig. PV fit ----
import pv
from scipy.special import lambertw
p = S["pv"]["SFAO:cocoon:aspin|SDM"]["best_params"]
Iph, I0, Rs, Rsh, n = p; a = n * pv.Vt
Vg = np.linspace(pv.V.min(), pv.V.max(), 200)
arg = Rs * I0 * Rsh / (a * (Rs + Rsh)) * np.exp(Rsh * (Rs * Iph + Rs * I0 + Vg) / (a * (Rs + Rsh)))
Ig = (Rsh * (Iph + I0) - Vg) / (Rs + Rsh) - a / Rs * np.real(lambertw(arg))
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
ax[0].plot(pv.V, pv.I, "o", ms=3.5, mfc="none", color="k", label="Measured (RTC France)")
ax[0].plot(Vg, Ig, "-", color="#c0392b", lw=1.2, label="SDM, SFAO best parameters")
ax[0].set_xlabel("Voltage (V)"); ax[0].set_ylabel("Current (A)"); ax[0].legend(fontsize=7, frameon=False)
ax[0].grid(alpha=.25, lw=.4)
rm = {}
for alg, lb in [("SFAO:cocoon:aspin", "SFAO"), ("L-SHADE", "L-SHADE"), ("jSO", "jSO"), ("CMA-ES", "CMA-ES"),
                ("PSO", "PSO"), ("GWO", "GWO"), ("WOA", "WOA")]:
    rows = [json.loads(l) for l in open("pv_results.jsonl")]
    rm[lb] = sorted(r["rmse"] for r in rows if r["algo"] == alg and r["model"] == "SDM")
ax[1].boxplot(list(rm.values()), patch_artist=True, widths=.6, flierprops=dict(markersize=2),
              medianprops=dict(color="k", lw=.8))
ax[1].set_yscale("log"); ax[1].set_xticks(range(1, len(rm) + 1)); ax[1].set_xticklabels(rm.keys(), rotation=40, fontsize=7)
ax[1].set_ylabel("SDM RMSE (30 runs)"); ax[1].grid(axis="y", alpha=.25, lw=.4)
fig.tight_layout(); fig.savefig("output/fig_pv.png"); plt.close(fig)

# ---- Fig. framework diagram ----
fig, ax = plt.subplots(figsize=(7.0, 3.4)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 50)
def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=c, ec="#333333", lw=.6))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.3, linespacing=1.25)
def arr(x1, y1, x2, y2, c="#333333"):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=.7, color=c))
W, H = 29, 10
X = [2, 35.5, 69]
box(X[0], 37, W, H, "Initialise $N_0$ worms uniformly\n(rearing-density cap, Eq. 3)", "#fdebd0")
box(X[1], 37, W, H, "M9 cocooning: shrink to $N_{target}$\n(mature worms first);\nM5 instar schedule + molting", "#fdebd0")
box(X[2], 37, W, H, "Per-iteration state: $\\rho$, $L$, $r_c$;\nM4 crowding/depletion $q^{eff}$;\nM8 matrix $B$, scales $d$", "#fdebd0")
box(X[2], 21, W, H, "Each worm, random order:\nM1 attraction $-$ M3 avoidance\n$+$ M4 expulsion $\\rightarrow$ body $b_i$", "#d6eaf8")
box(X[1], 21, W, H, "M2 head-swing: $m_s$ bites\n(arc / nibble / memory-guided)\ngreedy acceptance", "#d6eaf8")
box(X[0], 21, W, H, "Update archives: avoidance,\ndepletion, eggs, directions;\noperator success rates", "#d6eaf8")
box(X[0], 5, W, H, "M6 disease: infect, spread,\nrecover; cull and replace\n(egg archive / opposition)", "#d5f5e3")
box(X[1], 5, W, H, "M7 adaptive spinning ($p \\geq 0.8$):\nsample around $x^*$ with $B$, $d$;\n$\\sigma \\times 1.5$ or $\\times 0.9$", "#d5f5e3")
box(X[2], 5, W, H, "FE budget exhausted?\nyes: return $x^*$, $f^*$\nno: next iteration (to M9)", "#f2f3f4")
arr(X[0]+W+.6, 42, X[1]-.6, 42); arr(X[1]+W+.6, 42, X[2]-.6, 42); arr(X[2]+W/2, 36.4, X[2]+W/2, 31.6)
arr(X[2]-.6, 26, X[1]+W+.6, 26); arr(X[1]-.6, 26, X[0]+W+.6, 26); arr(X[0]+W/2, 20.4, X[0]+W/2, 15.6)
arr(X[0]+W+.6, 10, X[1]-.6, 10); arr(X[1]+W+.6, 10, X[2]-.6, 10)
ax.annotate("", (X[1]+W/2, 47.6), (X[2]+W+1.2, 10), arrowprops=dict(arrowstyle="-|>", lw=.6, color="#999999",
            connectionstyle="angle,angleA=90,angleB=0,rad=4", ls="--"))
ax.text(50, 0.8, "Orange: state per iteration    Blue: movement and feeding    Green: population management",
        ha="center", fontsize=6.2, color="#555555")
fig.savefig("output/fig_framework.png", bbox_inches="tight"); plt.close(fig)
# ---- Fig. structural bias (needs output/bias_test.json from bias_test.py) ----
import os
if os.path.exists("output/bias_test.json"):
    Bt = json.load(open("output/bias_test.json")); algos = list(Bt["results"])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    U = np.random.default_rng(0).random((100, 10))
    for i, a in enumerate(algos):
        X = np.array(Bt["positions"][a])
        ax[0].scatter(np.full(X.size, i) + np.random.default_rng(i).uniform(-.3, .3, X.size), X.ravel(), s=1.2, alpha=.35,
                      color="#c0392b" if a == "SFAO" else "#34495e")
    ax[0].scatter(np.full(U.size, len(algos)) + np.random.default_rng(9).uniform(-.3, .3, U.size), U.ravel(), s=1.2, alpha=.35, color="#999999")
    ax[0].set_xticks(range(len(algos) + 1)); ax[0].set_xticklabels(algos + ["Uniform"], rotation=30, fontsize=7)
    ax[0].set_ylabel("Final coordinate in [0,1]")
    dd = [np.linalg.norm(np.array(Bt["positions"][a]) - .5, axis=1) for a in algos] + [np.linalg.norm(U - .5, axis=1)]
    bp = ax[1].boxplot(dd, patch_artist=True, widths=.6, flierprops=dict(markersize=2), medianprops=dict(color="k", lw=.8))
    for p, a in zip(bp["boxes"], algos + ["Uniform"]):
        p.set_facecolor("#c0392b" if a == "SFAO" else ("#bbbbbb" if a == "Uniform" else "#5d6d7e")); p.set_alpha(.7)
    ax[1].set_xticks(range(1, len(algos) + 2)); ax[1].set_xticklabels(algos + ["Uniform"], rotation=30, fontsize=7)
    ax[1].set_ylabel("Distance of final point to centre")
    fig.tight_layout(); fig.savefig("output/fig_bias.png"); plt.close(fig)
print("figures written")
