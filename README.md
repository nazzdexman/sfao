# SFAO — Silkworm Feeding and Avoidance Optimization

Source code, experiment drivers and run-level results for the paper

> Thanapat, "Silkworm Feeding and Avoidance Optimization: A Mechanism-Audited Metaheuristic Benchmarked Against CEC Competition Winners up to 2024", *International Journal of Intelligent Engineering and Systems* (submitted).

SFAO is a population-based metaheuristic for box-constrained continuous minimization built from nine explicitly specified operators (attraction, adaptive head-swing local search, avoidance memory, crowding/depletion, instar schedule, disease-driven culling, adaptive cocoon spinning, directional memory, and population reduction with a budget-based density cap).

## Contents

| Path | What it is |
|---|---|
| `sfao.py` | SFAO implementation (class `SFAO`) |
| `sfao_final.py` | **The locked version used in the paper**: `SFAO(use_cocoon=True, food_per_worm=100, spin_mode="adaptive")` — label `SFAO:cocoon:aspin` in the result files. The class defaults in `sfao.py` are the earlier prototype. |
| `cec2017.py`, `cec2017_data/` | CEC2017 via the official C code (ctypes; prebuilt for Linux x64, macOS arm64, Windows x64) |
| `cec2022.py`, `cec2022_data/` | CEC2022 official Python port (checked against the official C code, max. difference < 5e-14) |
| `baselines.py`, `top_baselines.py`, `improved_baselines.py`, `recent_baselines.py`, `plug_disease.py` | Comparators: PSO, ABC, GWO, WOA, MFO; L-SHADE, jSO, CMA-ES (IPOP, `cma`); CLPSO; GJO, COA, SHO, WaOA, OOA (MEALPY 3.0.3) |
| `full_benchmark.py` | Runs any algorithm on CEC2017/CEC2022 with exact FE budgets |
| `lsrtde/` | Builds and runs L-SRTDE (CEC2024 winner) from its authors' code (see `lsrtde/README.md`) |
| `pv.py`, `run_pv.py` | Photovoltaic parameter extraction (RTC France, SDM/DDM) |
| `bap.py`, `run_bap_full.py`, `run_bap_discrete.py`, `dsfao_bap.py`, `bap_data/` | Berth allocation (Port of Barcelona 2024) |
| `bias_test.py` | Structural-bias test on f0 ~ U(0,1) |
| `analyze_all.py`, `paper_stats.py`, `paper_figs.py`, `analyze_bap_all.py` | Statistics, tables and figures of the paper |
| `results/` | Run-level results of every experiment in the paper (all seeds, gzipped JSON lines) |
| `figures/` | Figures as they appear in the paper |
| `reproduce_check.py` | Quick self-check (benchmark code + a few SFAO runs) |

## Installation

```bash
python -m pip install -r requirements.txt
```
Tested with Python 3.11, numpy 2.x, scipy 1.18, matplotlib 3.11, cma 4.5.0, mealpy 3.0.3. L-SRTDE additionally needs `git` and `g++`.

## Quick start

```python
import numpy as np
from sfao_final import sfao_final
sphere = lambda x: float(np.sum(x ** 2))
best_x, best_f, curve = sfao_final().optimize(sphere, dim=10, lb=-100, ub=100, max_fe=20000, seed=0)
```

Quick check of the benchmark code and SFAO (about one minute):
```bash
python reproduce_check.py
```

## Reproducing the tables and figures from the stored results

```bash
cd results
gunzip -k *.gz
python ../analyze_all.py      # Friedman ranks and W/T/L of all 16 algorithms (Tables 6-7)
python ../paper_stats.py      # Holm, CD, A12, classes, Wilson CI, ablation, ART, PV -> output/stats.json
mkdir -p output && cp bias_test.json output/
python ../paper_figs.py       # Figures 1-7 -> output/
python ../analyze_bap_all.py  # berth allocation (Table 19)
```

| Paper item | Source |
|---|---|
| Tables 6–13, Fig. 2–4 | `cec30`, `swarm_pc`, `swarm_recent`, `clpso22`, `c17_full`, `c17_d30`, `c17_swarm`, `lsrtde_results` |
| Table 14–15, Fig. 5 (ablation, ART 2×2) | `abl_pc`, `abl_mac`, `cell_2x2_pc` (+ `cec30` for the full version) |
| Table 16, Fig. 6 (structural bias) | `bias_test.json` |
| Table 18, Fig. 7 (photovoltaic) | `pv_results` |
| Table 19 (berth allocation) | `bap_results`, `bap_discrete`; the paper's "SFAO" is `SFAO:cocoon:aspin:cap100` (initial population capped by the density rule), "without the cap" is `SFAO:cocoon:aspin` |

Each line of a result file is one run: `suite`, `D`, `f`, `algo`, `seed`, `err` (final error, values < 1e-8 set to 0), and `curve_fe`/`curve` (best error at 60 log-spaced FE points).
Ablation labels: `SFAO:cocoon:aspin:no<mechanism>` with `attract, swing, toxic, crowd, molt, disease, spin, memory`; `SFAO:aspin` removes population reduction and the density cap.

## Re-running the experiments

```bash
# CEC2022 D=10 (200,000 FE) and CEC2017 D=10/30 (10,000*D FE), 30 runs
python full_benchmark.py --suite cec2022 --dims 10 --runs 30 --algos SFAO:cocoon:aspin L-SHADE jSO CMA-ES CLPSO ABC PSO GWO WOA MFO GJO COA SHO WaOA OOA --out my_cec2022.jsonl
python full_benchmark.py --suite cec2017 --dims 10 30 --runs 30 --algos SFAO:cocoon:aspin L-SHADE jSO CMA-ES --out my_cec2017.jsonl
# ablation example
python full_benchmark.py --suite cec2022 --dims 10 --runs 30 --algos SFAO:cocoon:aspin:nospin SFAO:cocoon:aspin:nomemory --out my_ablation.jsonl
# L-SRTDE
bash lsrtde/setup_lsrtde.sh && python lsrtde/run_lsrtde.py
# applications and bias test
python run_pv.py 30 50000
python run_bap_full.py 10 10000
python bias_test.py
```
Runs resume automatically if interrupted. Results are bit-identical on the same platform; across platforms, the last digits of `numpy.linalg.eigh` (directional memory) can differ and individual runs then diverge, without changing the 30-run statistics materially.

## Notes on benchmark correctness

- CEC2017 is evaluated through the official C code (`cec2017_data/cec17_lib.cpp`, only an export function and the data path added) and passes 174 reference checks. CEC2022 uses the official Python port, checked against the official C code at 200 points per function.
- Some widely used Python benchmark libraries return values that differ from the official code for several CEC2017/CEC2022 functions; they were not used.
- The CEC benchmark code and data come from P. N. Suganthan's official repositories (CEC2017-BoundContrained, 2022-SO-BO).

## Licences

- Code written for this project: MIT (see `LICENSE`).
- `lsrtde/`: files derived from L-SRTDE are GPL-3.0, as is the original code (see `lsrtde/README.md`). The original source is downloaded by the setup script and is not redistributed here.
- `bap_data/`: berth-allocation instances and best-known solutions by A. Santini, redistributed under their original GPL-3.0 licence (`bap_data/LICENSE_bap_data`).
- CEC benchmark code and data: as provided by the competition organisers.

## Citation

See `CITATION.cff`. Please cite the paper once it is published.
