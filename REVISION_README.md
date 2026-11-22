# SFAO IJIES revision reproducibility

Primary release: [SFAO v1.0.0](https://doi.org/10.5281/zenodo.24051942), Git commit `4471f7ca778eb9adab7b02dd8a24cc82bcc32b1a`.
Original supplied ZIP SHA-256: `697ade03b4909f9b5b5db1afa2c5fd306d18e688ab64f0c1ebdb513d0bdc2ae1`.

The package combines that source and its compressed run files with the revision scripts and evidence. The revision changes to `sfao.py` add diagnostics and an optional reflection control. The original defaults still use clipping. `revision_data/sfao_and_figure.patch` documents the changes relative to the supplied source. Checksums for the packaged files are in `SHA256SUMS`.

The revision was run with Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8 and GCC 14.2.0 on Linux. For the original comparator suites, use the original package requirements, including `mealpy==3.0.3`.

## Regenerate archived statistics and figures

From this directory:

```sh
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
python revision_audit.py
(
    cd results
    for f in *.jsonl.gz; do gzip -dc "$f" > "${f%.gz}"; done
    mkdir -p output
    cp bias_test.json output/
    python ../analyze_all.py
    python ../paper_stats.py
    python ../revision_tables.py
    python ../paper_figs.py
    python ../analyze_bap_all.py
)
```

Use `revision_output/Figure2.png` for revised Fig. 2. The older convergence figure emitted by `paper_figs.py` is retained for historical comparison and does not supply the revised figure. The Table 6–11 CSV files are regenerated to `results/output/revision_tables`. The tie-aware sensitivity CSV preserves signed effects, unadjusted tie-corrected Mann–Whitney p-values, exact median/mean differences and historical classifications. The Figure 2 audit emits a keyed run-level JSONL (including success flags and final errors), an audit JSON and the figure.

The archived record files contain sparse checkpoints and include stored zeroing or rounding. The L-SRTDE curve field used the next underlying checkpoint and can disagree with the separately recorded within-budget FINAL. On D30 its original driver used 300,001 search calls per run, but FINAL uses only the first 300,000. Rerun raw best-so-far arrays in `revision_data/traces` have one entry per objective evaluation and carry exact recorded FE for the first M7 sample where active. Seed labels do not guarantee the same outcome across numerical environments.

## Re-run benchmarks

For example, the focal full-budget comparison can be run with:

```sh
python full_benchmark.py --suite cec2017 --dims 30 --funcs 4 --runs 30 \
  --algos SFAO:cocoon:aspin jSO L-SHADE --out rerun_c17_f4.jsonl
python full_benchmark.py --suite cec2022 --dims 10 --funcs 2 --runs 30 \
  --algos SFAO:cocoon:aspin jSO L-SHADE --out rerun_c22_f2.jsonl
bash lsrtde/setup_lsrtde.sh
(cd lsrtde/build/run17 && ../lsrtde17 4 30 300000 0)
(cd lsrtde/build/run22 && ../lsrtde22 2 10 200000 0)
python lsrtde/run_lsrtde.py 4
```

L-SRTDE's build downloads pinned upstream source from commit `fa72910`; `lsrtde/driver.diff` is supplied. Its historical raw final values are distinct from the occasional extra search call. jSO and L-SHADE are the pinned Python reimplementations in this package.

## Re-run the revision experiments

```sh
mkdir -p reruns
python revision_pilot.py reruns/matched_refinement.jsonl
python revision_bias.py reruns/boundary_repair.jsonl
python revision_bap.py reruns/bap_cap_only.jsonl
python revision_trace.py reruns/traces
python revision_factorial.py reruns/traces/trace_runs.jsonl > reruns/factorial.json
git clone https://github.com/SichenTao/RDE26-SOP.git reruns/RDE26
git -C reruns/RDE26 checkout 877e68555ecbed73d05b98c57a196a3a0d1d3147
python revision_rde26.py reruns/RDE26 reruns/rde26.jsonl
python revision_bap_verify.py path/to/unpacked/Barcelona-v0.0.1 reruns/bap_reference_audit.csv
```

The final command takes the directory containing the `instances` and `bks` trees from [Barcelona v0.0.1](https://doi.org/10.5281/zenodo.17665596). It verifies dataset file identity and checks the GA candidate and archived schedule against the relative-position formulation. The RDE26 script clones no code itself. It prepares an isolated compilation copy of the user-supplied pinned upstream clone and instruments the evaluator for exact in-budget logging.

The stored revision outputs are in `revision_data/`. Retrieving `curve` from an NPZ record yields the best-so-far raw error at one-based FE, i.e. array position zero represents FE 1.


## Second revision (v1.1.0)

All files below are in `follow_up/`; `follow_up/README.md` lists them.

### RDE26-SOP on CEC2022 F2 (Section 6.8, Table S16)

`SFAO_RDE26_CEC2022_F2_Windows.zip` is the runner used under WSL2 on Windows (RDE26-SOP solver at upstream revision 877e68555ecbed73d05b98c57a196a3a0d1d3147, MIT licence and third-party notices included). `F2_WSL/` holds its per-seed table, manifest, evaluator check and the original result archive. `SFAO_RDE26_CEC2022_F2_Linux_results.zip` holds the Linux runs with per-evaluation traces; its SFAO runs are the full cell of the F2 factorial. Recompute every number of Section 6.8 and Table S16 with

```sh
cd follow_up
python revision_r2_tables.py
```

### Full-budget geometry experiment (Section 6.7, Table S10)

`R21_full_budget_geometry/SFAO_R21_runner.zip` is the runner exactly as executed; the SHA-256 digests of `run_r21.py`, `sfao_geom.py`, `sfao.py`, `cec2017.py`, `cec2022.py`, `PROTOCOL.md` and `analyze_r21.py` match `manifest.json` in `R21_results_M1.zip`. `PROTOCOL.md` fixes the arms, problems, seeds, outcomes and decision rule; it was written before the runs. `sfao_geom.py` differs from `sfao.py` only in the M7 sampling line (`sfao_geom.patch`), and `check_install.py` verifies that the learned arm at the 80% switch reproduces SFAO exactly. The 750 runs were executed on an Apple M1 computer with x86-64 Python 3.9.12 under Rosetta 2 and NumPy 2.0.2.

```sh
unzip SFAO_R21_runner.zip && cd SFAO_R21_runner
python check_install.py                 # on macOS with x86-64 Python, first run: sh build_cec17.sh
python run_r21.py results               # 750 runs; resumes if interrupted
python analyze_r21.py results           # pre-specified analysis
```

To re-analyse the stored runs, unzip `R21_results_M1.zip` into a folder that contains only the run files (remove `analysis_r21.json` first, because the analysis script reads every JSON file except `manifest.json`). The stored result is `R21_full_budget_geometry/analysis_r21.json`.

### Figure 2

`revision_audit.py` now assigns fixed colours to each method (CMA-ES no longer shares the red hue of SFAO) and applies the tick label size to minor ticks. The plotted data and `audit.json` are unchanged.
