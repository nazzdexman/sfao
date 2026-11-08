# L-SRTDE driver

L-SRTDE (Stanovov & Semenkin, CEC 2024 winner) was run from its authors' C++ code:
https://github.com/VladimirStanovov/L-SRTDE_CEC-2024 (commit `fa72910`).

`setup_lsrtde.sh` downloads that commit, applies `make_driver.py` and compiles `build/lsrtde17` and `build/lsrtde22`.
The algorithm itself is not changed. The driver only:

1. selects CEC2017 or CEC2022 with `-DBENCH=17|22`;
2. disables the time-complexity run;
3. takes `func D maxfe seed` on the command line and seeds the four `mt19937` generators from `seed`;
4. records the best error found **within** the budget (the original can exceed the budget by a few evaluations in the last generation);
5. records convergence at `MaxFEval/1000` steps (the original assumes a budget of 10,000·D);
6. comments out a duplicate `int GNVars;` in `cec22_test_func.cpp` so both files link together.

`driver.diff` shows the full difference between the original `L-SRTDE.cpp` and the generated driver.
`evalchk.cpp` checks that the benchmark code compiled with L-SRTDE agrees with ours (max. difference < 3e-14).

```bash
bash lsrtde/setup_lsrtde.sh
python lsrtde/run_lsrtde.py [workers]     # appends to results/lsrtde_results.jsonl, resumes if interrupted
```

Licence: the files in this folder are derived from L-SRTDE and are distributed under the GNU GPL v3.0, the licence of the original code.
