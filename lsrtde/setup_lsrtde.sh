#!/usr/bin/env bash
# Build the L-SRTDE driver used in the paper from the authors' original code.
# Usage: bash lsrtde/setup_lsrtde.sh   (run from the repository root; needs git, python3, g++)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
B="$HERE/build"; rm -rf "$B"; mkdir -p "$B"
git clone -q https://github.com/VladimirStanovov/L-SRTDE_CEC-2024.git "$B/src"
git -C "$B/src" checkout -q fa72910
python3 - "$B" <<'PY'
import sys, os
b = sys.argv[1]
for fn in ("L-SRTDE.cpp", "cec17_test_func.cpp", "cec22_test_func.cpp"):
    s = open(os.path.join(b, "src", fn), "rb").read().replace(b"\r\n", b"\n")       # normalise line endings
    if fn == "cec22_test_func.cpp":                                                  # GNVars is also defined in L-SRTDE.cpp
        assert s.count(b"\nint GNVars;") == 1
        s = s.replace(b"\nint GNVars;", b"\n// int GNVars;  (removed: also defined in L-SRTDE.cpp; benchmark code otherwise unchanged)")
    open(os.path.join(b, fn), "wb").write(s)
PY
cp "$HERE/make_driver.py" "$B/"
(cd "$B" && python3 make_driver.py)
for X in 17 22; do g++ -std=c++11 -O3 -DBENCH=$X "$B/lsrtde_driver.cpp" -o "$B/lsrtde$X" 2>/dev/null; done
mkdir -p "$B/run17" "$B/run22"
cp -r "$B/src/input_data" "$B/run17/input_data"
cp -r "$B/src/input_data_22" "$B/run22/input_data"
echo "built: $B/lsrtde17 $B/lsrtde22"
