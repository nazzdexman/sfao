"""
CEC2017 (bound-constrained) จากโค้ด C ทางการ (github.com/P-N-Suganthan/CEC2017-BoundContrained)
— เรียกโค้ด C ตรง ๆ ผ่าน ctypes (คอมไพล์ไว้แล้วสำหรับ Windows x64, macOS arm64/x64, Linux x64)
— แก้ในโค้ด C แค่สองจุด: ใส่โฟลเดอร์ข้อมูล และฟังก์ชันเรียกใช้จาก Python (คณิตศาสตร์ไม่แตะ)
— ใช้รุ่นที่อ่านไฟล์ด้วย %lf (รุ่น %Lf ในไฟล์ rar อ่านค่าผิดบน gcc/clang; บน MSVC ทั้งสองเหมือนกัน)

หมายเหตุ: opfunu (F*2017) และ cec2017-py (tilleyd) ให้ค่าไม่ตรงกับโค้ดทางการ — ห้ามใช้รายงานผล
ใช้งาน: f = CEC2017(fid, D); err = f(x)   (คืนค่า error = f(x) - 100*fid)
ตามธรรมเนียมการแข่ง ไม่ใช้ F2 (ไม่เสถียร)
"""
import ctypes
import os
import platform
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "cec2017_data")          # ข้างในมีโฟลเดอร์ input_data/
_LIB = None


def _libname():
    m = platform.machine().lower()
    if sys.platform.startswith("win"):
        return "cec17_win_x64.dll"
    if sys.platform == "darwin":
        return "libcec17_mac_arm64.dylib" if m in ("arm64", "aarch64") else "libcec17_mac_x64.dylib"
    return "libcec17_linux_x64.so"


def _load():
    global _LIB
    if _LIB is not None:
        return _LIB
    cands = [os.path.join(_DATA, "libcec17_local.so"), os.path.join(_DATA, _libname())]  # ตัวที่คอมไพล์เองบนเครื่องมาก่อน
    path = next(p for p in cands if os.path.exists(p))
    lib = ctypes.CDLL(path)
    lib.cec17_eval.restype = ctypes.c_double
    lib.cec17_eval.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
    lib.cec17_set_dir.argtypes = [ctypes.c_char_p]
    # โค้ด C ใช้ fopen แบบ ANSI: ใช้ path สัมพัทธ์ (ภาษาอังกฤษล้วน) ถ้าทำได้ กันปัญหาชื่อโฟลเดอร์ภาษาไทย
    try:
        d = os.path.relpath(_DATA, os.getcwd())
    except ValueError:
        d = _DATA
    if not d.isascii():
        d = _DATA
    lib.cec17_set_dir((d.replace("\\", "/") + "/").encode(sys.getfilesystemencoding(), "replace"))
    _LIB = lib
    return lib


class CEC2017:
    def __init__(self, fid, D):
        assert D in (10, 30, 50, 100) and 1 <= fid <= 30
        self.fid, self.D = fid, D
        self.lib = _load()
        self.bias = 100.0 * fid
        self.f_global = self.bias
        self._buf = (ctypes.c_double * D)()

    def raw(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        ctypes.memmove(self._buf, x.ctypes.data, 8 * self.D)
        return self.lib.cec17_eval(self._buf, self.D, self.fid)

    def __call__(self, x):
        """error = f(x) - f*"""
        return float(self.raw(x) - self.bias)


FUNCS = [1] + list(range(3, 31))      # 29 ฟังก์ชัน (ตัด F2 ตามธรรมเนียม)


def selfcheck(verbose=True):
    """เทียบค่ากับชุดอ้างอิงที่คำนวณจากโค้ด C ทางการ (174 จุด, D=10/30) — ต้องผ่านก่อนรันจริง"""
    import json
    ref = json.load(open(os.path.join(_DATA, "selfcheck.json")))
    worst = 0.0
    for r in ref:
        v = CEC2017(r["f"], r["D"]).raw(np.array(r["x"]))
        worst = max(worst, abs(v - r["v"]) / max(1.0, abs(r["v"])))
    ok = worst < 1e-9
    if verbose:
        print(f"CEC2017 self-check: {'PASS' if ok else 'FAIL'} (max rel diff {worst:.1e}, {len(ref)} points)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if selfcheck() else 1)
