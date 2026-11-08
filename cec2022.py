"""
CEC2022 จากโค้ดทางการ (github.com/P-N-Suganthan/2022-SO-BO, Python port โดย Abhishek Kumar)
— โหลดข้อมูล shift/rotation/shuffle ครั้งเดียว และเปลี่ยนเฉพาะ shift/rotate ให้ใช้ numpy (คณิตศาสตร์เหมือนเดิมทุกประการ)
— ตรวจแล้วว่าให้ค่าตรงกับฟังก์ชันทางการ cec22_test_func (ดู self_test)

หมายเหตุ: opfunu (cec_based.F*2022) ให้ค่า "ไม่ตรง" กับโค้ดทางการในหลายฟังก์ชัน
เช่น Zakharov (F1) ขาดน้ำหนัก i ในพจน์ sum(0.5*i*z) — ห้ามใช้รายงานผล CEC2022

ใช้งาน: f = CEC2022(fid, D); err = f(x)   (คืนค่า error = f(x) - f*)
"""
import os
import numpy as np

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cec2022_data")
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("CEC2022_official", os.path.join(_DIR, "CEC2022_official.py"))
_off = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_off)
# โค้ดทางการ (Python port) ใน F4 อ่านตัวแปร global y ค้างจากการเรียกก่อนหน้า ซึ่งไม่มีผลต่อค่าฟังก์ชัน
# (ค่า y ที่ถูกแก้ไม่ถูกนำไปใช้ต่อ เหมือนในโค้ด C) แต่ถ้า F4 ถูกเรียกเป็นครั้งแรกใน process จะ NameError
_off.y = np.zeros(100)

# ใช้ numpy แทนลูป Python (ผลลัพธ์เหมือนเดิม)
_off.shiftfunc = lambda x, nx, Os: np.asarray(x[:nx], float) - np.asarray(Os[:nx], float)
_off.rotatefunc = lambda x, nx, Mr: np.asarray(Mr, float)[:nx, :nx] @ np.asarray(x[:nx], float)

BIAS = {1: 300, 2: 400, 3: 600, 4: 800, 5: 900, 6: 1800, 7: 2000, 8: 2200, 9: 2300, 10: 2400, 11: 2600, 12: 2700}


class CEC2022:
    def __init__(self, fid, D):
        assert D in (2, 10, 20)
        self.fid, self.D = fid, D
        p = lambda n: os.path.join(_DIR, "input_data", n)
        self.M = np.loadtxt(p(f"M_{fid}_D{D}.txt"))
        Os = np.loadtxt(p(f"shift_data_{fid}.txt"))
        if fid < 9:
            self.Os = np.ravel(Os)[:D].copy()
        else:                       # โค้ดทางการใช้ cf_num=10 -> 9 แถวแรก
            self.Os = np.atleast_2d(Os)[:9, :D].reshape(-1).copy()
        self.SS = np.loadtxt(p(f"shuffle_data_{fid}_D{D}.txt")) if 6 <= fid <= 8 else None
        self.bias = BIAS[fid]
        self.f_global = self.bias

    def raw(self, x):
        x = np.asarray(x, float)
        D, Os, M, SS, f = self.D, self.Os, self.M, self.SS, self.fid
        o = _off
        if f == 1: return o.zakharov_func(x, D, Os, M, 1, 1)
        if f == 2: return o.rosenbrock_func(x, D, Os, M, 1, 1)
        if f == 3: return o.schaffer_F7_func(x, D, Os, M, 1, 1)
        if f == 4: return o.step_rastrigin_func(x, D, Os, M, 1, 1)
        if f == 5: return o.levy_func(x, D, Os, M, 1, 1)
        if f == 6: return o.hf02(x, D, Os, M, SS, 1, 1)
        if f == 7: return o.hf10(x, D, Os, M, SS, 1, 1)
        if f == 8: return o.hf06(x, D, Os, M, SS, 1, 1)
        if f == 9: return o.cf01(x, D, Os, M, 1, 1)
        if f == 10: return o.cf02(x, D, Os, M, 1, 1)
        if f == 11: return o.cf06(x, D, Os, M, 1, 1)
        if f == 12: return o.cf07(x, D, Os, M, 1, 1)

    def __call__(self, x):
        """error = f(x) - f*"""
        return float(self.raw(x))

    def x_opt(self):
        return self.Os[:self.D].copy()


def self_test(n=30, seed=0):
    """เทียบกับ cec22_test_func ทางการ (ฉบับไม่ดัดแปลง) ที่จุดสุ่มและใกล้จุดต่ำสุด"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("orig", os.path.join(_DIR, "CEC2022_official.py"))
    orig = importlib.util.module_from_spec(spec)
    cwd = os.getcwd(); os.chdir(_DIR)
    try:
        spec.loader.exec_module(orig)
        rng = np.random.default_rng(seed)
        worst = {}
        for D in (10, 20):
            for f in range(1, 13):
                F = CEC2022(f, D); xo = F.x_opt(); md = 0.0
                pts = [rng.uniform(-100, 100, D) for _ in range(n)] + \
                      [xo + rng.normal(0, s, D) for s in (0, 1e-6, 1e-2, 1, 10)]
                for x in pts:
                    ref = float(orig.cec22_test_func(x.copy(), D, 1, f)[0]) - F.bias
                    v = F(x)
                    md = max(md, abs(v - ref) / max(1.0, abs(ref)))
                worst[(f, D)] = (md, F(xo))
        return worst
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    for (f, D), (md, fo) in self_test().items():
        print(f"F{f:<2d} D={D}: max rel diff vs official = {md:.2e}   error at x* = {fo:.3g}")
