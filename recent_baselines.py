"""
อัลกอริทึมเลียนแบบสัตว์รุ่นใหม่ (2022–2023) จากไลบรารี MEALPY 3.0.3 (Van Thieu & Mirjalili, J. Syst. Archit. 2023)
ใช้งบ FE เท่ากันทุกตัว: นับการประเมินเอง และหยุดทันทีเมื่อครบงบ (ไม่ใช้ตัวนับของ mealpy)
  GJO     Golden Jackal Optimization        (Chopra & Ansari, Expert Syst. Appl. 2022)
  COA     Coati Optimization Algorithm      (Dehghani et al., Knowl.-Based Syst. 2023)
  SHO     Sea-Horse Optimizer               (Zhao et al., Appl. Intell. 2023)
  WaOA    Walrus Optimization Algorithm     (Trojovský & Dehghani, Sci. Rep. 2023)
  OOA     Osprey Optimization Algorithm     (Dehghani & Trojovský, Front. Mech. Eng. 2023)
ขนาดประชากร 30 (เท่ากับ PSO/GWO/WOA/MFO/ABC) พารามิเตอร์อื่นใช้ค่าเริ่มต้นของผู้เสนอ (ตามที่ mealpy ตั้ง)
"""
import numpy as np
from baselines import _Counter, _bounds


class _Budget(Exception):
    pass


def _run(cls_name, module, func, dim, lb, ub, max_fe, seed=None, N=30):
    import importlib
    from mealpy import FloatVar
    cls = getattr(importlib.import_module(f"mealpy.swarm_based.{module}"), cls_name)
    lb, ub = _bounds(lb, ub, dim)
    ev = _Counter(func, max_fe)

    def f(x):
        if ev.done:
            raise _Budget
        return ev(np.clip(np.asarray(x, float), lb, ub))

    problem = {"obj_func": f, "bounds": FloatVar(lb=lb.tolist(), ub=ub.tolist()),
               "minmax": "min", "log_to": None}
    # ตั้งจำนวน epoch ให้ตารางเวลา (t/T) ของแต่ละอัลกอริทึมสิ้นสุดพร้อมงบ FE:
    # วัดจำนวน FE ต่อ epoch ของอัลกอริทึมนั้นก่อน (บนฟังก์ชันง่าย ๆ ไม่นับงบ)
    k = _fe_per_epoch(cls, dim, N)
    T = max(1, min(100000, int(np.ceil((max_fe - N) / k))))
    model = cls(epoch=T, pop_size=N)
    try:
        model.solve(problem, seed=int(seed) if seed is not None else None)
    except _Budget:
        pass
    return ev.result()


_K = {}


def _fe_per_epoch(cls, dim, N):
    from mealpy import FloatVar
    key = (cls.__name__, dim, N)
    if key not in _K:
        cnt = [0]
        def g(x):
            cnt[0] += 1
            return float(np.sum(np.asarray(x) ** 2))
        res = {}
        for E in (2, 4):
            cnt[0] = 0
            cls(epoch=E, pop_size=N).solve({"obj_func": g, "bounds": FloatVar(lb=[-1.0] * dim, ub=[1.0] * dim),
                                            "minmax": "min", "log_to": None}, seed=0)
            res[E] = cnt[0]
        _K[key] = max(1.0, (res[4] - res[2]) / 2)
    return _K[key]


def gjo(func, dim, lb, ub, max_fe, seed=None):  return _run("OriginalGJO", "GJO", func, dim, lb, ub, max_fe, seed)
def coa(func, dim, lb, ub, max_fe, seed=None):  return _run("OriginalCoatiOA", "CoatiOA", func, dim, lb, ub, max_fe, seed)
def sho(func, dim, lb, ub, max_fe, seed=None):  return _run("OriginalSeaHO", "SeaHO", func, dim, lb, ub, max_fe, seed)
def waoa(func, dim, lb, ub, max_fe, seed=None): return _run("OriginalWaOA", "WaOA", func, dim, lb, ub, max_fe, seed)
def ooa(func, dim, lb, ub, max_fe, seed=None):  return _run("OriginalOOA", "OOA", func, dim, lb, ub, max_fe, seed)

RECENT = {"GJO": gjo, "COA": coa, "SHO": sho, "WaOA": waoa, "OOA": ooa}
