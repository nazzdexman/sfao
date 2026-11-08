"""
Berth Allocation Problem (BAP) — ข้อมูลจริงท่าเรือบาร์เซโลนา 2024
ชนิดปัญหา: hyb|dyn|fix|max(comp) (Bierwirth & Meisel 2015)
  - dynamic : เรือมาถึงไม่พร้อมกัน (ห้ามเข้าท่าก่อนเวลามาถึง)
  - hybrid  : ท่าแบ่งเป็นช่องยาว 1 หน่วย เรือหนึ่งลำใช้หลายช่องติดกันตามความยาวเรือ
  - fixed   : เวลาขนถ่ายขึ้นกับเรือ ไม่ขึ้นกับตำแหน่ง
  - objective: minimise makespan (เวลาที่เรือลำสุดท้ายเสร็จ)
ที่มา: https://github.com/alberto-santini/berth-allocation-problems (มี BKS + dual bound)

การเข้ารหัสสำหรับเมตาฮิวริสติกแบบต่อเนื่อง (random key, 2n มิติ ในช่วง [0,1])
  key[0:n]  = ลำดับความสำคัญของเรือ (น้อย = จัดก่อน)
  key[n:2n] = ตำแหน่งที่อยากจอด (สัดส่วนตามแนวท่า)
ตัวถอดรหัส: จัดเรือทีละลำตามลำดับ ให้เข้าท่าเร็วที่สุดเท่าที่ทำได้ (ไม่ก่อนเวลามาถึง)
ถ้าเร็วสุดเท่ากันหลายตำแหน่ง เลือกตำแหน่งที่ใกล้ตำแหน่งที่อยากจอดที่สุด
"""
import json
import os
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BAP_DATA", os.path.join(_here, "bap_data")
                      if os.path.isdir(os.path.join(_here, "bap_data")) else os.path.join(_here, "bap_data"))

HARD = ["bcn_36A_7", "bcn_36A_10", "bcn_36A_14", "bcn_36A_15", "bcn_36A_11", "bcn_36A_13",
        "bcn_36A_16", "bcn_36A_6", "bcn_36A_20", "bcn_36A_23", "bcn_36A_22", "bcn_36A_19",
        "bcn_36A_40", "bcn_36A_35", "bcn_24B_41", "bcn_24B_34", "bcn_24B_22", "bcn_24B_40",
        "bcn_24B_44", "bcn_24B_15"]


class BAP:
    def __init__(self, name, data_dir=DATA):
        d = json.load(open(f"{data_dir}/instances/hyb_dyn_fix_max-comp/{name}.json"))
        self.name = name
        self.n, self.B = d["n_ships"], d["n_berths"]
        self.a = np.array(d["arrival_time"], int)
        self.h = np.array(d["handling_time"], int)
        self.L = np.array(np.round(d["ship_len"]), int)
        assert all(x == 1 for x in d["berth_len"])
        self.lb_trivial = int((self.a + self.h).max())
        self.T_max = int(self.a.max() + self.h.sum() + 1)  # ขอบเวลาที่พอแน่นอน
        self.T = min(self.T_max, int(1.6 * self.lb_trivial + self.h.max()))  # ขอบที่ใช้จริง (ขยายเมื่อไม่พอ)
        try:
            b = json.load(open(f"{data_dir}/bks/hyb_dyn_fix_max-comp/bks-{name}.json"))
            self.bks, self.dual = b["makespan"], b["dual_bound"]
        except FileNotFoundError:
            self.bks = self.dual = None
        self.dim = 2 * self.n

    # ---------- ถอดรหัส ----------
    def decode(self, key):
        key = np.asarray(key)
        return self.decode_order(np.argsort(key[:self.n], kind="stable"), key[self.n:])

    def decode_order(self, order, pref):
        """ถอดรหัสจากลำดับเรือโดยตรง (ใช้กับอัลกอริทึมแบบจัดลำดับ)"""
        while True:
            out = self._decode(order, np.clip(pref, 0, 1), self.T)
            if out is not None:
                return out
            self.T = min(self.T_max, 2 * self.T)

    def objective_order(self, order, pref):
        start, _ = self.decode_order(order, pref)
        comp = start + self.h
        return comp.max() + 1e-3 * comp.mean() / self.T_max, start

    def _decode(self, order, pref, T):
        n, B = self.n, self.B
        busy = np.zeros((B, T + 1), np.int32)           # 1 = ช่องนั้นถูกใช้ในช่วงเวลานั้น
        start = np.zeros(n, int); pos = np.zeros(n, int)
        for i in order:
            Li, hi, ai = self.L[i], self.h[i], self.a[i]
            P = B - Li + 1                                   # ตำแหน่งซ้ายสุดที่เป็นไปได้
            if ai + hi > T:
                return None
            sub = busy[:, ai:]                               # ดูเฉพาะหลังเวลามาถึง
            W = sub.shape[1]
            cs = np.vstack([np.zeros((1, W), np.int32), np.cumsum(sub, axis=0)])
            block = (cs[Li:Li + P] - cs[0:P]) > 0            # (P, W) ช่วงตำแหน่ง×เวลาที่ติด
            ct = np.concatenate([np.zeros((P, 1), np.int32), np.cumsum(block, axis=1)], axis=1)
            win = ct[:, hi:W + 1] - ct[:, 0:W + 1 - hi]      # จำนวนช่วงที่ติดในหน้าต่าง [t, t+h)
            free = win == 0
            has = free.any(axis=1)
            if not has.any():
                return None
            t_first = np.where(has, free.argmax(axis=1), 10 ** 9) + ai
            tbest = t_first.min()
            cands = np.where(t_first == tbest)[0]
            p_target = pref[i] * (P - 1)
            p = cands[np.argmin(np.abs(cands - p_target))]
            busy[p:p + Li, tbest:tbest + hi] = 1
            start[i], pos[i] = tbest, p
        return start, pos

    def makespan(self, start):
        return int((start + self.h).max())

    def objective(self, key):
        start, _ = self.decode(np.asarray(key))
        comp = start + self.h
        # makespan เป็นหลัก + เวลาเสร็จเฉลี่ยเล็กน้อยเพื่อแยกคำตอบที่ makespan เท่ากัน
        return comp.max() + 1e-3 * comp.mean() / self.T_max

    def check(self, start, pos):
        """ตรวจความถูกต้องของคำตอบ: เวลามาถึง, ขอบท่า, ไม่ชนกัน"""
        assert np.all(start >= self.a), "เข้าท่าก่อนเรือมาถึง"
        assert np.all(pos >= 0) and np.all(pos + self.L <= self.B), "เกินขอบท่า"
        for i in range(self.n):
            for j in range(i + 1, self.n):
                t_ov = start[i] < start[j] + self.h[j] and start[j] < start[i] + self.h[i]
                s_ov = pos[i] < pos[j] + self.L[j] and pos[j] < pos[i] + self.L[i]
                assert not (t_ov and s_ov), f"เรือ {i} กับ {j} ชนกัน"
        return True

    def fcfs_key(self):
        """กฎพื้นฐานที่ท่าเรือใช้จริง: มาก่อนได้ก่อน (First-Come-First-Served)"""
        k = np.zeros(self.dim)
        k[:self.n] = np.argsort(np.argsort(self.a, kind="stable")) / self.n
        return k
