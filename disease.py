"""
กลไกโรคระบาดและการคัดออก (M6) แบบแยกเป็นโมดูล — เสียบเข้ากับอัลกอริทึมประชากรใดก็ได้
ใช้ทดสอบว่ากลไกนี้ช่วยอัลกอริทึมอื่น (PSO, DE) ได้จริง ไม่ได้ช่วยเฉพาะ SFAO

วงจร: ปกติ -> (ไม่พัฒนา T_s รอบ) ป่วย -> ทรุดลงทีละ delta -> ติดต่อตัวใกล้ ๆ ที่ค่าใกล้กัน
      -> หายเองถ้าพัฒนาได้ (ได้ภูมิ T_imm รอบ) / ถูกหยิบออกเมื่อความรุนแรงถึง 1 หรือระบาดทั้งฟาร์ม
อาการซึม: ตัวป่วยข้ามการประเมินด้วยโอกาส h และก้าวเล็กลงตาม (1-h)
"""
import numpy as np


class Disease:
    def __init__(self, N, T_s=10, delta=0.2, beta=0.3, rho_max=0.5, T_imm=5, tol=1e-3):
        self.N, self.T_s, self.delta, self.beta = N, T_s, delta, beta
        self.rho_max, self.T_imm, self.tol = rho_max, T_imm, tol
        self.stag = np.zeros(N, int)
        self.inf = np.zeros(N, bool)
        self.sev = np.zeros(N)
        self.imm = np.zeros(N, int)

    def resize(self, N):
        """สำหรับอัลกอริทึมที่ลดขนาดประชากร"""
        self.stag, self.inf = self.stag[:N], self.inf[:N]
        self.sev, self.imm = self.sev[:N], self.imm[:N]
        self.N = N

    def lethargy(self):
        """ระดับความซึมของแต่ละตัว (0 = ปกติ)"""
        return np.where(self.inf, self.sev, 0.0)

    def skip(self, rng):
        """ตัวที่ซึมจนไม่กินในรอบนี้"""
        return self.inf & (rng.random(self.N) < self.sev)

    def update(self, X, F, F_old, rng):
        """เรียกหลังจบแต่ละรอบ คืนดัชนีที่ต้องหยิบออก"""
        fb = F.min()
        improved = (F_old - F) > self.tol * (F_old - fb) + 1e-300
        self.stag = np.where(improved, 0, self.stag + 1)
        # เกณฑ์ซึมปรับตามอัตราสำเร็จของทั้งฝูง: อัลกอริทึมที่ปกติพัฒนาช้า (เช่น DE) ต้องรอนานกว่า
        rate = getattr(self, "rate", improved.mean())
        self.rate = 0.9 * rate + 0.1 * improved.mean()
        T_s = max(self.T_s, int(np.ceil(2.0 / max(self.rate, 0.02))))
        self.imm = np.maximum(0, self.imm - 1)
        rec = self.inf & improved
        self.inf[rec], self.sev[rec], self.imm[rec] = False, 0.0, self.T_imm
        self.sev[self.inf & ~improved] += self.delta
        new = (~self.inf) & (self.stag >= T_s) & (self.imm == 0)
        self.inf[new], self.sev[new] = True, 0.2
        spread = np.mean(np.linalg.norm(X - X.mean(0), axis=1)) + 1e-12
        r = 0.3 * spread
        sf = F.std() + 1e-300
        for i in np.where(self.inf)[0]:
            sus = np.where((~self.inf) & (self.imm == 0))[0]
            if not len(sus):
                break
            d2 = ((X[sus] - X[i]) ** 2).sum(1)
            p = self.beta * np.exp(-d2 / r ** 2) * np.exp(-np.abs(F[i] - F[sus]) / sf)
            hit = sus[rng.random(len(sus)) < p]
            self.inf[hit], self.sev[hit] = True, 0.2
        outbreak = self.inf.mean() > self.rho_max
        cull = np.where(self.inf & ((self.sev >= 1.0) | outbreak))[0]
        cull = cull[cull != np.argmin(F)]
        self.centroid = X[self.inf].mean(0) if self.inf.any() else X.mean(0)
        self.spread = spread
        self.elite = X[np.argsort(F)[:max(2, len(F) // 4)]].copy()   # คลังไข่: ตัวดี 25% บน
        return cull

    def replacement(self, x_old, lb, ub, rng):
        """ตัวใหม่แทนตัวที่ถูกหยิบออก: opposition + หนีจุดระบาด"""
        span = ub - lb
        D = len(x_old)
        if rng.random() < 0.5:     # ฟักจากไข่ของตัวดี (สำรวจรอบ ๆ พื้นที่ดีอื่น)
            base = self.elite[rng.integers(len(self.elite))]
            xn = base + self.spread * rng.standard_normal(D) / np.sqrt(D)
        else:                      # opposition + หนีจุดระบาด
            xn = lb + ub - x_old + 0.1 * span * rng.standard_normal(D)
            v = xn - self.centroid
            xn = xn + 0.5 * self.spread * v / (np.linalg.norm(v) + 1e-300)
        return np.clip(xn, lb, ub)

    def reset(self, i):
        self.inf[i], self.sev[i], self.stag[i], self.imm[i] = False, 0.0, 0, self.T_imm
