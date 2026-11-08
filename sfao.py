"""
SFAO: Silkworm Feeding & Avoidance Optimization (prototype)
============================================================
เมตาฮิวริสติกจากพฤติกรรมหนอนไหม (minimization)

กลไก
  M1  ดมกลิ่นอาหาร        Scent attraction (roulette ตามความแรงกลิ่น)
  M2  ส่ายหัวกินใบ         Head-swing local search (กัดเป็นแนวโค้งตามทิศที่หันหน้า)
  M3  หนีกลิ่นยา           Toxic avoidance (แรงผลักจากคลังคำตอบแย่ที่ค่อย ๆ จาง)
  M4  แย่งอาหารเมื่อแออัด  Crowding & depletion (ความจุใบ + ความจำใบที่ถูกกินแล้ว)
  M5  ไหมนอน/ลอกคราบ      Molting (เปลี่ยนวัย 5 วัย, ปักหมุด elite, ฟักใหม่บางส่วน)
  M6  โรคระบาด/คัดออก      Disease & culling (ป่วย -> ซึม -> ติดต่อ -> หยิบออก)
  +   ชักใยรูปเลข 8        Spinning (ขัดเกลาคำตอบช่วงท้าย)

ใช้งาน
  from sfao import SFAO
  best_x, best_f, curve = SFAO().optimize(f, dim=30, lb=-100, ub=100, max_fe=30000, seed=1)
"""
import numpy as np


class SFAO:
    def __init__(self, pop_size=30,
                 # M5 ตารางวัย: สัดส่วนงบ FE สะสมเมื่อจบแต่ละวัย (วัยหลังกินหนักกว่า)
                 stage_ends=(0.05, 0.15, 0.30, 0.55, 1.0),
                 reach=(1.0, 0.7, 0.45, 0.25, 0.12),   # L_s เทียบกับการกระจายตัวของฝูง
                 bites=(1, 2, 2, 3, 3),                 # m_s จำนวนคำต่อการส่ายหัว
                 alpha=(0.8, 1.0, 1.2, 1.4, 1.6),       # แรงดึงเข้าหาอาหาร
                 gamma=(1.0, 0.8, 0.6, 0.4, 0.2),       # แรงหนีกลิ่นยา
                 # M3
                 k_tox=3, rho_tox=0.9, tox_max=40,
                 # M4
                 capacity=None, rho_dep=0.95, dep_max=80, eta=1.0,
                 # M5
                 molt_reseed=0.0,
                 # M6
                 T_s=10, delta=0.2, beta=0.3, rho_max=0.5, T_imm=5,
                 # ชักใย
                 spin_start=0.8, spin_bites=4,
                 # ความจำทิศกิน (เรียนรู้ทิศจากการก้าวที่สำเร็จ)
                 mem_size=None, mem_reg=1e-6, mem_norm="unit",
                 attract_mode="scalar",
                 egg_max=10,
                 # สวิตช์ปิด/เปิดกลไก (ใช้ทำ ablation)
                 use_attract=True, use_swing=True, use_toxic=True,
                 use_crowd=True, use_molt=True, use_disease=True, use_spin=True,
                 use_memory=True,
                 # เข้าดักแด้: ฝูงเริ่มใหญ่แล้วเล็กลง (หนอนที่โตเต็มที่ชักใยออกจากฝูง)
                 use_cocoon=False, cocoon_init=18, cocoon_min=8, cocoon_cap=None, food_per_worm=None,
                 # ชักใยแบบปรับวงได้: วงแคบลงเมื่อเจอที่ดี (ผนังรังแน่นขึ้น) และใช้งบช่วงท้ายมากขึ้น
                 spin_mode="fixed", spin_share=0.5):
        self.N = pop_size
        self.stage_ends = np.array(stage_ends)
        self.reach, self.bites = reach, bites
        self.alpha, self.gamma = alpha, gamma
        self.k_tox, self.rho_tox, self.tox_max = k_tox, rho_tox, tox_max
        self.C = capacity if capacity is not None else max(2, pop_size // 6)
        self.rho_dep, self.dep_max, self.eta = rho_dep, dep_max, eta
        self.molt_reseed = molt_reseed
        self.T_s, self.delta, self.beta = T_s, delta, beta
        self.rho_max, self.T_imm = rho_max, T_imm
        self.spin_start, self.spin_bites = spin_start, spin_bites
        self.egg_max = egg_max
        self.bite_lp = 20     # ช่วงจำผลสำเร็จของวิธีกัด (รอบ)
        self.mem_size, self.mem_reg, self.mem_norm = mem_size, mem_reg, mem_norm
        self.attract_mode = attract_mode
        self.use = dict(attract=use_attract, swing=use_swing, toxic=use_toxic,
                        crowd=use_crowd, molt=use_molt, disease=use_disease,
                        spin=use_spin, memory=use_memory, cocoon=use_cocoon)
        self.cocoon_init, self.cocoon_min = cocoon_init, cocoon_min
        self.cocoon_cap = cocoon_cap   # เพดานฝูงเริ่มต้น (None = ไม่จำกัด)
        # ความหนาแน่นการเลี้ยง: อาหาร (งบ FE) ต่อหนอนหนึ่งตัวอย่างน้อยเท่านี้ -> ฝูงเริ่ม <= max_fe / food_per_worm
        self.food_per_worm = food_per_worm
        self.spin_mode, self.spin_share = spin_mode, spin_share

    # ------------------------------------------------------------------
    def optimize(self, func, dim, lb, ub, max_fe, seed=None, log=False):
        rng = np.random.default_rng(seed)
        lb = np.full(dim, lb, float) if np.isscalar(lb) else np.asarray(lb, float)
        ub = np.full(dim, ub, float) if np.isscalar(ub) else np.asarray(ub, float)
        span = ub - lb
        N, D = self.N, dim
        if self.use["cocoon"]:
            N = max(self.N, int(round(self.cocoon_init * dim)))   # ฝูงเริ่มต้น (หนอนวัยแรก)
            if self.cocoon_cap is not None:
                N = max(self.N, min(N, int(self.cocoon_cap)))
            if self.food_per_worm is not None:
                N = max(self.N, min(N, int(max_fe // self.food_per_worm)))
            N_min = max(4, min(self.cocoon_min, N))
            N0 = N
        self.fe = 0
        self.best_f, self.best_x = np.inf, None
        curve = []
        self.history = {"infected": [], "diversity": []}

        def evaluate(x):
            if self.fe >= max_fe:
                return np.inf
            v = float(func(x))
            self.fe += 1
            if v < self.best_f:
                self.best_f, self.best_x = v, x.copy()
            curve.append(self.best_f)
            return v

        def unit(v):
            n = np.linalg.norm(v)
            return v / n if n > 1e-300 else np.zeros_like(v)

        def rand_unit():
            return unit(rng.standard_normal(D))

        # ---------- เริ่มต้น ----------
        X = lb + rng.random((N, D)) * span
        F = np.array([evaluate(x) for x in X])
        stag = np.zeros(N, int)          # ความหิว/รอบที่ไม่พัฒนา
        infected = np.zeros(N, bool)     # M6 สถานะป่วย
        sev = np.zeros(N)                # ความรุนแรง h_i
        immune = np.zeros(N, int)        # ภูมิคุ้มกันคงเหลือ
        tox_Z, tox_I = np.empty((0, D)), np.empty(0)   # M3 คลังกลิ่นยา
        dep_P, dep_V = np.empty((0, D)), np.empty(0)   # M4 คลังใบหมด
        eggs = [X[np.argmin(F)].copy()]                 # คลังไข่จำศีล
        egg_f = [F.min()]
        stage = 0
        # สถิติความสำเร็จของการกัด 3 แบบ: [แนวโค้ง, แทะรายมิติ, ตามความจำทิศกิน]
        n_type = 3 if self.use["memory"] else 2
        bite_succ, bite_try = np.ones(n_type), np.full(n_type, 2.0)
        # ความจำทิศกิน: เก็บเวกเตอร์ก้าวที่ทำให้เจออาหารดีขึ้น
        M = self.mem_size or max(20, 3 * D)
        mem = np.zeros((M, D)); mem_n = 0; mem_ptr = 0
        Bm, dm = np.eye(D), np.ones(D)    # แกนหลักและสเกลที่เรียนรู้ได้
        spin_sigma = None                  # รัศมีวงชักใย (แบบปรับได้)

        while self.fe < max_fe:
            prog = self.fe / max_fe
            # ---------- เข้าดักแด้: หนอนที่โตเต็มที่ออกจากฝูง ----------
            if self.use["cocoon"]:
                target = int(round(N0 + (N_min - N0) * prog))
                k_out = len(X) - max(target, N_min)
                if k_out > 0:
                    order = np.argsort(F)
                    best_i = order[0]
                    # โตเต็มที่ = อิ่มแล้ว (ค่าดี) แต่หยุดพัฒนา; ถ้าไม่พอให้ตัวที่แย่ที่สุดออกก่อน
                    mature = [i for i in order[1:] if stag[i] >= 2]
                    rest = [i for i in order[::-1] if i != best_i and i not in mature]
                    out = (mature + rest)[:k_out]
                    for i in out:                      # เก็บไว้ในคลังรังไหม (ใช้ฟักเมื่อคัดตัวป่วย)
                        if not eggs or min(np.linalg.norm(e - X[i]) for e in eggs) > 1e-12:
                            eggs.append(X[i].copy()); egg_f.append(F[i])
                    while len(eggs) > self.egg_max:
                        w = int(np.argmax(egg_f)); eggs.pop(w); egg_f.pop(w)
                    keep = np.setdiff1d(np.arange(len(X)), out)
                    X, F = X[keep], F[keep]
                    stag, infected, sev, immune = stag[keep], infected[keep], sev[keep], immune[keep]
                    N = len(X)
            new_stage = int(np.searchsorted(self.stage_ends, prog, side="right"))
            new_stage = min(new_stage, 4)

            # ---------- M5 ไหมนอน/ลอกคราบ ----------
            if new_stage != stage:
                stage = new_stage
                if self.use["molt"]:
                    stag[:] = 0
                    dep_V *= 0.5
                    k = int(self.molt_reseed * N)
                    worst = np.argsort(F)[-k:] if k > 0 else []
                    spread = np.mean(np.linalg.norm(X - X.mean(0), axis=1)) + 1e-12
                    for w in worst:
                        egg = eggs[rng.integers(len(eggs))]
                        X[w] = np.clip(egg + 0.5 * spread * rng.standard_normal(D) / np.sqrt(D), lb, ub)
                        F[w] = evaluate(X[w])
                        infected[w], sev[w] = False, 0.0

            s = stage
            centroid = X.mean(0)
            spread = np.mean(np.linalg.norm(X - centroid, axis=1)) + 1e-12
            # ช่วงหัวเอื้อม: ผูกกับการกระจายตัวของฝูง แต่มีพื้นขั้นต่ำที่ค่อย ๆ ลดลง
            # (กันไม่ให้ฝูงหดตัวแล้วก้าวเล็กลงเรื่อย ๆ จนติดก่อนเวลา)
            floor = 0.1 * span.mean() * (1.0 - prog) ** 2
            L = self.reach[s] * max(spread, floor)
            m = self.bites[s]
            r_c = 0.3 * spread                          # รัศมีแออัด/ติดเชื้อ
            if self.use["memory"] and mem_n >= D:
                S_ = mem[:mem_n]
                nr = np.linalg.norm(S_, axis=1, keepdims=True) + 1e-300
                if self.mem_norm == "unit":      # จำแค่ทิศ
                    S_ = S_ / nr
                else:                            # จำทั้งทิศและขนาดสัมพัทธ์ของก้าว
                    S_ = S_ / np.median(nr)
                    S_ = S_ / np.maximum(1.0, np.linalg.norm(S_, axis=1, keepdims=True) / 3.0)
                Cm = S_.T @ S_ / mem_n
                Cm = Cm / (np.trace(Cm) / D + 1e-300) + self.mem_reg * np.eye(D)
                ev_, Bm = np.linalg.eigh(Cm)
                dm = np.sqrt(np.maximum(ev_, 1e-12))
                dm = dm / np.sqrt(np.mean(dm ** 2))        # สเกลเฉลี่ย = 1
            sigma = 0.5 * spread                        # รัศมีกลิ่นยา

            # ---------- คุณภาพอาหาร และ M4 อาหารที่ได้จริง ----------
            fb, fw = F.min(), F.max()
            q = (fw - F) / (fw - fb + 1e-300)
            Dm = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
            if self.use["crowd"]:
                n_nb = (Dm < r_c).sum(1) - 1
                crowd_pen = 1.0 / (1.0 + np.maximum(0, n_nb - self.C) / self.C)
                if len(dep_V):
                    dd = ((X[:, None, :] - dep_P[None, :, :]) ** 2).sum(-1)
                    depl = (dep_V[None, :] * np.exp(-dd / (r_c ** 2 + 1e-300))).max(1)
                else:
                    depl = np.zeros(N)
                q_eff = q * crowd_pen * (1.0 - 0.5 * depl)
            else:
                n_nb = np.zeros(N, int)
                q_eff = q

            improved = np.zeros(N, bool)
            for i in rng.permutation(N):
                h = sev[i] if infected[i] else 0.0
                # M6 อาการซึม: ตัวป่วยมีโอกาสไม่กิน (ไม่เปลืองงบประเมิน)
                if self.use["disease"] and infected[i] and rng.random() < h:
                    continue
                if self.fe >= max_fe:
                    break
                xi = X[i]

                # ---------- M1 ดมกลิ่นอาหาร ----------
                better = np.where(q_eff > q_eff[i])[0]
                if self.use["attract"] and len(better):
                    S = q_eff[better] / (1.0 + (Dm[i, better] / r_c) ** 2)
                    j = better[rng.choice(len(better), p=S / S.sum())]
                    food = X[j]
                else:
                    food = self.best_x
                face = unit(food - xi)
                if not face.any():
                    face = rand_unit()
                step = np.zeros(D)
                if self.use["attract"]:
                    # คลานตรงเข้าหาอาหาร (สุ่มระยะทางเดียวทั้งตัว = ไม่ขึ้นกับการหมุนแกน)
                    r_att = rng.random() if self.attract_mode == "scalar" else rng.random(D)
                    step += self.alpha[s] * r_att * (food - xi)

                # ---------- M3 หนีกลิ่นยา ----------
                if self.use["toxic"] and len(tox_I):
                    vec = tox_Z - xi
                    dist = np.linalg.norm(vec, axis=1) + 1e-300
                    w = tox_I * np.exp(-dist ** 2 / sigma ** 2)
                    step -= self.gamma[s] * L * (w[:, None] * vec / dist[:, None]).sum(0)

                # ---------- M4 ขับออกเมื่อใบแน่นเกินความจุ ----------
                if self.use["crowd"] and n_nb[i] > self.C:
                    nb = np.where(Dm[i] < r_c)[0]
                    rank = np.sum(F[nb] < F[i])
                    if rank >= self.C:   # แย่งไม่ทัน
                        step += self.eta * L * unit(xi - X[nb].mean(0) + 1e-9 * rng.standard_normal(D))

                body = np.clip(xi + (1.0 - h) * step, lb, ub)

                # ---------- M2 ส่ายหัวกินใบ ----------
                cand, ctype = [body], [-1]
                if self.use["swing"]:
                    u2 = rand_unit()
                    u2 = unit(u2 - u2.dot(face) * face)
                    rate = bite_succ / bite_try
                    pr = np.maximum(rate / rate.sum(), 0.1)
                    pr = pr / pr.sum()
                    for _ in range(m):
                        bt = rng.choice(n_type, p=pr)
                        if bt == 2:
                            # แบบ ค: กัดตามทิศที่เคยเจอใบดี (รูปร่างที่เรียนรู้จากความจำ)
                            z = rng.standard_normal(D)
                            y = body + (1.0 - h) * L * (Bm @ (dm * z)) / np.sqrt(D)
                            ctype.append(2)
                        elif bt == 0:
                            # แบบ ก: ส่ายหัวเป็นแนวโค้งตามทิศที่หันหน้า
                            th = rng.uniform(-np.pi / 2, np.pi / 2)
                            y = body + (1.0 - h) * L * rng.random() * (np.cos(th) * face + np.sin(th) * u2)
                            ctype.append(0)
                        else:
                            # แบบ ข: แทะรายมิติ (กัดคำเล็ก ๆ เปลี่ยนเฉพาะบางมิติ)
                            k_d = 1 + rng.binomial(D - 1, 0.1)
                            dims = rng.choice(D, k_d, replace=False)
                            y = body.copy()
                            y[dims] += (1.0 - h) * L * rng.standard_normal(k_d) / np.sqrt(k_d)
                            ctype.append(1)
                        cand.append(np.clip(y, lb, ub))
                elif not self.use["attract"]:
                    cand = [np.clip(xi + L * rng.standard_normal(D) / np.sqrt(D), lb, ub)]
                fc = [evaluate(c) for c in cand]
                k = int(np.argmin(fc))
                for t_, f_ in zip(ctype, fc):
                    if t_ >= 0:
                        bite_try[t_] += 1
                        bite_succ[t_] += f_ < F[i]
                if fc[k] < F[i]:
                    # กินอิ่มจริงไหม: พัฒนาน้อยกว่า 0.1% ของระยะห่างจากตัวดีที่สุด = ยังซึม
                    real = (F[i] - fc[k]) > 1e-3 * (F[i] - fb) + 1e-300
                    if self.use["memory"]:            # จำทิศที่กินแล้วอิ่ม
                        mem[mem_ptr] = cand[k] - xi
                        mem_ptr = (mem_ptr + 1) % M; mem_n = min(mem_n + 1, M)
                    X[i], F[i] = cand[k], fc[k]
                    if real:
                        improved[i] = True
                        stag[i] = 0
                    else:
                        stag[i] += 1
                    if self.use["crowd"]:   # ใบตรงนี้ถูกกินไปแล้ว
                        dep_P = np.vstack([dep_P, X[i]])
                        dep_V = np.append(dep_V, 1.0)
                else:
                    stag[i] += 1

            bite_succ *= 1 - 1 / self.bite_lp
            bite_try *= 1 - 1 / self.bite_lp
            bite_succ += 0.01; bite_try += 0.02

            # ---------- อัปเดตความจำ ----------
            if self.use["toxic"]:
                worst = np.argsort(F)[-self.k_tox:]
                tox_Z = np.vstack([tox_Z, X[worst]])
                tox_I = np.append(tox_I * self.rho_tox, np.ones(len(worst)))
                keep = tox_I > 0.05
                tox_Z, tox_I = tox_Z[keep][-self.tox_max:], tox_I[keep][-self.tox_max:]
            if self.use["crowd"] and len(dep_V):
                dep_V = dep_V * self.rho_dep
                keep = dep_V > 0.05
                dep_P, dep_V = dep_P[keep][-self.dep_max:], dep_V[keep][-self.dep_max:]

            # คลังไข่จำศีล: เก็บคำตอบดีที่อยู่ห่างกัน
            bi = int(np.argmin(F))
            if min(np.linalg.norm(e - X[bi]) for e in eggs) > r_c:
                eggs.append(X[bi].copy())
                egg_f.append(F[bi])
                if len(eggs) > self.egg_max:     # ทิ้งไข่ที่แย่ที่สุด (ใช้ค่าที่จำไว้ ไม่เสีย FE)
                    w = int(np.argmax(egg_f))
                    eggs.pop(w)
                    egg_f.pop(w)

            # ---------- M6 โรคระบาดและการคัดออก ----------
            if self.use["disease"]:
                immune = np.maximum(0, immune - 1)
                # หายเอง
                rec = infected & improved
                infected[rec], sev[rec], immune[rec] = False, 0.0, self.T_imm
                # ทรุดลง
                sev[infected & ~improved] += self.delta
                # เริ่มป่วย
                newly = (~infected) & (stag >= self.T_s) & (immune == 0)
                infected[newly], sev[newly] = True, 0.2
                # ติดต่อ
                sf = F.std() + 1e-300
                for i in np.where(infected)[0]:
                    sus = np.where((~infected) & (immune == 0))[0]
                    if not len(sus):
                        break
                    p = self.beta * np.exp(-Dm[i, sus] ** 2 / r_c ** 2) * np.exp(-np.abs(F[i] - F[sus]) / sf)
                    hit = sus[rng.random(len(sus)) < p]
                    infected[hit], sev[hit] = True, 0.2
                # ระบาดทั้งฟาร์ม -> คัดตัวป่วยทั้งหมด
                outbreak = infected.mean() > self.rho_max
                cull = np.where(infected & ((sev >= 1.0) | outbreak))[0]
                cull = cull[cull != np.argmin(F)]          # ไม่ทิ้งตัวดีที่สุด
                if len(cull):
                    sick_c = X[infected].mean(0)
                    for c in cull:
                        if self.fe >= max_fe:
                            break
                        if self.use["toxic"]:              # ฆ่าเชื้อพื้นที่
                            tox_Z = np.vstack([tox_Z, X[c]])
                            tox_I = np.append(tox_I, 1.0)
                        if rng.random() < 0.5 and len(eggs) > 1:
                            base = eggs[rng.integers(len(eggs))]
                            xn = base + spread * rng.standard_normal(D) / np.sqrt(D)
                        else:                              # opposition + หนีจุดระบาด
                            xn = lb + ub - X[c] + 0.1 * span * rng.standard_normal(D)
                            xn += 0.5 * L * unit(xn - sick_c)
                        X[c] = np.clip(xn, lb, ub)
                        F[c] = evaluate(X[c])
                        infected[c], sev[c], stag[c], immune[c] = False, 0.0, 0, self.T_imm

            # ---------- ชักใยรูปเลข 8 ----------
            if self.use["spin"] and prog >= self.spin_start and self.spin_mode == "adaptive":
                # ชักใยแบบปรับวง: สุ่มจุดรอบรังตามรูปร่างที่เรียนรู้ (ความจำทิศกิน)
                # เจอที่ดีกว่า -> วงกว้างขึ้นเล็กน้อย, ไม่เจอ -> วงแคบลง (กฎ ~1/5)
                if spin_sigma is None:
                    spin_sigma = max(L, 1e-12 * span.mean())
                n_spin = max(self.spin_bites, int(self.spin_share * N * self.bites[s]))
                for _ in range(n_spin):
                    if self.fe >= max_fe: break
                    z = rng.standard_normal(D)
                    step = (Bm @ (dm * z)) / np.sqrt(D) if self.use["memory"] else z / np.sqrt(D)
                    prev = self.best_x.copy(); fb_old = self.best_f
                    y = np.clip(prev + spin_sigma * step, lb, ub)
                    evaluate(y)
                    if self.best_f < fb_old:
                        spin_sigma *= 1.5
                        if self.use["memory"]:
                            mem[mem_ptr] = y - prev            # จำทิศที่ชักใยแล้วได้ผล
                            mem_ptr = (mem_ptr + 1) % M; mem_n = min(mem_n + 1, M)
                    else:
                        spin_sigma *= 0.9
                    spin_sigma = max(spin_sigma, 1e-15 * span.mean())
                bi = int(np.argmax(F))
                X[bi], F[bi] = self.best_x.copy(), self.best_f
            elif self.use["spin"] and prog >= self.spin_start:
                frac = (prog - self.spin_start) / (1 - self.spin_start)
                Lf = L * (1e-3 ** frac)
                if self.use["memory"]:
                    u1 = unit(Bm @ (dm * rng.standard_normal(D)))
                    u2 = unit(Bm @ (dm * rng.standard_normal(D)))
                else:
                    u1, u2 = rand_unit(), rand_unit()
                u2 = unit(u2 - u2.dot(u1) * u1)
                w0 = rng.uniform(0, 2 * np.pi)
                for k in range(self.spin_bites):
                    tau = w0 + 2 * np.pi * k / self.spin_bites
                    y = np.clip(self.best_x + Lf * (np.sin(tau) * u1 + 0.5 * np.sin(2 * tau) * u2), lb, ub)
                    evaluate(y)
                bi = int(np.argmax(F))           # แทนตัวแย่สุดด้วย elite ที่ขัดเกลาแล้ว
                X[bi], F[bi] = self.best_x.copy(), self.best_f

            self.history["infected"].append(infected.mean())
            self.history["diversity"].append(spread)

        curve = np.array(curve[:max_fe])
        return self.best_x, self.best_f, curve
