"""
D-SFAO: SFAO รุ่นจัดลำดับ (discrete) สำหรับ Berth Allocation Problem
=====================================================================
คำตอบของหนอนแต่ละตัว = (ลำดับเรือ π, ตำแหน่งที่อยากจอด pref ∈ [0,1]^n)
ตัวถอดรหัส: จัดเรือตามลำดับ π ให้เข้าท่าเร็วที่สุด (ไม่ก่อนมาถึง) ใกล้ตำแหน่งที่อยากจอดที่สุด

กลไก (เก็บเฉพาะที่ ablation บน CEC พบว่าช่วย; ตัดแย่งอาหาร/ไหมนอน/กลิ่นยาออก)
  M1 ดมกลิ่นอาหาร  : เลือกเพื่อนที่ดีกว่าแบบ roulette (อร่อย+ใกล้ = กลิ่นแรง) แล้ว "คลานเข้าหา"
                     = สลับลำดับทีละคู่ให้เหมือนลำดับของเพื่อนมากขึ้น (path relinking)
  M2 ส่ายหัวกินใบ  : กัดรอบตัวด้วย "คำ" 4 แบบ เลือกแบบที่เคยได้ผลบ่อยกว่า (ปรับเองตามอัตราสำเร็จ)
        ก. แนวโค้ง   = ย้ายเรือ 1 ลำไปตำแหน่งอื่นภายในระยะหัวเอื้อม (insertion)
        ข. แทะ       = สลับเรือ 2 ลำที่อยู่ใกล้กัน (swap)
        ค. ขอบใบ     = ย้ายเรือที่เสร็จช้าที่สุด (ตัวกำหนด makespan) ให้ขึ้นก่อน
        ง. ย้ายที่กิน = เปลี่ยนตำแหน่งจอดที่อยากได้ของเรือ 1 ลำ
     ความจำทิศกิน : จำว่าย้ายเรือลำไหนขึ้นก่อนแล้วดีขึ้น -> เลือกลำนั้นบ่อยขึ้น
  M6 โรคระบาด     : ตัวที่ไม่พัฒนานาน = ป่วย -> กัดน้อยลง (ซึม) -> ติดต่อตัวที่ลำดับคล้ายกันและค่าใกล้กัน
                     -> หยิบออกแล้วฟักใหม่ (จากไข่ของตัวดีที่ถูกเขย่าบางช่วง หรือสุ่มใหม่)
  ชักใย           : ช่วงท้าย ตัวดีที่สุดลองย้ายเรือวิกฤตไปทุกตำแหน่งก่อนหน้า (ค้นหาเฉพาะที่แบบละเอียด)
ตารางวัย: ระยะหัวเอื้อมแคบลง และจำนวนคำต่อรอบเพิ่มขึ้นตามวัย
"""
import numpy as np


class _Eval:
    def __init__(self, P, max_fe):
        self.P, self.max_fe, self.fe = P, max_fe, 0
        self.best = (np.inf, None, None, None); self.curve = []

    @property
    def done(self):
        return self.fe >= self.max_fe

    def __call__(self, order, pref):
        if self.done:
            return np.inf, None
        v, start = self.P.objective_order(order, pref)
        self.fe += 1
        if v < self.best[0]:
            self.best = (v, order.copy(), pref.copy(), start.copy())
        self.curve.append(self.best[0])
        return v, start


def _ranks(order):
    r = np.empty(len(order), int); r[order] = np.arange(len(order)); return r


def _ox(donor, base, a, b):
    """order crossover: ช่วง [a,b] มาจาก donor ที่ตำแหน่งเดิม ที่เหลือเรียงตาม base"""
    n = len(base); child = -np.ones(n, int); child[a:b + 1] = donor[a:b + 1]
    seg = set(donor[a:b + 1].tolist())
    child[child < 0] = [g for g in base if g not in seg]
    return child


def _insert(order, a, b):
    o = list(order); s = o.pop(a); o.insert(b, s); return np.array(o)


class DSFAO:
    def __init__(self, pop_size=20, stage_ends=(0.1, 0.3, 0.6, 1.0), reach=(0.5, 0.3, 0.15, 0.08),
                 bites=(2, 2, 3, 4), alpha=(0.6, 0.7, 0.8, 0.9), spin_start=0.85,
                 T_s=8, delta=0.25, beta=0.3, rho_max=0.5, T_imm=4,
                 use_attract=True, use_disease=True, use_spin=True, use_memory=True, use_critical=True,
                 crawl="segment", mating=True):
        self.N = pop_size; self.stage_ends = np.array(stage_ends)
        self.reach, self.bites, self.alpha = reach, bites, alpha
        self.spin_start = spin_start
        self.T_s, self.delta, self.beta, self.rho_max, self.T_imm = T_s, delta, beta, rho_max, T_imm
        self.use = dict(attract=use_attract, disease=use_disease, spin=use_spin,
                        memory=use_memory, critical=use_critical)
        self.crawl, self.mating = crawl, mating

    def optimize(self, P, max_fe, seed=None):
        rng = np.random.default_rng(seed)
        n, N = P.n, self.N
        ev = _Eval(P, max_fe)
        X = [rng.permutation(n) for _ in range(N)]
        PR = [rng.random(n) for _ in range(N)]
        F = np.empty(N); S = [None] * N
        for i in range(N):
            F[i], S[i] = ev(X[i], PR[i])
        stag = np.zeros(N, int); inf = np.zeros(N, bool); sev = np.zeros(N); imm = np.zeros(N, int)
        n_type = 4
        succ, tries = np.ones(n_type), np.full(n_type, 2.0)
        ship_mem = np.ones(n)                       # ความจำทิศกิน: เรือที่ย้ายขึ้นก่อนแล้วดี
        eggs = []

        while not ev.done:
            prog = ev.fe / max_fe
            s = min(int(np.searchsorted(self.stage_ends, prog, side="right")), len(self.reach) - 1)
            w = max(2, int(round(self.reach[s] * n)))     # ระยะหัวเอื้อม (จำนวนตำแหน่ง)
            m = self.bites[s]
            R = np.array([_ranks(o) for o in X], float) / n
            fb, fw = F.min(), F.max()
            q = (fw - F) / (fw - fb + 1e-12)
            Dm = np.sqrt(((R[:, None, :] - R[None, :, :]) ** 2).sum(-1))
            spread = Dm.mean() + 1e-12
            improved = np.zeros(N, bool)

            for i in rng.permutation(N):
                if ev.done: break
                h = sev[i] if inf[i] else 0.0
                if self.use["disease"] and inf[i] and rng.random() < h:
                    continue                                   # ซึม ไม่กินรอบนี้
                body, bpref = X[i].copy(), PR[i].copy()
                # ---- M1 ดมกลิ่นแล้วคลานเข้าหาอาหาร ----
                better = np.where(q > q[i])[0]
                if self.use["attract"] and len(better):
                    Sw = q[better] / (1 + (Dm[i, better] / (0.5 * spread)) ** 2)
                    j = better[rng.choice(len(better), p=Sw / Sw.sum())]
                    if self.crawl == "segment":
                        # คลานตามรอยเพื่อน: ลอกช่วงลำดับต่อเนื่องจากเพื่อน (คงลำดับที่เหลือของตัวเอง)
                        Lc = min(n - 1, max(2, int(round(self.alpha[s] * (0.5 + rng.random()) * n * (1 - h)))))
                        a0 = rng.integers(0, n - Lc + 1)
                        body = _ox(X[j], body, a0, a0 + Lc - 1)
                    else:
                        diff = np.where(body != X[j])[0]
                        k = int(np.ceil(self.alpha[s] * rng.random() * len(diff) * (1 - h)))
                        for idx in rng.permutation(diff)[:k]:
                            if body[idx] != X[j][idx]:
                                jdx = int(np.where(body == X[j][idx])[0][0])
                                body[idx], body[jdx] = body[jdx], body[idx]
                    mask = rng.random(n) < self.alpha[s] * rng.random()
                    bpref[mask] = PR[j][mask]
                fbody, sbody = ev(body, bpref) if not np.array_equal(body, X[i]) or not np.array_equal(bpref, PR[i]) else (F[i], S[i])
                cand = [(fbody, body, bpref, sbody, -1, -1)]
                # ---- M2 ส่ายหัวกัดรอบตัว ----
                rate = succ / tries; pr = np.maximum(rate / rate.sum(), 0.08)
                if not self.use["critical"]: pr[2] = 0
                pr = pr / pr.sum()
                ww = max(1, int(round(w * (1 - h))))
                for _ in range(m):
                    if ev.done: break
                    bt = rng.choice(n_type, p=pr)
                    o, p_, moved = body.copy(), bpref.copy(), -1
                    if bt == 0:                                   # ก. แนวโค้ง: insertion
                        if self.use["memory"]:
                            ship = rng.choice(n, p=ship_mem / ship_mem.sum())
                            a = int(np.where(o == ship)[0][0])
                        else:
                            a = rng.integers(n)
                        b = int(np.clip(a + rng.integers(-ww, ww + 1), 0, n - 1))
                        moved = o[a] if b < a else -1
                        o = _insert(o, a, b)
                    elif bt == 1:                                 # ข. แทะ: swap ใกล้ ๆ
                        a = rng.integers(n); b = int(np.clip(a + rng.integers(-ww, ww + 1), 0, n - 1))
                        o[a], o[b] = o[b], o[a]
                    elif bt == 2:                                 # ค. ขอบใบ: เรือวิกฤตขึ้นก่อน
                        st = sbody if sbody is not None else S[i]
                        comp = st + P.h
                        crit = np.argsort(-comp)[:3]
                        ship = crit[rng.integers(len(crit))]
                        a = int(np.where(o == ship)[0][0])
                        b = max(0, a - 1 - rng.integers(ww))
                        moved = ship; o = _insert(o, a, b)
                    else:                                         # ง. ย้ายที่กิน: เปลี่ยนตำแหน่งจอด
                        sh = rng.integers(n)
                        p_[sh] = rng.random() if rng.random() < 0.5 else np.clip(p_[sh] + rng.normal(0, 0.2), 0, 1)
                    fv, sv = ev(o, p_)
                    tries[bt] += 1; succ[bt] += fv < min(F[i], fbody)
                    if fv < min(F[i], fbody) and moved >= 0:
                        ship_mem[moved] += 1.0
                    cand.append((fv, o, p_, sv, bt, moved))
                best = min(cand, key=lambda c: c[0])
                if best[0] < F[i]:
                    real = (F[i] - best[0]) > 1e-9
                    X[i], PR[i], F[i], S[i] = best[1], best[2], best[0], best[3]
                    if real:
                        improved[i] = True; stag[i] = 0
                    else:
                        stag[i] += 1
                else:
                    stag[i] += 1
            succ = succ * 0.95 + 0.01; tries = tries * 0.95 + 0.02
            ship_mem = 1 + (ship_mem - 1) * 0.97

            # ไข่: เก็บตัวดีที่หลากหลาย
            bi = int(np.argmin(F))
            if not eggs or min(np.abs(_ranks(X[bi]) - _ranks(e[0])).sum() for e in eggs) > 0:
                eggs.append((X[bi].copy(), PR[bi].copy(), F[bi])); eggs = sorted(eggs, key=lambda e: e[2])[:6]

            # ---- M6 โรคระบาด ----
            if self.use["disease"]:
                imm = np.maximum(0, imm - 1)
                rec = inf & improved; inf[rec], sev[rec], imm[rec] = False, 0.0, self.T_imm
                sev[inf & ~improved] += self.delta
                new = (~inf) & (stag >= self.T_s) & (imm == 0); inf[new], sev[new] = True, 0.2
                sf = F.std() + 1e-12
                for i in np.where(inf)[0]:
                    sus = np.where((~inf) & (imm == 0))[0]
                    if not len(sus): break
                    p = self.beta * np.exp(-(Dm[i, sus] / (0.5 * spread)) ** 2) * np.exp(-np.abs(F[i] - F[sus]) / sf)
                    hit = sus[rng.random(len(sus)) < p]; inf[hit], sev[hit] = True, 0.2
                outbreak = inf.mean() > self.rho_max
                cull = [c for c in np.where(inf & ((sev >= 1.0) | outbreak))[0] if c != np.argmin(F)]
                for c in cull:
                    if ev.done: break
                    if self.mating and len(eggs) >= 2 and rng.random() < 0.6:
                        # ผีเสื้อผสมพันธุ์: ไข่ใหม่ได้ลำดับจากพ่อแม่ที่ดี 2 ตัว (order crossover)
                        i1, i2 = rng.choice(len(eggs), 2, replace=False)
                        a0, b0 = sorted(rng.integers(0, n, size=2))
                        o = _ox(eggs[i1][0], eggs[i2][0], a0, b0)
                        p_ = np.where(rng.random(n) < 0.5, eggs[i1][1], eggs[i2][1])
                        ia, ib = rng.integers(n, size=2); o = _insert(o, ia, ib)
                    elif rng.random() < 0.6 and eggs:              # ฟักจากไข่ + เขย่าบางช่วง
                        e = eggs[rng.integers(len(eggs))]
                        o = e[0].copy(); L = max(2, n // 5); a = rng.integers(0, n - L + 1)
                        o[a:a + L] = rng.permutation(o[a:a + L]); p_ = e[1].copy()
                        msk = rng.random(n) < 0.2; p_[msk] = rng.random(msk.sum())
                    else:
                        o, p_ = rng.permutation(n), rng.random(n)
                    X[c], PR[c] = o, p_
                    F[c], S[c] = ev(o, p_)
                    inf[c], sev[c], stag[c], imm[c] = False, 0.0, 0, self.T_imm

            # ---- ชักใย: ค้นหาละเอียดรอบตัวดีที่สุด ----
            if self.use["spin"] and prog >= self.spin_start and not ev.done:
                bv, bo, bp, bs = ev.best
                comp = bs + P.h
                ship = np.argsort(-comp)[rng.integers(3)]
                a = int(np.where(bo == ship)[0][0])
                for b in rng.permutation(max(1, a))[:6]:
                    if ev.done: break
                    o = _insert(bo, a, int(b)); fv, sv = ev(o, bp)
                    if fv < bv:
                        wi = int(np.argmax(F)); X[wi], PR[wi], F[wi], S[wi] = o, bp.copy(), fv, sv
                        break

        bv, bo, bp, bs = ev.best
        return bo, bp, bv, np.array(ev.curve[:max_fe])


# ===================== คู่แข่งแบบจัดลำดับ =====================
def ga_perm(P, max_fe, seed=None, N=50, pc=0.9, pm=0.3):
    """GA มาตรฐานสำหรับลำดับ: tournament, order crossover (OX), insertion/swap mutation, elitism"""
    rng = np.random.default_rng(seed); n = P.n; ev = _Eval(P, max_fe)
    X = [rng.permutation(n) for _ in range(N)]; PR = [rng.random(n) for _ in range(N)]
    F = np.array([ev(X[i], PR[i])[0] for i in range(N)])

    def tour():
        a, b = rng.integers(N, size=2); return a if F[a] < F[b] else b

    def ox(p1, p2):
        a, b = sorted(rng.integers(0, n, size=2)); child = -np.ones(n, int)
        child[a:b + 1] = p1[a:b + 1]; rest = [g for g in p2 if g not in set(p1[a:b + 1])]
        child[[i for i in range(n) if child[i] < 0]] = rest; return child

    while not ev.done:
        e = int(np.argmin(F)); NX, NP_, NF = [X[e].copy()], [PR[e].copy()], [F[e]]
        while len(NX) < N and not ev.done:
            p1, p2 = tour(), tour()
            if rng.random() < pc:
                c = ox(X[p1], X[p2]); m = rng.random(n) < 0.5; cp = np.where(m, PR[p1], PR[p2])
            else:
                c, cp = X[p1].copy(), PR[p1].copy()
            if rng.random() < pm:
                a, b = rng.integers(n, size=2)
                if rng.random() < 0.5: c = _insert(c, a, b)
                else: c[a], c[b] = c[b], c[a]
            if rng.random() < pm:
                k = rng.integers(n); cp[k] = rng.random()
            NX.append(c); NP_.append(cp); NF.append(ev(c, cp)[0])
        X, PR, F = NX, NP_, np.array(NF + [np.inf] * (N - len(NF)))[:N]
        if len(X) < N: break
    bv, bo, bp, _ = ev.best
    return bo, bp, bv, np.array(ev.curve[:max_fe])


def ils(P, max_fe, seed=None, ls_tries=None, kick=3):
    """Iterated Local Search: ค้นหาเฉพาะที่ด้วย insertion/swap แบบ first-improvement + เตะด้วยการสลับสุ่ม"""
    rng = np.random.default_rng(seed); n = P.n; ev = _Eval(P, max_fe)
    ls_tries = ls_tries or 2 * n
    o, p = rng.permutation(n), rng.random(n); f, _ = ev(o, p)
    best = (f, o, p)
    while not ev.done:
        fails = 0
        while fails < ls_tries and not ev.done:                 # local search
            a, b = rng.integers(n, size=2)
            r = rng.random()
            if r < 0.45: o2, p2 = _insert(o, a, b), p
            elif r < 0.8: o2 = o.copy(); o2[a], o2[b] = o2[b], o2[a]; p2 = p
            else: o2 = o; p2 = p.copy(); p2[a] = rng.random()
            f2, _ = ev(o2, p2)
            if f2 < f: f, o, p, fails = f2, o2, p2, 0
            else: fails += 1
        if f <= best[0]: best = (f, o, p)                       # ยอมรับเมื่อดีกว่าหรือเท่าเดิม
        o, p = best[1].copy(), best[2].copy()                   # เตะจากตัวดีที่สุด
        for _ in range(kick):
            a, b = rng.integers(n, size=2); o[a], o[b] = o[b], o[a]
        f, _ = ev(o, p)
    bv, bo, bp, _ = ev.best
    return bo, bp, bv, np.array(ev.curve[:max_fe])
