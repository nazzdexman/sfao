"""
SFAO รุ่นที่ล็อก (รุ่นที่ใช้รายงานผลในเปเปอร์ 1) = "SFAO:cocoon:aspin"

ค่าเริ่มต้นของคลาส SFAO ใน sfao.py *ไม่ใช่* รุ่นที่ล็อก (ค่าเริ่มต้นเป็นรุ่นต้นแบบ: ปิดดักแด้ ชักใยแบบคงที่)
รุ่นที่ล็อกเปิดเพิ่ม 3 ค่า ดังนี้ (ตรงกับ full_benchmark.get_algo):
    use_cocoon=True        M9 เข้าดักแด้ (ลดฝูงเชิงเส้น 18D -> 8)
    food_per_worm=100      กฎความหนาแน่นการเลี้ยง: ฝูงเริ่ม <= FE_max / 100
    spin_mode="adaptive"   M7 ชักใยแบบปรับวง (sigma x1.5 เมื่อสำเร็จ, x0.9 เมื่อไม่สำเร็จ)
กลไกที่เหลือทั้งหมดเปิดตามค่าเริ่มต้น (use_attract/swing/toxic/crowd/molt/disease/spin/memory = True)

ไฟล์นี้เป็นเพียงทางลัด ไม่ได้แก้ sfao.py (sfao.py คงไว้ตรงกับไฟล์ที่ใช้รันผลจริงทุกไบต์)
"""
from sfao import SFAO


def sfao_final(**override):
    kw = dict(use_cocoon=True, food_per_worm=100, spin_mode="adaptive")
    kw.update(override)
    return SFAO(**kw)


if __name__ == "__main__":
    import numpy as np
    sphere = lambda x: float(np.sum(x ** 2))
    best_x, best_f, curve = sfao_final().optimize(sphere, dim=10, lb=-100, ub=100, max_fe=20000, seed=0)
    print("sphere D=10, 20,000 FE -> best f =", best_f)
