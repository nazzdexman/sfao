"""
ตรวจซ้ำแบบเร็ว: (1) self-check โค้ด CEC2017 กับค่าอ้างอิงจากโค้ด C ทางการ 174 จุด
               (2) รัน SFAO รุ่นที่ล็อกบนบางฟังก์ชัน seed 0 ด้วยงบเต็ม แล้วพิมพ์ค่า error
ใช้:  python reproduce_check.py
หมายเหตุ: บนแพลตฟอร์มเดียวกัน (OS/numpy/BLAS เดียวกัน) ผลซ้ำได้ตรงทุกหลัก
          ข้ามแพลตฟอร์ม ค่าอาจต่างเล็กน้อย เพราะทศนิยมตำแหน่งท้าย ๆ ของ np.linalg.eigh (M8) ต่างกัน
          แล้วเส้นทางการค้นหาแยกออกจากกัน — ค่าเฉลี่ย 30 รอบไม่ควรเปลี่ยนในเชิงสาระ
"""
import time
import full_benchmark as fb
from cec2017 import selfcheck

print("CEC2017 self-check:", "PASS" if selfcheck() else "FAIL")
# ค่าที่รายงาน (รันบน PC Windows ของผู้เขียน) seed 0
REPORTED = {("cec2017", 10, 1): 580.8507334582318,
            ("cec2017", 10, 4): 1.61620619110181e-08,
            ("cec2017", 10, 21): 100.0000000000282,
            ("cec2022", 10, 2): 0.0}
for (suite, D, f), rep in REPORTED.items():
    t = time.time()
    r = fb.job((suite, D, f, "SFAO:cocoon:aspin", 0))
    print(f"{suite} D={D} F{f}: error = {r['err']:.10g}   (reported {rep:.10g})   {time.time() - t:.1f}s")
