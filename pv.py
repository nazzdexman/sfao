"""
หาค่าพารามิเตอร์เซลล์แสงอาทิตย์ (PV parameter extraction) — ชุดข้อมูลมาตรฐาน RTC France
(Easwarakhanthan et al., 1986) เซลล์ซิลิคอนเส้นผ่านศูนย์กลาง 57 มม. ที่ 1000 W/m², 33 °C, 26 จุดวัด I-V

แบบจำลอง
  SDM (single diode, 5 พารามิเตอร์): Iph, Isd, Rs, Rsh, n
      I = Iph - Isd[exp((V+I Rs)/(n Vt)) - 1] - (V+I Rs)/Rsh
  DDM (double diode, 7 พารามิเตอร์): Iph, Isd1, Isd2, Rs, Rsh, n1, n2
เป้าหมาย: ลด RMSE ของ residual (ใช้กระแสที่วัดได้ในสมการ ตามแบบที่งานส่วนใหญ่ใช้)
ขอบเขตมาตรฐานในวรรณกรรม: Iph [0,1] A, Isd [0,1] µA, Rs [0,0.5] Ω, Rsh [0,100] Ω, n [1,2]
ค่าดีที่สุดที่รายงานกันในวรรณกรรม: SDM RMSE ≈ 9.8602e-4 ; DDM RMSE ≈ 9.8248e-4
"""
import numpy as np

V = np.array([-0.2057, -0.1291, -0.0588, 0.0057, 0.0646, 0.1185, 0.1678, 0.2132, 0.2545, 0.2924,
              0.3269, 0.3585, 0.3873, 0.4137, 0.4373, 0.4590, 0.4784, 0.4960, 0.5119, 0.5265,
              0.5398, 0.5521, 0.5633, 0.5736, 0.5833, 0.5900])
I = np.array([0.7640, 0.7620, 0.7605, 0.7605, 0.7600, 0.7590, 0.7570, 0.7570, 0.7555, 0.7540,
              0.7505, 0.7465, 0.7385, 0.7280, 0.7065, 0.6755, 0.6320, 0.5730, 0.4990, 0.4130,
              0.3165, 0.2120, 0.1035, -0.0100, -0.1230, -0.2100])
k, q, T = 1.380649e-23, 1.602176634e-19, 33.0 + 273.15
Vt = k * T / q

BEST_KNOWN = {"SDM": 9.8602e-4, "DDM": 9.8248e-4}
BOUNDS = {
    "SDM": (np.array([0, 0, 0, 0, 1]), np.array([1, 1e-6, 0.5, 100, 2])),
    "DDM": (np.array([0, 0, 0, 0, 0, 1, 1]), np.array([1, 1e-6, 1e-6, 0.5, 100, 2, 2])),
}


def rmse_sdm(x):
    Iph, Isd, Rs, Rsh, n = x
    Vd = V + I * Rs
    with np.errstate(over="ignore"):
        r = Iph - Isd * (np.exp(Vd / (n * Vt)) - 1) - Vd / max(Rsh, 1e-12) - I
    v = float(np.sqrt(np.mean(r ** 2)))
    return v if np.isfinite(v) else 1e10


def rmse_ddm(x):
    Iph, I1, I2, Rs, Rsh, n1, n2 = x
    Vd = V + I * Rs
    with np.errstate(over="ignore"):
        r = (Iph - I1 * (np.exp(Vd / (n1 * Vt)) - 1) - I2 * (np.exp(Vd / (n2 * Vt)) - 1)
             - Vd / max(Rsh, 1e-12) - I)
    v = float(np.sqrt(np.mean(r ** 2)))
    return v if np.isfinite(v) else 1e10


PROBLEMS = {"SDM": rmse_sdm, "DDM": rmse_ddm}


def scaled(model):
    """ห่อให้ทุกอัลกอริทึมค้นหาในช่วง [0,1]^D เท่ากันหมด (ขอบเขตต่างกันหลายหลักทศนิยม)"""
    lo, hi = BOUNDS[model]; f = PROBLEMS[model]
    return (lambda u: f(lo + np.clip(u, 0, 1) * (hi - lo))), len(lo), lo, hi


if __name__ == "__main__":
    # ตรวจข้อมูล: พารามิเตอร์ดีที่สุดที่รายงานในวรรณกรรมสำหรับ SDM ควรให้ RMSE ≈ 9.86e-4
    x = [0.760776, 0.323021e-6, 0.036377, 53.718526, 1.481184]
    print("SDM RMSE at literature optimum:", rmse_sdm(np.array(x)))
