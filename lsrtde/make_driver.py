"""สร้าง lsrtde_driver.cpp จาก L-SRTDE.cpp ต้นฉบับ (github.com/VladimirStanovov/L-SRTDE_CEC-2024, commit fa72910)
แก้เฉพาะส่วน 'การตั้งค่าการทดลองและการบันทึกผล' ไม่แตะตัวอัลกอริทึม:
 1) เลือกชุดทดสอบด้วย -DBENCH=17|22 (แทนการแก้มือตามคอมเมนต์ CHANGE ของผู้แต่ง)
 2) ปิด TimeComplexity
 3) main ใหม่: รับ func D maxfe seed จาก argv, รัน 1 ครั้ง, ตั้ง seed ของ mt19937 ทั้ง 4 ตัวจาก seed
 4) cec_24 บันทึก best ภายในงบ (NFEval <= MaxFEval) แยกไว้ — ต้นฉบับไม่ตรวจงบภายในรุ่น จึงอาจเกินงบไม่กี่ครั้งตอนท้าย
 5) จุดบันทึกเส้นการลู่เข้าเปลี่ยนจาก 10000*D/1000 เป็น MaxFEval/1000 (ต้นฉบับสมมติงบ 10000D ซึ่งไม่ตรงกับ CEC2022 D10 = 200,000)
"""
src = open("L-SRTDE.cpp", encoding="utf-8", errors="surrogateescape").read()
def rep(a, b, n=1):
    global src
    c = src.count(a); assert c == n, (a, c); src = src.replace(a, b)
rep('const int Benchmark = 17; // 17 (CEC 2017 and CEC 2024) or 22 (CEC 2022)\n//#include "cec22_test_func.cpp"\n#include "cec17_test_func.cpp"',
    '#ifndef BENCH\n#define BENCH 17\n#endif\nconst int Benchmark = BENCH;\n#if BENCH == 22\n#include "cec22_test_func.cpp"\n#define CECFUNC cec22_test_func\n#else\n#include "cec17_test_func.cpp"\n#define CECFUNC cec17_test_func\n#endif')
rep('const int ResTsize1 = 29;', 'const int ResTsize1 = (BENCH == 22 ? 12 : 29);')
rep('bool TimeComplexity = true;', 'bool TimeComplexity = false;\ndouble bestWithinBudget; bool bestWithinInit;')
rep('    cec17_test_func(HostVector, tempF, GNVars, 1, func_num);\n    //cec22_test_func(HostVector, tempF, GNVars, 1, func_num);\t\n    NFEval++;\n',
    '    CECFUNC(HostVector, tempF, GNVars, 1, func_num);\n    NFEval++;\n    if(NFEval <= MaxFEval && (!bestWithinInit || tempF[0] < bestWithinBudget)) { bestWithinBudget = tempF[0]; bestWithinInit = true; }\n')
rep('    cec17_test_func(xopt, tempF, GNVars, 1, func_num);\n    //cec22_test_func(xopt, tempF, GNVars, 1, func_num);\n', '    CECFUNC(xopt, tempF, GNVars, 1, func_num);\n')
rep('stepsFEval[steps_k] = 10000.0/double(ResTsize2-1)*GNVars*(steps_k+1);', 'stepsFEval[steps_k] = double(MaxFEval)/double(ResTsize2-1)*(steps_k+1);')
rep('int main(int argc, char** argv)', 'int main_original(int argc, char** argv)')
rep('\t\t\t\tcec17_test_func(xtmp, tempF, GNVars, 1, func_num);\n\t\t\t\t//cec22_test_func(xtmp, tempF, GNVars, 1, func_num);\n', '\t\t\t\tCECFUNC(xtmp, tempF, GNVars, 1, func_num);\n')
src += r'''
// ---- added driver main: lsrtde FUNC D MAXFE SEED -> prints FINAL err, NFE, CURVE (1000 values)
int main(int argc, char** argv)
{
    if(argc < 5) { printf("usage: lsrtde func D maxfe seed\n"); return 1; }
    int func_num = atoi(argv[1]); GNVars = atoi(argv[2]); MaxFEval = atoi(argv[3]); unsigned s = (unsigned)atoi(argv[4]);
    generator_uni_i.seed(s*1000u+1u); generator_uni_r.seed(s*1000u+101u); generator_norm.seed(s*1000u+201u); generator_uni_i_3.seed(s*1000u+301u);
    static const double f22[13] = {0,300,400,600,800,900,1800,2000,2200,2300,2400,2600,2700};
    fopt = (Benchmark == 17) ? 100.0*func_num : f22[func_num];
    ResultsArray[ResTsize2-1] = MaxFEval; globalbestinit = false; bestWithinInit = false; LastFEcount = 0; NFEval = 0;
    int PopSize = 20;
    Optimizer OptZ;
    OptZ.Initialize(PopSize*GNVars, GNVars, func_num, 1);
    OptZ.MainCycle();
    OptZ.Clean();
    double e = bestWithinBudget - fopt; if(e < 1e-8) e = 0;
    printf("FINAL %.17g\nNFE %d\nCURVE", e, NFEval);
    for(int j=0;j!=ResTsize2-1;j++) printf(" %.10g", ResultsArray[j]);
    printf("\n");
    return 0;
}
'''
open("lsrtde_driver.cpp", "w", encoding="utf-8", errors="surrogateescape").write(src)
print("ok")
