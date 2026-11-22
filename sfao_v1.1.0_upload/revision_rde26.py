"""Build the pinned upstream RDE26-SOP entry with exact-budget trace instrumentation."""
import argparse
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np

PIN='877e68555ecbed73d05b98c57a196a3a0d1d3147'

def case(job):
    binary,cec,seed=job
    command=[str(binary),'4',str(seed),'300000','30','-100','100',str(cec)]
    lines=subprocess.check_output(command,text=True).splitlines()
    points=[(int(k),float(v)) for k,v in (line.split() for line in lines)]
    assert points[-1][0]==300000 and all(np.diff([i for i,_ in points])>0)
    cut=points[-1][1]
    return dict(algorithm='RDE26-SOP',f=4,D=30,seed=seed,budget=300000,final_error=cut,
                first_hit_fe={str(t):next((i for i,v in points if v<t),None) for t in (1.,.01,.001)},
                improvement_trace=points)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('repo',type=Path)
    p.add_argument('result',type=Path)
    a=p.parse_args()
    assert subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip()==PIN
    with tempfile.TemporaryDirectory() as temp:
        target=Path(temp)/'solver'
        shutil.copytree(a.repo/'RDE26_SOP/code/rde26-sop',target)
        cec=target/'src/evolutionary_computation/problems/cec2017_sop'
        r=target/'src/rdex_sandbox'
        p=r/'rdex_sop.cpp'
        source=p.read_text()
        source=source.replace('double cec_24(double* HostVector, int func_num) {',
          'double cec_24(double* HostVector, int func_num) {\n'
          '    if (NFEval >= MaxFEval) return std::numeric_limits<double>::infinity();\n')
        old='    NFEval++;\n    if (RbfRecordTrueSamples_global_sop)'
        assert source.count(old)==1
        source=source.replace(old,'    NFEval++;\n'
          '    static double matched_best = std::numeric_limits<double>::infinity();\n'
          '    if (tempF[0] < matched_best || NFEval == MaxFEval) {\n'
          '        matched_best = std::min(matched_best, tempF[0]);\n'
          '        std::cout.precision(17);\n'
          '        std::cout << NFEval << "\\t" << matched_best << "\\n";\n'
          '    }\n    if (RbfRecordTrueSamples_global_sop)')
        p.write_text(source)
        p=r/'rdex_sop_cli.cpp'
        source=p.read_text()
        start=source.index('        for (const auto& item')
        end=source.index('    } catch',start)
        p.write_text(source[:start]+source[end:])
        spec=importlib.util.spec_from_file_location('runner',target/'main.py')
        runner=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        binary=runner.build_binary(target,'g++')
        with Pool(6) as pool,a.result.open('w') as stream:
            for row in pool.imap_unordered(case,[(binary,cec,s) for s in range(30)]):
                print(json.dumps(row),file=stream,flush=True)

if __name__=='__main__':
    main()
