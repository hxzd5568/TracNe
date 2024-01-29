import sys
sys.path.append('..')
import os
from src.base_utils import Checkor
from src.isolatepattern import Ensurepattern
import re



case_path = '/home/zichaox/tvm/Fuzzfp/introcase/tests/'
args = sys.argv

if '/' in args[1]:
    dump_path = args[1]
    case_path = dump_path.split('out')[0]
    caseid = dump_path.split('out')[1][1:]
    caseids = [caseid]
elif '-' in args[1]:
    l,r = int(args[1].split('-')[0]),\
            int(args[1].split('-')[1])+1
    caseids = [str(i) for i in range(l,r,1)]
else:
    caseids = args[1:]
import time
t0 = time.time()
for caseid in caseids:
    dump_path = case_path+'/out/'+caseid
    print(dump_path)
    corrector = Ensurepattern(dump_path)
    corrector.ensurepattern()
    # try:
    #     dump_path = case_path+'/out/'+caseid
    #     print(dump_path)
    #     corrector = Corrector(dump_path)
    #     corrector.pipeline()
    # except Exception as e:
    #     print(e)
print('ensure using',time.time()-t0)
