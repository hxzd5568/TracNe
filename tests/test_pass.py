import sys
sys.path.append('..')
import os
from src.base_utils import Checkor
from src.pass_impact import Passcheckor
import re

case_path = './'
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
    corrector = Passcheckor(case_path,caseid)
    corrector.isolate()
    # corrector.isolatef()
    # corrector.isdiabled()
