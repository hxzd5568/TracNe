import os
import re
from argparse import Namespace, ArgumentParser

import numpy as np
import tvm
from tvm import parser, relay
import sys

sys.path.append('..')
from src.pliner import Pliner

case_path = os.getcwd()
args = sys.argv
if '-' in args[1]:
    l,r = int(args[1].split('-')[0]),\
            int(args[1].split('-')[1])+1
    caseids = [str(i) for i in range(l,r,1)]
else:
    caseids = args[1:]

import time
t0 = time.time()

for caseid in caseids:
    if caseid.isdigit():
        dump_path = case_path+'/out/'+caseid
    else:
        dump_path = case_path+'/dnn/out/'+caseid
    print(tvm.__version__)

    opt_level = 10
    with open(os.path.join(dump_path, 'code.txt'), 'r') as f:
        code = f.read()

    print(f'Reducing case {caseid}:')

    # Possibly load inputs and parameters
    inputs_path = os.path.join(dump_path, 'iinputs.npz')
    inputs = None
    if os.path.exists(inputs_path):
        with np.load(inputs_path) as f:
            inputs = dict(f.items())
    params = inputs
    mod = relay.parse(code)
    plinerins = Pliner(mod, filename = os.path.join(dump_path, 'code-pliner.txt'), inputarr= params)
    if len(re.findall('(%\d+)',mod.astext()))>0:
        opsn = int(re.findall('(%\d+)',mod.astext())[-1].strip('%'))+1
    else:
        opsn = 1
    # plinerins.unacceptable(675)
    plinerins.pliner(0, opsn)
    # mod2 = plinerins.pliner(0, opsn)
    # with open('./dnn/out/yolov8presoft/code.txt','w') as fp:
    #     fp.write(mod2.astext())
    t1 = time.time()
    print(caseid, 'using time' , t1-t0)
    t0= t1
    print('passed')
    
