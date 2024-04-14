import tvm
from tvm import relay,runtime
import os
import numpy as np
import queue
import shutil
import os.path
import random
from enum import IntEnum, auto
from tvm import transform, relay, parser, cpu, TVMError, IRModule
from tvm.contrib.graph_executor import GraphModule
from argparse import Namespace, ArgumentParser
from typing import Iterable, List, cast, Optional, Dict
from subprocess import run, CalledProcessError, TimeoutExpired
import sys
import re

sys.path.append('..')
from src.base_utils import Checkor
from src.fuzzer import Fuzzer
from multiprocessing import Process
from src.traceerror import Trace_error
import psutil
import os
import time
import datetime
# filePath= "out/6"
# ctime=os.path.getctime(filePath)
# create=datetime.datetime.fromtimestamp(ctime)
# nowdate=datetime.datetime.now()
# datetime.datetime
# print(create._day)
# print(nowdate._day)
# print(create._day==nowdate._day)
# exit()
case_path = os.getcwd()
case_id = os.path.basename(__file__).strip('.py')
args = sys.argv

if re.findall('torch',args[1]):
    model_name = ['transformer0',]#'resnet50','vgg11','vgg13','vgg16','inceptionv3','dpn68
    # 'resnet18','resnet34'
    for model in model_name:
        cmd = ['python', 'test_torchutils.py', model]
        try: #!! pack them
            rcode = run(cmd, timeout=60*80)
        except CalledProcessError:
            print(f'Error detected in initial check of case {model}.')
            keep_dir = True
        except TimeoutExpired:
            print(f'Case {model} timed out.')
    exit()
if '-' in args[1]:
    l,r = int(args[1].split('-')[0]),\
            int(args[1].split('-')[1])+1
    caseids = [str(i) for i in range(l,r,1)]
elif ',' in args[1]:
    caseids = args[1].split(',')
else:
    caseids = args[1:]

import time
t0 = time.time()
for caseid in caseids:
    if caseid.isdigit():
        dump_path = case_path+'/out/'+caseid
    else:
        dump_path = case_path+'/dnn/out/'+caseid
    print(dump_path)
    # if os.path.exists(dump_path+'/compiled_lib1.tar'):
    #     os.remove(dump_path+'/compiled_lib1.tar')
    for file in ['test_fuzzer.py', 'test_replay.py', 'test_traceerror.py', 'test_prop.py', 'test_pliner.py']:# 'test_fuzzer.py', 'test_replay.py', 'test_traceerror.py', 'test_fix.py', 'test_prop.py', 'test_pliner.py'
        cmd = ['python', file, caseid]
        try: #!! pack them
            rcode = run(cmd, timeout=180*4+30)
        except CalledProcessError:
            print(f'Error detected in initial check of case {case_id}.')
            keep_dir = True
        except TimeoutExpired:
            print(f'Case {case_id} timed out.')
# print('fix' ,time.time()-t0)
