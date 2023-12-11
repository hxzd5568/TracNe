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
import sys
import psutil
sys.path.append('..')
from src.base_utils import Checkor
from src.traceerror import Trace_error
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
    trace_error = Trace_error(dump_path)
    trace_error.get_trace_message()
    # try:
    #     trace_error.get_node_name()
    # except Exception as e:
    #     print(e.__class__.__name__, e)
    del trace_error
print(time.time()-t0)
