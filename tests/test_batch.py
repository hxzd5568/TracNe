import tvm
from tvm import relay, runtime
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

sys.path.append("..")
from src.base_utils import Checkor
from src.fuzzer import Fuzzer
from multiprocessing import Process
from src.traceerror import Trace_error
import psutil
import os
import time
import datetime

case_path = os.getcwd()
case_id = os.path.basename(__file__).strip(".py")
args = sys.argv
configargs = Namespace()


def _parse_args():
    global configargs
    p = ArgumentParser()
    p.add_argument("caseids", metavar="N", type=str, nargs="+", help="model ids")
    p.add_argument("--low", type=float, default=-5.0)
    p.add_argument("--high", type=float, default=5.0)
    p.add_argument(
        "--method", type=str, default="MEGA", choices=["DEMC", "MCMC", "MEGA"]
    )
    p.add_argument("--optlevel", type=int, default=5)
    p.add_argument("--granularity", type=int, default=64)
    # p.add_argument( '--name', type=str, help='such as --name 1')
    configargs = p.parse_args()


_parse_args()

if "-" in args[1]:
    l, r = int(args[1].split("-")[0]), int(args[1].split("-")[1]) + 1
    caseids = [str(i) for i in range(l, r, 1)]
elif "," in args[1]:
    caseids = args[1].split(",")
else:
    caseids = args[1:]

import time

t0 = time.time()
for caseid in caseids:
    dump_path = case_path + "/out/" + caseid
    print(dump_path)
    # if os.path.exists(dump_path+'/compiled_lib1.tar'):
    #     os.remove(dump_path+'/compiled_lib1.tar')
    for file in [
        "test_fuzzer.py",
        "test_replay.py",
        "test_traceerror.py",
        "test_prop.py",
    ]:  # 'test_fuzzer.py', 'test_replay.py', 'test_traceerror.py', 'test_fix.py', 'test_prop.py'
        if file == "test_fuzzer.py":
            cmd = ["python", file, caseid, "--method", configargs.method]
        else:
            cmd = ["python", file, caseid]
        try:  #!! pack them
            rcode = run(cmd, timeout=180 + 30)
        except CalledProcessError:
            print(f"Error detected in initial check of case {case_id}.")
            keep_dir = True
        except TimeoutExpired:
            print(f"Case {case_id} timed out.")
# print('fix' ,time.time()-t0)
