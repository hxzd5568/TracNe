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
import sys

sys.path.append("..")
from src.base_utils import Checkor
from src.fastfuzz import Fuzzer
from multiprocessing import Process
from src.traceerror import Trace_error
import psutil

# def get_current_memory_gb() -> int:
#     # 获取当前进程内存占用。
#     pid = os.getpid()
#     p = psutil.Process(pid)
#     info = p.memory_full_info()
#     return info.uss / 1024. / 1024. / 1024.

case_path = os.getcwd()
case_id = os.path.basename(__file__).strip(".py")
args = sys.argv
if "-" in args[1]:
    l, r = int(args[1].split("-")[0]), int(args[1].split("-")[1]) + 1
    caseids = [str(i) for i in range(l, r, 1)]
elif "," in args[1]:
    caseids = args[1].split(",")
else:
    caseids = args[1:]


for caseid in caseids:
    dump_path = case_path + "/out/" + caseid
    print(dump_path)
    fuzzer = Fuzzer(path=dump_path)
    fuzzer.fuzzps()
    # trace_error = Trace_error(dump_path)
    # # trace_error.get_trace_message()
    # try:
    #     trace_error.get_node_name()
    # except Exception as e:
    #     print(e.__class__.__name__, e)
    # del trace_error
    # try:

    #     continue
    # except Exception as e:
    #     print(e.__class__.__name__,':',e)
    #     # print (traceback.format_exc ())
    #     continue
