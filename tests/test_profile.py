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
from google.protobuf.json_format import MessageToDict

sys.path.append("..")
from src.base_utils import Checkor
from src.fuzztorch import Fuzzer
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
case_path = "./dnn/"
args = sys.argv

if "/" in args[1]:
    dump_path = args[1]
    flag = 1
    case_path = dump_path.split("out")[0]
    caseid = dump_path.split("out")[1][1:]
    caseids = [caseid]
elif "-" in args[1]:
    l, r = int(args[1].split("-")[0]), int(args[1].split("-")[1]) + 1
    caseids = [str(i) for i in range(l, r, 1)]
else:
    caseids = args[1:]


for caseid in caseids:
    if caseid.isdigit():
        dump_path = case_path + "/out/" + caseid
    else:
        dump_path = case_path + "/dnn/out/" + caseid
    print(dump_path)
    fuzzer = Fuzzer(
        path=case_path,
        low=0,
        high=1,
        case_id=caseid,
        fuzzmode="DEMC2",
    )
    # fuzzer.profile()
    # fuzzer.save_files()
    fuzzer.replay_error_int()
    # for i in range(3):
    # fuzzer.fuzzps()
    # fuzzer.replay_error()
    # exit()
    # fuzzer.bigflag =1
    # pfuzzer = Process(target=fuzzer.fuzzps)
    # try:
    #     pfuzzer.start()
    #     pfuzzer.join(timeout=60*120)
    #     pfuzzer.terminate()
    #     fuzzer.replay_error()
    # except Exception as e:
    #     print(e.__class__.__name__, e)
    # pfuzzer.close()
    # del pfuzzer
    del fuzzer

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
