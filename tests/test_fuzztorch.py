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

sys.path.append('..')
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

configargs = Namespace()

def _parse_args():
    global configargs
    p = ArgumentParser()
    p.add_argument('caseids', metavar='N', type=str, nargs='+',
                    help='model ids')
    p.add_argument( '--low', type=float, default=0.)
    p.add_argument( '--high', type=float, default=1.)
    p.add_argument( '--method', type=str, default='MEGA',choices=['DEMC', 'MCMC','MEGA'])
    p.add_argument('--optlevel', type=int,  default=5)
    p.add_argument('--granularity', type=int,  default=5)
    # p.add_argument( '--name', type=str, help='such as --name 1')
    configargs = p.parse_args()
_parse_args()

args1 = configargs.caseids
print(args1)
if '/' in args1[0]:
    dump_path = args1[0]
    flag =1
    case_path = dump_path.split('out')[0]
    caseid = dump_path.split('out')[1][1:]
    caseids = [caseid]
else:
    print('Wrong input path. A valid path is "./dnn/out/inceptionv3". ')


for caseid in caseids:
    print(dump_path)
    fuzzer = Fuzzer(path =case_path,case_id=caseid,
                    low= configargs.low,high= configargs.high,
                    fuzzmode=configargs.method, fuzzframe=False,
                    optlevel=configargs.optlevel,
                    fuseopsmax=configargs.granularity,)
    # fuzzer.replay_error()
    # for i in range(3):
    # fuzzer.fastv()
    # fuzzer.save_files()
    # fuzzer.replay_error_yolov8()
    # exit()
    fuzzer.bigflag =1
    # fuzzer.profile_nlp()
    # exit()
    # fuzzer.fastv()
    # fuzzer.fuzzps()

    pfuzzer = Process(target=fuzzer.fuzzps)
    try:
        pfuzzer.start()
        pfuzzer.join(timeout=60*30)
        pfuzzer.terminate()
        fuzzer.replay_error()
    except Exception as e:
        print(e.__class__.__name__, e)
    pfuzzer.close()
    del pfuzzer
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
