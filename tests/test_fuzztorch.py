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
from src.fuzztorch import Fuzzer
from multiprocessing import Process
from src.traceerror import Trace_error
import psutil

configargs = Namespace()
args = sys.argv
case_path = os.getcwd() + "/dnn/"


def _parse_args():
    global configargs
    p = ArgumentParser()
    p.add_argument(
        "caseids", metavar="model_dir_name", type=str, nargs="+", help="model ids"
    )
    p.add_argument("--low", type=float, default=0.0)
    p.add_argument("--high", type=float, default=1.0)
    p.add_argument(
        "--method", type=str, default="MEGA", choices=["DEMC", "MCMC", "MEGA"]
    )
    p.add_argument("--optlevel", type=int, default=5)
    p.add_argument("--granularity", type=int, default=5)
    # p.add_argument( '--name', type=str, help='such as --name 1')
    configargs = p.parse_args()


_parse_args()

for caseid in configargs.caseids:
    fuzzer = Fuzzer(
        path=case_path,
        case_id=caseid,
        low=configargs.low,
        high=configargs.high,
        fuzzmode=configargs.method,
        fuzzframe=False,
        optlevel=configargs.optlevel,
        fuseopsmax=configargs.granularity,
    )
    fuzzer.bigflag = 1

    pfuzzer = Process(target=fuzzer.fuzzps)
    try:
        pfuzzer.start()
        pfuzzer.join(timeout=60 * 30)
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
