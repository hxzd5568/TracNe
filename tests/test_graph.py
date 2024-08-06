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
from src.viz import save_irmod_viz
import re


def defuse_mod(mod):
    seq = tvm.transform.Sequential(
        [
            relay.transform.InferType(),
            relay.transform.DefuseOps(),
            relay.transform.InferType(),
        ]
    )
    with tvm.transform.PassContext(opt_level=4):
        return seq(mod)


def remove_virtarget(ncode):
    rncode = re.sub("({virtual_.*?}:)", ":", ncode, count=0, flags=re.M | re.S)
    rncode = re.sub("(virtual_.*?->)", ") ->", rncode, count=0, flags=re.M | re.S)
    return rncode


def remove_primary(code):
    return re.sub("(, Primitive\=.*?->)", ") ->", code, count=0, flags=re.M | re.S)


def remove_primary2(code):
    return re.sub("(, Primitive\=1)", "", code, count=0, flags=re.M | re.S)


def remove_comment(code):
    return re.sub("\/\*.*?\*\/", "", str(code), count=0, flags=re.M | re.S)


case_path = os.getcwd()
case_id = os.path.basename(__file__).strip(".py")
args = sys.argv


if "/" in args[1]:
    dump_path = args[1]
    case_path = dump_path.split("out")[0]
    caseid = dump_path.split("out")[1][1:]
    caseids = [caseid]
elif "-" in args[1]:
    l, r = int(args[1].split("-")[0]), int(args[1].split("-")[1]) + 1
    caseids = [str(i) for i in range(l, r, 1)]
elif "," in args[1]:
    caseids = args[1].split(",")
else:
    caseids = args[1:]
import time

t0 = time.time()
# import pdb
# pdb.set_trace()
for caseid in caseids:
    dump_path = case_path + "/out/" + caseid
    print(dump_path)

    def draw_contrast_mod():
        opt_mod = dump_path + "/optirmod.txt"
        unopt_mod = dump_path + "/unoptirmod.txt"
        print("dump_path", dump_path)
        with open(opt_mod) as f:
            mod = relay.parse(remove_primary2(remove_virtarget(f.read())))
        # save_irmod_viz(defuse_mod(mod),dump_path+'/relay_opt')
        print(remove_comment(defuse_mod(mod)))
        # save_irmod_viz(mod,dump_path+'/relay_comopt')
        del mod
        with open(unopt_mod) as f:
            mod = relay.parse(remove_primary2(remove_virtarget(f.read())))
        # save_irmod_viz(defuse_mod(mod),dump_path+'/relay_unopt')
        print(remove_comment(defuse_mod(mod)))

        # save_irmod_viz(mod,dump_path+'/relay_comunopt')

    draw_contrast_mod()
    # checkor = Checkor(path =case_path,case_id=caseid )
    # save_irmod_viz(checkor.mod,case_path+'/out/'+caseid+'/relay')
    # print('viz done')

    # checkor.random_difference_test()
