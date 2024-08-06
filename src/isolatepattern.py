"""



"""


import re
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
import json
import string
from multiprocessing import Process, Queue
import tvm.testing
import json
from .calculator2 import Calculate_error

comparebaseorder = 0


def remove_virtarget(ncode):
    rncode = re.sub("({virtual_.*?}:)", ":", ncode, count=0, flags=re.M | re.S)
    rncode = re.sub("(virtual_.*?->)", ") ->", rncode, count=0, flags=re.M | re.S)
    return rncode


def remove_primary(code):
    return re.sub("(, Primitive\=.*?->)", ") ->", code, count=0, flags=re.M | re.S)


def get_primfunc(opt_level, mod, target="llvm"):  # i.e. get tvm.ir.module.IRModule
    target, target_host = tvm.target.Target.canon_target_and_host(target)
    with tvm.transform.PassContext(
        opt_level=opt_level
    ):  # config={"relay.FuseOps.max_depth": 10}
        prim_mod, _ = relay.optimize(mod, target)
        code = remove_virtarget(str(prim_mod))
    return code


def read_primfunc(path):
    with open(path, "r") as fp:
        return fp.read()


class Ensurepattern:
    def __init__(self, case_path):
        self.case_path = case_path
        if "dnn" in self.case_path:
            self.dnn = True
        else:
            self.dnn = False
        self.opt_root = self.case_path + "/L5/"
        self.unopt_root = self.case_path + "/L1/"
        with open(f"{self.case_path}/code.txt", "r") as f:
            self.mod = relay.parse(f.read())
        self.optprim_mod = read_primfunc(f"{self.case_path}/optirmod.txt")
        self.unoptprim_mod = read_primfunc(f"{self.case_path}/unoptirmod.txt")
        with open("../src/op/opname.json", "r") as fp:
            self.opname = json.load(fp)
        with open("../src/op/opcoding.json", "r") as fp:
            self.opcoding = json.load(fp)
        self.qnn = False
        if len(re.findall(r"qnn\.", str(self.mod), flags=re.S | re.M)) > 0:
            self.qnn = True

    # add a new fix attribute
    # #if prenode is reshape, and there are more than one reshape node, then it may find incorrect reshape
    # because the debug info is not clear for reshape_nop
    def get_topo_name_index(self, keys: List[str]):
        topo_name = dict()
        for key in keys:
            name, index = key.split("____")[0], int(key.split("____")[1].split(":")[1])
            topo_name[name] = index
        return topo_name

    def rm_invalid(self, str):
        pat = "(_e[a-f0-9]+e)"
        # print(str)
        str = re.sub(pat, "", str, count=0, flags=re.M | re.S)
        # print(str)
        return str

    def handle_strings(self, hash: str):
        def get_modstr_from_hash(hash):
            pattern = "fn.*" + hash + ".*?}"
            # pattern = hash+'.*?}'
            matched = re.search(pattern, self.unoptprim_mod, flags=re.M | re.S)
            if matched is not None:
                l, r = matched.span()
                ocode = self.unoptprim_mod
            else:
                matched = re.search(pattern, self.optprim_mod, flags=re.M | re.S)
                l, r = matched.span()
                ocode = self.optprim_mod
            origin_func = ocode[l:r]
            return origin_func

        def get_function_fromhash(hash, ocode):
            hash = hash
            assert hash != "" and hash is not None
            pattern1 = hash + ".*?\}"
            pattern2 = hash[::-1] + "(.*?nf.*?)"
            matched1 = re.search(pattern1, ocode, flags=re.M | re.S)
            a = matched1.group()
            matched = re.search(
                pattern2, ocode[: matched1.span()[1]][::-1], flags=re.M | re.S
            )
            b = matched.group(1)[::-1]
            origin_func = b + a
            return remove_primary(origin_func)

        kernelstr = get_modstr_from_hash(hash)

        return get_function_fromhash(hash, kernelstr)

    def reorderfunc_withindex(self, funcname_alls) -> List[str]:
        def fitness(item):
            ten = int(item.split("____")[1].split(":")[1])
            one = int(item.split("____")[2].split(":")[1])
            return ten * 10 + one

        return sorted(funcname_alls, key=lambda item: fitness(item))

    def get_shape(self, key):
        gdata = self.kerneldict
        for node in gdata["nodes"]:
            index = node["name"].rstrip("_")
            if key == index:
                shape = node["shape"].copy()
                return shape
        print(key, " no shape")
        exit()

    def encode_obj(self, obj):
        obj_dict = {}
        obj_dict.update(obj.__dict__)
        return obj_dict

    # look op params' key

    def lookup_keys(self, path):
        data = relay.load_param_dict(bytearray(open(path, "rb").read()))
        for k, v in data.items():
            tvm_array = v.numpy()
            print(k, tvm_array.shape)

    def parse(self, funcstr):
        # get base number
        print(funcstr)
        matchedall = re.findall(
            r"(%\d+ = )", funcstr[funcstr.find("{") :]
        )  # change handle base mnumber
        if len(matchedall) >= 1:
            self.base_num = int(re.findall(r"(\d+)", matchedall[0])[0])
            self.onum = int(re.findall(r"(\d+)", matchedall[-1])[0]) + 1
        else:
            matchedall = re.findall(r"(%\d+)", funcstr)
            if len(matchedall) >= 1:
                self.base_num = int(re.findall(r"(%\d+)", funcstr)[0].strip("%"))
                self.onum = self.base_num
                print(self.base_num, self.onum)
            else:
                return relay.parse(
                    funcstr.replace("fn", '#[version = "0.0.5"]\ndef @main')
                )

    def ensurepattern(self, report: Queue = None) -> str:
        # prepare params for inference

        graph_json_path = os.path.join(
            self.unopt_root, "_tvmdbg_device_CPU_0/_tvmdbg_graph_dump.json"
        )
        with open(graph_json_path, "r") as gfile:
            self.kerneldict = json.load(gfile)
        unnoptparams_path = os.path.join(
            self.unopt_root, "_tvmdbg_device_CPU_0/output_tensors.params"
        )
        params1: Dict[str, int] = relay.load_param_dict(
            bytearray(open(unnoptparams_path, "rb").read())
        )
        nodes = self.kerneldict["nodes"]
        preindexs = []
        for node in nodes:
            if (
                "hash" in node["attrs"].keys()
                and node["attrs"]["hash"] == "34a7fc534547cc03"
            ):
                preindexs = node["inputs"]
        print(preindexs)
        print(list(params1.keys()))
        print(type(params1))
        inputs = {}
        for i in preindexs:
            for key in list(params1.keys()):
                if key.split("____")[0] == i:
                    inputs[i] = params1[key]

        # prepare model

        modstr = self.handle_strings("34a7fc534547cc03")
        modstr_head = re.search(r"(fn.*?{)", modstr)
        allkeys = re.findall(r"(%p\d+)", modstr_head.group(0))
        # nkeys = matchedall.group()
        print(allkeys)
        inputs2 = {}
        for k, item in enumerate(list(inputs.items())):
            inputs2[k] = item
        # parse modelstr
        mod = self.parse(modstr)

        # calculate error

        cal2 = Calculate_error(mod=self.mod)
        # path_params = os.path.join(self.case_path, 'inputs.npz')
        # with np.load(path_params) as f:
        #         loaded_params = dict(f.items())
        error = cal2.test_consistent(isolatepass=[], inputs=inputs2)
        if error == 0:
            print(error, "not right isolation")
        else:
            print(error, "isolate rightly", mod)

        # ensure calculate error > 0
