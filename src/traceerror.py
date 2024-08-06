"""

only need a tempdir to analyse
config={"relay.FuseOps.max_depth": max_fused_ops

L1:

new_x____topo-index:0____output-num:0 (4, 3)
new_y____topo-index:1____output-num:0 (4, 3)
tvmgen_default_fused_abs____topo-index:2____output-num:0 (4, 3)
tvmgen_default_fused_sqrt____topo-index:3____output-num:0 (4, 3)
tvmgen_default_fused_divide____topo-index:4____output-num:0 (4, 3)
tvmgen_default_fused_sum____topo-index:5____output-num:0 (4,)
tvmgen_default_fused_tan____topo-index:6____output-num:0 (4,)
tvmgen_default_fused_nn_relu____topo-index:7____output-num:0 (4,)

L5:

new_x____topo-index:0____output-num:0 (4, 3)
new_y____topo-index:1____output-num:0 (4, 3)
tvmgen_default_fused_abs_rsqrt_multiply_sum____topo-index:2____output-num:0 (4,)
tvmgen_default_fused_tan_nn_relu____topo-index:3____output-num:0 (4,)

match op: topo-index:x+node_ops-1 = topo-index:y
x from L5  y from L1

{topo-index:x,{next-node-topo-index:y, errormessage}}
struct errormessage:{error,[output1,output5],node-name(of x),fix-method}

"""
# we can get a rough compare by comparing the same actually topoindex.
# But becuase of fuseops it is too rough.

# But it is benefical because error always immese from one pattern not a op.
# then we enhance the precision of subgraph and rerun.

# question solved.
# ops:g

import re
import tvm
from tvm import relay, runtime, te
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

comparebaseorder = 0
strict_mode = False


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


class Matchinfo:
    def __init__(
        self,
        opt_order: int,
        unopt_order: int,
    ):
        self.opt_order = opt_order
        self.unopt_order = unopt_order


class Errormessage:
    def __init__(
        self, error: np = -1, hash="", opindex: int = -1, node_name_hint: str = ""
    ):
        self.error = error
        self.params_keys = {"unopt_params_keys": [""], "opt_params_keys": [""]}
        self.params = {"unopt_params": [""], "opt_params": [""]}
        self.node_name_hint = node_name_hint
        self.opindex = opindex
        self.hash = hash
        self.layouts = ["", ""]


class Trace_item(Errormessage):
    def __init__(
        self,
        namenode,
        topoindex: int,
        pre_topoindex: List[int],
        errormessage=Errormessage(),
    ):
        self.nodename = namenode
        self.topoindex = topoindex
        self.pre_topoindex = pre_topoindex.copy()
        self.errormessage = errormessage


class Actualnodeinfo:
    def __init__(
        self,
        nodeindex: int,
        t_order: int,
        coding: np.uint64,
        shape: List[int],
        tpnum: int,
        layout: str,
        params_keys: List[str],
        hash="",
    ):  # params: List[np.ndarray]
        self.nodeindex = nodeindex
        self.torder = t_order
        self.coding = coding
        self.hash = hash
        self.shape = shape
        self.tpnum = tpnum
        self.layout = layout
        self.params_keys = params_keys.copy()  # many outputs


class Trace_error:
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
    def get_topo_structure(
        self, graph_json_path: str, topo_name_index: Dict[str, int]
    ) -> List[Trace_item]:

        # graph node order is consistent with nature number order
        trace_messages: Dict[int, Trace_item] = dict()
        with open(graph_json_path, "r") as gfile:
            gdata = json.load(gfile)
        self.kerneldict = gdata
        length = len(gdata["nodes"])
        for i in range(length):
            node = gdata["nodes"][i]
            index = i  # topo_name_index[node['name'].rstrip('_')]
            if node["op"] == "param":  # local params also does not have inputs
                trace_messages[index] = Trace_item(
                    node["name"], index, [-1], Errormessage().__dict__
                ).__dict__
            else:
                hash = node["attrs"]["hash"]
                trace_messages[index] = Trace_item(
                    node["name"],
                    index,
                    [
                        topo_name_index[i.rstrip("_")] for i in node["inputs"]
                    ],  # ! may need reshape_nop handler
                    Errormessage(hash=hash).__dict__,
                ).__dict__
                # self.hash2optindex[hash ]= index
        return trace_messages

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

    def handle_strings(self, funcname_all: str):
        # print(':'*10)
        # special name + regular name + from_kernel_name(when ops_number>9)
        def get_hash_from_name(funcname):
            nodes = self.kerneldict["nodes"]
            for node in nodes:
                if funcname == node["name"].rstrip("_"):
                    return node["attrs"]["hash"]

        def get_modstr_from_hash(hash):
            # pattern = 'fn.*'+hash+'.*?}'
            pattern = hash + ".*?}"
            matched = re.search(pattern, self.unoptprim_mod, flags=re.M | re.S)
            if matched is not None:
                l, r = matched.span()
                ocode = self.unoptprim_mod
            else:
                matched = re.search(pattern, self.optprim_mod, flags=re.M | re.S)
                l, r = matched.span()
                ocode = self.optprim_mod
            origin_func = ocode[l:r]
            return remove_primary(origin_func)

        def getops_from_kernel(text):
            pat = r"(?P<value>\/\*.*?\*\/)"
            text = re.sub(pat, "", text, count=0, flags=re.S | re.M)
            pat2 = r"[a-zA-Z_]+[.a-zA-Z0-9_]+\("
            ops = re.findall(pat2, text, flags=re.S | re.M)
            ops = [i.rstrip("(").replace(".", "_") for i in ops]
            return ops

        def getconst_from_kernel(text):
            pat2 = r"f\ \/\*\ ty\="
            ops = re.findall(pat2, text, flags=re.S | re.M)
            ops = len(ops)
            return ops

        def getlayout_from_kernel(text):
            pat = r"(?P<value>\/\*.*?\*\/)"
            text = re.sub(pat, "", text, count=0, flags=re.S | re.M)
            if "out_layout" in text and 'out_layout=""' not in text:
                pat2 = r'out_layout="(.*?)"'
                layout = re.search(pat2, text, flags=re.S | re.M).group(1)
                # print(re.search(pat2,text,flags=re.S|re.M).group(1))
                # exit()
            elif "data_layout" in text and 'data_layout=""' not in text:
                pat2 = r'data_layout="(.*?)"'
                layout = re.search(pat2, text, flags=re.S | re.M).group(1)
            else:
                pat2 = r', layout="(.*?)"'
                layout = re.search(pat2, text, flags=re.S | re.M)
                if layout is not None:
                    layout = layout.group(1)
                else:
                    layout = ""
            return layout

        funcname = funcname_all.split("____")[0]
        # print(funcname)
        lendefault = len("tvmgen_default_fused_")
        current_name = funcname[lendefault:].rstrip(string.digits).rstrip("_")
        while current_name.split("_")[-1].isnumeric():
            current_name = current_name.rstrip(string.digits).rstrip("_")
        current_name = self.rm_invalid(current_name)
        index = int(funcname_all.split("____")[1].split(":")[1])
        op_numbers = 0
        coding = np.uint64(0)
        ops = []  # record ops along kernel order

        def call_kernel_analysis():
            hash = get_hash_from_name(funcname)
            kernelstr = get_modstr_from_hash(hash)
            ops = getops_from_kernel(kernelstr)
            constnum = getconst_from_kernel(kernelstr)
            layout = getlayout_from_kernel(kernelstr)
            return hash, ops, constnum, layout

        signature, ops, currtpnum, layout = call_kernel_analysis()
        op_numbers = len(ops)
        for op in ops:
            if op in self.opname:
                coding += np.uint64(self.opcoding[op])
            else:
                print("can not handle op:", op)
                exit()
        # print(op_numbers, index, coding)
        # print(ops)
        return signature, op_numbers, index, coding, currtpnum, layout

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
        # print(key,' no shape')
        exit()

    def key2prekeys(self, tracem: Dict[int, Trace_item]):
        key2pkdict = dict()
        for item in tracem.values():
            key2pkdict[item["nodename"].rstrip("_")] = item["pre_topoindex"]
        return key2pkdict

    def get_actual_index_param(
        self,
        params: Dict[str, np.ndarray],
    ) -> Dict[int, Actualnodeinfo]:
        actualparams = dict()
        keys = self.reorderfunc_withindex(params.keys())
        nodeindex = 0
        opindex = 0  # topo order
        baseindex = 0
        order = 0
        basecoding = np.uint64(0)
        tpnum = 0  # temp time node has how many global params
        codingdict = dict()
        for key in keys:
            # distinguish placehold and function node
            if "tvmgen_default" in key or "reshape_nop" in key:
                # preccessors =  self.key2pkdict[key.split('____')[0].split(':')[0]]
                (
                    hash,
                    addops,
                    nodeindex,
                    coding,
                    currtpnum,
                    layout,
                ) = self.handle_strings(key)
                opindex = baseindex + addops - 1
                baseindex += addops
                basecoding = basecoding + np.uint64(coding)
                # codingdict[nodeindex]= basecoding
                tpnum += currtpnum
            else:
                nodeindex = int(key.split("____")[1].split(":")[1])
                opindex = baseindex + 1 - 1
                baseindex += 1
                tpnum += 1
                layout = ""
                hash = ""
            # get shape
            shape = self.get_shape(key.split("____")[0].split(":")[0])
            # populate
            if order not in actualparams.keys():
                actualparams[order] = Actualnodeinfo(
                    nodeindex, opindex, basecoding, shape, tpnum, layout, [key], hash
                )
            else:
                # actualparams[order].params.append(params[key])
                actualparams[order].params_keys.append(key)
            order += 1
        return actualparams

    def locate_naninf(self, modstr: str):
        print("enter locating")
        if modstr == "1":
            dump_root = self.case_path + "/L1/"
        else:
            dump_root = self.case_path + "/L5/"
        # binary find  using a list [nodeindex: key]
        params_path = os.path.join(
            dump_root, "_tvmdbg_device_CPU_0/output_tensors.params"
        )
        params: Dict[str, np.array] = relay.load_param_dict(
            bytearray(open(params_path, "rb").read())
        )
        keys = self.reorderfunc_withindex(list(params.keys()))
        lens = len(keys)
        # locate last nonnan
        def isnan(y_true):
            if np.isinf(y_true).any() == 1 or np.isnan(y_true).any() == 1:
                return True
            else:
                return False

        def binary_search(l, r):
            if l >= r:
                return l
            m = int((l + r) / 2)
            if (not isnan(params[keys[m]].numpy())) and isnan(
                params[keys[m + 1]].numpy()
            ):
                return m
            if isnan(params[keys[m]].numpy()):
                binary_search(l, m - 1)
            else:
                binary_search(m + 1, r)

        lastindex = binary_search(0, int(lens - 1))
        fixportpath = os.path.join(dump_root, "Locate_NAN_Report")
        with open(fixportpath, "a") as fp:
            fp.write("Located")
            fp.write("\n\n\n" + "*" * 50 + "\n")
            fp.write("The first nan/inf incurs in pattern:", keys[lastindex + 1])
            fp.write("\n\n\n" + "*" * 50 + "\n")
            fp.write("The input arr is:\n\n")
            fp.write(params[keys[lastindex]])
            fp.write("\n\n\n" + "*" * 50 + "\n")
            fp.write("The output arr is:\n\n")
            fp.write(params[keys[lastindex + 1]])

    def MSE(
        self,
        y_true,
        y_pred,
    ):  # precision along with  tf.keras.metrics.MeanRelativeError
        if np.isinf(y_true).any() == 1 or np.isnan(y_true).any() == 1:
            print("y_true have inf\\nan:locating...")
            # self.locate_naninf('1')
            return 0
        if np.isinf(y_pred).any() == 1 or np.isnan(y_pred).any() == 1:
            print("y_pred have inf\\nan:locating...")
            return 0
            # self.locate_naninf('5')
        else:
            pass
        d = np.abs(y_true.astype(np.float64) - y_pred)
        relative_error = np.average(
            d / (np.abs(y_true).astype(np.float64) + 1e-8) * np.not_equal(y_true, 0)
            + np.equal(y_true, 0) * d
        )
        if self.dnn is None:
            relative_error = np.average(
                d / (np.abs(y_true).astype(np.float64) + 1e-8) * np.not_equal(y_true, 0)
                + np.equal(y_true, 0) * d
            )
        else:
            relative_error = np.average(
                d
                / (np.abs(y_true).astype(np.float64) + 1e-8)
                * np.abs(y_true)
                / np.mean(np.abs(y_true))
            )
        return relative_error

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

    def get_equivalent_character(self):
        # get all unique float-ps
        s1 = str(self.optprim_mod)
        s2 = str(self.unoptprim_mod)
        kfps = re.findall(r"\d+\.*\d+f", s1)
        vfps = re.findall(r"\d+\.*\d+f", s2)
        fps = list(set(kfps) & set(vfps))
        self.eqhash = dict()
        for fp in fps:
            a = re.search(
                f'{fp[::-1]}.*?"([0-9a-f]*)"=hsah', s1[::-1], re.S | re.M
            ).group(1)[::-1]
            b = re.search(
                f'{fp[::-1]}.*?"([0-9a-f]*)"=hsah', s2[::-1], re.S | re.M
            ).group(1)[::-1]
            self.eqhash[a] = b  # find corresponding hash {opthash-> unopthash}

    def get_eqnodeindex(
        self, opttracem: Dict[int, Trace_item], unopttracem: Dict[int, Trace_item]
    ):
        # prepare hash2nodeindex
        hash2optindex = dict()
        hash2unoptindex = dict()
        for k, v in opttracem.items():
            hash2optindex[v.hash] = k
        for k, v in unopttracem.items():
            hash2unoptindex[v.hash] = k
        print("5", hash2optindex)
        print("1", hash2unoptindex)
        # get equal index
        list1 = []
        self.eqnodeindex = dict()
        for hash5, hash1 in self.eqhash.items():
            index5 = hash2optindex[hash5]
            index1 = hash2unoptindex[hash1]
            if index1 not in list1:
                self.eqnodeindex[index5] = index1
            else:
                tmp = dict(zip(self.eqnodeindex.values(), self.eqnodeindex.keys()))
                if index1 in self.eqnodeindex.keys():
                    del self.eqnodeindex[tmp[index1]]
            list1.append(index1)
        # print(self.eqnodeindex)

    def get_trace_message(self, report: Queue = None) -> str:
        # first step
        self.get_equivalent_character()

        # second, prepare actualnode and hash
        params_path = os.path.join(
            self.opt_root, "_tvmdbg_device_CPU_0/output_tensors.params"
        )
        graph_json_path = os.path.join(
            self.opt_root, "_tvmdbg_device_CPU_0/_tvmdbg_graph_dump.json"
        )
        params: Dict[str, np.array] = relay.load_param_dict(
            bytearray(open(params_path, "rb").read())
        )
        keys = self.reorderfunc_withindex(params.keys())
        names = [(i, key.split("____")[0].split(":")[0]) for i, key in enumerate(keys)]
        # print('opt5 k:',names)
        # params check
        # path_params = os.path.join(self.case_path, 'inputs.npz')
        # with np.load(path_params) as f:
        #     loaded_params = dict(f.items())
        # self.eparams = loaded_params
        # for k, v in self.eparams.items():
        #     for ck in params.keys():
        #         if k in ck:
        #             print(k, ck)
        #             tvm.testing.assert_allclose(v,params[ck].numpy() , rtol=1e-8, atol=1e-8)
        topo_name_index = self.get_topo_name_index(params.keys())
        index_topo_name = dict([val, key] for key, val in topo_name_index.items())
        trace_messages: Dict[int, Trace_item] = self.get_topo_structure(
            graph_json_path, topo_name_index
        )
        # populate_errormessages(trace_messages, params)
        # self.key2pkdict =  self.key2prekeys(trace_messages)
        # get precessor
        # self.precessordict = getprecessor(graph_json_path)
        optparams = self.get_actual_index_param(
            params,
        )
        # print('op topo:', [(value.torder, value.nodeindex)
        # for key, value in optparams.items()])

        # unopt
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
        # topo_name_index1 = self.get_topo_name_index(params1.keys())
        # trace_messages1: Dict[int,Trace_item] = self.get_topo_structure(
        #     graph_json_path, topo_name_index1)
        # self.key2pkdict =  self.key2prekeys(trace_messages1)
        unoptparams = self.get_actual_index_param(params1)
        keys = self.reorderfunc_withindex(params1.keys())
        # names = [(i,key.split('____')[0].split(':')[0]) for i,key in enumerate(keys)]
        # print('opt1 k:',names)
        self.pararms5 = params
        del params
        self.pararms1 = params1
        del params1
        diff = 0
        self.matchinfos = []
        # get sign nodes
        self.get_eqnodeindex(optparams, unoptparams)
        # match
        def imprecsionmatch(optorder, optnode, unoptparams):
            global comparebaseorder
            # sign points
            if optorder in self.eqnodeindex.keys():
                uk = self.eqnodeindex[optorder]
                shape1 = unoptparams[uk].shape
                shape5 = optnode.shape
                if np.prod(shape1) == np.prod(
                    shape5
                ):  # np.prod(ushape)!=np.prod(shape) / ushape!=shape
                    self.matchinfos.append(Matchinfo(optorder, uk))
                    comparebaseorder = uk + 1
                    # synchronize encoding case1: opt< unopt
                    optcode = optnode.coding
                    unoptcode = unoptparams[uk].coding
                    delta = 0
                    if optcode < unoptcode:
                        delta = unoptcode - optcode
                    elif optcode > unoptcode:
                        # handle unoptparams
                        tdelta = optcode - unoptcode
                        tkey = uk
                        while tkey in unoptparams.keys():
                            unoptparams[tkey].coding += np.uint64(tdelta)
                            tkey += 1
                    else:
                        pass
                    print(optorder, uk, "******")
                    return uk, True, delta
            if strict_mode:
                return -1, False, 0
            for uk, unoptnode in list(unoptparams.items())[comparebaseorder:]:
                unoptcoding = unoptnode.coding
                optcoding = optnode.coding
                utpnums = unoptnode.tpnum
                tpnums = optnode.tpnum
                uleng = unoptnode.torder
                leng = optnode.torder
                # modify coding amendment
                if self.qnn:
                    pass
                else:
                    if tpnums < utpnums:
                        optcoding += np.uint64((utpnums - tpnums) * 2)
                    else:
                        unoptcoding += np.uint64((tpnums - utpnums) * 2)
                # get diff
                diff = (
                    float(optcoding - unoptcoding)
                    if optcoding > unoptcoding
                    else float(unoptcoding - optcoding)
                )
                # get tolerance
                if self.qnn:
                    tolerance = 2 + 1 * abs(leng) / 20.0  # may modify  #5
                else:
                    tolerance = 1 + abs(leng) / 10.0  # may modify  #10
                # print('tolerance',tolerance)
                if diff < tolerance:
                    # shape compare
                    ushape = unoptnode.shape
                    shape = optnode.shape
                    if np.prod(ushape) != np.prod(
                        shape
                    ):  # np.prod(ushape)!=np.prod(shape) / ushape!=shape
                        continue
                    if uk + 1 == len(list(unoptparams.items())):
                        self.matchinfos.append(Matchinfo(optorder, uk))
                        comparebaseorder = uk + 1
                        return uk, True, 0
                    if self.qnn:  # next diff check:
                        unoptnode2 = unoptparams[uk + 1]
                        unoptcoding = unoptnode2.coding
                        optcoding = optnode.coding
                        utpnums = unoptnode2.tpnum
                        tpnums = optnode.tpnum
                        uleng = unoptnode2.torder
                        leng = optnode.torder
                        # modify coding amendment
                        if self.qnn:
                            pass
                        else:
                            if tpnums < utpnums:
                                optcoding += np.uint64((utpnums - tpnums) * 2)
                            else:
                                unoptcoding += np.uint64((tpnums - utpnums) * 2)
                        # get diff
                        diff2 = (
                            float(optcoding - unoptcoding)
                            if optcoding > unoptcoding
                            else float(unoptcoding - optcoding)
                        )
                        ushape2 = unoptnode2.shape
                        if diff2 < diff and (
                            np.prod(ushape) == np.prod(ushape2)
                        ):  # np.prod(ushape)==np.prod(ushape2) / ushape2== ushape or ushape==ushape2
                            self.matchinfos.append(Matchinfo(optorder, uk + 1))
                            comparebaseorder = uk + 2
                            return uk + 1, True, 0
                        else:
                            self.matchinfos.append(Matchinfo(optorder, uk))
                            comparebaseorder = uk + 1
                            return uk, True, 0
                    else:
                        self.matchinfos.append(Matchinfo(optorder, uk))
                        comparebaseorder = uk + 1
                        return uk, True, 0
            return -1, False, 0

        def clayout(a_np, k):
            a, b, c, d = a_np.shape
            Dtype = str(a_np.dtype)
            input_tensor = te.placeholder((a, b, c, d), dtype=Dtype, name="input")
            C = te.compute(
                (a, int(b / k), c, d, k),
                lambda n, c, h, w, c3: input_tensor[n, c * k + c3, h, w],
                name="output",
            )
            s = te.create_schedule(C.op)
            target = "llvm"
            func = tvm.build(s, [input_tensor, C], target=target)
            b_np = np.zeros(shape=(a, int(b / k), c, d, k)).astype(Dtype)
            a_tvm = tvm.nd.array(
                a_np,
            )
            b_tvm = tvm.nd.array(
                b_np,
            )
            func(a_tvm, b_tvm)
            b_result = b_tvm.asnumpy()
            return b_result

        def metric(outs1, outs5, l1, l5):
            diff = 0.0
            outshape1 = outs1[0].numpy().shape
            outshape2 = outs5[0].numpy().shape
            if str(outshape1) == str(outshape2) or str(outshape1) in str(outshape2):
                for ro, o in zip(outs1, outs5):
                    diff = max(
                        diff, self.MSE(ro.numpy().flatten(), o.numpy().flatten())
                    )
            else:
                if (l1 == "" or l1 == "NCHW") and (
                    re.match(r"NCHW\dc", l5) is not None or l5 == "NCHWc"
                ):
                    diff = 0
                    optp = outs5[0].numpy()
                    opt_ct = clayout(outs1[0].numpy(), outshape2[-1])
                    diff = self.MSE(
                        opt_ct.flatten(), optp.flatten()
                    )  # diff = self.MSE(opt_ct,optp)
                elif (
                    np.size(outshape1) == 4
                    and np.size(outshape2) == 5
                    and (
                        outshape1[0] == outshape2[0]
                        and outshape1[2] == outshape2[2]
                        and outshape1[3] == outshape2[3]
                    )
                ):
                    diff = 0
                    optp = outs5[0].numpy()
                    opt_ct = clayout(outs1[0].numpy(), outshape2[-1])
                    diff = self.MSE(
                        opt_ct.flatten(), optp.flatten()
                    )  # diff = self.MSE(opt_ct,optp)
                else:
                    print("undefined comparision")
                    diff = -1
                print(diff)
            return diff

        for key, indexparams in optparams.items():
            trace_messages[indexparams.nodeindex]["errormessage"][
                "node_name_hint"
            ] = index_topo_name[indexparams.nodeindex]
            outs5 = [self.pararms5[i] for i in indexparams.params_keys]
            if key + 1 < len(optparams.keys()):
                if (
                    indexparams.nodeindex + 1 in index_topo_name.keys()
                    and "fused_layout_transform"
                    in index_topo_name[indexparams.nodeindex + 1]
                    and optparams[key].coding == optparams[key + 1].coding
                ):  # optparams[key+1].:
                    continue
            unopt_ordernum, flag, delta = imprecsionmatch(
                key, optparams[key], unoptparams
            )  # fuzzy match
            if delta != 0:
                tkey = key
                while tkey in optparams.keys():
                    optparams[tkey].coding += np.uint64(delta)
                    tkey += 1
            if flag == False:
                continue
            print("match", key, unopt_ordernum, flag)
            outs1 = [self.pararms1[i] for i in unoptparams[unopt_ordernum].params_keys]
            diff = metric(
                outs1, outs5, unoptparams[unopt_ordernum].layout, optparams[key].layout
            )
            # print(trace_messages.keys())
            trace_messages[indexparams.nodeindex]["errormessage"]["error"] = float(diff)
            trace_messages[indexparams.nodeindex]["errormessage"]["opindex"] = key
            trace_messages[indexparams.nodeindex]["errormessage"]["params_keys"][
                "unopt_params_keys"
            ] = unoptparams[unopt_ordernum].params_keys
            trace_messages[indexparams.nodeindex]["errormessage"]["params_keys"][
                "opt_params_keys"
            ] = indexparams.params_keys
            trace_messages[indexparams.nodeindex]["errormessage"]["layouts"][
                0
            ] = unoptparams[unopt_ordernum].layout
            trace_messages[indexparams.nodeindex]["errormessage"]["layouts"][
                1
            ] = optparams[key].layout
            if diff > 0:
                trace_messages[indexparams.nodeindex]["errormessage"]["params"][
                    "unopt_params"
                ] = [np.array2string(i.numpy()) for i in outs1]
                trace_messages[indexparams.nodeindex]["errormessage"]["params"][
                    "opt_params"
                ] = [np.array2string(i.numpy()) for i in outs5]
            # trace_messages[indexparams.nodeindex]['errormessage']['node_name_hint'] = \
            #     index_topo_name[indexparams.nodeindex] if indexparams.nodeindex in index_topo_name.keys() else 'reshape_nop'
        global comparebaseorder
        comparebaseorder = 0
        dump_path = os.path.join(self.case_path, "trace.json")
        with open(dump_path, "w") as fp:
            json.dump(
                trace_messages,
                fp,
                default=self.encode_obj,
                indent=4,
                sort_keys=True,
            )
        if report is not None:
            report.put({"trace stored in file:", dump_path})
        else:
            print("trace stored in file:", dump_path)
        for key, value in optparams.items():
            print("op opt_topo:", "key:", key, value.__dict__)
        for key, value in unoptparams.items():
            print("op unopt_topo:", "key:", key, value.__dict__)
        return dump_path
