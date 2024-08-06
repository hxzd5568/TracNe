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
from typing import Iterable, List, cast, Optional, Dict, Any
import sys
from multiprocessing import Process, Queue
from tvm.relay.backend import Runtime, Executor, graph_executor_codegen
import re
import json
from tvm.relay.transform import InferType, ToMixedPrecision, mixed_precision
from .calculator2 import Calculate_error
from threading import Thread

TensorDict = Dict[str, np.ndarray]
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
Required_pass1 = [
    "EliminateCommonSubexpr",
    "CombineParallelDense",
    "CombineParallelBatchMatmul",
    "CombineParallelConv2D",
]


def buildlib1(mod, params=None, Disabled_pass=["SimplifyExpr"]):
    with transform.PassContext(
        opt_level=1, required_pass=Required_pass1, disabled_pass=Disabled_pass
    ):
        lib1 = relay.build(mod, target, params=params)
    return lib1


def MSE(
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
    return relative_error


def rewrelay(hash: str, ocode, rcode, newf_lastnum, base_num, onum):
    # match mod funciton with f2
    hash = hash
    pattern1 = hash + ".*?\}"
    pattern2 = hash[::-1] + "(.*?nf)"
    matched = re.search(pattern1, ocode, flags=re.M | re.S)
    a = matched.group()
    _, r = matched.span()
    matched = re.search(pattern2, ocode[::-1], flags=re.M | re.S)
    b = matched.group(1)[::-1]
    _, l = matched.span()
    l = len(ocode) - l
    origin_func = b + a
    # update int number
    rl, rr = re.search("fn.*?\}", rcode, flags=re.M | re.S).span()
    new_func = rcode[rl:rr]
    addfactor = newf_lastnum - onum
    # back change handle_number
    def update_external(matched):
        value = int(matched.group("value").strip("%"))
        # print(str(value))
        if value >= onum:
            return "%" + str(value + addfactor)
        else:
            return "%" + str(value)

    def update_internal(matched):
        value = int(matched.group("value").strip("%"))
        return "%" + str(value + base_num)

    # print(ocode[r:])
    poststr = re.sub(
        "(?P<value>%\d+)", update_external, ocode[r:], count=0, flags=re.M | re.S
    )
    # print(poststr)
    prestr = re.sub(
        "(?P<value>%\d+)", update_external, ocode[:l], count=0, flags=re.M | re.S
    )
    new_func = re.sub(
        "(?P<value>%\d+)", update_internal, new_func, count=0, flags=re.M | re.S
    )
    ncode = prestr + new_func + poststr
    return ncode


def remove_virtarget(ncode):
    rncode = re.sub("({virtual_.*?}:)", ":", ncode, count=0, flags=re.M | re.S)
    rncode = re.sub("(virtual_.*?->)", ") ->", rncode, count=0, flags=re.M | re.S)
    return rncode


def remove_primary(code):
    return re.sub("(, Primitive\=.*?->)", ") ->", code, count=0, flags=re.M | re.S)


def run_module(mod, mod_params: Dict[str, Any]) -> List:
    opt_level = 5
    with transform.PassContext(opt_level=opt_level, disabled_pass=["SimplifyExpr"]):
        mod5 = relay.build(
            mod,
            target,  # params=params
        )
    graph_mod = GraphModule(mod5["default"](dev))
    graph_mod.run(**mod_params)
    length = graph_mod.get_num_outputs()
    return [graph_mod.get_output(i).numpy() for i in range(length)]


# auto debug funcname
#                ----> prim_mod 's hash  ---> mod's call(function)
#                ----> cast precision(to mixed)
#                ----> fn_level valid  ----> rewrite relaytxt ----> fuzz[process]
#                ----> ./fixed/fixed_relay.txt, ./fixed/fixed_relay.tar, .
#                       /fixed error report(error, cases num)


class Corrector:
    def __init__(self, dumppath) -> None:
        self.dumpath = dumppath
        self.dictpath = os.path.join(dumppath, "trace.json")
        modpath = os.path.join(dumppath, "code.txt")
        optmodpath = os.path.join(dumppath, "optirmod.txt")
        unoptmodpath = os.path.join(dumppath, "unoptirmod.txt")
        with open(self.dictpath, "r") as fp:
            self.trace_error_dict = json.load(fp)
        with open(modpath, "r") as fp:
            self.mod = relay.parse(fp.read())
            self.pbatch = int(
                int(re.findall("(%\d+)", str(self.mod))[-1].strip("%")) / 10 + 1
            )
        with open(optmodpath, "r") as fp:
            self.optirmod = relay.parse(fp.read())
        with open(unoptmodpath, "r") as fp:
            self.unoptirmod = relay.parse(fp.read())
        self.Dtype = re.search("main.+?Tensor\[.+?\),\ (.+?)\]", str(self.mod)).group(1)
        self.tolerance = (
            1e-3 if self.Dtype == "float16" else 1e-6
        )  # should be checked RE and MRE ratio
        self.debpath1 = os.path.join(
            dumppath, "L1/_tvmdbg_device_CPU_0/_tvmdbg_graph_dump.json"
        )
        self.debpath5 = os.path.join(
            dumppath, "L5/_tvmdbg_device_CPU_0/_tvmdbg_graph_dump.json"
        )
        # with open(self.debpath1, 'r') as fp:
        #     self.nodedict1 = json.load(fp)
        # with open(self.debpath5, 'r') as fp:
        #     self.nodedict5 = json.load(fp)
        self.prim_mod = self.optirmod.astext()
        self.obj_func = None
        # a function's base num and handle nums should be recorded
        # and then adjusted for preventing collision
        self.base_num = 1000
        self.newf_lastnum = 0
        self.returnty = ""
        self.eouts = None
        self.fixbatchnums = 0
        self.Disabled_pass = ["SimplifyExpr"]

    def verify_mixed_precision_(
        self,
        mod: tvm.runtime.Module,
        eouts: Dict[str, Any] = None,
        mixed_precision_dtype="float64",
        rtol: float = 1e-10,
        atol: float = 0,
        keep_orig_output_dtype=True,
    ) -> tvm.runtime.Module:
        # result_fp32 = run_module(mod)
        mod = InferType()(mod)
        if not keep_orig_output_dtype:
            amp_mod = ToMixedPrecision(mixed_precision_dtype)(mod)
            # result_amp = run_module(amp_mod, mod_params)
        else:  # this path
            with tvm.transform.PassContext(
                config={"relay.ToMixedPrecision.keep_orig_output_dtype": True}
            ):
                amp_mod = ToMixedPrecision(mixed_precision_dtype)(mod)
        fixportpath = os.path.join(self.dumpath, "Fix_Report")
        with open(fixportpath, "w") as fp:
            pass
        fixportpath = os.path.join(self.dumpath, "Conflict_Report")
        with open(fixportpath, "w") as fp:
            pass
        return amp_mod

    def hack_build_graph(self, target="llvm"):
        def get_mod_hash_callfunction():
            # [I.GlobalVar("tvmgen_default_fused_exp_exp_exp_exp_add"),
            irmodule_funcname_list = self.irmodule.get_global_vars()
            funcnames: str = []
            lens = len(irmodule_funcname_list)
            for svar in irmodule_funcname_list:
                # print(svar)
                matched = re.search(r"\"tvm.*?\"", str(svar))
                l, r = matched.span()
                funcnames.append(str(svar)[l + 1 : r].rstrip('"\)'))
            return [
                self.irmodule.__getitem__(funcnames[i]).attrs["hash"]
                for i in range(lens)
            ]

        def get_function_fromhash(hash, ocode):
            hash = hash
            assert hash != "" and hash is not None
            pattern1 = hash + ".*?\}"
            pattern2 = hash[::-1] + "(.*?nf.*?\d+%)"
            matched1 = re.search(pattern1, ocode, flags=re.M | re.S)
            a = matched1.group()
            matched = re.search(
                pattern2, ocode[: matched1.span()[1]][::-1], flags=re.M | re.S
            )
            b = matched.group(1)[::-1]
            origin_func = b + a
            return remove_primary(origin_func)

        target, target_host = tvm.target.Target.canon_target_and_host(target)
        # with tvm.transform.PassContext(opt_level=1):# config={"relay.FuseOps.max_depth": 10}
        #     self.prim_mod, _ = relay.optimize(self.mod, target)
        #     print(self.prim_mod)
        # with tvm.transform.PassContext(opt_level=5):# config={"relay.FuseOps.max_depth": 10}
        #     self.prim_mod, _ = relay.optimize(self.mod, target)
        #     print(self.prim_mod)
        # grc = graph_executor_codegen.GraphExecutorCodegen(None, target)
        # _, lowered_funcs, _ = grc.codegen(self.prim_mod, self.prim_mod["main"])
        # self.rmod = relay.backend._backend.build(lowered_funcs, target)
        # arr = grc._get_irmodule()
        # target, self.irmodule = list(arr.items())[0]
        # print(self.irmodule)
        # self.get_prim_mod_hashs(self.irmodule)
        # f2 = self.prim_mod.__getitem__('main').body.op.body
        # print(type(f2))
        funcstr = get_function_fromhash(self.hash, self.prim_mod)
        print("funcstr:", funcstr)
        # get base number
        matchedall = re.findall(
            r"(%\d+ = )", funcstr[funcstr.find("{") :]
        )  # change handle base mnumber
        if len(matchedall) >= 1:
            self.base_num = int(re.findall(r"(\d+)", matchedall[0])[0])
            self.onum = int(re.findall(r"(\d+)", matchedall[-1])[0]) + 1
        else:
            self.base_num = int(re.findall(r"(%\d+)", funcstr)[0].strip("%"))
            self.onum = self.base_num

        def functomod(code):
            l = re.search("fn", code).span()[0]
            code = '#[version = "0.0.5"]\ndef @main ' + code[l + 2 :]
            print("code:", code)
            return code

        mod = relay.parse(functomod(funcstr))
        return mod

    # def enhance_call_function(self):
    #     mod_low = tvm.IRModule.from_expr(self.obj_func)
    #     with tvm.transform.PassContext(
    #         config={"relay.ToMixedPrecision.keep_orig_output_dtype": True}
    #     ):
    #         mod_high = relay.transform.ToMixedPrecision('float64')(mod_low)
    #     self.obj_func = mod_high['main']

    def prim_valid_fix(self):
        # get error input and then test
        # general test
        calculator = Calculate_error(self.mod)
        inp_path = os.path.join(self.dumpath, "inputs.npz")

        # prepare tar1
        if os.path.exists(self.dumpath + "/compiled_lib1.tar"):
            self.factorymod1 = tvm.runtime.load_module(
                self.dumpath + "/compiled_lib1.tar"
            )
            gmod1 = GraphModule(self.factorymod1["default"](dev))

        error_after_fix = calculator.replay_withlocatenan_withtar1(inp_path, gmod1)
        if error_after_fix > self.tolerance:
            if os.path.exists(self.dumpath + "-h/code.txt"):
                with open(self.dumpath + "-h/code.txt", "r") as f:
                    originmh = relay.parse(f.read())
                hlib1 = buildlib1(originmh)
                hgmod1 = GraphModule(hlib1["default"](dev))
                error_after_fix2 = calculator.replay_withlocatenan_withtar1(
                    inp_path, hgmod1
                )
                if error_after_fix2 < self.tolerance:
                    print("opt are more precious", error_after_fix2)
                else:
                    factorymod5 = tvm.runtime.load_module(
                        self.dumpath + "/compiled_lib5.tar"
                    )
                    tar5 = GraphModule(factorymod5["default"](dev))
                    error_after_fix3 = calculator.replay_withlocatenan_withtar5(
                        inp_path, hgmod1, tar5
                    )
                    print("fix error", error_after_fix2)
                    print("before fix error", error_after_fix3)
            else:
                with open(self.dumpath + "/code.txt", "r") as f:
                    originmh = relay.parse(f.read())
                originmh = self.verify_mixed_precision_(originmh)
                hlib1 = buildlib1(originmh)
                hgmod1 = GraphModule(hlib1["default"](dev))
                error_after_fix2 = calculator.replay_withlocatenan_withtar1(
                    inp_path, hgmod1
                )
                if error_after_fix2 < self.tolerance:
                    print("opt are more precious", error_after_fix2)
                else:
                    factorymod5 = tvm.runtime.load_module(
                        self.dumpath + "/compiled_lib5.tar"
                    )
                    tar5 = GraphModule(factorymod5["default"](dev))
                    error_after_fix3 = calculator.replay_withlocatenan_withtar5(
                        inp_path, hgmod1, tar5
                    )
                    print("fix error", error_after_fix2)
                    print("before fix error", error_after_fix3)
            fixportpath = os.path.join(self.dumpath, "Conflict_Report")
            with open(fixportpath, "a") as fp:
                fp.write("Contraversy")
                fp.write("\n\n\n" + "*" * 50 + "\n")
                fp.write("fixed method: Targeted repair")
                fp.write("\n\n\n" + "*" * 50 + "\n")
                fp.write(
                    "After repairing, error prone input's mean relatively error is: %.10f"
                    % (error_after_fix)
                )  #
                fp.write("\n\n\n" + "*" * 50 + "\n")
                fp.write("The best recommendation:\n\n")
                fp.write(self.mod.astext())
                fp.write("\n\n\n" + "*" * 50 + "\n")
                fp.write("The repair subgraph:\n\n")
                fp.write(self.obj_func.astext())
            return False

        error = calculator.random_difference_test()
        fixportpath = os.path.join(self.dumpath, "Fix_Report")
        with open(fixportpath, "a") as fp:
            fp.write("Fixed")
            fp.write("\n\n\n" + "*" * 50 + "\n")
            fp.write("fixed method: Targeted repair")
            fp.write("\n\n\n" + "*" * 50 + "\n")
            fp.write(
                "After repairing, error prone input's mean relatively error is: %.10f"
                % (error_after_fix)
            )  #
            fp.write("\n\n\n" + "*" * 50 + "\n")
            fp.write("After repairing, mean relatively error is: %.10f" % (error))  #
            fp.write("\n\n\n" + "*" * 50 + "\n")
            fp.write("The best recommendation:\n\n")
            fp.write(self.mod.astext())
            fp.write("\n\n\n" + "*" * 50 + "\n")
            fp.write("The repair subgraph:\n\n")
            fp.write(self.obj_func.astext())
        return True

    def defuse_mod(self, mod):
        seq = tvm.transform.Sequential(
            [
                relay.transform.InferType(),
                relay.transform.DefuseOps(),
                relay.transform.InferType(),
            ]
        )
        with tvm.transform.PassContext(opt_level=4):
            return seq(mod)

    def mod_fix_and_check(self, input):
        # self.mod_low = self.defuse_mod(self.mod_low)
        # find return type
        returnty = re.findall("(->\ Tensor\[.*?\])", str(self.obj_func))[0]
        self.returnty = re.findall("(float\d+)", returnty)[0]
        self.obj_func = self.verify_mixed_precision_(
            self.obj_func, mod_params=input, eouts=self.eouts
        )
        ss = str(self.obj_func)
        # handle_latest_cast_fp type

        def changereturn(matched):
            # -> Tensor[(4), float32]
            returnty = re.findall("(float\d+)", matched.group("value"))[0]
            value = matched.group("value").replace(returnty, self.returnty)
            return value

        ss = re.sub("(?P<value>->\ Tensor\[.*?\])", changereturn, ss)
        matched = re.findall("(cast\(%\d+.*?\))", ss)[-1]
        beginindex = ss.find(matched)
        stail = ss[beginindex:]
        shead = ss[:beginindex]
        stail = re.sub("(?P<value>float\d+)", changereturn, stail)
        ss = shead + stail
        self.obj_func = relay.parse('#[version = "0.0.5"]\n' + ss)
        matched = re.findall("(%\d+\ =)", ss)[-1]
        self.newf_lastnum = int(matched.strip("%=")) + 1 + self.base_num
        return True

    def mod_fix(
        self,
        input=None,
    ):
        # self.mod_low = self.defuse_mod(self.mod_low)
        # find return type
        returnty = re.findall("(->\ Tensor\[.*?\])", str(self.obj_func))[0]
        self.returnty = re.findall("(float\d+)", returnty)[0]
        self.obj_func = self.verify_mixed_precision_(self.obj_func, eouts=self.eouts)
        ss = str(self.obj_func)
        # handle_latest_cast_fp type

        def changereturn(matched):
            # -> Tensor[(4), float32]
            returnty = re.findall("(float\d+)", matched.group("value"))[0]
            value = matched.group("value").replace(returnty, self.returnty)
            return value

        code = re.search(r"\{.*\}", ss, flags=re.S | re.M).group(0)
        sslen = len(re.findall("\n", code))
        ss = re.sub("(?P<value>->\ Tensor\[.*?\])", changereturn, ss)
        # last expr
        if sslen > 2 and "cast" in code.split("\n")[-2]:
            matched = re.findall("(cast\(%\d+.*?\))", ss)[-1]
            beginindex = ss.find(matched)
            stail = ss[beginindex:]
            shead = ss[:beginindex]
            stail = re.sub("(?P<value>float\d+)", changereturn, stail)
            ss = shead + stail
            self.obj_func = relay.parse('#[version = "0.0.5"]\n' + ss)
        else:
            assert "cast" not in code.split("\n")[-2]
            alterloc = ss.find(code.split("\n")[-2])
            sh, sm, st = ss[:alterloc], code.split("\n")[-2], "  \n}"
            lastnum = int(re.findall(r"%(\d) = ", sh)[-1])
            sm = (
                "  %"
                + str(lastnum + 1)
                + " = "
                + sm
                + ";\n"
                + f'  cast(%{lastnum+1},dtype="{self.returnty}")'
            )
            ss = sh + sm + st
            self.obj_func = relay.parse('#[version = "0.0.5"]\n' + sh + sm + st)
        print(self.obj_func)
        # get last num
        matched = re.findall("(%\d+\ =)", ss, flags=re.S | re.M)[-1]
        self.newf_lastnum = int(matched.strip("%=")) + 1 + self.base_num

    def pipeline(self):
        #         """
        #         #handetest
        #         """

        #         code = """
        # #[version = "0.0.5"]
        # def @main(%p03: Tensor[(3, 4, 1, 4), float16] /* ty=Tensor[(3, 4, 1, 4), float16] span=from_string:3:12 */) -> Tensor[(3, 4, 1, 1), float16] {
        #   %0 = cast(%p03, dtype="float32") /* ty=Tensor[(3, 4, 1, 4), float32] */;
        #   %1 = mean(%0, axis=[2, 3], keepdims=True) /* ty=Tensor[(3, 4, 1, 1), float32] span=from_string:3:7 */;
        #   cast(%1, dtype="float16")
        # }

        #             """
        #         print(code)
        #         code = re.search(r'\{.*\}',code,flags= re.S|re.M).group(0)
        #         print(code)
        #         print(len(re.findall('\n', code)))

        #         code = remove_virtarget(code)
        #         code = remove_primary(code)
        # self.mod = self.defuse_mod(relay.parse(code))
        #         print(code)
        #         opt_level = 5
        #         with transform.PassContext(opt_level=opt_level):
        #              mod5 = relay.build(self.mod, target, #params=params
        #                     )
        #         calculator = Calculate_error(self.mod)
        #         calculator.random_difference_test()
        #         exit()
        # get incorrect node and hash
        # check all non-zero nodes in the path
        # collect all sensitive subgraph
        def get_out_node():
            nodelen = len(self.trace_error_dict) - 1
            outputnode = [True] * (nodelen + 1)
            prenodestrue = []
            for i in range(nodelen + 1):
                prenodestrue += self.trace_error_dict[str(i)]["pre_topoindex"]
            prenodestrue = sorted(list(set(prenodestrue)))
            prenodestrue.remove(-1)

            for i in prenodestrue:
                outputnode[i] = False
            outputindex = []
            for i in range(len(outputnode)):
                if outputnode[i]:
                    outputindex.append(i)
            return outputindex

        outputindex = get_out_node()
        print("outputindex", outputindex)

        print("self.pbatch", self.pbatch)

        def fix_sensi(outputindex):
            for k, i in enumerate(outputindex):
                print("current output", k, i)
                if self.trace_error_dict[str(i)]["errormessage"]["error"] > 0:
                    prenode_index = self.trace_error_dict[str(i)]["pre_topoindex"]
                    sensi_nodes = [i]
                else:
                    assert self.trace_error_dict[str(i)]["errormessage"]["error"] == 0
                    print("node outputs 0")
                    continue
                    currentindex = i - 1
                    while currentindex != 0:
                        if (
                            self.trace_error_dict[str(currentindex)]["errormessage"][
                                "error"
                            ]
                            > 0
                        ):
                            prenode_index = self.trace_error_dict[str(currentindex)][
                                "pre_topoindex"
                            ]
                            sensi_nodes = [currentindex]
                            break
                        currentindex -= 1
                # trace back along each chain
                # handle current chain
                if -1 in prenode_index:
                    prenode_index.remove(-1)
                zero_nodes = [
                    i
                    for i in prenode_index
                    if self.trace_error_dict[str(i)]["errormessage"]["error"] == 0
                ]
                print("zero_nodes", zero_nodes)
                for i in zero_nodes:
                    prenode_index.remove(i)
                sensi_nodes += prenode_index.copy()
                while len(prenode_index) != 0:  # bfs
                    idx = prenode_index[0]
                    print(idx)
                    print("prenode_index", prenode_index)
                    newidx = self.trace_error_dict[str(idx)]["pre_topoindex"]
                    for nidx in newidx:
                        if (
                            nidx != -1
                            and self.trace_error_dict[str(nidx)]["errormessage"][
                                "error"
                            ]
                            != 0
                        ):
                            if nidx not in sensi_nodes:
                                sensi_nodes.append(nidx)
                                prenode_index.append(nidx)
                    prenode_index.remove(idx)
                sensi_nodes.sort()

                print("sensi_nodes", sensi_nodes)
                # fix end assert
                # fix can interupt
                fixflag = 0
                temp_fix_node = sensi_nodes[0 : self.pbatch]  # 5self.pbatch
                sensi_nodes = sensi_nodes[self.pbatch :]
                # begin amp each nodes and then compile
                changeflag = False
                for temp_idx in temp_fix_node:
                    value = self.trace_error_dict[str(temp_idx)]
                    self.hash = value["errormessage"]["hash"]
                    if self.hash == "" or self.hash is None:
                        print("error because of constant fold")
                        continue
                    changeflag = True
                    self.obj_func = self.hack_build_graph()
                    # whether fix
                    ss0 = str(self.obj_func)
                    code = re.search(
                        r"\{.*\}", str(self.obj_func), flags=re.S | re.M
                    ).group(0)
                    sslen0 = len(re.findall("\n", code))
                    if sslen0 == 3 and "layout_transform" in ss0:
                        continue
                    if (
                        sslen0 > 2
                        or sslen0 == 2
                        and (
                            "softmax" in ss0
                            or "tan" in ss0
                            or "exp" in ss0
                            or "erf" in ss0
                        )
                    ):
                        self.mod_fix()
                        print(
                            "self.newf_lastnum, self.base_num, self.onum",
                            self.newf_lastnum,
                            self.base_num,
                            self.onum,
                        )
                        mod = rewrelay(
                            self.hash,
                            remove_virtarget(self.prim_mod),
                            str(self.obj_func).replace("main", "fn"),
                            self.newf_lastnum,
                            self.base_num,
                            self.onum,
                        )
                        self.prim_mod = mod
                self.fixbatchnums += 1  # control controversy
                # bind subgraph inputs
                """
                with open(params_path, "rb") as fp:
                    params: Dict[str, np.array] = \
                        relay.load_param_dict(bytearray(fp.read()))
                    for k in prenode_inpkeys:
                        prenode_inputs.append(params[k])
                    del params

                unoptparams_path = os.path.join(self.dumpath,
                                        'L1/_tvmdbg_device_CPU_0/output_tensors.params')
                self.eouts = []
                with open(unoptparams_path, "rb") as fp:
                    params: Dict[str, np.array] = \
                        relay.load_param_dict(bytearray(fp.read()))
                    for k in eoutskeys:
                        self.eouts.append(params[k])
                    del params

                # get incorrect node's inputs
                def bind_inputs():
                    inputs = dict()
                    names = [i.name_hint for i in self.obj_func['main'].params]
                    types = [i.type_annotation for i in self.obj_func['main'].params]
                    for (name, type) in zip(names,types):
                        for kth, input in enumerate(prenode_inputs):
                            if tuple(type.shape)==input.shape:
                                inputs[name] = input
                                del prenode_inputs[kth]
                                continue
                    return inputs
                inputs = bind_inputs()
                flag = self.mod_fix_and_check(inputs)
                """
            if changeflag:
                print("sure", mod)
                self.mod = self.defuse_mod(relay.parse(mod))

                if self.prim_valid_fix():
                    print(">>>>>>>>>>>>fix succeed")
                    exit()  # fix and exit
                else:
                    if self.fixbatchnums >= 1:
                        print("************controversy****** ")
                        exit()  # controversy and return
                    else:
                        pass
            else:
                print("no change")

        fix_sensi(outputindex)
