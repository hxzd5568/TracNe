import tvm
from tvm import relay, runtime
import os
import numpy as np
import queue
import shutil
import os.path
import random
from tvm import transform, relay, parser, cpu, TVMError, IRModule
from tvm.contrib.graph_executor import GraphModule
from argparse import Namespace, ArgumentParser
from typing import Iterable, List, cast, Optional, Dict

TensorDict = Dict[str, np.ndarray]
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
import time

Required_pass1 = [
    "EliminateCommonSubexpr",
    "CombineParallelDense",
    "CombineParallelBatchMatmul",
    "CombineParallelConv2D",
]
Disabled_pass5 = [
    "FastMath",
]  # [ 'AlterOpLayout', 'CanonicalizeCast']


def MRE(
    y_true,
    y_pred,
):  # precision along with  tf.keras.metrics.MeanRelativeError
    print("y_true", y_true)
    print("y_pred", y_pred)
    try:
        np.testing.assert_allclose(y_true, y_pred, rtol=1e-11, atol=1e-11)
    except Exception as e:
        print(e)
    d = np.abs(y_true.astype(np.float64) - y_pred)

    relative_error = np.average(
        d
        / (np.abs(y_true).astype(np.float64) + 1e-9)
        * np.abs(y_true)
        / np.mean(np.abs(y_true))
    )
    # print(y_true,y_pred,relative_error)
    return relative_error


def RE(
    y_true,
    y_pred,
):  # precision along with  tf.keras.metrics.MeanRelativeError
    d = np.abs(y_true.astype(np.float64) - y_pred)
    relative_error = np.max(
        d / (np.abs(y_true).astype(np.float64) + 1e-8)
    )  # * np.abs(y_true) / np.mean(np.abs(y_true))
    return relative_error


def run_gmod(
    gmod: GraphModule, inputs: Dict[str, np.ndarray] = None
) -> List[np.ndarray]:
    if inputs is not None:
        gmod.run(**inputs)
    else:
        gmod.run()
    return [gmod.get_output(i).numpy() for i in range(gmod.get_num_outputs())]


def build_workload(mod, params=None, Disabled_pass=["SimplifyExpr"]):
    with transform.PassContext(
        opt_level=1, required_pass=Required_pass1, disabled_pass=Disabled_pass
    ):
        lib1 = relay.build(mod, target, params)
        lib1 = GraphModule(lib1["default"](dev))
    with transform.PassContext(
        opt_level=5, disabled_pass=Disabled_pass5
    ):  # disabled_pass=Disabled_pass
        lib5 = relay.build(mod, target, params)
        lib5 = GraphModule(lib5["default"](dev))
    return lib1, lib5


def run_diff(lib1, lib5, inputarr):
    outs1 = run_gmod(lib1, inputarr)
    outs5 = run_gmod(lib5, inputarr)
    tempdiff = 0
    for i, (ro, o) in enumerate(zip(outs1, outs5)):
        diff = MRE(ro, o)
        tempdiff = max(diff, tempdiff)
        # print(ro,o,'error',tempdiff)
    return tempdiff


def isolate(mod, i):
    a = relay.analysis.extract_intermdeiate_expr(mod, i)  # 		# API 2
    return a


class Pliner:
    def __init__(self, mod, filename, inputarr, logging=""):
        self.mod = mod
        self.logging = logging
        self.filename = filename
        self.inputarr = inputarr

    def unacceptable(self, j):
        mod2 = isolate(self.mod, j)
        lib1, lib5 = build_workload(mod2, params=self.inputarr)
        error = run_diff(lib1, lib5, self.inputarr)
        if error > 1e-9:
            print("unacceptable Yes")
            print(error)
            return True
        else:
            print("unacceptable No")
            return False

    def pliner(self, i, j):
        print(i, j)
        if abs(j - i) <= 2:
            self.logging += f"final isolation: error between {i} and {j}.\n"
            print(self.logging)
            with open(self.filename, "w") as fp:
                fp.write(self.logging)
            return
        m = int((j+i)/2) #,668, 790,685
        # int((j+i)/3*2)
        print("now detect at: ", m)
        mod2 = isolate(self.mod, m)
        lib1, lib5 = build_workload(mod2, params=self.inputarr)
        error = run_diff(lib1, lib5, self.inputarr)
        print("error", error)
        exit(0)
        if error > 1e-9:
            self.logging += (
                f"error between {i} and {m}. The error after {m} is {error}\n"
            )
            self.pliner(i, m)
        else:
            self.logging += (
                f"error between {m} and {j}. The error after {m} is {error}\n"
            )
            self.pliner(m, j)
