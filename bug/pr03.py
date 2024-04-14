######################################
######  A buggy name rule
######################################
import tvm
from tvm import relay,runtime
import os
import numpy as np
import queue
import shutil
import os.path
import random
from tvm import transform, relay, parser, cpu, TVMError, IRModule
from argparse import Namespace, ArgumentParser
from typing import Iterable, List, cast, Optional, Dict
import sys
from tvm.contrib.debugger.debug_executor import GraphModuleDebug
from tvm.contrib.graph_executor import GraphModule
sys.path.append('../')
TensorDict = Dict[str, np.ndarray]
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
import time
Required_pass1 = ['EliminateCommonSubexpr','CombineParallelDense','CombineParallelBatchMatmul','CombineParallelConv2D']

def build_workload(mod, params=None, Disabled_pass=['SimplifyExpr']):
        with transform.PassContext(opt_level=1,
                                   config={"relay.FuseOps.max_depth": 1},
                                   required_pass=Required_pass1,disabled_pass=Disabled_pass):
            lib1 = relay.build(mod, target)
        with transform.PassContext(opt_level=5,
                                   config={"relay.FuseOps.max_depth": 1},
                                   ):#disabled_pass=Disabled_pass
            lib5 = relay.build(mod, target)
        return lib1, lib5

def replay_withdebugger(mod, params=None):
        factorymod1, factorymod5 = build_workload(\
                mod,params= params)
        GraphModuleDebug( factorymod1["debug_create"]("default", dev),
                            [dev],factorymod1["get_graph_json"](), dump_root='./pr03'+'/L1/')
        GraphModuleDebug( factorymod5["debug_create"]("default", dev),
                            [dev],factorymod5["get_graph_json"](), dump_root='./pr03'+'/L5/')

def test_mod1():
    def mod1():
        shape = (4,3)
        x = relay.var("x", shape=shape, dtype="float32")
        y = relay.var("y", shape=shape, dtype="float32")
        m = relay.sqrt(relay.abs(y))
        n = relay.divide(x,m)
        s = relay.reshape(n, newshape=(3, 4))
        l = relay.round(relay.nn.relu(relay.tan(relay.sum(s,axis=[1]))))
        g= relay.reshape(l, newshape=(1, 3))
        return tvm.IRModule.from_expr(relay.tan(g))
    mod = mod1()
    # replay(mod,params)
    replay_withdebugger(mod)
print('case 1')
test_mod1()

# The following is a snippet of executor graph.
# ...
# {
#     "op": "tvmgen_default_fused_sum",
#     "name": "tvmgen_default_fused_sum",
#     "attrs": {
#         ...
#     },
#     "inputs": [
#         "reshape_nop"
#     ],
#     ...
# },
# {
#     "op": "tvmgen_default_fused_tan",
#     "name": "tvmgen_default_fused_tan",
#     ...
#     "inputs": [
#         "tvmgen_default_fused_sum"
#     ],
#     ...
# },
# {
#     "op": "tvmgen_default_fused_tan_1",
#     "name": "tvmgen_default_fused_tan_1",
#     ...
#     "inputs": [
#         "reshape_nop"
#     ],
#     ...
# }
# ...

# From it, the nodes named tvmgen_default_fused_tan_1 and tvmgen_default_fused_sum have the same input node.
# However, this graph is wrong. From relay ir, we know tan_1 and sum have different predecessors. 
# This error occurs because the reshape node is named without distinguishing between different reshapes, and instead uses a uniform name. 

# It is required the addition of an ordinal number to the individual reshape in the codegen procedure.
# After modifying graph_executor_codegen.cc, the dumped graph is able to correctly represent the data flow relationships.

# ...
# {
#     "op": "tvmgen_default_fused_sum",
#     "name": "tvmgen_default_fused_sum",
#     "attrs": {
#         ...
#     },
#     "inputs": [
#         "reshape_nop_0"
#     ],
#     ...
# },
# {
#     "op": "tvmgen_default_fused_tan",
#     "name": "tvmgen_default_fused_tan",
#     ...
#     "inputs": [
#         "tvmgen_default_fused_sum"
#     ],
#     ...
# },
# {
#     "op": "tvmgen_default_fused_tan_1",
#     "name": "tvmgen_default_fused_tan_1",
#     ...
#     "inputs": [
#         "reshape_nop_1"
#     ],
# }...