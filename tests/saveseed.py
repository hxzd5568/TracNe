import tvm
from tvm import relay
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
from tvm.relay.testing import create_workload
from tvm.relay.build_module import bind_params_by_name
def initializer(_, param):
    param = np.zeros(param.shape)

Boundlow, Boundhigh = -50, 50
TensorDict = Dict[str, np.ndarray]
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
Dtype = 'float16'
Input_size = (4,3)
Input_size2 = (3,3)
Disabled_pass = 'SimplifyExpr'
case_id = '7'
## basic operations
## !!! update pass

def build_mod(mod: IRModule, opt_level: int, params: Optional[TensorDict] = None):
    if opt_level <5:
        with transform.PassContext(opt_level=opt_level,disabled_pass=[Disabled_pass]):
            # graph1, lib1, params1 = relay.build(mod, target, params=params)
            lib = relay.build(mod, target='llvm', params=params)
    else:
        with transform.PassContext(opt_level=opt_level):
            # graph5, lib5, params5 = relay.build(mod, target, params=params)
            mod = tvm.relay.transform.InferType()(mod)
            combine_pass = tvm.relay.transform.SimplifyExpr()
            mod = combine_pass(mod)

            lib = relay.build(mod, target='llvm', params=params)
    return GraphModule(lib['default'](cpu()))

def MRE(y_true, y_pred,):
    if np.isinf(y_true).any()==1 or np.isinf(y_pred).any()==1:
        print('y_true, y_pred have inf')
        return 0
    elif np.isnan(y_true).any()==1 or np.isnan(y_pred).any()==1:
        print('y_true, y_pred have nan')

        return 0
    else:
        relative_error = np.average(np.abs(y_true - y_pred) / (y_true + 1e-7))
    relative_error = np.average(np.abs(y_true - y_pred) / (y_true + 1e-7))
    return relative_error

def run_gmod(gmod: GraphModule, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
    gmod.run(**inputs)
    return [gmod.get_output(i).numpy() for i in range(gmod.get_num_outputs())]

def generate_inputs(main_fn:relay.function.Function):
    length = len(main_fn.params)
    inputarr = dict()
    for i in range(length):
        varx = main_fn.params[i]

        var_tyx = varx.checked_type
        size=[int(d) for d in var_tyx.shape]
        if random.random()<0.3:
            inputarr[varx.name_hint] = np.clip(np.random.normal(size=size).astype(var_tyx.dtype),
                                           Boundlow, Boundhigh)
        elif random.random()<0.6:
            inputarr[varx.name_hint] = np.clip(np.random.uniform(size=size).astype(var_tyx.dtype),
                                Boundlow, Boundhigh)
        elif random.random()<0.9:
            inputarr[varx.name_hint] = np.clip(np.random.standard_cauchy(size=size).astype(var_tyx.dtype),
                                Boundlow, Boundhigh)
        else:
            inputarr[varx.name_hint] = np.clip(np.random.standard_cauchy(size=size).astype(var_tyx.dtype)
                                               *np.random.normal(size=size).astype(var_tyx.dtype),
                                Boundlow, Boundhigh)

    return inputarr

def rundiff(main_fn:relay.function.Function, gmod1:GraphModule,gmod5:GraphModule):
    diff = np.zeros(1)
    for i in range(20):
        inputarr = generate_inputs(main_fn)
        outs1 = run_gmod(gmod1,inputarr)
        outs5 = run_gmod(gmod5,inputarr)
        tempdiff = np.zeros(1)
        for i, (o, ro) in enumerate(zip(outs1, outs5)):
            tempdiff =max( MRE(ro,o),tempdiff)
        diff = max(diff, tempdiff)
    return diff

#############################################################
#################################   model def
def _get_positive_scale(size):
    return np.random.uniform(0.5, 1, size=size).astype("float32")


def construct():
    def before(x, w1, w2, b1, b2):
        args = [x, w1, w2, b1, b2]
        y1 = relay.nn.dense(x, w1)
        y2 = relay.nn.dense(x, w2)
        y1 = relay.add(y1, b1)
        y2 = relay.add(y2, b2)
        y = relay.Tuple((y1, y2))
        return relay.Function(args, y)

    def expected(x, w1, w2, b1, b2, is_2d_bias):
        args = [x, w1, w2, b1, b2]
        x_stacked = relay.stack((x, x), axis=0)
        w = relay.stack((w1, w2), axis=0)
        y = relay.nn.batch_matmul(x_stacked, w)

        if not is_2d_bias:
            b1 = relay.expand_dims(b1, 0)
            b2 = relay.expand_dims(b2, 0)

        b = relay.stack((b1, b2), axis=0)
        y = relay.add(y, b)
        (y1, y2) = relay.split(y, 2)
        y1 = relay.squeeze(y1, [0])
        y2 = relay.squeeze(y2, [0])
        y = relay.Tuple((y1, y2))
        return relay.Function(args, y)

    def check(i, j, k, is_2d_bias):
        x = relay.var("x", shape=(i, k))
        w1 = relay.var("w1", shape=(j, k))
        w2 = relay.var("w2", shape=(j, k))

        if is_2d_bias:
            b1 = relay.var("b1", shape=(i, j))
            b2 = relay.var("b2", shape=(i, j))
        else:
            b1 = relay.var("b1", shape=(j,))
            b2 = relay.var("b2", shape=(j,))

        return before(x, w1, w2, b1, b2)
        # y = run_opt_pass(y_before, transform.CombineParallelDense(min_num_branches=2))
        # y_expected = expected(x, w1, w2, b1, b2, is_2d_bias)
        # y_expected = run_opt_pass(y_expected, transform.InferType())
        # tvm.ir.assert_structural_equal(y, y_expected, map_free_vars=True)

    return check(3, 5, 4, False)

     


##############################################################
#func = relay.Function([x,v], z)
# mod = tvm.IRModule.from_expr(origin_exprs )
case_path = os.path.join('./out/', '1019' )
if not os.path.exists(case_path):
    os.mkdir(case_path)
else:
    shutil.rmtree(case_path)
    os.mkdir(case_path)

mod = tvm.IRModule.from_expr( construct())
with open(f'{case_path}/code.txt','w')as fp:
    fp.write(mod.astext())

## !!! pass path
case_path = os.path.join('./out/', case_id)
if not os.path.exists(case_path):
    os.mkdir(case_path)
with open(f'{case_path}/code.txt', 'w') as f:
    f.write(mod.astext())

with open(f'{case_path}/code.txt', 'r') as f:
    mod = relay.parse(f.read())

try:
    gmod1 = build_mod(mod, 0)
except:
    gmod1 = build_mod(mod, 1)
gmod5 = build_mod(mod, 10)
main_fn = mod['main']
prediff = rundiff(main_fn,gmod1,gmod5)
print('rel error: %.10f' %(prediff))

## expose error

'''
n = 32
c1_val = np.random.uniform(size=n).astype(Dtype)
c2_val = np.random.uniform(size=n).astype(Dtype)
c3_val = np.random.uniform(size=n).astype(Dtype)
x = relay.var("x", shape=(n,), dtype=Dtype)
c1 = relay.const(c1_val)
c2 = relay.const(c2_val)
c3 = relay.const(c3_val)
origin_exprs =  (x + c1) * c2 + c3

rel error: 0.0003302097 float16
rel error: 0.0000000427 float32
rel error: 0            float64

-------
'''

'''
def test_simplify_add():
    x = relay.var("x", shape=(1, 3, 100, 100), dtype="float32")

    def before():
        return relay.add(x, x)

    def expected():
        s = relay.const(2.0)
        return relay.multiply(x, s)

    opt = run_opt_pass(before(), transform.SimplifyExpr())
    ref = run_infer_type(expected())
    assert tvm.ir.structural_equal(opt, ref)
'''
'''
def test_callback():
    def before():
        x = relay.var("x", shape=(1, 16))
        y1 = relay.nn.relu(x)
        y2 = relay.nn.relu(x)
        y1 = relay.add(y1, relay.const(1.0, "float32"))
        y2 = relay.add(y2, relay.const(1.0, "float32"))
        y = relay.add(y1, y2)
        f = relay.Function([x], y)
        return f

    def expected():
        x = relay.var("x", shape=(1, 16))
        y = relay.nn.relu(x)
        y1 = relay.add(y, relay.const(1.0, "float32"))
        y2 = relay.add(y, relay.const(1.0, "float32"))
        y = relay.add(y1, y2)
        f = relay.Function([x], y)
        return run_opt_pass(f, transform.InferType())

    def fskip(expr):
        if isinstance(expr, relay.expr.Call) and expr.op.name == "add":
            return True
        return False

    z = before()
    z = run_opt_pass(z, transform.EliminateCommonSubexpr(fskip))
    assert tvm.ir.structural_equal(z, expected())
'''
