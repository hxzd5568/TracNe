import tvm
from tvm import relay,runtime
import numpy as np
import queue
import shutil
import os.path
import random
from tvm.contrib.debugger.debug_executor import GraphModuleDebug
from enum import IntEnum, auto
from tvm import transform, relay, parser, cpu, TVMError, IRModule
from tvm.contrib.graph_executor import GraphModule
from argparse import Namespace, ArgumentParser
from typing import Iterable, List, cast, Optional, Dict
from multiprocessing import Process, Queue
from scipy.optimize import basinhopping, differential_evolution, Bounds, dual_annealing
from threading import Thread
import re
import os
import platform
import sys
from packaging import version as package_version
import pytest
import numpy as np
import tvm.testing
from tvm.contrib import graph_executor
from tvm.contrib.nvcc import have_fp16
from tvm.contrib import cudnn, utils
# from relay.utils.tag_span import _create_span, _set_span, _verify_structural_equal_with_span
import torch
from torch.nn import Module
from torch.nn import functional as F
import torchvision

case_path = ''
TensorDict = Dict[str, np.ndarray]
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
low, high = -5, 5
sys.setrecursionlimit(10000)


torch.set_printoptions(precision=9)
np.set_printoptions(precision=9)

def verify_model_with_input(
    test_func,
    input_data,
    *,
    path=None,
    input_dict=None,
    custom_convert_map=None,
    rtol=1e-5,
    atol=1e-5,
    assert_shape_only=False,
    validate_structural_equal=True,

):
    """Generic function to generate and compare Pytorch and TVM output"""
    input_dict = input_dict or {}
    custom_convert_map = custom_convert_map or {}

    trace = torch.jit.trace(test_func, [input.clone() for input in input_data])
    input_names = [f"input{idx}" for idx, _ in enumerate(input_data)]
    input_shapes = list(zip(input_names, [inp.shape for inp in input_data]))
    with tvm.testing.disable_span_filling():
        mod, params = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
    if validate_structural_equal:
        with tvm.testing.enable_span_filling():
            mod_with_span, _ = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
        assert tvm.ir.structural_equal(mod, mod_with_span, map_free_vars=True)

    # with tvm.transform.PassContext(opt_level=3):
    #         target ="llvm"
    #         dev = tvm.device(target, 0)
    #         lib = relay.build(mod, target=target, params=params)
    #         relay_model = graph_executor.GraphModule(lib["default"](dev))
    # # for i in range(25):
    #         input_data = np.float32([[ 0.26805357,  1.00831976 ,-1.00951968]])
    #         inputs= {"input0": input_data}
    #         relay_model.run(**inputs)
    #         compiled_output = relay_model.get_output(0).numpy()
    #         baseline_outputs = test_func(torch.from_numpy(input_data).float()).float()
    #         if assert_shape_only is False:
    #             tvm.testing.assert_allclose(baseline_outputs, compiled_output, rtol=1e-43, atol=1e-43)
    #             try:
    #                 assert( np.equal(baseline_outputs, compiled_output).all() == True)
    #             except:
    #                 print(baseline_outputs, compiled_output,input_data)
    # to c
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, tvm.target.Target('c'), params=params)
    if path is not None:
        lib.export_library('targetcode/'+f"{path}compiled_model.tar")
    else:
        lib.export_library('targetcode/'+"compiled_model.tar")



def savetorch(model):
    global case_path
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(case_path+'/model_scripted.pt')



def test_dnn():
    # 1. get input_shape range
    # 2. repare tmod, mod1, mod5
    # 3. fuzz
    import time
    t0 =time.time()

    torch.set_grad_enabled(False)

    class Sqrt1(Module):
        def forward(self, *args):
            return torch.sqrt(args[0])

    class RSqrt1(Module):
        def forward(self, *args):
            return torch.rsqrt(args[0])

    class Ceil1(Module):
        def forward(self, *args):
            return torch.ceil(args[0])

    class Floor1(Module):
        def forward(self, *args):
            return torch.floor(args[0])

    class Round1(Module):
        def forward(self, *args):
            return torch.round(args[0])

    class Cos1(Module):
        def forward(self, *args):
            return torch.cos(args[0])

    class Sin1(Module):
        def forward(self, *args):
            return torch.sin(args[0])

    class Tan1(Module):
        def forward(self, *args):
            return torch.tan(args[0])

    class Tanh1(Module):
        def forward(self, *args):
            return torch.tanh(args[0])

    class Acos1(Module):
        def forward(self, *args):
            return torch.acos(args[0])

    class Asin1(Module):
        def forward(self, *args):
            return torch.asin(args[0])

    class Atan1(Module):
        def forward(self, *args):
            return torch.atan(args[0])

    class Log1(Module):
        def forward(self, *args):
            return torch.log(args[0])

    class Exp1(Module):
        def forward(self, *args):
            return torch.exp(args[0])

    class Erf1(Module):
        def forward(self, *args):
            return torch.erf(args[0])

    class Trunc1(Module):
        def forward(self, *args):
            return torch.trunc(args[0])

    class Sign1(Module):
        def forward(self, *args):
            return torch.sign(args[0])

    class Neg1(Module):
        def forward(self, *args):
            return torch.neg(args[0])

    class Sinh1(Module):
        def forward(self, *args):
            return torch.sinh(args[0])

    class Cosh1(Module):
        def forward(self, *args):
            return torch.cosh(args[0])

    class Log2_1(Module):
        def forward(self, *args):
            return torch.log2(args[0])

    class Log10_1(Module):
        def forward(self, *args):
            return torch.log10(args[0])

    class Log1p_1(Module):
        def forward(self, *args):
            return torch.log1p(args[0])

    class Square(Module):
        def forward(self, *args):
            return torch.square(args[0])
    input_data = torch.tensor(1+9e-20)


    input_shape = [1,3]
    input_data = [torch.rand(input_shape).float()]
    m = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), dilation=(3, 1),padding_mode='reflect')#replicate,reflect

    # input = torch.randn(1)
    # output = m(input)
    # verify_model_with_input(m.float().eval(), input_data=[input])
    input_data = torch.randn(1)
    path = 1
    verify_model_with_input(Square().float().eval(), input_data=input_data,path=str(path))
    path += 1
    verify_model_with_input(Sqrt1().float().eval(), input_data=input_data,path=str(path))
    path += 1

    verify_model_with_input(Cos1().float().eval(), input_data=input_data,path=str(path))
    path += 1

    verify_model_with_input(Cosh1().float().eval(), input_data=input_data,path=str(path))
    path += 1

    verify_model_with_input(Sin1().float().eval(), input_data=input_data,path=str(path))
    path += 1

    verify_model_with_input(Sinh1().float().eval(), input_data=input_data,path=str(path))
    path += 1

    verify_model_with_input(Log2_1().float().eval(), input_data=input_data,path=str(path))
    path += 1

    verify_model_with_input(Log10_1().float().eval(), input_data=input_data,path=str(path))
    path += 1

    verify_model_with_input(Log1p_1().float().eval(), input_data=input_data,path=str(path))
    path += 1

    verify_model_with_input(Exp1().float().eval(), input_data=input_data,path=str(path))

    path += 1
    verify_model_with_input(Erf1().float().eval(), input_data=input_data,path=str(path))


test_dnn()
"""

'''
inline float log2f(float __x) { return __ocml_log2_f32(__x); }

for (int32_t i0 = 0; i0 < m; ++i0) {
((float*)B)[(i0 * stride_1)] = log2f(((float*)A)[(i0 * stride)]);
}
'''


Mismatched elements: 6 / 900 (0.667%)
Max/mean absolute difference: 2.9802322e-08,5.5361955e-11
Max/mean relative difference: 9.4958914e-08,5.121599e-10
 x: array([[[-2.09939432e+00, -1.20508111e+00, -7.00299680e-01],
        [-1.20246673e+00, -3.07873225e+00, -3.32036972e-01],
        [-3.51132822e+00, -1.33616483e+00, -1.23283780e+00],...
 y: array([[[-2.09939432e+00, -1.20508111e+00, -7.00299680e-01],
        [-1.20246673e+00, -3.07873225e+00, -3.32036972e-01],
        [-3.51132822e+00, -1.33616483e+00, -1.23283780e+00],...
Traceback (most recent call last):
"""
