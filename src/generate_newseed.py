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
from tvm.contrib.debugger.debug_executor import GraphModuleDebug
from .base_utils import Checkor
from .calculator2 import Calculate_error
from .seed import seed3,seed1,seed2
import re
import sys
sys.path.append('../')
Boundlow, Boundhigh = -5, 5
TensorDict = Dict[str, np.ndarray]
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
Disabled_pass = 'SimplifyExpr'
case_id = 191
base_id =191
Max_opr_num = 30
outputsize = 3
path = './tests/out'

#############################################################
#################################   error propagation model

# ops :[negative, abs, ceil, floor, round, trunc, exp, sin, cos, tan, sigmoid,
#  tanh, add, subtract, multiply, divide, maximum, minimum, sum, mean, min, max,
# expand_dims, squeeze, reshape, transpose, concatenate, split, strided_slice,
# nn.relu, nn.leaky_relu, nn.prelu, nn.bias_add, nn.softmax, nn.conv1d, nn.conv2d,
#  nn.conv3d, nn.conv1d_transpose, nn.conv2d_transpose, nn.conv3d_transpose,
# nn.max_pool1d, nn.max_pool2d, nn.max_pool3d, nn.avg_pool1d, nn.avg_pool2d,
# nn.avg_pool3d, nn.global_avg_pool2d, nn.adaptive_max_pool1d, nn.adaptive_max_pool2d,
#  nn.adaptive_max_pool3d, nn.adaptive_avg_pool1d, nn.adaptive_avg_pool2d, nn.adaptive_avg_pool3d, nn.upsampling,
# nn.upsampling3d, nn.pad, nn.layer_norm, nn.instance_norm, nn.group_norm, nn.batch_norm, nn.dense, nn.batch_flatten]

import sys
sys.path.append('../')
from numpy.random import Generator, PCG64
from gencog.expr.ty import  DataType, ValueType, BOOL, INT, FLOAT
from gencog.graph import GraphGenerator, print_relay
from gencog.spec import OpRegistry
from tvm.relay.analysis import free_vars, free_type_vars
from gencog.debug import ModuleError
# import seed model      ---> get output size and wrap seedmodel as a call
# expand(without conv2d) ---> randomerror > 2* seedmodel --->   save after 20 cycles

def ge_seed():
    #############################################################
    # #################################   model def
    # x = relay.var("x", shape=Input_size, dtype=Dtype)
    # v = relay.var("v", shape=Input_size, dtype=Dtype)
    # w = relay.abs(v)
    # y = relay.sqrt(w)
    # z = relay.divide(x, y)
    # func = relay.Function([x,v], z)
    # mod = tvm.IRModule()
    # mod['main'] = func
    ##############################################################
    global case_id
    global base_id
    global path

    while case_id<=base_id+1:
        dim = random.randint(2,3)
        Input_size = []
        if dim==2:
            for i in range(dim):
                Input_size.append(random.randint(100,160))
        elif dim == 3:
            Input_size.append(random.randint(40,60))
        rng = Generator(PCG64(seed=random.randint(0,1e6)))
        gen = GraphGenerator(OpRegistry.ops(), rng)
        dtype = 'float32' if random.random() < 0.95 else 'float16'
        tolerance = 5e-6  if dtype == 'float16' else  5e-9
        if dtype=='float16':
            graph = gen.imagegenerate(imageshape=Input_size, imagedtype=DataType.f(16),Max_opr_num=Max_opr_num) # 1st:5+expand 10 =15; 2:
        else:
            graph = gen.imagegenerate(imageshape=Input_size, imagedtype=DataType.f(32),Max_opr_num=Max_opr_num)
        if (len(graph.outputs_)>outputsize):
            continue
        code = print_relay(graph)
        mod = relay.parse(code)
        func = mod['main']

        case_path = os.path.join(path,str(case_id))
        if(not os.path.exists(case_path)):
            os.mkdir(case_path)
        casepath = case_path+'/code.txt'
        seq = tvm.transform.Sequential(
            [
                relay.transform.InferType(),
                relay.transform.DefuseOps(),
                relay.transform.InferType(),
            ]
        )
        # with tvm.transform.PassContext(opt_level=4):
        #     mod2 = seq(mod)
        print('one')
        try:
            checkor = Calculate_error(mod = mod)
            if(checkor.random_difference_test()>tolerance):
                print('------- succeed generate------')
                with open(casepath,'w') as fp:
                    fp.write(mod.astext())
                    case_id+=1
            else:
                continue
        except Exception as e:
            print(e)
            continue
