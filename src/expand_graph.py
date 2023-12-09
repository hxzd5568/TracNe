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
from .fastfuzz import Fuzzer
from .calculator2 import Calculate_error
from .seed import seed3,seed1,seed2
import re
import sys
sys.path.append('../')
outputnum = 4
Boundlow, Boundhigh = -5, 5
TensorDict = Dict[str, np.ndarray]
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
Input_size = (4,3)
Input_size2 = (3,3)
Disabled_pass = 'SimplifyExpr'
case_id = 1086
Input_size = (3,4,4)

seedlist = [1051,1047,8,9,1050]#1051,1047,8,9,
lastdirnum = 1090
Max_opr_num = 30
Dtype = 'float32'
path = './tests/out'

## basic operations
## !!! update pass
a:IRModule
b:GraphModule
c:GraphModuleDebug
d:relay.function.Function
e:relay.Type
e:relay.Var.name_hint
e:relay.Var.checked_type
f:tvm.ir.tensor_type.TensorType
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

def expand():
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
    global path
    seedpath = './tests/out/'
    seedss = []
    for i in seedlist:
        seedp = seedpath+str(i)+'/code.txt'
        with open(seedp,'r')as fp:
            seedss.append(relay.parse(fp.read()))
    for ten, mod in enumerate(seedss):# seed1,seed2,
        # mod = relay.parse(seed)
        # print( mod.show())
        fn = mod['main']
        pass_params0 = []
        for i in fn.params[:]:
                pass_params0.append(relay.var('pz'+i.name_hint,i.checked_type))
        newop = relay.Call(fn,pass_params0)
        Last_Input = fn.ret_type
        inputshape = [i.value for i in list(Last_Input.shape)]
        print(inputshape)
        # # get shape
        # shapes_str = re.findall(r'\[.*?\]',inputstr)[:]
        # shapes = []
        # for shapestr in shapes_str:
        #     shapes.append( [int(i) for i in shapestr.strip('\'').strip('[').strip(']').split(',')])
        # print(shapes)
        # # get type
        # types_str = re.findall(r'\".*?\"',inputstr)[:]
        # types = []
        # for shapestr in types_str:
        #     types.append( shapestr.strip('\'').strip('"'))
        # print(types)
        # a = relay.TensorType([1, 2], 'float16')
        # b = relay.TensorType([1, 2], 'float16')
        # c = tvm.ir.type.TupleType([a,b])
        # print(c)
        while case_id<=lastdirnum:
            rng = Generator(PCG64(seed=random.randint(0,1e6)))
            gen = GraphGenerator(OpRegistry.ops(), rng)
            if Last_Input.dtype == 'float16':
                graph = gen.imagegenerate(imageshape=inputshape, imagedtype=DataType.f(16),Max_opr_num=Max_opr_num) # 1st:5+expand 10 =15; 2:
            else:
                graph = gen.imagegenerate(imageshape=inputshape, imagedtype=DataType.f(32),Max_opr_num=Max_opr_num)
            if (len(graph.outputs_)>outputnum):
                continue
            code = print_relay(graph)
            #mod2 = parser.parse(code)
            mod2 = relay.parse(code)
            func2 = mod2['main']
            pass_params = []
            for i in func2.params[1:]:
                pass_params.append(relay.var('p'+i.name_hint,i.checked_type))
            params = pass_params
            new_op2 = relay.Call(func2, [newop]+params)
            new_f = relay.Function(pass_params0+params,new_op2)
            mod = tvm.IRModule.from_expr(new_f)
            case_path = os.path.join(path,str(case_id))
            if(not os.path.exists(case_path)):
                os.mkdir(case_path)
            else:
                if os.path.exists(case_path+'/compiled_lib1.tar'):
                    os.remove(case_path+'/compiled_lib1.tar')
            casepath = case_path+'/code.txt'
            seq = tvm.transform.Sequential(
                [
                    relay.transform.InferType(),
                    relay.transform.DefuseOps(),
                    relay.transform.InferType(),
                ]
            )
            with tvm.transform.PassContext(opt_level=4):
                mod2 = seq(mod)
            try:
                if  not tvm.ir.structural_equal(mod, mod2):
                    checkor = Calculate_error(mod = mod2)
                    if(checkor.random_difference_test()>1e-9 ):
                        print('------- succeed generate------')
                        print(casepath)
                        with open(casepath,'w') as fp:
                            fp.write(mod2.astext())
                            case_id+=1
                        break
                    else:
                        continue
                else :
                    print('-------!!! not succeed undefuse------')
                    continue
            except Exception as e:
                continue
