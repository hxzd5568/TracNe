######################################
######  Two buggy optimization passes
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
from tvm.contrib.graph_executor import GraphModule
from argparse import Namespace, ArgumentParser
from typing import Iterable, List, cast, Optional, Dict
import sys
sys.path.append('../')
TensorDict = Dict[str, np.ndarray]
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
import time
Required_pass1 = ['EliminateCommonSubexpr','CombineParallelDense','CombineParallelBatchMatmul','CombineParallelConv2D']

def MSE(y_true, y_pred,):  #precision along with  tf.keras.metrics.MeanRelativeError
        d = np.abs(y_true.astype(np.float64) - y_pred)
        relative_error = np.average( d \
                    / (np.abs(y_true).astype(np.float64) + 1e-8) )
        return relative_error

def SE(y_true, y_pred,):  #precision along with  tf.keras.metrics.MeanRelativeError
        d = np.abs(y_true.astype(np.float64) - y_pred)
        relative_error = np.max( d \
                    / (np.abs(y_true).astype(np.float64) + 1e-8))# * np.abs(y_true) / np.mean(np.abs(y_true))
        return relative_error

def run_gmod( gmod: GraphModule, inputs: Dict[str, np.ndarray]=None) -> List[np.ndarray]:
        if inputs is not None:
            gmod.run(**inputs)
        else:
            gmod.run()
        return [gmod.get_output(i).numpy() for i in range(gmod.get_num_outputs())]

def build_workload(mod, params=None, Disabled_pass=['SimplifyExpr']):
        with transform.PassContext(opt_level=1, required_pass=Required_pass1,disabled_pass=Disabled_pass):
            lib1 = relay.build(mod, target)
        with transform.PassContext(opt_level=5):#disabled_pass=Disabled_pass
            lib5 = relay.build(mod, target)
        return lib1, lib5

def replay(mod,params):
        factorymod1, factorymod5 = build_workload(\
                mod,params= params)
        gmod1 = GraphModule(factorymod1["default"](dev))
        gmod5 = GraphModule(factorymod5["default"](dev))
        outs1 = run_gmod(gmod1,params)
        outs5 = run_gmod(gmod5,params)
        tdiff = 0.
        for (ro,o) in zip(outs1,outs5):
            diff =  MSE(ro,o)
            tdiff = max(tdiff,diff)
        print('mean relative error = ' ,tdiff)
        tdiff2 = 0.
        for (ro,o) in zip(outs1,outs5):
            diff =  SE(ro,o)
            tdiff2 = max(tdiff2,diff)
        print('max relative error = ' ,tdiff2)

def test_mod1():
    def mod1():
        shape = (4,3)
        x = relay.var("x", shape=shape, dtype="float32")
        y = relay.var("y", shape=shape, dtype="float32")
        m = relay.sqrt(relay.abs(y))
        n = relay.divide(x,m)
        l = relay.round(relay.nn.relu(relay.tan(relay.sum(n,axis=[1]))))
        return tvm.IRModule.from_expr(l)
    params = {'x': np.array([[-3.0407448 ,  5.        ,  1.4677091 ],
       [ 5.        , -0.08194685,  3.0596933 ],
       [ 5.        ,  5.        ,  3.7800522 ],
       [ 5.        ,  3.1617928 ,  5.        ]], dtype=np.float32), 'y': np.array([[-0.11967325  , -0.018634353 ,  0.1582024   ],
       [-0.09131396  , -0.0047433637, -0.020964164 ],
       [-0.08089028  , -0.01746996  , -0.008808094 ],
       [ 0.1787599   ,  0.1756186   ,  0.041228298 ]], dtype=np.float32)}
    mod = mod1()
    replay(mod,params)
print('case 1')
test_mod1()

# mean relative error   =  0.14285714265306124
# max relative error =   0.571428570612245


def test_mod2():
    def mod2():
        n = 16
        c1_val = np.ones(shape=n).astype("float32")/1.0
        c2_val = np.ones(shape=n).astype("float32")/100.0
        c3_val = np.ones(shape=n).astype("float32")/10000.0

        x = relay.var("x", shape=(n,), dtype="float32")
        c1 = relay.const(c1_val)
        c2 = relay.const(c2_val)
        c3 = relay.const(c3_val)
        return tvm.IRModule.from_expr(c2 + (c1 + x) + c3,)
    params = {'x': np.array([-1.0100999  , -1.0346043  , -1.9652936  ,  5.         ,
        5.         ,  5.         ,  5.         ,  5.         ,
        0.3813362  ,  5.         , -0.052576065,  5.         ,
        3.8130388  ,  5.         , -5.         ,  5.         ],
      dtype=np.float32)}
    mod = mod2()
    replay(mod,params)
print('case 2')
test_mod2()

def test_mod3():
    def mod3():
        data = relay.var("data", shape=(1, 3, 3, 8), dtype="float32")
        in_bias= relay.var("in_bias", shape=(3, 1 ,1), dtype="float32")
        weight= relay.var("weight", shape=(3, 3, 3, 3), dtype="float32")
        f = relay.const(3.0)
        m = relay.nn.conv2d(data, weight, padding=[1, 1, 1, 1], channels=3, kernel_size=[3, 3])
        n = relay.add(m, in_bias)
        l = relay.nn.relu(n)
        k = relay.multiply(l, f,)
        return tvm.IRModule.from_expr(k)
    mod = mod3()
    params = {'data': np.array([[[[ 4.1163635  ,  3.2228088  ,  0.017242432, -4.548645   ,
          -2.5224304  ,  4.209137   , -4.1366577  ,  0.9109497  ],[ 3.2025146  , -1.8534851  ,  2.7986145  , -0.64559937 ,
           1.1576843  , -4.189911   ,  0.09902954 ,  4.2404175  ],[-4.8649597  ,  4.2799377  , -3.9460754  ,  2.3535156  ,
          -3.7632751  , -4.7947693  ,  2.372284   ,  1.6668701  ]],[[ 4.610138   ,  3.8053894  ,  3.8381958  , -4.450531   ,
          -2.3736572  ,  4.4615173  ,  0.6541443  ,  3.7913513  ],[-4.0896606  , -3.0641174  , -2.592926   , -1.4572144  ,
          -2.4858093  ,  2.6922607  , -0.069732666,  1.1351013  ],[-3.5592651  ,  0.49713135 , -4.1394043  , -2.407074   ,
           3.5751343  , -4.7013855  , -0.10864258 , -4.7891235  ]],[[-2.8282166  , -0.7815552  , -4.8573303  ,  3.753662   ,
          -1.3938904  ,  0.66101074 , -4.884033   , -0.05050659 ],
         [ 2.443695   , -0.6170654  ,  3.6932373  , -2.158661   , -3.4761047  ,  2.8678894  ,  1.6082764  ,  0.010681152],[-4.1098022  ,  2.013092   , -3.005371   , -4.393463   ,
           0.23513794 ,  0.9436035  , -3.8816833  ,  2.2740173  ]]]],
      dtype=np.float32), 'weight': np.array([[[[-0.036727745, -0.08940563 ,  0.05051272 ],
         [-0.0333286  ,  0.21914756 , -0.12153826 ],[-0.07797686 ,  0.009578303,  0.073190555]],
        [[-0.2484066  ,  0.000455132, -0.21513228 ],[-0.018433733, -0.2710633  , -0.08242117 ],
         [-0.13191707 , -0.098520845,  0.22458874 ]],[[ 0.035224877,  0.16323657 , -0.09220455 ],
         [ 0.22315206 , -0.072422385, -0.12208488 ],[ 0.19187246 , -0.07688672 ,  0.003581573]]],
       [[[ 0.027952887,  0.061361663, -0.024426105],[ 0.07162253 , -0.14932604 ,  0.06435288 ],
         [-0.008233514,  0.08305748 , -0.17171563 ]],[[-0.09118227 , -0.014926386,  0.014232294],
         [-0.1564838  , -0.069641225,  0.061708245],
         [ 0.072204806,  0.11291813 ,  0.11106545 ]],[[-0.24633893 ,  0.057572417,  0.08551504 ],
         [-0.12791695 , -0.011270673,  0.13212293 ],[-0.1458071  ,  0.1949932  ,  0.007811096]]],
       [[[-0.10390888 , -0.049856093, -0.08322507 ],[-0.046482403, -0.17208004 ,  0.08034821 ],
         [ 0.0440737  , -0.003398159,  0.09506506 ]],[[-0.08517007 ,  0.010996343,  0.12853315 ],
         [-0.023498693,  0.071802065, -0.042154644],[-0.004092929,  0.12372954 ,  0.0930078  ]],[[-0.055090867, -0.07654511 , -0.03721075 ],
         [ 0.030530095, -0.07391206 ,  0.036769908],[-0.13674302 , -0.056045175, -0.010012659]]]], dtype=np.float32), 'in_bias': np.array([[[-0.10464073]],
       [[ 0.0493139 ]],
       [[-0.10237443]]], dtype=np.float32)}
    replay(mod,params)
test_mod3()
# mean relative error =  0.007166008869682966
# max relative error =  0.2915852779342002