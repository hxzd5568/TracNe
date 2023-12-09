
import numpy as np
import tvm
from tvm import relay,runtime
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
import sys
from multiprocessing import Process, Queue
from tvm.contrib.debugger.debug_executor import GraphModuleDebug
import re
from tvm.relay.backend import Runtime, Executor, graph_executor_codegen
TensorDict = Dict[str, np.ndarray]
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)

def MSE(self,y_true, y_pred,):  #precision along with  tf.keras.metrics.MeanRelativeError
        if np.isinf(y_true).any()==1 or np.isnan(y_true).any()==1:
            print('y_true have inf\\nan:locating...')
            #locate_naninf('1')
            return 0

        if np.isinf(y_pred).any()==1 or np.isnan(y_pred).any()==1:
            print('y_pred have inf\\nan:locating...')
            return 0
            #locate_naninf('5')
        else:
            pass
        d = np.abs(y_true.astype(np.float64) - y_pred)
        relative_error = np.average( d \
                    / (np.abs(y_true).astype(np.float64) + 1e-8) * y_true / np.mean(np.abs(y_true)))
        return relative_error
def SE(self,y_true, y_pred,):  #precision along with  tf.keras.metrics.MeanRelativeError
        if np.isinf(y_true).any()==1 or np.isnan(y_true).any()==1:
            print('y_true have inf\\nan:locating...')
            #locate_naninf('1')
            return 0

        if np.isinf(y_pred).any()==1 or np.isnan(y_pred).any()==1:
            print('y_pred have inf\\nan:locating...')
            return 0
            #locate_naninf('5')
        else:
            pass
        d = np.abs(y_true.astype(np.float64) - y_pred)
        relative_error = np.max( d \
                    / (np.abs(y_true).astype(np.float64) + 1e-8) * y_true / np.mean(np.abs(y_true)))
        return relative_error
def build_workload(mod1,mod5,params=None, Disabled_pass=['SimplifyExpr']):
        with transform.PassContext(opt_level=1, disabled_pass=Disabled_pass):
            lib1 = relay.build(mod1, target, params=params)
        with transform.PassContext(opt_level=5):
            lib5 = relay.build(mod5, target, params=params)
        return lib1, lib5
def save_model( gmod1,gmod5,case_path):
    gmod1.export_library(case_path+"/compiled_lib1.tar")
    gmod5.export_library(case_path+"/compiled_lib5.tar")
def save_arr( params,case_path):
                    print('type',type(list(params.values())[0]))
                    inputarr = dict()
                    for k, v in params.items():
                        inputarr[k]=v.numpy()
                    path_params = os.path.join(case_path, 'oinputs.npz')
                    np.savez(path_params, **inputarr)
def Dumprelay(mod1,mod5,case_path, params=None):
        factorymod1, factorymod5 = build_workload(\
                mod1,mod5,params= params)
        save_model(factorymod1,factorymod5, case_path)
        if params is not None:
            path_params = os.path.join(case_path, 'oinputs.npz')
            np.savez(path_params, **params)
