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
from typing import Iterable, List, cast, Optional, Dict, Any
import sys
from multiprocessing import Process, Queue
from tvm.relay.transform import InferType, ToMixedPrecision, mixed_precision
import re

TensorDict = Dict[str, np.ndarray]
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
Unopt_level = 1

def run_gmod(gmod: GraphModule, inputs: Dict[str, np.ndarray]=None) -> List[np.ndarray]:
        if inputs is not None:
            gmod.run(**inputs)
        else:
            gmod.run()
        return [gmod.get_output(i).numpy() for i in range(gmod.get_num_outputs())]

# build run difference-test utils
def build_workload(mod ,params=None, Disabled_pass=['SimplifyExpr'],
                   isolatepass=[]):
        with transform.PassContext(opt_level=1, disabled_pass=Disabled_pass):
            lib1 = relay.build(mod, target, params=params)
        with transform.PassContext(opt_level=5,disabled_pass=isolatepass):
            lib5 = relay.build(mod, target, params=params)
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

class Calculate_error:

    def __init__(self,mod,params= None,
                 disabled_pass = ['SimplifyExpr'],lowbound:float=-5, highbound:float=5):
        self.Boundlow, self.Boundhigh = lowbound, highbound
        self.Disabled_pass = disabled_pass
        self.mod = mod
        self.Dtype = re.search('main.+?Tensor\[.+?\),\ (.+?)\]',str(self.mod)).group(1)
        self.eparams = None
        # self.factorymod1, self.factorymod5 = build_workload(\
        #         self.mod, params= params)
        # self.gmod1 = GraphModule(self.factorymod1["default"](dev))
        # self.gmod5 = GraphModule(self.factorymod5["default"](dev))
        # self.tolerance = 1e-3 if self.Dtype=='float16' else 1e-6  # should be checked RE and MRE ratio
        # print('tolerance,type',self.tolerance,self.Dtype)

    def MSE(self,y_true, y_pred,):  #precision along with  tf.keras.metrics.MeanRelativeError
        if np.isinf(y_true).any()==1 or np.isnan(y_true).any()==1:
            print('y_true have inf\\nan:locating...')
            #self.locate_naninf('1')
            return 0

        if np.isinf(y_pred).any()==1 or np.isnan(y_pred).any()==1:
            print('y_pred have inf\\nan:locating...')
            return 0
            #self.locate_naninf('5')
        else:
            pass
        d = np.abs(y_true.astype(np.float64) - y_pred)
        relative_error = np.average( d \
                / (np.abs(y_true).astype(np.float64) + 1e-8) * np.not_equal(y_true, 0)\
                 + np.equal(y_true, 0)* d )
        return relative_error

    def replay_withlocatenan(self, inp_case_path):
        with np.load(inp_case_path) as f:
            loaded_params = dict(f.items())
        self.eparams = loaded_params
        outs1 = run_gmod(self.gmod1,loaded_params)
        outs5 = run_gmod(self.gmod5,loaded_params)
        return self.MSE(outs1[0],outs5[0])
    def replay_withlocatenan_withtar1(self, inp_case_path,tar1):
        with np.load(inp_case_path) as f:
            loaded_params = dict(f.items())
        self.eparams = loaded_params
        loaded_params1 = dict()
        for k,v in loaded_params.items():
            loaded_params1[k]= v.astype('float64')
        outs1 = run_gmod(tar1,loaded_params)
        outs5 = run_gmod(self.gmod5,loaded_params)
        return self.MSE(outs1[0],outs5[0])

    def replay_withlocatenan_withtar5(self, inp_case_path,tar1,tar5):
        with np.load(inp_case_path) as f:
            loaded_params = dict(f.items())
        self.eparams = loaded_params
        loaded_params1 = dict()
        for k,v in loaded_params.items():
            loaded_params1[k]= v.astype('float64')
        outs1 = run_gmod(tar1,loaded_params)
        outs5 = run_gmod(tar5,loaded_params)
        return self.MSE(outs1[0],outs5[0])

    def build_mod(self, mod: IRModule, opt_level: int,
                  isolatepass=[],params=None):
        if opt_level <=1:
            with transform.PassContext(opt_level=opt_level,disabled_pass=[self.Disabled_pass]):
                lib = relay.build(mod, target, params=params)
        else:
            with transform.PassContext(opt_level=opt_level, disabled_pass=isolatepass):
                lib = relay.build(mod, target, params=params)
        return lib
    def build_mod2(self, mod: IRModule, opt_level: int, params: Optional[TensorDict] = None):
        global target
        global dev
        if opt_level <5:
            with transform.PassContext(opt_level=opt_level,disabled_pass=[self.Disabled_pass]):
                graph1, lib1, params1 = relay.build(mod, target, params=params)
                module = tvm.contrib.graph_executor.create(graph1, lib1, dev)
                module.set_input(**params1)
                #lib = relay.build(mod, target='llvm', params=params)
        else:
            with transform.PassContext(opt_level=opt_level):
                graph5, lib5, params5 = relay.build(mod, target, params=params)
                module = tvm.contrib.graph_executor.create(graph5, lib5, dev)
                module.set_input(**params5)
                # mod = tvm.relay.transform.InferType()(mod)
                # combine_pass = tvm.relay.transform.FoldScaleAxis()
                # mod = combine_pass(mod)
                #lib = relay.build(mod, target='llvm', params=params)
        return module

    def generate_inputs_shape(self, main_fn:relay.function.Function,inputs_num:int=1):
        length = inputs_num
        inputs_shape=dict()
        for i in range(length):
            varx = main_fn.params[i]
            var_tyx = varx.checked_type
            size=[int(d) for d in var_tyx.shape]
            inputs_shape[varx.name_hint] = size
        return inputs_shape

    def generate_inputs(self, main_fn:relay.function.Function, arrs:List[np.array]= None):
        if arrs is None:  # random test
            length = len(main_fn.params)
            inputarr = dict()
            for i in range(length):
                varx = main_fn.params[i]
                var_tyx = varx.checked_type
                size=[int(d) for d in var_tyx.shape]
                inputarr[varx.name_hint] = np.clip(
                    np.random.normal(0,1,size=size).astype(var_tyx.dtype), self.Boundlow, self.Boundhigh)

        else:   # {spec inputs} +random test
            speclen = len(arrs)
            length = len(main_fn.params)-speclen
            inputarr = dict()
            for i in range(length):
                varx = main_fn.params[i+speclen]
                var_tyx = varx.checked_type
                size=[int(d) for d in var_tyx.shape]
                inputarr[varx.name_hint] = np.clip(
                    np.random.normal(size=size).astype(var_tyx.dtype)
                    , self.Boundlow, self.Boundhigh)
            for j in range(speclen):
                varx = main_fn.params[j]
                var_tyx = varx.checked_type
                size=[int(d) for d in var_tyx.shape]
                inputarr[varx.name_hint] = np.clip(
                    arrs[j].astype(var_tyx.dtype)
                    , self.Boundlow, self.Boundhigh)# one 2 many arrs[i]
        return  inputarr
    # with fuzz params satisfying normal distribution

    def rundiff_nohalt(self,main_fn:relay.function.Function,
                 arrs:List[np.array]=None):# input:list of np
        outdiff = np.zeros(1)
        for i in range(10):
            inputarr = self.generate_inputs(main_fn, arrs)
            # backup for check
            self.eparams = inputarr
            # run
            outs1 = run_gmod(self.gmod1, inputarr)
            outs5 = run_gmod(self.gmod5, inputarr)
            tempdiff = np.zeros(1)
            for i, (ro, o) in enumerate(zip(outs1, outs5)):
                diff =  self.MSE(ro,o)
                tempdiff = max(diff, tempdiff)
            outdiff = max(diff, tempdiff)
        return outdiff

    def random_difference_test(self):
        main_fn = self.mod['main']
        prediff = self.rundiff_nohalt(main_fn)
        print('rel error: %.12f' %(prediff))
        return prediff

    def test_consistent(self,isolatepass=[],inputs= None):
        # mod = self.mod
        # rmod1 = self.build_mod(mod, 1)
        # rmod5 = self.build_mod(mod, 10, isolatepass=isolatepass)
        self.factorymod1, self.factorymod5 = build_workload(\
                self.mod, params= None,isolatepass=isolatepass)
        self.gmod1 = GraphModule(self.factorymod1["default"](dev))
        self.gmod5 = GraphModule(self.factorymod5["default"](dev))
        main_fn = self.mod['main']
        for i in range(20):
            if inputs is  None:
                inputarr = self.generate_inputs(main_fn)
            else:
                inputarr = inputs
            outs1 = run_gmod(self.gmod1,inputarr)
            outs5 = run_gmod(self.gmod5,inputarr)
            tempdiff = 0
            for i, (ro, o) in enumerate(zip(outs1, outs5)):
                diff =  self.MSE(ro,o)
                tempdiff = max(diff, tempdiff)
            outdiff = max(diff, tempdiff)
        return outdiff
