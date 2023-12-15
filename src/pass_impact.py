# pass-related study
from random import sample
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
import json
from multiprocessing import Process, Queue
from tvm.contrib.debugger.debug_executor import GraphModuleDebug
import re
from tvm.relay.backend import Runtime, Executor, graph_executor_codegen
import torch
from .calculator2 import Calculate_error
np.set_printoptions(precision=15)# threshold=sys.maxsize
storepoint = 0.15
TensorDict = Dict[str, np.ndarray]
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
Required_pass1 = ['EliminateCommonSubexpr','CombineParallelDense','CombineParallelBatchMatmul','CombineParallelConv2D']
sys.path.append('../')
def remove_virtarget(ncode):
    rncode = re.sub('({virtual_.*?}:)', ':', ncode,count=0, flags=re.M|re.S)
    rncode = re.sub('(virtual_.*?->)', ') ->', rncode,count=0, flags=re.M|re.S)
    return rncode
def remove_primary(code):
    return  re.sub('(, Primitive\=.*?->)', ') ->', code,count=0, flags=re.M|re.S)

class Passcheckor:
    def __init__(self,path:str,case_id:str,params= None,
                 disabled_pass = ['SimplifyExpr'],lowbound:float=-5, highbound:float=5,
                 fuseopsmax =64,
                 fuzzmode='DEMC2', optlevel =5,
                 required_pass = ['DenseToSparse']):
        self.path = path
        self.case_id = case_id
        self.Disabled_pass = disabled_pass
        self.Required_pass = required_pass
        self.Required_pass1 = Required_pass1
        self.dump_path = os.path.join(self.path,'out')
        self.case_path = os.path.join(self.dump_path,case_id)
        self.OPTLEVEL = optlevel
        self.fuseopsmax = fuseopsmax
        if 'dnn' in self.case_path:
            self.dnn = 1
        else:
            self.dnn = None
        if not os.path.exists(self.dump_path):
            os.mkdir(self.dump_path)
        if not os.path.exists(self.case_path):
            os.mkdir(self.case_path)
        if not os.path.exists(f'{self.case_path}/reduce.txt'):
            print("""Please first store the isolated buggy sub-graph \n
                  to reduce.txt according to error_accumulation.txt""")
            exit()
        with open(f'{self.case_path}/reduce.txt', 'r') as f:
            self.mod = relay.parse(f.read())
        with open('../src/pass/tvm.json','r')as fp:
            self.passes = json.load(fp)
    def isdiabled(self):
        candis = []
        for p in self.passes:
            try:
                optprim_mod = self.get_primfunc(self.OPTLEVEL,disabled=p)
                flag = 1
            except:
                flag= 0
            if not flag:
                print(p,'can not be disabled')
                continue
            candis.append(p)
        return candis
    def isolate(self):
        print(self.mod)
        cal2 = Calculate_error(mod=self.mod)
        def bisect(mod, passes, candidate):
            print('passes',passes,'candidate',candidate)
            if len(passes)==1 and len(candidate)==0 :
                print('isolated pass is', passes[0])
                return
            if len(passes)==0 and len(candidate)==1 :
                print('isolated pass is', candidate[0])
                return
            # optprim_mod = self.get_primfunc(self.OPTLEVEL,disabled=passes)
            # unoptprim_mod = self.get_primfunc(1)
            # print('pass')
            # tvm.ir.structural_equal(optprim_mod, unoptprim_mod)]
            path_params = os.path.join(self.case_path, 'inputs.npz')
            with np.load(path_params) as f:
                loaded_params = dict(f.items())
            error = cal2.test_consistent(isolatepass = passes,inputs= loaded_params)
            print(error)
            if error!=0:
                if len(candidate)==1:
                    bisect(mod, candidate, [])
                else:
                    temp = sample(candidate,int(len(candidate)/2))
                    candidate = list(set(candidate)-set(temp))
                    passes = temp
                    bisect(mod, passes, candidate)
            else:
                if len(passes)==1:
                    print('isolated pass is', passes[0])
                    return
                else:
                    temp = sample(passes,int(len(passes)/2))
                    candidate = list(set(passes)-set(temp))
                    passes = temp
                    bisect(mod, passes, candidate)
        temp = sample(self.passes,int(len(self.passes)/2))
        candidate = list(set(self.passes)-set(temp))
        passes = temp
        bisect(self.mod, passes, candidate)
    def isthepass(self):
        for p in self.passes:
            self.ispass(p)
    def ispass(self,passs):
        cal2 = Calculate_error(mod=self.mod)
        path_params = os.path.join(self.case_path, 'inputs.npz')
        with np.load(path_params) as f:
                loaded_params = dict(f.items())
        error = cal2.test_consistent(isolatepass = [passs],inputs= loaded_params)
        if error ==0:
            print(error,'yes',passs)
        else :
            print(error,'not',passs)

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
    def isolatef(self):
        print(self.mod)
        cal2 = Calculate_error(mod=self.mod)
        def bisect(mod, passes, candidate):
            print('passes',passes,'candidate',candidate)
            if len(passes)==1 and len(candidate)==0 :
                print('isolated pass is', passes[0])
                return
            if len(passes)==0 and len(candidate)==1 :
                print('isolated pass is', candidate[0])
                return
            optprim_mod = self.get_primfunc(self.OPTLEVEL,disabled=passes)
            unoptprim_mod = self.get_primfunc(1)
            print('pass')
            # tvm.ir.structural_equal(optprim_mod, unoptprim_mod)
            if optprim_mod==unoptprim_mod:
                if len(candidate)==1:
                    bisect(mod, candidate, [])
                else:
                    temp = sample(candidate,int(len(candidate)/2))
                    candidate = list(set(candidate)-set(temp))
                    passes = temp
                    bisect(mod, passes, candidate)
            else:
                if len(passes)==1:
                    print('isolated pass is', passes[0])
                    return
                else:
                    temp = sample(passes,int(len(passes)/2))
                    candidate = list(set(passes)-set(temp))
                    passes = temp
                    bisect(mod, passes, candidate)

        temp = sample(self.passes,int(len(self.passes)/2))
        candidate = list(set(self.passes)-set(temp))
        passes = temp
        bisect(self.mod, passes, candidate)
    def build_mod(self, mod: IRModule, opt_level: int, params: Optional[TensorDict] = None):
        if opt_level == 0:
            with transform.PassContext(opt_level=opt_level+1,
                                       required_pass=self.Required_pass1,
                                       disabled_pass=self.Disabled_pass):
                lib = relay.build(mod, target, params=params)
        else:
            with transform.PassContext(opt_level=opt_level,required_pass=self.Required_pass):
                lib = relay.build(mod, target, params=params)
        return lib
    def get_primfunc(self, opt_level,target='llvm', disabled=[]):# i.e. get tvm.ir.module.IRModule
        target, target_host = tvm.target.Target.canon_target_and_host(target)
        if opt_level>=2:
            with tvm.transform.PassContext(opt_level=opt_level,disabled_pass=disabled,
                                           config={"relay.FuseOps.max_depth": self.fuseopsmax}):
                prim_mod, _ = relay.optimize(self.mod, target)
                code = remove_virtarget(prim_mod.astext())
            return code
        else:
            with tvm.transform.PassContext(opt_level=opt_level,
                                           config={"relay.FuseOps.max_depth": self.fuseopsmax},
                                           required_pass=self.Required_pass1,
                                           disabled_pass=self.Disabled_pass):# config={"relay.FuseOps.max_depth": 10}
                prim_mod, _ = relay.optimize(self.mod, target)
                code = remove_virtarget(prim_mod.astext())
            return code