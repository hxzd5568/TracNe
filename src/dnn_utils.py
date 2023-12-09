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

Disabled_pass = ['SimplifyExpr']
TensorDict = Dict[str, np.ndarray]
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
def normalname(mod):  # return new_mod and if changed flag
    changeflag = []
    mod = mod.astext()
    pat = '(?P<value>%[a-zA-Z_]+[.a-zA-Z_0-9]+)'
    def update_internal(matched):
        changeflag.append(1)
        return matched.group('value').replace('_','').replace('.','')
    mod = re.sub(pat, update_internal, mod,count=0, flags=re.M|re.S)
    pat2 = '(?P<value>%p)'
    def changep(matched):
        changeflag.append(1)
        return matched.group('value').replace('p','n')
    mod = re.sub(pat2, changep, mod,count=0, flags=re.M|re.S)
    return relay.parse(mod),len(changeflag)!=0

# build run difference-test utils
def build_workload(path,params=None):
        with open(path, 'r') as f:
            mod = relay.parse(f.read())
        with transform.PassContext(opt_level=1, disabled_pass=Disabled_pass):
            lib1 = relay.build(mod, target, params=params)
        with transform.PassContext(opt_level=5):
            lib5 = relay.build(mod, target, params=params)
        return lib1, lib5
def save_model( gmod1,gmod5,case_path):
    gmod1.export_library(case_path+"/compiled_lib1.tar")
    gmod5.export_library(case_path+"/compiled_lib5.tar")
class Checkor:
    def __init__(self,path:str,case_id:str,mod, params= None,
                 disabled_pass = ['SimplifyExpr'],lowbound:float=0, highbound:float=1):
        self.Boundlow, self.Boundhigh = lowbound, highbound

        self.path = path
        self.case_id = case_id
        self.Disabled_pass = disabled_pass
        self.dump_path = os.path.join(self.path,'out')
        self.case_path = os.path.join(self.dump_path,case_id)
        if not os.path.exists(self.dump_path):
            os.mkdir(self.dump_path)
        if not os.path.exists(self.case_path):
            os.mkdir(self.case_path)
        self.convflag = 0
        self.bigflag = 0
        self.mod = mod
        if ('conv' in str(self.mod)):
            self.convflag = 1
        if (int(re.findall('(%\d+)',str(self.mod))[-1].strip('%'))>70):
            self.bigflag = 1
        self.Dtype = re.search('Tensor\[.+?\),\ (.+?)\]',str(self.mod)).group(1)
        self.eparams = None
        if os.path.exists(self.case_path+"/compiled_lib1.tar") and \
            os.path.exists(self.case_path+"/compiled_lib5.tar"):
            self.factorymod1 = tvm.runtime.load_module(self.case_path+"/compiled_lib1.tar")
            self.factorymod5 = tvm.runtime.load_module(self.case_path+"/compiled_lib5.tar")
        else:
            self.factorymod1, self.factorymod5 = build_workload(\
                f'{self.case_path}/code.txt',params= params)
            save_model(self.factorymod1,self.factorymod5, self.case_path)
        self.gmod1 = GraphModule(self.factorymod1["default"](dev))
        self.gmod5 = GraphModule(self.factorymod5["default"](dev))
        self.tolerance = 1e-2 if self.Dtype=='float16' else 1e-5
        print('tolerance,type',self.tolerance,self.Dtype)
        # f: normalize_model_name
        # wflag = False
        # self.mod ,wflag= normalname(self.mod)
        # if wflag:
        #     with open(f'{self.case_path}/code.txt','w') as fp:
        #         fp.write(self.mod.astext())


    ## basic operations
    def run_gmod(self, gmod: GraphModule, inputs: Dict[str, np.ndarray]=None) -> List[np.ndarray]:
        if inputs is not None:
            gmod.run(**inputs)
        else:
            gmod.run()
        return [gmod.get_output(i).numpy() for i in range(gmod.get_num_outputs())]

    def locate_naninf(self,modstr:str):
        dump_root= self.case_path+'/NAN_error/'
        print('enter',dump_root)
        if modstr=='1':
            print(type(self.factorymod1))

            graph_mod = GraphModuleDebug( self.factorymod1["debug_create"]("default", dev),
                                [dev],self.factorymod1["get_graph_json"](), dump_root=dump_root)
        else:
            print(type(self.factorymod5))
            graph_mod = GraphModuleDebug( self.factorymod5["debug_create"]("default", dev),
                                [dev],self.factorymod5["get_graph_json"](), dump_root=dump_root)
        self.run_gmod(graph_mod,self.eparams)
        # binary find  using a list [nodeindex: key]
        def reorderfunc_withindex(funcname_alls) -> List[str]:
            def compare(item1, item2):
                return (fitness(item1) < fitness(item2))
            def fitness(item):
                ten = int(item.split('____')[1].split(':')[1])
                one = int(item.split('____')[2].split(':')[1])
                return ten*10+one
            return sorted(funcname_alls,key=lambda item: fitness(item))
        params_path = os.path.join(dump_root,
                                '_tvmdbg_device_CPU_0/output_tensors.params')
        params: Dict[str, np.array] = relay.load_param_dict(bytearray(open(
                    params_path, "rb").read()))
        keys  = reorderfunc_withindex(list(params.keys()))
        lens = len(keys)
        # locate last nonnan
        def isnan(y_true):
            if np.isinf(y_true).any()==1 or np.isnan(y_true).any()==1:
                return True
            else:
                return False

        def binary_search(l,r):
            if(l>=r):
                return l
            m = int((l+r)/2)
            if((not isnan(params[keys[m]].numpy())) and isnan(params[keys[m+1]].numpy())):
                return m
            if(isnan(params[keys[m]].numpy()) ):
                binary_search(l,m-1)
            else:
                binary_search(m+1, r)
        lastindex = binary_search(0,int(lens-1))
        fixportpath = os.path.join(dump_root, 'Locate_Report_'+modstr)
        with open(fixportpath,'a') as fp:
                fp.write('Located')
                fp.write('\n\n\n'+'*'*50+'\n')
                fp.write('The first nan/inf incurs in pattern:'+str(keys[lastindex+1]))
                fp.write('\n\n\n'+'*'*50+'\n')
                fp.write('The input arr is:\n\n')
                fp.write(str(params[keys[lastindex]]))
                fp.write('\n\n\n'+'*'*50+'\n')
                fp.write('The output arr is:\n\n')
                fp.write(str(params[keys[lastindex+1]]))


    def MSE(self,y_true, y_pred,):  #precision along with  tf.keras.metrics.MeanRelativeError
        if np.isinf(y_true).any()==1 or np.isnan(y_true).any()==1:
            print('y_true have inf\\nan:locating...')
            #self.locate_naninf('1')
            # return 0

        if np.isinf(y_pred).any()==1 or np.isnan(y_pred).any()==1:
            print('y_pred have inf\\nan:locating...')
            # return 0
            #self.locate_naninf('5')
        else:
            pass
        relative_error = np.average(np.abs(y_true - y_pred).astype(np.float64)
                                    / (np.abs(y_true).astype(np.float64) + 1e-8))
        return relative_error

    def MRE(self, y_true, y_pred,): # signal exposing numerical bugs
        flag = 0
        aexception = np.isinf(y_true).any()==1 or np.isnan(y_true).any()==1
        bexception = np.isinf(y_pred).any()==1 or np.isnan(y_pred).any()==1
        if (aexception and not bexception) :
            # flag = 3
            # print('y_true have inf\\nan:locating...')
            # self.locate_naninf('1')
            return 0,0
        elif not aexception and bexception:
            # print('y_pred have inf\\nan:locating...')
            # flag = 3
            # self.locate_naninf('5')
            return 0,0
        elif aexception and bexception:
            # flag = 10
            # print('y_true and y_pred have inf\\nan:locating...')
            # self.locate_naninf('1')
            # self.locate_naninf('5')
            return 0,0
        else:
            pass
        # relative_error = np.average(np.abs(y_true - y_pred).astype(np.float64)\
        #                             / (np.abs(y_true).astype(np.float64) + 1e-8))
        # size = np.size(y_true)
        # zeronum = size- np.count_nonzero(y_true)
        # relative_error = relative_error / (1e8)*(zeronum) * size
        relative_error = np.average( np.abs(y_true - y_pred).astype(np.float64)\
                                    / (np.abs(y_true).astype(np.float64) + 1e-8) * np.not_equal(y_true, 0) )
        if relative_error > 1.0: #!!!
            print("relative error is:", relative_error)# y_true,y_pred
            flag = 1
        return relative_error,flag

    def replay(self):
        path_params = os.path.join(self.case_path, 'inputs.npz')
        with np.load(path_params) as f:
            loaded_params = dict(f.items())
        opt_level =0
        mod = self.mod
        try:
            with transform.PassContext(opt_level=opt_level+self.bigflag,disabled_pass=self.Disabled_pass):
                mod1 = relay.build(mod, target, #params=params
                                    )
        except:
            with transform.PassContext(opt_level=opt_level+1,disabled_pass=self.Disabled_pass):
                mod1 = relay.build(mod, target, #params=params
                                    )
        # level 5
        opt_level =5
        with transform.PassContext(opt_level=opt_level):
            mod5 = relay.build(mod, target, #params=params
                            )
        # self.factorymod1 = mod1
        # self.factorymod5 = mod5
        self.eparams = loaded_params
        outs1 = self.run_gmod(GraphModule(mod1["default"](dev)),loaded_params)
        outs5 = self.run_gmod(GraphModule(mod5["default"](dev)),loaded_params)
        print('expected, actual, self.MSE = ', self.MSE(outs1[0],outs5[0]))

    def replay_withlocatenan(self):
        path_params = os.path.join(self.case_path, 'inputs.npz')
        with np.load(path_params) as f:
            loaded_params = dict(f.items())
        self.eparams = loaded_params
        outs1 = self.run_gmod(self.gmod1,loaded_params)
        outs5 = self.run_gmod(self.gmod5,loaded_params)
        print('expected, actual, self.MSE = ', self.MSE(outs1[0],outs5[0]))

    def replay_withdebugger(self,report:Queue=None):
        path_params = os.path.join(self.case_path, 'inputs.npz')
        graph_mod1 = GraphModuleDebug( self.factorymod1["debug_create"]("default", dev),
                            [dev],self.factorymod1["get_graph_json"](), dump_root=self.case_path+'/L1/')
        graph_mod5 = GraphModuleDebug( self.factorymod5["debug_create"]("default", dev),
                            [dev],self.factorymod5["get_graph_json"](), dump_root=self.case_path+'/L5/')
        with np.load(path_params) as f:
            loaded_params = dict(f.items())
        self.eparams = loaded_params
        outs1 = self.run_gmod(graph_mod1,loaded_params)
        outs5 = self.run_gmod(graph_mod5,loaded_params)
        tdiff = 0.
        for (ro,o) in zip(outs1,outs5):
            diff =  self.MSE(ro,o)
            tdiff = max(tdiff,diff)
        print('expected, actual, self.MSE = ' ,tdiff)
        if report is not None:
            report.put('done_debugger')

    def replay_withnewgmod(self, mod: IRModule=None):
        path_params = os.path.join(self.case_path, 'inputs.npz')
        with np.load(path_params) as f:
            loaded_params = dict(f.items())
        # level 1
        opt_level =0
        #params = self.generate_inputs(mod['main'])
        if mod is None:
            mod = self.mod
        try:
            with transform.PassContext(opt_level=opt_level+self.bigflag,disabled_pass=self.Disabled_pass):
                mod1 = relay.build(mod, target, #params=params
                                    )
        except:
            with transform.PassContext(opt_level=opt_level+1,disabled_pass=self.Disabled_pass):
                mod1 = relay.build(mod, target, #params=params
                                    )
        # level 5
        opt_level =5
        with transform.PassContext(opt_level=opt_level):
            mod5 = relay.build(mod, target, #params=params
                            )
        graph_mod1 = GraphModule(mod1["default"](dev))
        graph_mod5 = GraphModule(mod5["default"](dev))
        outs1 = self.run_gmod(graph_mod1,loaded_params)
        outs5 = self.run_gmod(graph_mod5,loaded_params)
        tdiff = 0.
        for (ro,o) in zip(outs1,outs5):
            diff =  self.MSE(ro,o)
            tdiff = max(tdiff,diff)
        print('expected, actual, self.MSE = ' ,tdiff)

    def build_mod(self, mod: IRModule, opt_level: int, params: Optional[TensorDict] = None):
        if opt_level <5:
            with transform.PassContext(opt_level=opt_level+self.bigflag,disabled_pass=self.Disabled_pass):
                lib = relay.build(mod, target, params=params)
        else:
            with transform.PassContext(opt_level=opt_level):
                lib = relay.build(mod, target, params=params)
        return lib
    def generate_inputs_nameshape(self, main_fn:relay.function.Function,inputs_num:int=1):
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
                    np.random.normal(size=size).astype(var_tyx.dtype), self.Boundlow, self.Boundhigh)
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
    def rundiff(self,main_fn:relay.function.Function,
                 arrs:List[np.array]=None):# input:list of np
        outdiff = 0.
        for i in range(10):
            inputarr = self.generate_inputs(main_fn, arrs)
            self.eparams = inputarr
            outs1 = self.run_gmod(self.gmod1, inputarr)
            outs5 = self.run_gmod(self.gmod5, inputarr)
            tempdiff = 0.
            for i, (ro, o) in enumerate(zip(outs1, outs5)):
                diff , flag =  self.MRE(ro,o)
                if flag==1 :
                    # dump_arrs
                    path_params = os.path.join(self.case_path, 'inputs.npz')
                    np.savez(path_params, **inputarr)
                    print('find enormous error:', diff)
                    print(self.case_path)
                    exit()
                tempdiff = max(diff, tempdiff)
            outdiff = max(diff, tempdiff)
        return outdiff
    def rundiff_nohalt(self,main_fn:relay.function.Function,
                 arrs:List[np.array]=None):# input:list of np
        outdiff = np.zeros(1)
        for i in range(10):
            inputarr = self.generate_inputs(main_fn, arrs)
            # backup for check
            self.eparams = inputarr
            # run
            outs1 = self.run_gmod(self.gmod1, inputarr)
            outs5 = self.run_gmod(self.gmod5, inputarr)
            tempdiff = np.zeros(1)
            for i, (ro, o) in enumerate(zip(outs1, outs5)):
                diff =  self.MSE(ro,o)
                tempdiff = max(diff, tempdiff)
            outdiff = max(diff, tempdiff)
        return outdiff

    def random_difference_test(self):
        main_fn = self.mod['main']
        prediff = self.rundiff_nohalt(main_fn)
        print('rel error: %.10f' %(prediff))
        return prediff
