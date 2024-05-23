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
import torch
np.set_printoptions(precision=15)# threshold=sys.maxsize
storepoint = 0.15
TensorDict = Dict[str, np.ndarray]
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
Required_pass1 = ['EliminateCommonSubexpr','CombineParallelDense','CombineParallelBatchMatmul','CombineParallelConv2D']
Disabled_pass5 =['SimplifyExpr'] #['AlterOpLayout','ForwardFoldScaleAxis']#[ 'AlterOpLayout', 'CanonicalizeCast']#['AlterOpLayout','ForwardFoldScaleAxis']#[ 'AlterOpLayout', 'CanonicalizeCast']#['CanonicalizeOps','BackwardFoldScaleAxis', 'FoldConstant', 'FastMath', 'ForwardFoldScaleAxis', 'SimplifyExpr']
sys.path.append('../')
def run_tmod(
    model,
    input_data=None,
):
    """Assert that the output of a compiled model matches with that of its
    baseline."""
    global case_path
    with torch.no_grad():
        if isinstance(input_data,List):
            baseline_input = input_data
        else:
            baseline_input = [input_data]
        baseline_outputs = model(*[input.clone() for input in baseline_input])

    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.cpu().numpy(),)
    return baseline_outputs


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
def build_workload(path,path2, params=None, Disabled_pass=[],OPTLEVEL=5,fuseopsmax=64):#'SimplifyExpr'
        print('come here')
        with open(path, 'r') as f:
            mod = relay.parse(f.read())
        with transform.PassContext(opt_level=1, required_pass=Required_pass1,
                                   config={"relay.FuseOps.max_depth": fuseopsmax},
                                   disabled_pass=['SimplifyExpr']):
            lib1 = relay.build(mod, target, params=params)
        with transform.PassContext(opt_level=5,config={"relay.FuseOps.max_depth":fuseopsmax},
                                disabled_pass=Disabled_pass5, ):
            lib5 = relay.build(mod, target, params=params)
        lib1.export_library(path2+"/compiled_lib1.tar")
        lib5.export_library(path2+"/compiled_lib5.tar")
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
class Checkor:
    def __init__(self,path:str,case_id:str,params= None,
                 disabled_pass = ['SimplifyExpr'],lowbound:float=-5, highbound:float=5,
                 fuseopsmax =64,
                 fuzzmode='MEGA', optlevel =5,
                 required_pass = ['DenseToSparse']):
        self.fuzzmode = fuzzmode
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
        if self.dnn:
            self.Boundlow, self.Boundhigh = 0, 1
        else:
            self.Boundlow, self.Boundhigh = lowbound, highbound
        if not os.path.exists(self.dump_path):
            os.mkdir(self.dump_path)
        if not os.path.exists(self.case_path):
            os.mkdir(self.case_path)
        with open(f'{self.case_path}/code.txt', 'r') as f:
            self.mod = relay.parse(f.read())
            if ('conv' in str(self.mod)):
                self.convflag = 1
            if len(re.findall('(%\d+)',str(self.mod)))>0:
                opsn = int(re.findall('(%\d+)',str(self.mod))[-1].strip('%'))
            else:
                opsn = 1
            if (opsn > 100):
                self.bigflag = 1


        self.Dtype = re.search('main.+?Tensor\[.+?\),\ (.+?)\]',str(self.mod)).group(1)
        self.eparams = None
        if os.path.exists(self.case_path+"/compiled_lib1.tar") and \
            os.path.exists(self.case_path+"/compiled_lib5.tar"):
            self.factorymod1 = tvm.runtime.load_module(self.case_path+"/compiled_lib1.tar")
            self.factorymod5 = tvm.runtime.load_module(self.case_path+"/compiled_lib5.tar")
        else:
            self.factorymod1, self.factorymod5 = build_workload(\
                f'{self.case_path}/code.txt',self.case_path,params= params,
                fuseopsmax=fuseopsmax, OPTLEVEL=optlevel)
        if params is not None:
            # save_arr(params, self.case_path)
            # for float 16 mode
            path_params = os.path.join(self.case_path, 'oinputs.npz')
            np.savez(path_params, **params)

        self.gmod1 = GraphModule(self.factorymod1["default"](dev))
        self.gmod5 = GraphModule(self.factorymod5["default"](dev))
        self.tolerance = (1e-3 if self.Dtype=='float16' else 1e-6 ) if self.dnn is None else 1e-7 # should be checked RE and MRE ratio
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
                return ten*10 + one
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

    def generalMSE(self,y_true, y_pred,):  #precision along with  tf.keras.metrics.MeanRelativeError
        if not isinstance(y_true,np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred,np.ndarray):
            y_pred = np.array(y_pred)
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
        if self.dnn is None:
            relative_error = np.average( d \
                    / (np.abs(y_true).astype(np.float64) + 1e-8) * np.not_equal(y_true, 0)\
                        + np.equal(y_true, 0)* d )
        else:
            relative_error = np.average( d \
                    / (np.abs(y_true).astype(np.float64) + 1e-8) * np.abs(y_true) / np.mean(np.abs(y_true)))
        return relative_error
    def SE(self,y_true, y_pred,):  #precision along with  tf.keras.metrics.MeanRelativeError
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
        if self.dnn is None:
            relative_error = np.max( d \
                    / (np.abs(y_true).astype(np.float64) + 1e-8) * np.not_equal(y_true, 0)\
                        + np.equal(y_true, 0)* d )
        else:
            relative_error = np.max( d \
                    / (np.abs(y_true).astype(np.float64) + 1e-8) * np.abs(y_true) / np.mean(np.abs(y_true)))
        return relative_error
    def MRE(self, y_true, y_pred): # signal exposing numerical bugs
        flag = 0
        # print('l_true,l_pred', np.max(y_true), np.max(y_pred) )
        # y = np.sort(y_true[0]) # sort array
        # y = y[::-1] # reverse sort order
        # y = y[0:5] # take a slice of the first 5
        # print('top 5', y)
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
        d = np.abs(y_true.astype(np.float64) - y_pred)
        if self.dnn is None:
            relative_error = np.average( d \
                    / (np.abs(y_true).astype(np.float64) + 1e-8) * np.not_equal(y_true, 0)\
                        + np.equal(y_true, 0)* d )
        else:
            relative_error = np.average( d \
                    / (np.abs(y_true).astype(np.float64) + 1e-8) * np.abs(y_true) / np.mean(np.abs(y_true)))
        if  self.dnn is None and relative_error > 0.9 or self.dnn is not None and relative_error > storepoint:
            #!!!or self.dnn is not None and l_true!=l_pred
            print("[fuzzer]unacceptable relative error is:", relative_error)# y_true,y_pred
            flag = 1
        return relative_error,flag

    def replay(self,mode):
        path_params = os.path.join(self.case_path, mode+'inputs.npz')
        with np.load(path_params) as f:
            loaded_params = dict(f.items())

        outs1 = self.run_gmod(self.gmod1,loaded_params)
        outs5 = self.run_gmod(self.gmod5,loaded_params)
        tdiff = 0.
        for (ro,o) in zip(outs1,outs5):
            diff =  self.MSE(ro,o)
            tdiff = max(tdiff,diff)
        print('expected, actual, self.MSE = ' ,tdiff)
        tdiff2 = 0.
        for (ro,o) in zip(outs1,outs5):
            diff =  self.SE(ro,o)
            tdiff2 = max(tdiff2,diff)
        print('expected, actual, self.SE = ' ,tdiff2)
        if self.dnn is None:
            with open('./tests/out/error_all.txt','a') as fp:
                fp.write('\n'+self.case_id+','+str(self.fuzzmode)+','+str(tdiff)+','+str(tdiff2))

    def replay_withlocatenan(self):
        path_params = os.path.join(self.case_path, 'inputs.npz')
        with np.load(path_params) as f:
            loaded_params = dict(f.items())
        self.eparams = loaded_params
        outs1 = self.run_gmod(self.gmod1,loaded_params)
        outs5 = self.run_gmod(self.gmod5,loaded_params)
        print('expected, actual, self.MSE = ', self.MSE(outs1[0],outs5[0]))

    def replay_withsample(self):
        path_params = os.path.join(self.case_path, 'inputs.npz')
        with np.load(path_params) as f:
            loaded_params = dict(f.items())
        self.eparams = loaded_params
        key = list(loaded_params.keys())[0]
        tmp = loaded_params[key]
        diff = []*int(4e4+2)
        outs1 = self.run_gmod(self.gmod1,loaded_params)
        outs5 = self.run_gmod(self.gmod5,loaded_params)
        diff.append(self.MSE(outs1[0],outs5[0]))
        for i in range(int(2e4)):
            delta = np.random.uniform(0,1e-3,size = list(loaded_params.values())[0].shape)
            loaded_params[key]= delta + tmp
            outs1 = self.run_gmod(self.gmod1,loaded_params)
            outs5 = self.run_gmod(self.gmod5,loaded_params)
            diff.append(self.MSE(outs1[0],outs5[0]))
        for i in range(int(2e4)):
            delta = np.random.uniform(0,1e-4,size = list(loaded_params.values())[0].shape)
            loaded_params[key]= delta + tmp
            outs1 = self.run_gmod(self.gmod1,loaded_params)
            outs5 = self.run_gmod(self.gmod5,loaded_params)
            diff.append(self.MSE(outs1[0],outs5[0]))
        diffs = dict()
        diffs['samples'] = np.array(diff)
        path_params = os.path.join(self.case_path, 'sample2.npz')
        np.savez(path_params, **diffs)

    def replay_withdebugger(self,report:Queue=None):
        import time
        # with open('temp.txt','w') as fp:
        #     print(self.factorymod1["get_graph_json"](),file=fp)
        if os.path.exists(self.case_path+'/L1/'):
            shutil.rmtree(self.case_path+'/L1/')
        if os.path.exists(self.case_path+'/L5/'):
            shutil.rmtree(self.case_path+'/L5/')
        path_params = os.path.join(self.case_path, 'inputs.npz')
        graph_mod1 = GraphModuleDebug( self.factorymod1["debug_create"]("default", dev),
                            [dev],self.factorymod1["get_graph_json"](), dump_root=self.case_path+'/L1/')
        graph_mod5 = GraphModuleDebug( self.factorymod5["debug_create"]("default", dev),
                            [dev],self.factorymod5["get_graph_json"](), dump_root=self.case_path+'/L5/')
        with np.load(path_params) as f:
            loaded_params = dict(f.items())
        print(loaded_params)
        # set for sparse optimization
        # if os.path.exists(self.case_path+'/params5.npz'):
        #     with np.load(self.case_path+'/params5.npz') as f:
        #         params5 = dict(f.items())
        #     outs5 = self.run_gmod(graph_mod5,params5)
        # else:
        print('params5 use input.npz')
        outs5 = self.run_gmod(graph_mod5,loaded_params)
        self.eparams = loaded_params
        # newp = dict()
        # k,v = list(loaded_params.keys())[0],list(loaded_params.values())[0]
        # newp[k] = v
        # outs1 = self.run_gmod(graph_mod1,newp)# loaded_params
        # outs5 = self.run_gmod(graph_mod5,newp)
        outs1 = self.run_gmod(graph_mod1,loaded_params)


        tdiff = 0.
        for (ro,o) in zip(outs1,outs5):
            diff =  self.MSE(ro,o)
            tdiff = max(tdiff,diff)
        print('expected, actual, self.MSE = ' ,tdiff)
        tdiff2 = 0.
        for (ro,o) in zip(outs1,outs5):
            diff =  self.SE(ro,o)
            tdiff2 = max(tdiff2,diff)
        print('expected, actual, self.SE = ' ,tdiff2)
        if self.dnn is None:
            if not os.path.exists('./out/error_all.txt'):
                with open('./out/error_all.txt','w') as fp:
                    pass
            with open('./out/error_all.txt','a') as fp:
                fp.write('\n'+self.case_id+','+str(self.fuzzmode)+','+str(tdiff)+','+str(tdiff2))
        # total error

        # if 1:#os.path.exists(self.case_path+'/model_scripted.pt')
        #     # baseline_model = torch.load(self.case_path +'/model_scripted.pt').float().eval()
        #     from torch.nn import Module

        #     class Log2_1(Module):
        #         def forward(self, *args):
        #             return torch.log2(args[0])
        #     baseline_model = Log2_1().float().eval()
        #     outst = run_tmod(baseline_model,torch.from_numpy(list(loaded_params.values())[0]).float())
        #     tempdiff1 = 0
        #     tempdiff2 = 0
        #     for i, (ro, o) in enumerate(zip(outst, outs5)):
        #         diff  =  self.MSE(ro,o)
        #         tempdiff1 = max(diff, tempdiff1)
        #         diff = self.SE(ro,o)
        #         tempdiff2 = max(diff, tempdiff2)
        # print('total error,mse,se are',tempdiff1,tempdiff2)
        if report is not None:
            report.put('done_debugger')

    def replay_tansformer(self,report:Queue=None):
        import time
        from torch.nn import Module
        path_params = os.path.join(self.case_path, 'inputs.npz')
        graph_mod1 = self.gmod1
        graph_mod5 = self.gmod5
        with np.load(path_params) as f:
            loaded_params = dict(f.items())
        tgtt = list(loaded_params.values())[0]
        if os.path.exists(self.case_path+'/params5.npz'):
            with np.load(self.case_path+'/params5.npz') as f:
                params5 = dict(f.items())
            params5['input1'] = tgtt
            outs5 = self.run_gmod(graph_mod5,params5)
        else:
            print('params5 use input.npz')
            outs5 = self.run_gmod(graph_mod5,loaded_params)
        self.eparams = loaded_params
        outs1 = self.run_gmod(graph_mod1,loaded_params)
        # define a fc layer
        fclayerw = np.random.normal(loc=0,scale=0.1,size=(64 , 10000))
        outs5 = np.swapaxes(outs5[0],0,1)
        outs1 = np.swapaxes(outs1[0],0,1)
        outs5 = np.matmul(outs5, fclayerw)
        outs1 = np.matmul(outs1, fclayerw)
        tdiff = 0.
        tdiff2 = 0.
        for (ro,o) in zip(outs1,outs5):
            diff  =  self.MSE(ro,o)
            tdiff = max(tdiff,diff)
            diff  =  self.SE(ro,o)
            tdiff2 = max(tdiff2, diff)
        print('optimization MSE,SE = ' ,tdiff, tdiff2)
        tgt = list(loaded_params.values())[0]
        src = list(loaded_params.values())[1]
        print('src',src.shape)
        # total error
        if os.path.exists(self.case_path+'/model_scripted.pt'):#
            baseline_model = torch.load(self.case_path +'/model_scripted.pt').float().eval()
            outst = run_tmod(baseline_model,[torch.from_numpy(src\
                            ,).float(), torch.from_numpy(tgt).float()])
            tempdiff1 = 0
            tempdiff2 = 0
            outst = np.swapaxes(outst[0],0,1)
            outst = np.matmul(outst, fclayerw)
            for i, (ro, o) in enumerate(zip(outst, outs5)):
                diff  =  self.MSE(ro,o)
                tempdiff1 = max(diff, tempdiff1)
                diff = self.SE(ro,o)
                tempdiff2 = max(diff, tempdiff2)
        print('total error,mse,se are',tempdiff1,tempdiff2)
        tempdiff1 = 0
        tempdiff2 = 0
        for i, (ro, o) in enumerate(zip(outst, outs1)):
                diff  =  self.MSE(ro,o)
                tempdiff1 = max(diff, tempdiff1)
                diff = self.SE(ro,o)
                tempdiff2 = max(diff, tempdiff2)
        print('framework error,mse,se are',tempdiff1,tempdiff2)
        totalpredicts = 0.
        incorrect = 0.
        # calculate incorrect prediction rate
        nb, nsen = outs1.shape[0], outs1.shape[1]
        for i in range(nb):
            for j in range(nsen):
                inda = np.argsort(outs5[i][j])[::-1][0:3]
                indb = np.argsort(outs1[i][j])[::-1][0:3]
                # print(inda,indb)
                totalpredicts += 1
                incorrect +=  np.equal(inda,indb).all()
        print('incorrect rate', (totalpredicts - incorrect)/totalpredicts)

        if report is not None:
            report.put('done_debugger')

    def replay_dnn(self,report:Queue=None):
        import time
        from torch.nn import Module
        path_params = os.path.join(self.case_path, 'inputs.npz')
        graph_mod1 = self.gmod1
        graph_mod5 = self.gmod5
        with np.load(path_params) as f:
            loaded_params = dict(f.items())
        tgtt = list(loaded_params.values())[0]
        # if os.path.exists(self.case_path+'/params5.npz'):
        #     with np.load(self.case_path+'/params5.npz') as f:
        #         params5 = dict(f.items())
        #     params5['input1'] = tgtt
        #     outs5 = self.run_gmod(graph_mod5,params5)
        # else:
        print('params5 use input.npz')
        outs5 = self.run_gmod(graph_mod5,loaded_params)
        self.eparams = loaded_params
        outs1 = self.run_gmod(graph_mod1,loaded_params)
        tdiff = 0.
        tdiff2 = 0.
        for (ro,o) in zip(outs1,outs5):
            diff  =  self.MSE(ro,o)
            tdiff = max(tdiff,diff)
            diff  =  self.SE(ro,o)
            tdiff2 = max(tdiff2, diff)
        print('optimization MSE,SE = ' ,tdiff, tdiff2)
        tgt = list(loaded_params.values())[0]
        # total error
        if os.path.exists(self.case_path+'/model_scripted.pt'):#
            baseline_model = torch.load(self.case_path +'/model_scripted.pt').float().eval()
            outst = run_tmod(baseline_model,torch.from_numpy(tgt).float())
            tempdiff1 = 0
            tempdiff2 = 0
            for i, (ro, o) in enumerate(zip(outst, outs5)):
                diff  =  self.MSE(ro,o)
                tempdiff1 = max(diff, tempdiff1)
                diff = self.SE(ro,o)
                tempdiff2 = max(diff, tempdiff2)
        print('total error,mse,se are',tempdiff1,tempdiff2)
        tempdiff1 = 0
        tempdiff2 = 0
        for i, (ro, o) in enumerate(zip(outst, outs1)):
                diff  =  self.MSE(ro,o)
                tempdiff1 = max(diff, tempdiff1)
                diff = self.SE(ro,o)
                tempdiff2 = max(diff, tempdiff2)
        print('framework error,mse,se are',tempdiff1,tempdiff2)
        totalpredicts = 0.
        correct = 0.
        # calculate incorrect prediction rate
        inda = np.argsort(outs5[0][0])[::-1][0:10]
        indb = np.argsort(outs1[0][0])[::-1][0:10]

        # print(inda,indb)
        # print(outs1[0][0][inda])
        # print(outs5[0][0][inda])
        totalpredicts += 10
        correct +=  np.count_nonzero(np.equal(inda,indb))
        print('incorrect rate', (totalpredicts - correct)/totalpredicts)

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
            with transform.PassContext(opt_level=opt_level+1,disabled_pass=self.Disabled_pass):
                mod1 = relay.build(mod, target, #params=params
                                    )
        except:
            with transform.PassContext(opt_level=opt_level+1,disabled_pass=self.Disabled_pass):
                mod1 = relay.build(mod, target, #params=params
                                    )
        # level 5
        opt_level =self.OPTLEVEL
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
        if opt_level == 0:
            with transform.PassContext(opt_level=opt_level+1,required_pass=self.Required_pass1,disabled_pass=self.Disabled_pass):
                lib = relay.build(mod, target, params=params)
        else:
            with transform.PassContext(opt_level=opt_level,required_pass=self.Required_pass):
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
                inputarr[varx.name_hint] = \
                    np.random.normal(0,0.1,size=size).astype(var_tyx.dtype)
        else:   # {spec inputs} +random test
            speclen = len(arrs)
            length = len(main_fn.params)-speclen
            inputarr = dict()
            for i in range(length):
                varx = main_fn.params[i+speclen]
                var_tyx = varx.checked_type
                size=[int(d) for d in var_tyx.shape]
                inputarr[varx.name_hint] = \
                    np.random.normal(0,0.1,size=size).astype(var_tyx.dtype)
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
