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
from .mutate_utils import generaltactics, simpleDE, simulated_annealing, genetic_algorithm
from .base_utils import Checkor
from multiprocessing import Process, Queue
from scipy.optimize import basinhopping, differential_evolution, Bounds, dual_annealing
from threading import Thread
import re
import torch

speed1 ,speed2,speed3, speed4= 1,1,1,5## for non quant model 6,24,1,40  is good configuration, for quant model, 2,2 1,16 is good
topk = 100
langbatch = 2 #  {torch_transformer0: 10, oberts: 2, otheroberta:1}
dictnums = 30520 #30521 for mask / 5e4 for qa
TensorDict = Dict[str, np.ndarray]
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
# /home/zichaox/tvm/tests/python/frontend/onnx/test_forward.py
# /home/zichaox/tvm/Fuzzfp/introcase/src/fuzztorch.py
import sys
sys.path.append('../../..')
from .onnx_utils import get_ort_out
import time
def time_it(func):
    def inner():
        start = time.time()
        func()
        end = time.time()
        print('using time:{}secs'.format(end-start))
    return inner
def boundarr(x,inputsize):
        arr = np.empty( shape = inputsize).flatten()
        arr.fill(x)
        return arr
def print_fun(x, f, accepted):
        print("at de minimum %.7f accepted %d" % (-f, int(accepted)),x)
        #pass

def print_fun2( x, f, context):
        print("at anneal minimum %.7f" % (-f),x)
        if(f<1e-9):
            return True
class MyTakeStep:
   def __init__(self, stepsize=0.5):
       self.stepsize = stepsize
       self.rng = np.random.default_rng()
   def __call__(self, x):
       s = self.stepsize
       x[0 ] += self.rng.uniform(-2.*s, 2.*s)
       x[1:] += self.rng.uniform(-s, s, x[1:].shape)
       return x
# another method
"""

https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping
https://docs.scipy.org/doc/scipy-1.0.0/reference/optimize.html
https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution
"""

'''
fuzzing methods:

1. use 4 tatics to mutate the seed until diff no longer increase
    confuse:
    [ random loss] on a flattened tensor's first n element at a random rate
    [ random enhance] on a flattened tensor's first n element at a random rate
    [ random add noise] on a flattened tensor's first n element at a random rate
    diffuse:
    [ random loss] roll the tensor along the axis

2. cross the top performance seed with the curthy_random seed
3. let some seeds listed below enter the queue
    top performance seed
    good newly produced descendant
    some not well-performed seed
4. run until the queue for best seeds are empty
5. if the same difference shows up more than 100 times in the loop of 3, change the flatten direction.
6. if the same difference shows up more than 4 times in the loop of 5, end the process.
'''
def remove_virtarget(ncode):
    rncode = re.sub('({virtual_.*?}:)', ':', ncode,count=0, flags=re.M|re.S)
    rncode = re.sub('(virtual_.*?->)', ') ->', rncode,count=0, flags=re.M|re.S)
    return rncode
def remove_primary(code):
    return  re.sub('(, Primitive\=.*?->)', ') ->', code,count=0, flags=re.M|re.S)

def run_tmod(
    model,
    input_data=None,
):
    """Assert that the output of a compiled model matches with that of its
    baseline."""
    with torch.no_grad():
        if isinstance(input_data, list):
            baseline_input = input_data
        elif isinstance(input_data, torch.Tensor) or not input_data.shape:
            baseline_input = [input_data]
        else:
            print(type(input_data),'not valid')
            exit()
        baseline_outputs = model(*[input.clone() for input in baseline_input])

    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.cpu().numpy(),)
    return baseline_outputs
from torch.nn import Module

class Log2_1(Module):
        def forward(self, *args):
            return torch.log2(args[0])
class Fuzzer(Checkor):
    def __init__(self,path:str,case_id:str,low:float=0,high:float=1, fuseopsmax=5,\
                 params = None, fuzzmode:str ='MEGA',fuzzframe = False,optlevel=5):
        super().__init__(path=path,case_id=case_id,params=params,lowbound=low,
                         highbound=high, fuseopsmax=fuseopsmax,
                         fuzzmode=fuzzmode,optlevel=optlevel)
        self.case_path = os.path.join(path+'/out', case_id)
        self.usedseed =[None]*20
        self.Maxfuzztrytimes = 50
        self.Lastfuzztimes = 10
        self.seedq = queue.Queue()
        self.params = params
        self.tempname = None
        self.tempshape = None
        self.fuzzmode = fuzzmode
        self.randweight = None
        self.torchm = None
        self.usetorch = os.path.exists(self.case_path +'/model_scripted.pt')
        if self.usetorch:
            self.torchm = torch.load(self.case_path +'/model_scripted.pt').float().eval()
        self.useonnx = os.path.exists(self.case_path +'/model.onnx')
        if self.useonnx:
            import onnx
            self.onnx_model = onnx.load(self.case_path+"/model.onnx")
        self.frame = fuzzframe
        # self.torchm = torch.load(self.case_path +'/model_scripted.pt').float().eval()

    def get_primfunc(self, opt_level,target='llvm'):# i.e. get tvm.ir.module.IRModule
        target, target_host = tvm.target.Target.canon_target_and_host(target)
        if opt_level==5:
            with tvm.transform.PassContext(opt_level=opt_level):# config={"relay.FuseOps.max_depth": 10}
                prim_mod, _ = relay.optimize(self.mod, target)
                code = remove_virtarget(prim_mod.astext())
            return code
        else:
            with tvm.transform.PassContext(opt_level=opt_level,required_pass=self.Required_pass1,disabled_pass=self.Disabled_pass):# config={"relay.FuseOps.max_depth": 10}
                prim_mod, _ = relay.optimize(self.mod, target)
                code = remove_virtarget(prim_mod.astext())
            return code
    def rundiff(self,main_fn:relay.function.Function,
                 arrs:List[np.array]=None,changeweight=False):# input:list of np
        inputarr = self.eparams
        inputarr[self.tempkey] = arrs[0]
        self.eparams = inputarr
        outs1 = self.run_gmod(self.gmod1,inputarr)
        outs5 = self.run_gmod(self.gmod5,inputarr)
        tempdiff = 0
        for i, (ro, o) in enumerate(zip(outs1, outs5)):
            diff , flag =  self.MRE(ro,o)
            if flag>0:

                self.handle_exit(flag,diff)
            tempdiff = max(diff, tempdiff)
        return tempdiff
    def rundiff_changew(self,main_fn:relay.function.Function,
                 arrs:List[np.array]=None):# input:list of np
        inputarr = self.generate_inputs(main_fn, arrs,)
        # tvm.contrib.graph_executor.GraphModule
        # GraphExecutorFactoryModule
        self.eparams = inputarr
        outs1 = self.run_gmod(self.gmod1,inputarr)
        outs5 = self.run_gmod(self.gmod5,inputarr)
        tempdiff = 0
        for i, (ro, o) in enumerate(zip(outs1, outs5)):
            diff , flag =  self.MRE(ro,o)
            if flag>0:
                self.handle_exit(flag,diff)
            tempdiff = max(diff, tempdiff)
        return tempdiff,inputarr
    def cross(self, main_fn:relay.function.Function,inputo:np,inputseed:np):
        input = inputo.copy()
        input2 = inputo.copy()
        originshape = inputo.shape
        #get input
        input = np.reshape(input,originshape)
        input2 = np.reshape(input2,originshape)
        origindiff = self.rundiff(main_fn,[input] )
        input = input.flatten()
        input2 = input2.flatten()
        length = input.shape[0]
        sindex = random.randint(int(length/3),int(length/2))
        pattern = inputseed[random.randint(0,9)]
        if pattern is None:
            print(inputseed)
            print('pattern is nonetype: ',pattern is None)

        if input is None:
            print(input)
            print('input is nonetype: ',input is None)
        input[sindex:]=pattern.flatten()[sindex:]
        input2[:sindex]=pattern.flatten()[:sindex]
        input = np.reshape(input,originshape)
        input2 = np.reshape(input2,originshape)
        # valide input
        diff = self.rundiff(main_fn,[input] )
        diff2 = self.rundiff(main_fn,[input2] )

        if  diff > origindiff:
            return input,1
        else:
            if  diff2 > origindiff:
                return input2,1
            else:
                return input2,0
    def save_files(self):
        optprim_mod = self.get_primfunc(5)
        unoptprim_mod = self.get_primfunc(1)
        with open(f'{self.case_path}/optirmod.txt', 'w') as f: # str less #[version = "0.0.5"]\n
            f.write( optprim_mod)
        with open(f'{self.case_path}/unoptirmod.txt', 'w') as f:
            f.write(unoptprim_mod)
    def fastv(self):
        #Input_size
        self.oDtype = self.Dtype
        print('self.dnn', self.dnn)
        main_fn = self.mod['main']
        plen = len(main_fn.params)

        Input_sizes = self.generate_inputs_nameshape(main_fn,plen)
        # initial weight
        diff = 0
        # begin

        for tempkey, Input_size in Input_sizes.items():  # for each params
            self.Input_size = Input_size
            self.tempkey = tempkey
            if 'transformer' in self.case_path:
                t_sizes = list(Input_sizes.values())
                keys = list(Input_sizes.keys())
                if 'int' in self.Dtype:
                    self.Input_size = (t_sizes[0][0]+t_sizes[1][0],t_sizes[0][1]) # int input
                else:
                    self.Input_size = (t_sizes[0][0]+t_sizes[1][0],t_sizes[0][1],t_sizes[0][2]) # float input
            print('temp fuzzing params: ',self.tempkey,self.Input_size)

            # random default
            for i in range(3):
                arr = np.random.uniform(0,1,size=self.Input_size)
                inputarr = dict()
                if 'transformer' in self.case_path:
                        a,b = np.vsplit(np.reshape(arr, self.Input_size),[langbatch])
                        if self.oDtype == 'int64':
                            if 'int' in str(a.dtype):
                                inputarr[keys[0]] = a
                            else:
                                inputarr[keys[0]] = (a*dictnums).astype('int64')
                            inputarr[keys[1]] = np.zeros(shape =t_sizes[1]).astype(self.oDtype)
                            inputarr[keys[1]][0][0:30] = 1
                        else:
                            inputarr['input0'] = a
                            inputarr['input1'] = b
                else:
                        # inputarr[self.tempkey] = (np.reshape(arr, self.Input_size)*255).astype('uint8')
                        inputarr[self.tempkey] = np.reshape(arr, self.Input_size)

                # path_params = os.path.join(self.case_path, 'inputs.npz')
                # with np.load(path_params) as f:
                #     inputarr = dict(f.items())
                #     print(inputarr)

                outs1  =self.run_gmod(self.gmod1)
                outs5 = self.run_gmod(self.gmod5)
                tdiff = 0.
                tdiff2 = 0.
                # np.testing.assert_allclose(outs1,outs5,rtol=1e-42)

                for (ro,o) in zip(outs1,outs5):
                    diff  =  self.MSE(ro,o)
                    tdiff = max(tdiff,diff)
                    diff  =  self.SE(ro,o)
                    tdiff2 = max(tdiff2, diff)
                print('optimization MSE,SE =' ,tdiff, tdiff2)
            for i in range(3):
                arr =np.clip(np.random.normal(loc=0,scale=0.2,size=self.Input_size),0,1)
                inputarr = dict()
                if 'transformer' in self.case_path:
                        a,b = np.vsplit(np.reshape(arr, self.Input_size),[langbatch])
                        if self.oDtype == 'int64':
                            if 'int' in str(a.dtype):
                                inputarr[keys[0]] = a
                            else:
                                inputarr[keys[0]] = (a*dictnums).astype('int64')
                            inputarr[keys[1]] = np.zeros(shape =t_sizes[1]).astype(self.oDtype)
                            inputarr[keys[1]][0][0:30] = 1
                        else:
                            inputarr['input0'] = a
                            inputarr['input1'] = b
                else:
                    # inputarr[self.tempkey] = np.reshape(arr, self.Input_size)
                    inputarr[self.tempkey] = np.random.randint(0,255,size = self.Input_size,dtype = self.Dtype)
                # print('YES,',keys,inputarr[keys[0]])
                outs1  =self.run_gmod(self.gmod1,inputarr)
                outs5 = self.run_gmod(self.gmod5,inputarr)
                tdiff = 0.
                tdiff2 = 0.
                # np.testing.assert_allclose(outs1,outs5,rtol=1e-42)
                for (ro,o) in zip(outs1,outs5):
                    diff  =  self.MSE(ro,o)
                    tdiff = max(tdiff,diff)
                    diff  =  self.SE(ro,o)
                    tdiff2 = max(tdiff2, diff)
                print('optimization MSE,SE =' ,tdiff, tdiff2)
            break
    def profile_nlp(self):
        #Input_size
        self.oDtype = self.Dtype
        print('self.dnn', self.dnn)
        main_fn = self.mod['main']
        plen = len(main_fn.params)

        Input_sizes = self.generate_inputs_nameshape(main_fn,plen)
        # initial weight
        diff = 0
        # begin
        if True:

            self.Input_size = list(Input_sizes.values())[0]
            self.tempkey =list(Input_sizes.keys())[0]
            if 'transformer' in self.case_path:
                t_sizes = list(Input_sizes.values())
                keys = list(Input_sizes.keys())
                if 'int' in self.Dtype:
                    self.Input_size = (t_sizes[0][0]+t_sizes[1][0],t_sizes[0][1]) # int input
                else:
                    self.Input_size = (t_sizes[0][0]+t_sizes[1][0],t_sizes[0][1],t_sizes[0][2]) # float input
            print('temp fuzzing params: ',self.tempkey,self.Input_size)

            for i in range(1):
                x0 = np.random.uniform(0,1,size=self.Input_size)
                inputarr = dict()
                if 'transformer' in self.case_path:
                        a,b = np.vsplit(np.reshape(x0, self.Input_size),[langbatch])
                        if self.oDtype == 'int64':
                            if 'int' in str(a.dtype):
                                inputarr[keys[0]] = a
                            else:
                                inputarr[keys[0]] = (a*dictnums).astype('int64')
                            inputarr[keys[1]] = np.zeros(shape =t_sizes[1]).astype(self.oDtype)
                            inputarr[keys[1]][0][0:30] = 1
                        else:
                            inputarr['input0'] = a
                            inputarr['input1'] = b
                else:
                    arr = np.random.uniform(0,1,size=self.Input_size)
                    inputarr[self.tempkey] = np.reshape(arr, self.Input_size)
                    # inputarr[self.tempkey] = np.random.randint(0,255,size = self.Input_size,dtype = self.Dtype)
                # path_params = os.path.join(self.case_path, 'inputs.npz')
                # with np.load(path_params) as f:
                #     inputarr = dict(f.items())
                #     print(inputarr)

                outs1  =self.run_gmod(self.gmod1,inputarr)
                outs5 = self.run_gmod(self.gmod5,inputarr)
                tdiff = 0.
                tdiff2 = 0.
                # np.testing.assert_allclose(outs1,outs5,rtol=1e-42)

                # for (ro,o) in zip(outs1,outs5):
                #     diff  =  self.MSE(ro,o)
                #     tdiff = max(tdiff,diff)
                #     diff  =  self.SE(ro,o)
                #     tdiff2 = max(tdiff2, diff)
                # print('optimization MSE,SE =' ,tdiff, tdiff2)
                # inda = np.argsort(outs5[0][0])[::-1][0:topk]
                # indb = np.argsort(outs1[0][0])[::-1][0:topk]
                # # print(inda,indb)
                # totalpredicts = float(topk)
                # correct =  np.count_nonzero(np.equal(inda,indb))

                # print(f'top{topk} incorrect rate', (totalpredicts - correct)/totalpredicts)
                # print(inda,indb)
                # print(outs5[0][0][inda])
                # print(outs1[0][0][indb])
                # print(self.MSE(outs5[0][0][inda],outs1[0][0][indb]))
                # print(outs5[0][0][:,4:84].shape)
                
                inda = np.array(sorted(outs5[0][0].tolist(),reverse=1)[0:topk]) # [0][0][0][4:84] for yolov8, [0][0] for resnet [0][0][0] for 
                print(outs5[0])
                indb = np.array(sorted(outs5[0][0].tolist(),reverse=1)[0:topk])
                indb[1:]= indb[:-1]
                print(inda,indb)
                print('the distance between classes is',np.mean(np.abs((indb-inda)/inda)))
                # inda = np.argsort(outs5[0][0])[0:5]
                # indb = np.argsort(outs1[0][0])[0:5]


                # # print(inda,indb)
                # totalpredicts = 5.0
                # correct =  np.count_nonzero(np.equal(inda,indb))
                # print('tail 5 incorrect rate', (totalpredicts - correct)/totalpredicts)
                # print(inda,indb)
                # print(outs5[0][0][inda])
                # print(outs1[0][0][indb])

    def handle_exit(self,flag,diff):
            print('-------------------------------')
            if flag==1 :
            # dump_arrs
                path_params = os.path.join(self.case_path, 'inputs.npz')
                np.savez(path_params, **self.eparams)
                print('find enomous error',diff)
                import time
                print('time0',time.time())
                with open(self.case_path+'/Fuzzstate','a') as fp:
                    fp.write('correctness bug')
                    exit()
            if flag==3:
                with open(self.case_path+'/Fuzzstate','a') as fp:
                    fp.write('exception bug')
                exit()
            if flag ==10:
                with open(self.case_path+'/Fuzzstate','a') as fp:
                    fp.write('invalid result')
                exit()
            if not os.path.exists((f'{self.case_path}/optirmod.txt')):
                self.save_files()
            exit()
    def replay_error(self,report:Queue=None):
        #!!!

        from torch.nn import Module
        path_params = os.path.join(self.case_path, 'inputs.npz')
        graph_mod1 = self.gmod1
        graph_mod5 = self.gmod5
        with np.load(path_params) as f:
            loaded_params = dict(f.items())
        print('loaded_params', list(loaded_params.values())[0].dtype)
        outs5 = self.run_gmod(graph_mod5,loaded_params)
        outs1 = self.run_gmod(graph_mod1,loaded_params)


        tdiff = 0.
        tdiff2 = 0.
        for (ro,o) in zip(outs1,outs5):
            diff  =  self.MSE(ro,o)
            print(diff)
            tdiff = max(tdiff,diff)
            diff  =  self.SE(ro,o)
            tdiff2 = max(tdiff2, diff)
        print('optimization MSE,SE =' ,tdiff, tdiff2)
        if os.path.exists(self.case_path+"/compiled_lib0.tar"):
            self.factorymod0 = tvm.runtime.load_module(self.case_path+"/compiled_lib0.tar")
            self.gmod0 = GraphModule(self.factorymod0["default"](dev))
            outs0 = self.run_gmod(graph_mod1,loaded_params)
            tdiff = 0.
            tdiff2 = 0.
            for (ro,o) in zip(outs0,outs1):
                diff  =  self.MSE(ro,o)
                tdiff = max(tdiff,diff)
                diff  =  self.SE(ro,o)
                tdiff2 = max(tdiff2, diff)
            print('prune MSE,SE =' ,tdiff, tdiff2)
            tdiff = 0.
            tdiff2 = 0.
            for (ro,o) in zip(outs0,outs5):
                diff  =  self.MSE(ro,o)
                tdiff = max(tdiff,diff)
                diff  =  self.SE(ro,o)
                tdiff2 = max(tdiff2, diff)
            print('total MSE,SE =' ,tdiff, tdiff2)
        if 'transformer' in self.case_path:
            # define a fc layer
            fclayerw = np.random.normal(loc=0,scale=0.1,size=(64 , 10000))
            outs5 = np.swapaxes(outs5[0],0,1)
            outs1 = np.swapaxes(outs1[0],0,1)
            outs0 = np.swapaxes(outs0[0],0,1)
            outs5 = np.matmul(outs5, fclayerw)
            outs1 = np.matmul(outs1, fclayerw)
            outs0 = np.matmul(outs0, fclayerw)
            nb, nsen = outs1.shape[0], outs1.shape[1]
            totalpredicts = 0
            correct = 0
            for i in range(nb):
                for j in range(nsen):
                    inda = np.argsort(outs5[i][j])[::-1][0:5]
                    indb = np.argsort(outs0[i][j])[::-1][0:5]
                    # print(inda,indb)
                    totalpredicts += 5
                    correct +=  np.count_nonzero(np.equal(inda,indb))
            print('top5 incorrect rate', (totalpredicts - correct)/totalpredicts)
            totalpredicts = 0
            correct = 0
            for i in range(nb):
                for j in range(nsen):
                    inda = np.argsort(outs5[i][j])[::-1][0:topk]
                    indb = np.argsort(outs0[i][j])[::-1][0:topk]
                    # print(inda,indb)
                    totalpredicts += topk
                    correct +=  np.count_nonzero(np.equal(inda,indb))
            print('top 10 incorrect rate', (totalpredicts - correct)/totalpredicts)
        else:
            inda = np.argsort(outs5[1][0])[::-1][0:topk]
            indb = np.argsort(outs1[1][0])[::-1][0:topk]
            totalpredicts = 0.0
            correct = 0.0
            totalpredicts += float(topk)
            correct +=  np.count_nonzero(np.equal(inda,indb))
            print(inda,indb)
            print(self.MSE(outs5[1][0][inda],outs1[1][0][indb]))
            print(f'top{topk} incorrect rate', (totalpredicts - correct)/totalpredicts)
        import onnx
        # !!! renew path
        onnx_model = onnx.load("/root/.cache/sparsezoo/e37ce71a-d72f-42ec-b87f-21994e8fb6df/deployment/model.onnx")
        outs0 = get_ort_out(onnx_model,[list(loaded_params.values())[0]])

        tempdiff1 = 0
        tempdiff2 = 0
        for i, (ro, o) in enumerate(zip(outs0, outs1)):
                diff  =  self.MSE(ro,o)
                tempdiff1 = max(diff, tempdiff1)
                diff = self.SE(ro,o)
                tempdiff2 = max(diff, tempdiff2)
        print('framework error,mse,se are',tempdiff1,tempdiff2)

        tempdiff1 = 0
        tempdiff2 = 0
        for i, (ro, o) in enumerate(zip(outs0, outs5)):
                diff  =  self.MSE(ro,o)
                tempdiff1 = max(diff, tempdiff1)
                diff = self.SE(ro,o)
                tempdiff2 = max(diff, tempdiff2)
        print('total error,mse,se are',tempdiff1,tempdiff2)

        # torch calculate
        '''
        tgt = list(loaded_params.values())[0]
        src = list(loaded_params.values())[1]
        print('src',src.shape)
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
    '''
    def replay_error_int(self,report:Queue=None):
        #!!!
        self.oDtype = self.Dtype
        from torch.nn import Module
        path_params = os.path.join(self.case_path, 'inputs.npz')
        graph_mod1 = self.gmod1
        graph_mod5 = self.gmod5
        # with np.load(path_params) as f:
        #     loaded_params = dict(f.items())
        # arr = list(loaded_params.values())[0]
        # print('loaded_params', arr.dtype)
        inputarr = dict()
        main_fn = self.mod['main']
        plen = len(main_fn.params)

        Input_sizes = self.generate_inputs_nameshape(main_fn,plen)
        inkeys = list(Input_sizes.keys())
        t_sizes = list(Input_sizes.values())
        # a,b = np.vsplit(np.reshape(arr, t_sizes[0]),[langbatch])
        a = (np.random.uniform(size=(1,384))* dictnums).astype('int64')

        if self.oDtype == 'int64':
            if 'int' in str(a.dtype):
                inputarr[inkeys[0]] = a
            else:
                inputarr[inkeys[0]] = (a*dictnums).astype('int64')
            inputarr[inkeys[1]] = np.ones(shape =t_sizes[1]).astype(self.oDtype)
            # inputarr[inkeys[1]][0][100] = 0
            # inputarr[inkeys[1]][0][510:] = 1
            # inputarr[inkeys[2]] = np.zeros(shape =t_sizes[2]).astype(self.oDtype)

        outs5 = self.run_gmod(graph_mod5,inputarr)
        outs1 = self.run_gmod(graph_mod1,inputarr)


        tdiff = 0.
        tdiff2 = 0.
        for (ro,o) in zip(outs1,outs5):
            diff  =  self.MSE(ro,o)
            print(diff)
            tdiff = max(tdiff,diff)
            diff  =  self.SE(ro,o)
            tdiff2 = max(tdiff2, diff)
        print('optimization MSE,SE =' ,tdiff, tdiff2)

        inda = np.argsort(outs5[0][0])[::-1][0:topk]
        indb = np.argsort(outs1[0][0])[::-1][0:topk]
        totalpredicts = 0.0
        correct = 0.0
        totalpredicts += float(topk)
        correct +=  np.count_nonzero(np.equal(inda,indb))
        # print(inda,indb)
        # print(self.MSE(outs5[1][0][inda],outs1[1][0][indb]))
        print(f'top{topk} incorrect rate', (totalpredicts - correct)/totalpredicts)
        import onnx
        # !!! renew path
        # onnx_model = onnx.load("/root/.cache/sparsezoo/321dac91-0d0a-4815-bc95-9ec13ef997aa/deployment/model.onnx")
        # onnx_model = onnx.load('/root/.cache/sparsezoo/9887ee48-6cd3-4885-8105-51213ac2ff6f/deployment/model.onnx')
        onnx_model = onnx.load('/root/.cache/sparsezoo/9273788e-f3a8-41f8-8e93-e3249621866c/deployment/model.onnx')
        # onnxinput1 = np.ones(shape=(2,384)).astype('int64')
        # onnxinput2 =  np.zeros(shape=(2,384)).astype('int64')
        # onnxinput1[0][0:50] = inputarr[inkeys[0]][0]
        # onnxinput1[1][0:50] = inputarr[inkeys[0]][1]
        # onnxinput2[0][0:50] = inputarr[inkeys[1]][0]
        # onnxinput2[1][0:50] = inputarr[inkeys[1]][1]
        # onnxinput2 = onnxinput2 %50264
        # onnxinput1 = onnxinput1%50264
        onnxinput1 = inputarr[inkeys[0]]
        onnxinput2 = inputarr[inkeys[1]]
        # onnxinput3 = inputarr[inkeys[2]]
        outs0 = get_ort_out(onnx_model,[onnxinput1,onnxinput2])#onnxinput3
        # temp00 =np.zeros(shape=(2,50)).astype('int64')
        # temp01 =np.zeros(shape=(2,50)).astype('int64')
        # temp00[0][:] = outs0[0][0][0:50]
        # temp00[1][:] = outs0[0][1][0:50]

        # temp01[0][:] = outs0[1][0][0:50]
        # temp01[1][:] = outs0[1][1][0:50]
        # outs0 = [temp00,temp01]
        tempdiff1 = 0
        tempdiff2 = 0
        for i, (ro, o) in enumerate(zip(outs0, outs1)):
                diff  =  self.MSE(ro,o)
                tempdiff1 = max(diff, tempdiff1)
                diff = self.SE(ro,o)
                tempdiff2 = max(diff, tempdiff2)
        print('framework error,mse,se are',tempdiff1,tempdiff2)

        tempdiff1 = 0
        tempdiff2 = 0
        for i, (ro, o) in enumerate(zip(outs0, outs5)):
                diff  =  self.MSE(ro,o)
                tempdiff1 = max(diff, tempdiff1)
                diff = self.SE(ro,o)
                tempdiff2 = max(diff, tempdiff2)
        print('total error,mse,se are',tempdiff1,tempdiff2)
        totalpredicts = 0.0
        correct = 0.0
        inda = np.argsort(outs1[0][0])[::-1][0:topk]
        indb = np.argsort(outs0[0][0])[::-1][0:topk]

        totalpredicts += float(topk)
        correct +=  np.count_nonzero(np.equal(inda,indb))
        # print(inda,indb)
        # print(self.MSE(outs5[1][0][inda],outs1[1][0][indb]))
        print(f'total top{topk} incorrect rate', (totalpredicts - correct)/totalpredicts)
    
    def profile_error(self,report:Queue=None):
        #!!!

        from torch.nn import Module
        path_params = os.path.join(self.case_path, 'inputs.npz')
        graph_mod1 = self.gmod1
        graph_mod5 = self.gmod5
        with np.load(path_params) as f:
            loaded_params = dict(f.items())
        print('loaded_params', list(loaded_params.values())[0].dtype)
        x0 = np.random.uniform(0,1,size=self.Input_size).astype(self.Dtype)
        outs5 = self.run_gmod(graph_mod5,loaded_params)
        outs1 = self.run_gmod(graph_mod1,loaded_params)


        tdiff = 0.
        tdiff2 = 0.
        for (ro,o) in zip(outs1,outs5):
            diff  =  self.MSE(ro,o)
            print(diff)
            tdiff = max(tdiff,diff)
            diff  =  self.SE(ro,o)
            tdiff2 = max(tdiff2, diff)
        print('optimization MSE,SE =' ,tdiff, tdiff2)
        if os.path.exists(self.case_path+"/compiled_lib0.tar"):
            self.factorymod0 = tvm.runtime.load_module(self.case_path+"/compiled_lib0.tar")
            self.gmod0 = GraphModule(self.factorymod0["default"](dev))
            outs0 = self.run_gmod(graph_mod1,loaded_params)
            tdiff = 0.
            tdiff2 = 0.
            for (ro,o) in zip(outs0,outs1):
                diff  =  self.MSE(ro,o)
                tdiff = max(tdiff,diff)
                diff  =  self.SE(ro,o)
                tdiff2 = max(tdiff2, diff)
            print('prune MSE,SE =' ,tdiff, tdiff2)
            tdiff = 0.
            tdiff2 = 0.
            for (ro,o) in zip(outs0,outs5):
                diff  =  self.MSE(ro,o)
                tdiff = max(tdiff,diff)
                diff  =  self.SE(ro,o)
                tdiff2 = max(tdiff2, diff)
            print('total MSE,SE =' ,tdiff, tdiff2)
        if 'transformer' in self.case_path:
            # define a fc layer
            fclayerw = np.random.normal(loc=0,scale=1,size=(64 , 10000))
            outs5 = np.swapaxes(outs5[0],0,1)
            outs5 = np.matmul(outs5, fclayerw)
            nb, nsen = outs1.shape[0], outs1.shape[1]
            totalpredicts = 0
            correct = 0
            inda = outs5[1][0][::-1][0:topk]
            indb = outs5[1][0][::-1][0:topk]
            indb[1:]= indb[:-1]
            print('the distance between classes is',np.mean(np.abs((indb-inda)/inda)))
            # for i in range(nb):
            #     for j in range(nsen):
            #         inda = np.argsort(outs5[i][j])[::-1][0:5]
            #         indb = np.argsort(outs0[i][j])[::-1][0:5]
            #         # print(inda,indb)
            #         totalpredicts += 5
            #         correct +=  np.count_nonzero(np.equal(inda,indb))
            # print('top5 incorrect rate', (totalpredicts - correct)/totalpredicts)
            # totalpredicts = 0
            # correct = 0
            # for i in range(nb):
            #     for j in range(nsen):
            #         inda = np.argsort(outs5[i][j])[::-1][0:topk]
            #         indb = np.argsort(outs0[i][j])[::-1][0:topk]
            #         # print(inda,indb)
            #         totalpredicts += topk
            #         correct +=  np.count_nonzero(np.equal(inda,indb))
            # print('top 10 incorrect rate', (totalpredicts - correct)/totalpredicts)
        else:
            print(outs5[0].shape)
            inda = np.array(sorted(outs5[0][0].tolist(),reverse=1)[0:topk])

            indb = np.array(sorted(outs5[0][0].tolist(),reverse=1)[0:topk])
            indb[1:]= indb[:-1]
            print(inda,indb)
            print('The distance between classes is',np.mean(np.abs((indb-inda)/inda)))
        exit()
            # inda = np.argsort(outs5[1][0])[::-1][0:topk]
            # indb = np.argsort(outs1[1][0])[::-1][0:topk]
            # totalpredicts = 0.0
            # correct = 0.0
            # totalpredicts += float(topk)
            # correct +=  np.count_nonzero(np.equal(inda,indb))
            # print(inda,indb)
            # print(self.MSE(outs5[1][0][inda],outs1[1][0][indb]))
            # print(f'top{topk} incorrect rate', (totalpredicts - correct)/totalpredicts)



    def replay_error_int(self,report:Queue=None):
        #!!!
        self.oDtype = self.Dtype
        from torch.nn import Module
        path_params = os.path.join(self.case_path, 'inputs.npz')
        graph_mod1 = self.gmod1
        graph_mod5 = self.gmod5
        # with np.load(path_params) as f:
        #     loaded_params = dict(f.items())
        # arr = list(loaded_params.values())[0]
        # print('loaded_params', arr.dtype)
        inputarr = dict()
        main_fn = self.mod['main']
        plen = len(main_fn.params)

        Input_sizes = self.generate_inputs_nameshape(main_fn,plen)
        inkeys = list(Input_sizes.keys())
        t_sizes = list(Input_sizes.values())
        # a,b = np.vsplit(np.reshape(arr, t_sizes[0]),[langbatch])
        a = (np.random.uniform(size=(1,384))* dictnums).astype('int64')

        if self.oDtype == 'int64':
            if 'int' in str(a.dtype):
                inputarr[inkeys[0]] = a
            else:
                inputarr[inkeys[0]] = (a*dictnums).astype('int64')
            inputarr[inkeys[1]] = np.ones(shape =t_sizes[1]).astype(self.oDtype)
            # inputarr[inkeys[1]][0][100] = 0
            # inputarr[inkeys[1]][0][510:] = 1
            # inputarr[inkeys[2]] = np.zeros(shape =t_sizes[2]).astype(self.oDtype)

        outs5 = self.run_gmod(graph_mod5,inputarr)
        outs1 = self.run_gmod(graph_mod1,inputarr)


        tdiff = 0.
        tdiff2 = 0.
        for (ro,o) in zip(outs1,outs5):
            diff  =  self.MSE(ro,o)
            print(diff)
            tdiff = max(tdiff,diff)
            diff  =  self.SE(ro,o)
            tdiff2 = max(tdiff2, diff)
        print('optimization MSE,SE =' ,tdiff, tdiff2)

        inda = np.argsort(outs5[0][0])[::-1][0:topk]
        indb = np.argsort(outs1[0][0])[::-1][0:topk]
        totalpredicts = 0.0
        correct = 0.0
        totalpredicts += float(topk)
        correct +=  np.count_nonzero(np.equal(inda,indb))
        # print(inda,indb)
        # print(self.MSE(outs5[1][0][inda],outs1[1][0][indb]))
        print(f'top{topk} incorrect rate', (totalpredicts - correct)/totalpredicts)
        import onnx
        # !!! renew path
        # onnx_model = onnx.load("/root/.cache/sparsezoo/321dac91-0d0a-4815-bc95-9ec13ef997aa/deployment/model.onnx")
        # onnx_model = onnx.load('/root/.cache/sparsezoo/9887ee48-6cd3-4885-8105-51213ac2ff6f/deployment/model.onnx')
        onnx_model = onnx.load('/root/.cache/sparsezoo/9273788e-f3a8-41f8-8e93-e3249621866c/deployment/model.onnx')
        # onnxinput1 = np.ones(shape=(2,384)).astype('int64')
        # onnxinput2 =  np.zeros(shape=(2,384)).astype('int64')
        # onnxinput1[0][0:50] = inputarr[inkeys[0]][0]
        # onnxinput1[1][0:50] = inputarr[inkeys[0]][1]
        # onnxinput2[0][0:50] = inputarr[inkeys[1]][0]
        # onnxinput2[1][0:50] = inputarr[inkeys[1]][1]
        # onnxinput2 = onnxinput2 %50264
        # onnxinput1 = onnxinput1%50264
        onnxinput1 = inputarr[inkeys[0]]
        onnxinput2 = inputarr[inkeys[1]]
        # onnxinput3 = inputarr[inkeys[2]]
        outs0 = get_ort_out(onnx_model,[onnxinput1,onnxinput2])#onnxinput3
        # temp00 =np.zeros(shape=(2,50)).astype('int64')
        # temp01 =np.zeros(shape=(2,50)).astype('int64')
        # temp00[0][:] = outs0[0][0][0:50]
        # temp00[1][:] = outs0[0][1][0:50]

        # temp01[0][:] = outs0[1][0][0:50]
        # temp01[1][:] = outs0[1][1][0:50]
        # outs0 = [temp00,temp01]
        tempdiff1 = 0
        tempdiff2 = 0
        for i, (ro, o) in enumerate(zip(outs0, outs1)):
                diff  =  self.MSE(ro,o)
                tempdiff1 = max(diff, tempdiff1)
                diff = self.SE(ro,o)
                tempdiff2 = max(diff, tempdiff2)
        print('framework error,mse,se are',tempdiff1,tempdiff2)

        tempdiff1 = 0
        tempdiff2 = 0
        for i, (ro, o) in enumerate(zip(outs0, outs5)):
                diff  =  self.MSE(ro,o)
                tempdiff1 = max(diff, tempdiff1)
                diff = self.SE(ro,o)
                tempdiff2 = max(diff, tempdiff2)
        print('total error,mse,se are',tempdiff1,tempdiff2)
        totalpredicts = 0.0
        correct = 0.0
        inda = np.argsort(outs1[0][0])[::-1][0:topk]
        indb = np.argsort(outs0[0][0])[::-1][0:topk]

        totalpredicts += float(topk)
        correct +=  np.count_nonzero(np.equal(inda,indb))
        # print(inda,indb)
        # print(self.MSE(outs5[1][0][inda],outs1[1][0][indb]))
        print(f'total top{topk} incorrect rate', (totalpredicts - correct)/totalpredicts)

    def replay_error_yolov8(self,report:Queue=None):
        import onnx
        graph_mod1 = self.gmod1
        graph_mod5 = self.gmod5
        # !!! renew path
        onnx_model = onnx.load("/root/.cache/sparsezoo/e37ce71a-d72f-42ec-b87f-21994e8fb6df/deployment/model.onnx")


        # with np.load(path_params) as f:
        #     loaded_params = dict(f.items())
        from .yolov8_utils import get_input
        filepath = '/home/zichaox/tvm/models/ils2012'
        files = os.listdir(filepath)
        nums = 0.0
        zeropredict = 0.0
        shortpredict = 0.0
        wrong = 0.0
        correct = 0.0
        locerrors = 0.0
        sizeerrors = 0.0
        totalnums = 0.0
        opt_errors = []
        frame_errors = []
        total_errors = []
        for fi in files[50:1050]:
            fi_d = os.path.join(filepath, fi)
            if not os.path.isdir(fi_d):
                print('-'*30, ' handle', fi_d)
                input1 = get_input(fi_d)
                input2 = input1.copy()
                input = np.concatenate((input1,input2),axis=0)

                loaded_params2 = dict()
                loaded_params2['images'] = input

                outs5 = self.run_gmod(graph_mod5,loaded_params2)
                outs1 = self.run_gmod(graph_mod1,loaded_params2)
                outs0 = get_ort_out(onnx_model,[input])

                tempdiff1 = 0
                for i, (ro, o) in enumerate(zip(outs1, outs5)):
                        diff  =  self.MSE(ro,o)
                        tempdiff1 = max(diff, tempdiff1)
                # print('opt error,mse,se are',tempdiff1,tempdiff2)
                opt_errors.append(tempdiff1)
                tempdiff1 = 0
                for i, (ro, o) in enumerate(zip(outs0, outs1)):
                        diff  =  self.MSE(ro,o)
                        tempdiff1 = max(diff, tempdiff1)
                # print('framework error,mse,se are',tempdiff1,tempdiff2)
                frame_errors.append(tempdiff1)
                tempdiff1 = 0
                for i, (ro, o) in enumerate(zip(outs0, outs5)):
                        diff  =  self.MSE(ro,o)
                        tempdiff1 = max(diff, tempdiff1)
                # print('total error,mse,se are',tempdiff1,tempdiff2)
                total_errors.append(tempdiff1)
                #out1[0],out[-1]  -> (Tensor[(2, 116, 8400)(2, 144, 80, 80), (2, 144, 40, 40), \
                # (2, 144, 20, 20),(2, 32, 8400), (2, 32, 160, 160)
                totalnums += 1
                from .yolov8_utils import get_yolo_result
                result5 = get_yolo_result([outs5[0],outs5[-1]])
                result1 = get_yolo_result([outs1[0],outs1[-1]])
                result0 = get_yolo_result([outs0[0],outs0[-1]])
                if len(result0) != 0:
                    nums+=1.0
                    if len(result5) == 0 :
                        zeropredict += 1
                    elif len(result5)<len(result0):
                        shortpredict += 1
                    else:
                        if result5[0][4]!= result0[0][4]:
                            wrong+=1.0
                            print('results5 are:',result5[0][4],result5[0][5])
                            print('results0 are:',result0[0][4])
                            if len(result1)!=0:
                                print('results1 are:',result1[0][4])
                        else:
                            correct += 1
                            locerrors += max(self.generalMSE(result5[0][0],result0[0][0]),self.generalMSE(result5[0][1],result0[0][1]))
                            sizeerrors += self.generalMSE(result5[0][3]*result5[0][2],result0[0][2]*result0[0][3])
        print('totalnums', totalnums)
        print('valid samples',nums)
        print('incorrect rate', wrong/nums)
        print('no detection rate', zeropredict/nums)
        print('short detection rate', shortpredict /nums)
        print('mean locerrors',locerrors/correct)
        print('mean size',sizeerrors/correct)
        errordict = dict()
        errordict['0']= np.array(opt_errors)
        errordict['1']= np.array(frame_errors)
        errordict['2']= np.array(total_errors)

        np.savez(self.case_path+'/trace_ils.npz',**errordict)
    def fuzzps(self, mod:IRModule=None,):
        self.oDtype = self.Dtype
        if 'int64' in self.Dtype:
            self.Dtype ='float32'
            intflag = 1
        else:
            intflag = None

        #Input_size
        print('self.dnn', self.dnn)
        main_fn = self.mod['main']
        plen = len(main_fn.params)
        Input_sizes = self.generate_inputs_nameshape(main_fn,plen)
        # initial weight
        diff = 0

        # begin
        for tempkey, Input_size in Input_sizes.items():  # for each params
            self.Input_size = Input_size
            self.tempkey = tempkey
            if 'transformer' in self.case_path:
                t_sizes = list(Input_sizes.values())
                inkeys = list(Input_sizes.keys())
                if 'int' in self.oDtype:
                    self.Input_size = (t_sizes[0][0]+t_sizes[1][0],t_sizes[0][1]) # int input
                else:
                    self.Input_size = (t_sizes[0][0]+t_sizes[1][0],t_sizes[0][1],t_sizes[0][2]) # float input
            # inputarr1 = np.random.randint(0,1,size =t_sizes[1]).astype(self.Dtype)
            print('temp fuzzing params: ',self.tempkey,self.Input_size)
            keep_dir = False
            def savearr(arr):
                    inputarr = dict()
                    if 'transformer' in self.case_path:
                        a,b = np.vsplit(np.reshape(arr, self.Input_size),[langbatch])
                        if self.oDtype == 'int64':
                            if 'int' in str(a.dtype):
                                inputarr[inkeys[0]] = a
                            else:
                                inputarr[inkeys[0]] = (a*dictnums).astype('int64')
                            inputarr[inkeys[1]] = np.zeros(shape =t_sizes[1]).astype(self.oDtype)
                            inputarr[inkeys[1]][0][0:30] = 1

                            # inputarr[inkeys[1]] = inputarr1
                        else:
                            inputarr['input0'] = a
                            inputarr['input1'] = b
                    else:
                        # inputarr[self.tempkey] = (np.reshape(arr, self.Input_size)*255).astype('uint8')
                        inputarr[self.tempkey] = np.reshape(arr, self.Input_size)
                    print('find accuracy bug')
                    path_params = os.path.join(self.case_path, 'inputs.npz')
                    if self.frame is False:
                        np.savez(path_params, **inputarr)
                        print(path_params)
            def fuzzfn(arr:np.array):# input:list of np factorymod1: tvm.runtime.Module,factorymod5: tvm.runtime.Module,
                inputarr = dict()
                if 'transformer' in self.case_path:
                    a,b = np.vsplit(np.reshape(arr, self.Input_size),[langbatch])
                    if self.oDtype == 'int64':
                        if 'int' in str(a.dtype):
                            inputarr[inkeys[0]] = a
                        else:
                            inputarr[inkeys[0]] = (a*dictnums).astype('int64')
                        inputarr[inkeys[1]] = np.zeros(shape =t_sizes[1]).astype(self.oDtype)
                        inputarr[inkeys[1]][0][0:30] = 1

                        # inputarr[inkeys[2]] = np.zeros(shape =t_sizes[1]).astype(self.oDtype)
                        if float(np.count_nonzero(a))/float(np.size(a))<0.3:
                            return 0
                        # inputarr[inkeys[1]] = (b*dictnums).astype(self.Dtype)
                    else:
                        inputarr['input0'] = a
                        inputarr['input1'] = b
                else:
                    # inputarr[self.tempkey] = (np.reshape(arr, self.Input_size)*255).astype('uint8')
                    inputarr[self.tempkey] = np.reshape(arr, self.Input_size)

                self.eparams = inputarr
                if self.frame and self.usetorch :
                    self.torchm = torch.load(self.case_path +'/model_scripted.pt').float().eval()
                    if len(inputarr) == 1:
                        outs1 = run_tmod(self.torchm,torch.from_numpy(list(inputarr.values())[0]).float())
                    else:
                        outs1 = run_tmod(self.torchm,[torch.from_numpy(list(inputarr.values())[0]).float(),\
                                                      torch.from_numpy(list(inputarr.values())[1]).float()])
                elif self.frame and self.useonnx :
                    if len(inputarr) == 1:
                        outs1 = get_ort_out(self.onnx_model,[list(inputarr.values())[0]])
                    else:
                        outs1 = get_ort_out(self.onnx_model,[list(inputarr.values())[0],\
                                                             list(inputarr.values())[1]])
                else:
                    # if 'prune' in self.case_path:
                    #     # outs1  =self.run_gmod(self.gmod0,inputarr)
                    # else:
                    outs1 = self.run_gmod(self.gmod1,inputarr)
                if self.frame:
                    outs5 = self.run_gmod(self.gmod1,inputarr)
                else:
                    outs5 = self.run_gmod(self.gmod5,inputarr)
                del inputarr
                tempdiff = 0
                for i, (ro, o) in enumerate(zip(outs1, outs5)):
                    diff , flag =  self.MRE(ro,o)
                    if flag>0:
                        self.handle_exit(flag,diff)
                    tempdiff = max(diff, tempdiff)
                return -tempdiff

            if(self.fuzzmode == 'CD'):
                # dump norm output diff
                if mod is not None:
                    mod = mod
                else:
                    mod = self.mod
                self.tempkey = main_fn.params[0].name_hint
                Input_size =list(self.generate_inputs_nameshape(main_fn,1).values())[0]

                #  Inputs_num -> Input_size
                ninput = np.random.normal(0.5, 1, size = Input_size)
                nndiff,_ = self.rundiff_changew(main_fn, [ninput] )
                # dump curthy output diff
                cinput = np.random.standard_cauchy(size = Input_size)
                cdiff,_ = self.rundiff_changew(main_fn, [cinput] )
                # Prepare ab/rel error
                # if(nndiff!=nndiff or cdiff != cdiff):
                #         print('nan in run')
                # if (nndiff ==np.inf or cdiff == np.inf):
                #             print('inf in run')
                seed = np.random.uniform(self.Boundlow,self.Boundhigh, size=Input_size)

                # inital imageseed
                prediff,self.params = self.rundiff_changew(main_fn,[seed] )
                iter = 1
                inputarr = seed*np.random.uniform(0.95,1.05,size=Input_size)
                self.seedq.put(inputarr)
                newdiff     = prediff
                self.diff = prediff
                # print('193')
                while( newdiff <= prediff ):
                    if(iter==5 and self.diff < 1e-10):
                        print(f'optimization seems true : case {self.case_id}')
                        break
                    if(iter>=10 and self.diff < 2e-5):
                        print(f'failed to find potensial fuzz input for : case {self.case_id}')
                        break
                    inputarr = seed*np.random.uniform(0.95,1.05,size=Input_size)
                    newdiff, inparams = self.rundiff_changew(main_fn,[inputarr] )
                    if newdiff >prediff:
                        self.params = inparams
                        self.diff = newdiff
                    iter+=1
                    # print(newdiff)
                with open(self.case_path+'/Fuzzlog','w') as fp:
                        fp.write('ciff:' +str(cdiff)+'\n')
                        fp.write('niff:' +str(nndiff)+'\n')
                self.seedq.put(inputarr.copy())
                # fuzz with 6 tactics
                tac = generaltactics()
                func1 = [tac.f1,tac.f0,tac.f2,tac.f0,tac.f3,tac.f0,tac.f4,tac.f0]  # confuse and diffuse
                func2 = [tac.f0,tac.f4,tac.f2,tac.f1,tac.f5,tac.f3,]  # confuse and diffuse
                lastdiff = self.diff  # last biggest diff
                seedinitdiff = 0
                if inputarr is not None:
                    lastinputarr = inputarr
                else:
                    lastinputarr = np.random.uniform(self.Boundlow,self.Boundhigh, size=Input_size)
                oldestdiff = self.diff
                with open(self.case_path+'/Fuzzlog','w') as fp:
                    fp.write(self.fuzzmode+'diff:' +str(oldestdiff)+'\n')
                outermost_cycle =0
                outermost_diff = self.diff
                diffuseorder = 'C'
                func = func2
                # main process
                # while outermost_cycle !=4 :         # begin order loop
                trys =0
                for i in range(20):
                    self.usedseed[i] = np.random.laplace(size=Input_size).astype(self.Dtype)
                    self.seedq.put(self.usedseed[i])
                while not self.seedq.empty():

                    trys += 1
                    if trys > self.Maxfuzztrytimes:
                        break
                    imageseed = self.seedq.get()
                    #!!! MUTATE  SEED SELECTION
                    # update self.usedseed
                    if len(self.usedseed)==20:
                        self.usedseed[random.randint(0,19)]=imageseed
                    else:
                        self.usedseed.append(imageseed)
                    # turn tatics
                    for mutateseed in func:
                        last_mutatetimes = self.Lastfuzztimes
                        # mutate under the same tactic
                        while(last_mutatetimes!=0):
                            inputarr = mutateseed(imageseed, order= diffuseorder)
                            diff = self.rundiff(main_fn,[inputarr] )

                            if(diff>lastdiff):
                                last_mutatetimes = self.Lastfuzztimes
                                lastdiff = diff
                                lastinputarr = inputarr
                            else:
                                last_mutatetimes -= 1
                    # print(inputarr.shape)
                    if (lastdiff > seedinitdiff * (1 + 8e-2)):
                        print('enter queue type 1: %.10f'%(lastdiff), seedinitdiff)
                        seedinitdiff = lastdiff
                        self.seedq.put(lastinputarr)
                        # print(lastinputarr.shape, self.usedseed[0].shape)
                        for i in range(10):
                            cr_imageseed, noneflag = self.cross(main_fn,lastinputarr,self.usedseed)
                            if(noneflag!= None):
                                self.usedseed[random.randint(0,19)]=cr_imageseed
                                self.seedq.put(cr_imageseed)

                    else :
                        if(np.random.random()<0.2 and diff > 0.9*lastdiff):
                            self.seedq.put(inputarr)
                            cr_imageseed, noneflag = self.cross(main_fn,inputarr,self.usedseed)
                            if(noneflag!= None):
                                self.seedq.put(cr_imageseed)
                            print('enter queue type 2: ',diff)
                            print('trys:',trys)
                    # print('283')
                    if ( lastdiff > 10*oldestdiff or lastdiff > 0.01):
                        # dump_arrs
                        savearr(retx)
                        if not os.path.exists((f'{self.case_path}/optirmod.txt')):
                            self.save_files()
                        if(lastdiff> 0.1):
                            return
                        with open(self.case_path+'/Fuzzlog','a') as fp:
                            fp.write(self.fuzzmode+'diff'+ str(lastdiff )+'\n')
                    # if lastdiff > outermost_diff:   # pre order loop
                    #     outermost_diff = lastdiff
                    #     outermost_cycle=0
                    # else:
                    #     outermost_cycle+=1
                    # if diffuseorder == 'C':
                    #         diffuseorder = 'F'
                    # if diffuseorder == 'F':
                    #         diffuseorder = 'C'      # finish order loop
                print('rel error: %.10f' %(lastdiff))
                while(not self.seedq.empty()):
                    self.seedq.get()
                self.diff = lastdiff
                inputarr = None
            elif self.fuzzmode == 'MCMC':
                # print('358')
                minimizer_kwargs = {"method":"L-BFGS-B"}
                if not self.seedq.empty():
                    x0 = self.seedq.get().flatten()
                else:
                    x0 = np.random.laplace(size=Input_size).astype(self.Dtype).flatten()
                # ret = basinhopping(fuzzfn, x0, minimizer_kwargs=minimizer_kwargs,
                #                 niter=200, callback=print_fun,)
                lw =boundarr(self.Boundlow,inputsize=self.Input_size)
                up =boundarr(self.Boundhigh,self.Input_size)
                ret = dual_annealing(fuzzfn, bounds=list(zip(lw, up)),maxiter=30, callback=print_fun)
                if -ret.fun> self.tolerance:
                    print('find')
                    exit()
                self.seedq.put(ret.x)
            elif self.fuzzmode == 'DEMC':
                if not self.seedq.empty():
                    x0 = self.seedq.get().flatten()
                else:
                    x0 = np.random.laplace(size=Input_size).astype(self.Dtype).flatten()

                minimizer_kwargs = {"method":"L-BFGS-B"}
                lw =boundarr(self.Boundlow,inputsize=self.Input_size)
                up =boundarr(self.Boundhigh,self.Input_size)
                bounds = list(zip(lw, up))
                print('bounds',bounds[0])
                print(x0.shape)
                ret = differential_evolution(fuzzfn, bounds=bounds,
                                seed=1, maxiter= 10,init='halton')
                print("global minimum: x , f(x) = " ,ret.x,',',ret.fun,'\n')

                # mcmc
                ret = basinhopping(fuzzfn, ret.x , minimizer_kwargs=minimizer_kwargs,stepsize = 0.02,
                                 niter=40, callback=print_fun)
                self.seedq.put(ret.x)
            elif self.fuzzmode == 'MEGA':
                if not self.seedq.empty():
                    x0 = self.seedq.get().flatten()
                else:
                    x0 = np.random.uniform(0,1,size=self.Input_size).astype(self.Dtype)

                # x0 = torch.from_numpy(x0).float()
                print(self.Input_size)
                lw =boundarr(self.Boundlow, inputsize=self.Input_size).astype(self.Dtype)
                up =boundarr(self.Boundhigh, inputsize=self.Input_size).astype(self.Dtype)
                bounds = list(zip(lw, up))
                seeds = dict()
                t0 =time.time()
                for i in range(speed1):
                    print('another de finding')
                    for retx, retf in simpleDE(fuzzfn,x0=x0.flatten(), bounds=bounds,its=speed2,normalflag=self.dnn,dtype= self.Dtype):#15
                        print('global minimum: x , f(x) = ',retx,-retf)
                    savearr(retx)
                    retf = -retf
                    seeds[str(retf)]= retx
                seeds = dict(sorted(seeds.items(), key=lambda item: float(item[0]),reverse=True))
                retf, retx = list(seeds.items())[0]
                retf = float(retf)
                t1 = time.time()
                print(retf, 'de using time',t1-t0)
                # savearr(retx)
                bretf, bretx = list(seeds.items())[0]
                bretf = float(bretf)
                bseeds = list(seeds.values())
                seeds.clear()
                savearr(bretx)
                if not os.path.exists((f'{self.case_path}/optirmod.txt')):
                    self.save_files()



                # --------------------- fuzz weight ------
                # if self.randweight:
                #     self.Boundlow = -5
                #     self.Boundhigh = 5
                #     self.eparams[self.tempkey]  = np.reshape(bretx,self.Input_size)
                #     continue
                for i in range(speed3):
                    print('another me finding')
                    if intflag:
                        enclen = 14
                    else:
                        enclen = 16
                    for retx, retf in genetic_algorithm(fuzzfn, x0 = bseeds, bounds=bounds,its=speed4,n_bits=enclen,\
                                                        n_pop=50, r_mut=1/16.0/len(bounds)*2,type= self.Dtype,intflag=intflag):#15
                        print('global minimum: x , f(x) = ',-retf)
                    retf = -retf
                    seeds[str(retf)]= retx

                seeds = dict(sorted(seeds.items(), key=lambda item: float(item[0]),reverse=True))
                retf, retx = list(seeds.items())[0]
                retf = float(retf)
                t2= time.time()
                print(retf,'me using time',t2-t1)
                if bretf<retf:
                    bretf = retf
                    bretx = retx
                if  bretf > self.tolerance:
                    print('find',bretf)
                savearr(bretx)
                print('total time',time.time()-t0)
                exit()

                # mcmc fuzzing -------------------------------
                '''
                print('-'*50, ' mcmc ','\n')
                retout={}
                # def anneal
                for i in range(3):
                    print('another mc finding')
                    print('boundsshape:', len(bounds),len(bounds[0]))
                    for retx2, retf2 in simulated_annealing(fuzzfn, bounds=bounds,x0 = bretx, its=150):
                        print('global minimum: x , f(x) = ',retx2,-retf2)
                    retf2 = -retf2
                    retout[str(retf2)]= retx2
                retout = dict(sorted(retout.items(), key=lambda item: float(item[0]),reverse=True))
                retf2, retx2 = list(retout.items())[0]
                retf2 = float(retf2)
                print(retf2,'mc using time',time.time()-t2)
                if  bretf < retf2:
                    bretx = retx2
                    bretf = retf2
                '''

                # f: basin-hopping
                # for mod in ['Powell','Nelder-Mead']:
                #     t0 = time.time()
                #     minimizer_kwargs = {"method":mod}
                #     ret = basinhopping(func = fuzzfn,x0= x0, minimizer_kwargs=minimizer_kwargs,
                #                     niter=20, callback=print_fun,)
                #     if -ret.fun > retf:
                #         if -ret.fun/retf>10 or -ret.fun>1e-2:
                #             print(mod,'using time', time.time()-t0)
                #         retf = -ret.fun
                #         retx = ret.x
                #     print(mod,'minimum',retf,retx)
                #     exit()
                #     mcmc: result better than before
                # f:anneal
                # def handle_threadmcmc():
                #     rets = dual_annealing(func=fuzzfn, initial_temp=10,bounds=list(zip(lw, up)),
                #                             maxiter=5,callback = print_fun2,
                #                             x0=retx ,no_local_search=True)# no_local_search=True
                #     retout[-rets.fun]= rets.x
                #     print('total anneal time',time.time()-t0 )
                # threadmcmc = Process(target=handle_threadmcmc)
                # threadmcmc.start()
                # threadmcmc.join(timeout=5)
                # threadmcmc.kill()

                # f:patch fuzz
                # def getnewshape(inputsize:List[int]):
                #     shape = self.Input_size
                #     modnum = 3 if len(shape)>3 else 4
                #     return [i % modnum  for i in shape]
                # newshape = getnewshape(self.Input_size)
                # print('newshape',newshape)
                # nx0 = x0.flatten()
                # self.x0 =x0
                # newsize = np.prod(newshape)
                # print('newsize',newsize)
                # originsize = np.prod(self.Input_size)
                # newsize = originsize % 64 if originsize<64 else 64
                # patchtensor = nx0[:int(newsize)]
                # print(patchtensor,x0)
                # minimizer_kwargs = {"method":"Powell"}
                # ret = basinhopping(func = fuzzfn2,x0= patchtensor, minimizer_kwargs=minimizer_kwargs,
                #                     niter=20, callback=print_fun,)

                # if os.path.exists(self.case_path+'/inputs.npz'):
                #     exit()
                if  bretf > self.tolerance:
                    print('find',bretf)
                    savearr(bretx)
                    # threadmcmc.kill()
                    print('total time',time.time()-t0)
                    exit()
                else:
                    print('no accuracy bug')
                    # ret = dual_annealing(fuzzfn, initial_temp=100,bounds=list(zip(lw, up)),maxiter=10,callback=print_fun2,
                    #                       x0 =retx,no_local_search=True)# no_local_search=True
                self.seedq.put(bretx)
                exit()# just find first input
            else:
                pass
            with open(self.case_path+'/Fuzzstate','a') as fp:
                fp.write('numerical error'+ str(bretf))
        return
'''
'Nelder-Mead' (see here)       x 4.11-4min>10min  -0.0007778525193955139  1.5 min*5  pure shape too slow
'Powell' (see here)           -0.00064    1.55min*5 x              too slow   best
'CG' (see here)               -0.00030      too slow
'BFGS' (see here)             -0.00028      require gradient
'Newton-CG' (see here)   x
'L-BFGS-B' (see here)  # 2min -0.00023          may be  best
'TNC' (see here)               0.00020          may be  ?
'COBYLA' (see here)      10mi  0.00066    not fit
'SLSQP' (see here)       20min 0.00084
'trust-constr'(see here)
'dogleg' (see here)
'trust-ncg' (see here)  x
'trust-exact' (see here)
'trust-krylov' (see here)
differential evolution ???????
mcmc                    ??????
anneal why more slow with the time goes on          ok!
'''
