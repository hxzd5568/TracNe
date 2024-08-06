import tvm
from tvm import relay, runtime
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
import time
from .mutate_utils import (
    generaltactics,
    simpleDE,
    simulated_annealing,
    genetic_algorithm,
)


speed1, speed2, speed3, speed4 = 2, 10, 1, 15


TensorDict = Dict[str, np.ndarray]
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)


def boundarr(x, inputsize):
    arr = np.empty(shape=inputsize).flatten()
    arr.fill(x)
    return arr


def save_model(gmod1, gmod5, case_path):
    gmod1.export_library(case_path + "/compiled_lib1.tar")
    gmod5.export_library(case_path + "/compiled_lib5.tar")


def save_arr(params, case_path):
    print("type", type(list(params.values())[0]))
    inputarr = dict()
    for k, v in params.items():
        inputarr[k] = v.numpy()
    path_params = os.path.join(case_path, "oinputs.npz")
    np.savez(path_params, **inputarr)


def run_gmod(
    gmod: GraphModule, inputs: Dict[str, np.ndarray] = None
) -> List[np.ndarray]:
    ninputs = dict()

    for k, v in inputs.items():
        ninputs[k] = tvm.nd.array(v)
    a, b, c = (
        list(ninputs.values())[0],
        list(ninputs.values())[1],
        list(ninputs.values())[2],
    )
    gmod(a, b, c)
    return [c.numpy()]


class Fuzzer:
    def __init__(
        self,
        path,
        fmod1=None,
        fmod5=None,
        input=None,  # params1,params5
        lowbound: float = -5,
        highbound: float = 5,
    ):
        self.case_path = path
        self.Boundlow, self.Boundhigh = lowbound, highbound
        # self.params1 = params1
        # self.params5 = params5
        if fmod1 is not None:
            self.fmod1 = fmod1
            self.fmod5 = fmod5
            self.input = input
        else:
            self.fmod1 = tvm.runtime.load_module(self.case_path + "/compiled_lib1.tar")
            self.fmod5 = tvm.runtime.load_module(self.case_path + "/compiled_lib5.tar")
        self.gmod1 = self.fmod1
        self.gmod5 = self.fmod5
        self.dnn = None

    def handle_exit(self, flag, diff):
        exit()

    def MRE(self, y_true, y_pred):  # signal exposing numerical bugs
        flag = 0
        aexception = np.isinf(y_true).any() == 1 or np.isnan(y_true).any() == 1
        bexception = np.isinf(y_pred).any() == 1 or np.isnan(y_pred).any() == 1
        if aexception and not bexception:
            # flag = 3
            # print('y_true have inf\\nan:locating...')
            # self.locate_naninf('1')
            return 0, 0
        elif not aexception and bexception:
            # print('y_pred have inf\\nan:locating...')
            # flag = 3
            # self.locate_naninf('5')
            return 0, 0
        elif aexception and bexception:
            # flag = 10
            # print('y_true and y_pred have inf\\nan:locating...')
            # self.locate_naninf('1')
            # self.locate_naninf('5')
            return 0, 0
        else:
            pass
        d = np.abs(y_true.astype(np.float64) - y_pred)
        if self.dnn is None:
            relative_error = np.average(
                d / (np.abs(y_true).astype(np.float64) + 1e-8) * np.not_equal(y_true, 0)
                + np.equal(y_true, 0) * d
            )
        else:
            l_true = np.argmax(y_true[0])
            l_pred = np.argmax(y_pred[0])
            relative_error = np.average(
                d
                / (np.abs(y_true).astype(np.float64) + 1e-8)
                * np.abs(y_true)
                / np.mean(np.abs(y_true))
            )
        if (
            self.dnn is None
            and relative_error > 0.9
            or self.dnn is not None
            and l_true != l_pred
        ):
            print(
                "[fuzzer]unacceptable relative error is:", relative_error
            )  # y_true,y_pred
            flag = 1
        return relative_error, flag

    def fuzzps(
        self,
        mod: IRModule = None,
    ):
        def fuzzfn(
            arr: np.array,
        ):  # input:list of np factorymod1: tvm.runtime.Module,factorymod5: tvm.runtime.Module,
            inputarr = self.eparams
            inputarr[self.tempkey] = np.reshape(arr, self.Input_size)
            self.eparams = inputarr
            # inputarr = np.reshape(arr, self.Input_size)
            outs1 = run_gmod(self.gmod1, inputarr)
            outs5 = run_gmod(self.gmod5, inputarr)
            tempdiff = 0
            for i, (ro, o) in enumerate(zip(outs1, outs5)):
                diff, flag = self.MRE(ro, o)
                if flag > 0:
                    self.handle_exit(flag, diff)
                tempdiff = max(diff, tempdiff)
            return -tempdiff

        # Input_size
        path_params = os.path.join(self.case_path, "oinputs.npz")
        with np.load(path_params) as f:
            loaded_params = dict(f.items())
        keyss = list(loaded_params.keys())
        valuess = list(loaded_params.values())
        self.dtype = valuess[0].dtype

        plen = len(keyss)
        Input_sizes = [i.shape for i in valuess]
        # initial weight
        diff = 0

        if os.path.exists(os.path.join(self.case_path, "params5.npz")):
            with np.load(os.path.join(self.case_path, "params5.npz")) as f:
                self.params5 = dict(f.items())
            path_params = os.path.join(self.case_path, "inputs.npz")
            with np.load(path_params) as f:
                self.params = dict(f.items())
                self.eparams = self.params
        else:
            path_params = os.path.join(self.case_path, "oinputs.npz")
            with np.load(path_params) as f:
                self.params = dict(f.items())
                self.eparams = self.params
        # begin
        for Input_size in Input_sizes:  # for each params
            self.Input_size = Input_size
            self.tempkey = keyss[0]

            self.eparams = self.params
            print("temp fuzzing params: ", self.tempkey, self.Input_size)

            for i in range(1):
                lw = boundarr(self.Boundlow, inputsize=self.Input_size)
                up = boundarr(self.Boundhigh, inputsize=self.Input_size)
                bounds = list(zip(lw, up))
                seeds = dict()
                t0 = time.time()
                for i in range(speed1):
                    print("another de finding")
                    for retx, retf in simpleDE(
                        fuzzfn, bounds=bounds, its=speed2, dtype=self.dtype
                    ):  # 15
                        print("global minimum: x , f(x) = ", retx, -retf)
                    retf = -retf
                    seeds[str(retf)] = retx
                seeds = dict(
                    sorted(seeds.items(), key=lambda item: float(item[0]), reverse=True)
                )
                retf, retx = list(seeds.items())[0]
                retf = float(retf)
                t1 = time.time()
                print(retf, "de using time", t1 - t0)
                bretf, bretx = list(seeds.items())[0]
                bretf = float(bretf)
                bseeds = list(seeds.values())
                seeds.clear()

                # --------------------- fuzz weight ------
                # if self.randweight:
                #     self.Boundlow = -5
                #     self.Boundhigh = 5
                #     self.eparams[self.tempkey]  = np.reshape(bretx,self.Input_size)
                #     continue
                if self.dnn:
                    for i in range(speed3):
                        print("another me finding")
                        for retx, retf in genetic_algorithm(
                            fuzzfn,
                            x0=bseeds,
                            bounds=bounds,
                            its=speed4,
                            n_pop=50,
                            r_mut=1 / 16.0 / len(bounds) * 2,
                            type=self.dtype,
                        ):  # 15
                            print("global minimum: x , f(x) = ", -retf)
                        retf = -retf
                        seeds[str(retf)] = retx
                else:
                    for i in range(speed3):
                        print("another me finding")
                        for retx, retf in genetic_algorithm(
                            fuzzfn,
                            x0=bseeds,
                            bounds=bounds,
                            its=speed4,
                            r_mut=1 / 16.0 / len(bounds) * 2,
                            type=self.dtype,
                        ):  # 15
                            print("global minimum: x , f(x) = ", retx, -retf)
                        retf = -retf
                        seeds[str(retf)] = retx
                seeds = dict(
                    sorted(seeds.items(), key=lambda item: float(item[0]), reverse=True)
                )
                retf, retx = list(seeds.items())[0]
                retf = float(retf)
                t2 = time.time()
                print(retf, "me using time", t2 - t1)
                exit()

                # mcmc fuzzing -------------------------------
                """
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
                """

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

            else:
                pass
            with open(self.case_path + "/Fuzzstate", "a") as fp:
                fp.write("numerical error" + str(bretf))
        return
