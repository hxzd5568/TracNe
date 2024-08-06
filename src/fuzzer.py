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
from .mutate_utils import (
    generaltactics,
    simpleDE,
    simulated_annealing,
    genetic_algorithm,
)
from .base_utils import Checkor
from multiprocessing import Process, Queue
from scipy.optimize import basinhopping, differential_evolution, Bounds, dual_annealing
from threading import Thread
import torch
import re
import sys

sys.path.append("../")
TensorDict = Dict[str, np.ndarray]
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
speed1, speed2, speed3, speed4 = 2, 10, 2, 6  # 5, 24, 2, 16 for traditional programs
mcmcstorepoint = 0
Disabled_pass5 = [
    "SimplifyExpr"
]  # ['AlterOpLayout','ForwardFoldScaleAxis']#[ 'AlterOpLayout', 'CanonicalizeCast']
import time


def time_it(func):
    def inner():
        start = time.time()
        func()
        end = time.time()
        print("using time:{}secs".format(end - start))

    return inner


def boundarr(x, inputsize):
    arr = np.empty(shape=inputsize).flatten()
    arr.fill(x)
    return arr


def print_fun(x, f, accepted):
    print("at de minimum %.7f accepted %d" % (-f, int(accepted)), x)
    # pass


def print_fun2(x, f, context):
    print("at anneal minimum %.7f" % (-f), x)
    if f < 1e-9:
        return True


class MyTakeStep:
    def __init__(self, stepsize=0.5):
        self.stepsize = stepsize
        self.rng = np.random.default_rng()

    def __call__(self, x):
        s = self.stepsize
        x[0] += self.rng.uniform(-2.0 * s, 2.0 * s)
        x[1:] += self.rng.uniform(-s, s, x[1:].shape)
        return x


# another method
"""
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping
https://docs.scipy.org/doc/scipy-1.0.0/reference/optimize.html
https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution
"""

"""
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
"""


def remove_virtarget(ncode):
    rncode = re.sub("({virtual_.*?}:)", ":", ncode, count=0, flags=re.M | re.S)
    rncode = re.sub("(virtual_.*?->)", ") ->", rncode, count=0, flags=re.M | re.S)
    return rncode


def remove_primary(code):
    return re.sub("(, Primitive\=.*?->)", ") ->", code, count=0, flags=re.M | re.S)


class Fuzzer(Checkor):
    def __init__(
        self,
        path: str,
        case_id: str,
        low: float = -5,
        high: float = 5,
        fuseopsmax=64,
        params=None,
        fuzzmode: str = "MEGA",
        fuzzframe=False,
        optlevel=5,
    ):
        super().__init__(
            path=path,
            case_id=case_id,
            params=params,
            lowbound=low,
            highbound=high,
            fuseopsmax=fuseopsmax,
            fuzzmode=fuzzmode,
            optlevel=optlevel,
        )
        self.case_path = os.path.join(path + "/out", case_id)
        self.usedseed = [None] * 20
        self.Maxfuzztrytimes = 50
        self.Lastfuzztimes = 10
        self.seedq = queue.Queue()
        self.params = params
        self.tempname = None
        self.tempshape = None
        self.randweight = None
        self.usetorch = os.path.exists(self.case_path + "/model_scripted.pt")
        if self.usetorch:
            self.torchm = (
                torch.load(self.case_path + "/model_scripted.pt").float().eval()
            )
        self.useonnx = os.path.exists(self.case_path + "/model.onnx")
        if self.useonnx:
            import onnx

            self.onnx_model = onnx.load(self.case_path + "/model.onnx")
        self.torchm = None
        self.frame = fuzzframe

    def get_primfunc(self, opt_level, target="llvm"):  # i.e. get tvm.ir.module.IRModule
        target, target_host = tvm.target.Target.canon_target_and_host(target)
        print("self.fuseopsmax is:", self.fuseopsmax)
        if opt_level >= 2:
            with tvm.transform.PassContext(
                opt_level=opt_level,
                disabled_pass=Disabled_pass5,
                config={"relay.FuseOps.max_depth": self.fuseopsmax},
            ):
                prim_mod, _ = relay.optimize(self.mod, target)
                code = remove_virtarget(prim_mod.astext())
            return code
        else:
            with tvm.transform.PassContext(
                opt_level=opt_level,
                config={"relay.FuseOps.max_depth": self.fuseopsmax},
                required_pass=self.Required_pass1,
                disabled_pass=self.Disabled_pass,
            ):  # config={"relay.FuseOps.max_depth": 10}
                prim_mod, _ = relay.optimize(self.mod, target)
                code = remove_virtarget(
                    prim_mod.astext()
                )  # ,remove_virtarget(str(prim_mod))
            return code

    def rundiff(
        self,
        main_fn: relay.function.Function,
        arrs: List[np.array] = None,
        changeweight=False,
    ):  # input:list of np
        inputarr = self.eparams
        inputarr[self.tempkey] = arrs[0]
        self.eparams = inputarr
        outs1 = self.run_gmod(self.gmod1, inputarr)
        outs5 = self.run_gmod(self.gmod5, inputarr)
        tempdiff = 0
        for i, (ro, o) in enumerate(zip(outs1, outs5)):
            diff, flag = self.MRE(ro, o)
            if flag > 0:

                self.handle_exit(flag, diff)
            tempdiff = max(diff, tempdiff)
        return tempdiff

    def rundiff_changew(
        self, main_fn: relay.function.Function, arrs: List[np.array] = None
    ):  # input:list of np
        inputarr = self.generate_inputs(
            main_fn,
            arrs,
        )
        # tvm.contrib.graph_executor.GraphModule
        # GraphExecutorFactoryModule
        self.eparams = inputarr
        outs1 = self.run_gmod(self.gmod1, inputarr)
        outs5 = self.run_gmod(self.gmod5, inputarr)
        tempdiff = 0
        for i, (ro, o) in enumerate(zip(outs1, outs5)):
            diff, flag = self.MRE(ro, o)
            if flag > 0:
                self.handle_exit(flag, diff)
            tempdiff = max(diff, tempdiff)
        return tempdiff, inputarr

    def cross(self, main_fn: relay.function.Function, inputo: np, inputseed: np):
        input = inputo.copy()
        input2 = inputo.copy()
        originshape = inputo.shape
        # get input
        input = np.reshape(input, originshape)
        input2 = np.reshape(input2, originshape)
        origindiff = self.rundiff(main_fn, [input])
        input = input.flatten()
        input2 = input2.flatten()
        length = input.shape[0]
        sindex = random.randint(int(length / 3), int(length / 2))
        pattern = inputseed[random.randint(0, 9)]
        if pattern is None:
            print(inputseed)
            print("pattern is nonetype: ", pattern is None)

        if input is None:
            print(input)
            print("input is nonetype: ", input is None)
        input[sindex:] = pattern.flatten()[sindex:]
        input2[:sindex] = pattern.flatten()[:sindex]
        input = np.reshape(input, originshape)
        input2 = np.reshape(input2, originshape)
        # valide input
        diff = self.rundiff(main_fn, [input])
        diff2 = self.rundiff(main_fn, [input2])

        if diff > origindiff:
            return input, 1
        else:
            if diff2 > origindiff:
                return input2, 1
            else:
                return input2, 0

    def save_files(self):
        optprim_mod = self.get_primfunc(self.OPTLEVEL)
        unoptprim_mod = self.get_primfunc(1)
        with open(
            f"{self.case_path}/optirmod.txt", "w"
        ) as f:  # str less #[version = "0.0.5"]\n
            f.write(optprim_mod)
        with open(f"{self.case_path}/unoptirmod.txt", "w") as f:
            f.write(unoptprim_mod)

    def handle_exit(self, flag, diff):
        print("-------------------------------")
        if not os.path.exists((f"{self.case_path}/optirmod.txt")):
            self.save_files()
        if flag == 1:
            # dump_arrs
            if self.fuzzmode == "MEGA":
                path_params = os.path.join(self.case_path, "inputs.npz")
            else:
                path_params = os.path.join(self.case_path, self.fuzzmode + "inputs.npz")
            np.savez(path_params, **self.eparams)
            print("find enomous error", diff)
            import time

            print("time0", time.time())
            with open(self.case_path + "/Fuzzstate", "a") as fp:
                fp.write("correctness bug")
                exit()
        if flag == 3:
            with open(self.case_path + "/Fuzzstate", "a") as fp:
                fp.write("exception bug")
            exit()
        if flag == 10:
            with open(self.case_path + "/Fuzzstate", "a") as fp:
                fp.write("invalid result")
            exit()

    def fuzzps(
        self,
        mod: IRModule = None,
    ):
        # Input_size
        print("self.Boundlow,self.Boundhigh", self.Boundlow, self.Boundhigh)
        print("self.dnn", self.dnn)
        main_fn = self.mod["main"]
        plen = len(main_fn.params)
        Input_sizes = list(self.generate_inputs_nameshape(main_fn, plen).items())
        # initial weight
        diff = 0

        if os.path.exists(os.path.join(self.case_path, "params5.npz")):
            with np.load(os.path.join(self.case_path, "params5.npz")) as f:
                self.params5 = dict(f.items())
            path_params = os.path.join(self.case_path, "inputs.npz")
            with np.load(path_params) as f:
                self.params = dict(f.items())
                self.eparams = self.params
        # elif self.dnn:
        #     path_params = os.path.join(self.case_path, 'oinputs.npz')
        #     with np.load(path_params) as f:
        #         self.params  = dict(f.items())
        # else:
        _, inparams = self.rundiff_changew(
            main_fn,
        )
        self.params = inparams
        for i in range(10):
            newdiff, inparams = self.rundiff_changew(
                main_fn,
            )

            if diff < newdiff:
                diff = newdiff
                self.params = inparams
        if self.randweight:
            for i in range(10):
                newdiff, inparams = self.rundiff_changew(
                    main_fn,
                )
                if diff < newdiff:
                    self.params = inparams

        # path_params = './tests/out/5/inputs.npz'
        # with np.load(path_params) as f:
        #         self.params  = dict(f.items())
        #         self.eparams = self.params
        # begin
        for tempkey, Input_size in Input_sizes:  # for each params
            self.Input_size = Input_size
            self.tempkey = tempkey
            self.eparams = self.params
            print("temp fuzzing params: ", self.tempkey, self.Input_size)
            keep_dir = False

            def savearr(arr, label=None):
                if isinstance(arr, np.ndarray) or isinstance(arr, List):
                    inputarr = self.params
                    inputarr[self.tempkey] = np.reshape(arr, self.Input_size)
                else:
                    inputarr = arr
                print("find accuracy bug")
                if label is None:
                    filename = "inputs.npz"
                else:
                    filename = label + "inputs.npz"
                path_params = os.path.join(self.case_path, filename)
                np.savez(path_params, **inputarr)
                print(self.case_path)

            def fuzzfn(
                arr: np.array,
            ):  # input:list of np factorymod1: tvm.runtime.Module,factorymod5: tvm.runtime.Module,
                inputarr = self.eparams
                if not os.path.exists(os.path.join(self.case_path, "params5.npz")):
                    inputarr[self.tempkey] = np.reshape(arr, self.Input_size).astype(
                        self.Dtype
                    )
                    self.eparams = inputarr
                    outs1 = self.run_gmod(self.gmod1, inputarr)
                    outs5 = self.run_gmod(self.gmod5, inputarr)
                else:
                    self.eparams[self.tempkey] = np.reshape(arr, self.Input_size)
                    self.params5[self.tempkey] = np.reshape(arr, self.Input_size)
                    outs1 = self.run_gmod(self.gmod1, self.eparams)
                    outs5 = self.run_gmod(self.gmod5, self.params5)
                tempdiff = 0
                for i, (ro, o) in enumerate(zip(outs1, outs5)):
                    diff, flag = self.MRE(ro, o)
                    if flag > 0:
                        self.handle_exit(flag, diff)
                    tempdiff = max(diff, tempdiff)
                return -tempdiff

            if self.fuzzmode == "CD":
                # dump norm output diff
                if mod is not None:
                    mod = mod
                else:
                    mod = self.mod
                self.tempkey = main_fn.params[0].name_hint
                Input_size = list(self.generate_inputs_nameshape(main_fn, 1).values())[
                    0
                ]

                #  Inputs_num -> Input_size
                ninput = np.random.normal(0.5, 1, size=Input_size)
                nndiff, _ = self.rundiff_changew(main_fn, [ninput])
                # dump curthy output diff
                cinput = np.random.standard_cauchy(size=Input_size)
                cdiff, _ = self.rundiff_changew(main_fn, [cinput])
                # Prepare ab/rel error
                # if(nndiff!=nndiff or cdiff != cdiff):
                #         print('nan in run')
                # if (nndiff ==np.inf or cdiff == np.inf):
                #             print('inf in run')
                seed = np.random.uniform(self.Boundlow, self.Boundhigh, size=Input_size)

                # inital imageseed
                prediff, self.params = self.rundiff_changew(main_fn, [seed])
                iter = 1
                inputarr = seed * np.random.uniform(0.95, 1.05, size=Input_size)
                self.seedq.put(inputarr)
                newdiff = prediff
                self.diff = prediff
                # print('193')
                while newdiff <= prediff:
                    if iter == 5 and self.diff < 1e-10:
                        print(f"optimization seems true : case {self.case_id}")
                        break
                    if iter >= 10 and self.diff < 2e-5:
                        print(
                            f"failed to find potensial fuzz input for : case {self.case_id}"
                        )
                        break
                    inputarr = seed * np.random.uniform(0.95, 1.05, size=Input_size)
                    newdiff, inparams = self.rundiff_changew(main_fn, [inputarr])
                    if newdiff > prediff:
                        self.params = inparams
                        self.diff = newdiff
                    iter += 1
                    # print(newdiff)
                with open(self.case_path + "/Fuzzlog", "w") as fp:
                    fp.write("ciff:" + str(cdiff) + "\n")
                    fp.write("niff:" + str(nndiff) + "\n")
                self.seedq.put(inputarr.copy())
                # fuzz with 6 tactics
                tac = generaltactics()
                func1 = [
                    tac.f1,
                    tac.f0,
                    tac.f2,
                    tac.f0,
                    tac.f3,
                    tac.f0,
                    tac.f4,
                    tac.f0,
                ]  # confuse and diffuse
                func2 = [
                    tac.f0,
                    tac.f4,
                    tac.f2,
                    tac.f1,
                    tac.f5,
                    tac.f3,
                ]  # confuse and diffuse
                lastdiff = self.diff  # last biggest diff
                seedinitdiff = 0
                if inputarr is not None:
                    lastinputarr = inputarr
                else:
                    lastinputarr = np.random.uniform(
                        self.Boundlow, self.Boundhigh, size=Input_size
                    )
                oldestdiff = self.diff
                with open(self.case_path + "/Fuzzlog", "w") as fp:
                    fp.write(self.fuzzmode + "diff:" + str(oldestdiff) + "\n")
                outermost_cycle = 0
                outermost_diff = self.diff
                diffuseorder = "C"
                func = func2
                # main process
                # while outermost_cycle !=4 :         # begin order loop
                trys = 0
                for i in range(20):
                    self.usedseed[i] = np.random.laplace(size=Input_size).astype(
                        self.Dtype
                    )
                    self.seedq.put(self.usedseed[i])
                while not self.seedq.empty():

                    trys += 1
                    if trys > self.Maxfuzztrytimes:
                        break
                    imageseed = self.seedq.get()
                    #!!! MUTATE  SEED SELECTION
                    # update self.usedseed
                    if len(self.usedseed) == 20:
                        self.usedseed[random.randint(0, 19)] = imageseed
                    else:
                        self.usedseed.append(imageseed)
                    # turn tatics
                    for mutateseed in func:
                        last_mutatetimes = self.Lastfuzztimes
                        # mutate under the same tactic
                        while last_mutatetimes != 0:
                            inputarr = mutateseed(imageseed, order=diffuseorder)
                            diff = self.rundiff(main_fn, [inputarr])

                            if diff > lastdiff:
                                last_mutatetimes = self.Lastfuzztimes
                                lastdiff = diff
                                lastinputarr = inputarr
                            else:
                                last_mutatetimes -= 1
                    # print(inputarr.shape)
                    if lastdiff > seedinitdiff * (1 + 8e-2):
                        print("enter queue type 1: %.10f" % (lastdiff), seedinitdiff)
                        seedinitdiff = lastdiff
                        self.seedq.put(lastinputarr)
                        # print(lastinputarr.shape, self.usedseed[0].shape)
                        for i in range(10):
                            cr_imageseed, noneflag = self.cross(
                                main_fn, lastinputarr, self.usedseed
                            )
                            if noneflag != None:
                                self.usedseed[random.randint(0, 19)] = cr_imageseed
                                self.seedq.put(cr_imageseed)

                    else:
                        if np.random.random() < 0.2 and diff > 0.9 * lastdiff:
                            self.seedq.put(inputarr)
                            cr_imageseed, noneflag = self.cross(
                                main_fn, inputarr, self.usedseed
                            )
                            if noneflag != None:
                                self.seedq.put(cr_imageseed)
                            print("enter queue type 2: ", diff)
                            print("trys:", trys)
                    # print('283')
                    if lastdiff > 10 * oldestdiff or lastdiff > 0.01:
                        # dump_arrs
                        if not os.path.exists((f"{self.case_path}/optirmod.txt")):
                            self.save_files()
                        savearr(retx)
                        path_params = os.path.join(self.case_path, "inputs.npz")
                        self.params[self.tempkey] = inputarr
                        np.savez(path_params, **self.params)
                        if lastdiff > 0.1:
                            return
                        with open(self.case_path + "/Fuzzlog", "a") as fp:
                            fp.write(self.fuzzmode + "diff" + str(lastdiff) + "\n")
                    # if lastdiff > outermost_diff:   # pre order loop
                    #     outermost_diff = lastdiff
                    #     outermost_cycle=0
                    # else:
                    #     outermost_cycle+=1
                    # if diffuseorder == 'C':
                    #         diffuseorder = 'F'
                    # if diffuseorder == 'F':
                    #         diffuseorder = 'C'      # finish order loop
                print("rel error: %.10f" % (lastdiff))
                while not self.seedq.empty():
                    self.seedq.get()
                self.diff = lastdiff
                inputarr = None
            elif self.fuzzmode == "MCMC":
                # print('358')
                t0 = time.time()
                a = 1
                for i in self.Input_size:
                    a *= i
                print("shape size", a)
                if a > 40:
                    minimizer_kwargs = {"method": "L-BFGS-B"}
                else:
                    minimizer_kwargs = {"method": "Powell"}
                if not self.seedq.empty():
                    x0 = self.seedq.get().flatten()
                else:
                    x0 = np.random.laplace(size=Input_size).astype(self.Dtype).flatten()

                lw = boundarr(self.Boundlow, inputsize=self.Input_size)
                up = boundarr(self.Boundhigh, self.Input_size)

                ret = dual_annealing(
                    fuzzfn, bounds=list(zip(lw, up)), maxiter=1, callback=print_fun
                )
                savearr(ret.x, self.fuzzmode)
                print("best", ret.fun)
                if -ret.fun > self.tolerance:
                    print("find")
                print("MCMC using time", time.time() - t0)
                exit()
            elif self.fuzzmode == "random":
                maxdiff = -1
                for i in range(10):
                    if random.random() < 0.5:
                        x0 = (
                            np.random.uniform(size=Input_size)
                            .astype(self.Dtype)
                            .flatten()
                        )
                    else:
                        x0 = np.clip(
                            np.random.normal(size=Input_size)
                            .astype(self.Dtype)
                            .flatten(),
                            self.Boundlow,
                            self.Boundhigh,
                        )
                    inputarr = self.eparams
                    arr = x0
                    if not os.path.exists(os.path.join(self.case_path, "params5.npz")):
                        inputarr[self.tempkey] = np.reshape(
                            arr, self.Input_size
                        ).astype(self.Dtype)
                        self.eparams = inputarr
                        outs1 = self.run_gmod(self.gmod1, inputarr)
                        outs5 = self.run_gmod(self.gmod5, inputarr)
                    else:
                        self.eparams[self.tempkey] = np.reshape(arr, self.Input_size)
                        self.params5[self.tempkey] = np.reshape(arr, self.Input_size)
                        outs1 = self.run_gmod(self.gmod1, self.eparams)
                        outs5 = self.run_gmod(self.gmod5, self.params5)
                    tempdiff = 0
                    for i, (ro, o) in enumerate(zip(outs1, outs5)):
                        diff, flag = self.MRE(ro, o)
                        if flag > 0:
                            self.handle_exit(flag, diff)
                        tempdiff = max(diff, tempdiff)
                    if tempdiff > maxdiff:
                        maxdiff = tempdiff
                        maxinput = x0
                # replay
                loaded_params = self.eparams
                loaded_params[self.tempkey] = np.reshape(
                    maxinput, self.Input_size
                ).astype(self.Dtype)
                outs1 = self.run_gmod(self.gmod1, loaded_params)
                outs5 = self.run_gmod(self.gmod5, loaded_params)
                tdiff = 0.0
                for (ro, o) in zip(outs1, outs5):
                    diff = self.MSE(ro, o)
                    tdiff = max(tdiff, diff)
                print("expected, actual, self.MSE = ", tdiff)
                tdiff2 = 0.0
                for (ro, o) in zip(outs1, outs5):
                    diff = self.SE(ro, o)
                    tdiff2 = max(tdiff2, diff)
                print("expected, actual, self.SE = ", tdiff2)
                if self.dnn is None:
                    with open("./tests/out/error_all.txt", "a") as fp:
                        fp.write(
                            "\n"
                            + self.case_id
                            + ","
                            + str(self.fuzzmode)
                            + ","
                            + str(tdiff)
                            + ","
                            + str(tdiff2)
                        )
                exit()
            elif self.fuzzmode == "DEMC":
                t0 = time.time()

                x0 = np.random.laplace(size=Input_size).astype(self.Dtype).flatten()
                minimizer_kwargs = {"method": "L-BFGS-B"}
                lw = boundarr(self.Boundlow, inputsize=self.Input_size)
                up = boundarr(self.Boundhigh, self.Input_size)
                bounds = list(zip(lw, up))
                print("bounds", bounds[0])
                print(x0.shape)
                ret = differential_evolution(
                    fuzzfn, bounds=bounds, seed=1, maxiter=80, init="random"
                )
                # print("global minimum: x , f(x) = " ,ret.x,',',ret.fun,'\n')
                savearr(ret.x, self.fuzzmode)
                best = -ret.fun
                # mcmc
                ret = basinhopping(
                    fuzzfn, ret.x, stepsize=0.02, niter=80, callback=print_fun
                )
                if -ret.fun > best:
                    savearr(ret.x, self.fuzzmode)
                print("ret.fun , best", ret.fun, best)
                print("DEMC using time", time.time() - t0)

                exit()
                self.seedq.put(ret.x)
            elif self.fuzzmode == "MEGA":
                # get random seed
                maxdiff = -1
                for i in range(10):
                    if random.random() < 0.5:
                        x0 = (
                            np.random.uniform(size=Input_size)
                            .astype(self.Dtype)
                            .flatten()
                        )
                    else:
                        x0 = np.clip(
                            np.random.normal(size=Input_size)
                            .astype(self.Dtype)
                            .flatten(),
                            self.Boundlow,
                            self.Boundhigh,
                        )
                    inputarr = self.eparams
                    arr = x0
                    if not os.path.exists(os.path.join(self.case_path, "params5.npz")):
                        inputarr[self.tempkey] = np.reshape(
                            arr, self.Input_size
                        ).astype(self.Dtype)
                        self.eparams = inputarr
                        outs1 = self.run_gmod(self.gmod1, inputarr)
                        outs5 = self.run_gmod(self.gmod5, inputarr)
                    else:
                        self.eparams[self.tempkey] = np.reshape(arr, self.Input_size)
                        self.params5[self.tempkey] = np.reshape(arr, self.Input_size)
                        outs1 = self.run_gmod(self.gmod1, self.eparams)
                        outs5 = self.run_gmod(self.gmod5, self.params5)
                    tempdiff = 0
                    for i, (ro, o) in enumerate(zip(outs1, outs5)):
                        diff, flag = self.MRE(ro, o)
                        if flag > 0:
                            self.handle_exit(flag, diff)
                        tempdiff = max(diff, tempdiff)
                    if tempdiff > maxdiff:
                        maxdiff = tempdiff
                        maxinput = x0

                if not self.seedq.empty():
                    x0 = self.seedq.get().flatten()
                else:
                    x0 = self.eparams[self.tempkey]
                x0 = maxinput
                lw = boundarr(self.Boundlow, inputsize=self.Input_size)
                up = boundarr(self.Boundhigh, inputsize=self.Input_size)
                bounds = list(zip(lw, up))
                seeds = dict()
                t0 = time.time()
                for i in range(speed1):
                    print("another de finding")
                    for retx, retf in simpleDE(
                        fuzzfn,
                        x0=x0,
                        bounds=bounds,
                        its=speed2,
                        normalflag=self.dnn,
                        dtype=self.Dtype,
                    ):  # 15
                        print("global minimum: x , f(x) = ", retx, -retf)
                    retf = -retf
                    seeds[str(retf)] = retx
                if len(seeds) > 0:
                    seeds = dict(
                        sorted(
                            seeds.items(), key=lambda item: float(item[0]), reverse=True
                        )
                    )
                    retf, retx = list(seeds.items())[0]
                    retf = float(retf)
                    t1 = time.time()
                    print(retf, "de using time", t1 - t0)
                    bretf, bretx = list(seeds.items())[0]
                    bretf = float(bretf)
                    bseeds = list(seeds.values())
                    seeds.clear()
                    savearr(bretx)
                    if bretf > (self.tolerance):
                        if not os.path.exists((f"{self.case_path}/optirmod.txt")):
                            self.save_files()

                # --------------------- fuzz weight ------
                # if self.randweight:
                #     self.Boundlow = -5
                #     self.Boundhigh = 5
                #     self.eparams[self.tempkey]  = np.reshape(bretx,self.Input_size)
                #     continue
                # bseeds = [np.random.uniform(size=Input_size).astype(self.Dtype).flatten()]
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
                            type=self.Dtype,
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
                            type=self.Dtype,
                        ):  # 15
                            print("global minimum: x , f(x) = ", retx, -retf)
                            if bretf < -retf:
                                bretf = -retf
                                bretx = retx
                                savearr(retx)
                        retf = -retf
                        seeds[str(retf)] = retx
                seeds = dict(
                    sorted(seeds.items(), key=lambda item: float(item[0]), reverse=True)
                )
                retf, retx = list(seeds.items())[0]
                retf = float(retf)
                t2 = time.time()

                print(retf, "me using time", t2 - t1)
                if bretf > self.tolerance:
                    print("find", bretf)
                    print("total time", time.time() - t0)
                    exit()
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

                if bretf > self.tolerance:
                    print("find", bretf)
                    savearr(bretx)
                    # threadmcmc.kill()
                    print("total time", time.time() - t0)
                    exit()
                else:
                    print("no accuracy bug")
                    # ret = dual_annealing(fuzzfn, initial_temp=100,bounds=list(zip(lw, up)),maxiter=10,callback=print_fun2,
                    #                       x0 =retx,no_local_search=True)# no_local_search=True
                self.seedq.put(bretx)
                exit()  # just find first input
            else:
                pass
            with open(self.case_path + "/Fuzzstate", "a") as fp:
                fp.write("numerical error" + str(bretf))
        return


"""
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
"""
