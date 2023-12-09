import os
import tvm
import time
import itertools
import numpy as np
import tensorflow as tf
from tvm import relay, runtime
from tvm.contrib import graph_executor
from tvm.relay import data_dep_optimization as ddo
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)
from typing import Iterable, List, cast, Optional, Dict
import torch

import scipy.sparse as sp
import re
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("tensorflow will use experimental.set_memory_growth(True)")
    except RuntimeError as e:
        print("experimental.set_memory_growth option is not available: {}".format(e))


name = "huggingface/prunebert-base-uncased-6-finepruned-w-distil-squad"
batch_size = 1
seq_len = 128
target = "llvm"
measure_sparse = True
bs_r = 4
sparsity = 0.85
case_path = './dnn/out/transformerprune'
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
TensorDict = Dict[str, np.ndarray]
rpass = ['FastMath','FoldScaleAxis','SimplifyExpr','DivToMul']
Disabled_pass=['SimplifyExpr']

def run_func(func, params, x):
    with tvm.transform.PassContext(opt_level=5,required_pass=rpass):
        graph, lib, new_params = relay.build(func, "llvm", params=params)

    from tvm.contrib import graph_executor

    dev = tvm.cpu(0)
    dtype = "float32"
    m = graph_executor.create(graph, lib, dev)
    # set inputs
    m.set_input("data", tvm.nd.array(x.astype(dtype)))
    m.set_input(**new_params)
    # execute
    m.run()
    # get outputs
    tvm_output = m.get_output(0)
    return tvm_output.numpy()
def remove_virtarget(ncode):
    rncode = re.sub('({virtual_.*?}:)', ':', ncode,count=0, flags=re.M|re.S)
    rncode = re.sub('(virtual_.*?->)', ') ->', rncode,count=0, flags=re.M|re.S)
    return rncode
def get_primfunc( mod, opt_level,target='llvm'):# i.e. get tvm.ir.module.IRModule
        target, target_host = tvm.target.Target.canon_target_and_host(target)
        if opt_level==5:
            with tvm.transform.PassContext(opt_level=opt_level,required_pass=rpass):# config={"relay.FuseOps.max_depth": 10}
                prim_mod, _ = relay.optimize(mod, target)
                code = remove_virtarget(prim_mod.astext())
            return code
        else:
            with tvm.transform.PassContext(opt_level=opt_level,disabled_pass=['SimplifyExpr']):# config={"relay.FuseOps.max_depth": 10}
                prim_mod, _ = relay.optimize(mod, target)
                code = remove_virtarget(prim_mod.astext())
            return code
def run_relay_graph(mod, params, target, dev,):
    with relay.build_config(opt_level=5,required_pass=rpass):
        lib = relay.build(mod, target=target, params=params)
    # input_shape = shape_dict["input_1"]
    # dummy_data = np.random.uniform(size=input_shape, low=0, high=input_shape[1]).astype("int32")
    lib.export_library(case_path+"/compiled_lib5.tar")
    optprim_mod = get_primfunc(mod, 5)
    with open(f'{case_path}/optirmod.txt', 'w') as f: # str less #[version = "0.0.5"]\n
            f.write( optprim_mod)
    exit()
    m = graph_executor.GraphModule(lib["default"](dev))
    m.set_input(0, dummy_data)
    m.run()
    tvm_output = m.get_output(0)
    print(m.benchmark(dev, repeat=5, number=5))
    exit()#!!!
    return tvm_output
def build_workload(path,params=None, Disabled_pass=['SimplifyExpr']):
        with open(path, 'r') as f:
            mod = relay.parse(f.read())
        with tvm.transform.PassContext(opt_level=1, disabled_pass=Disabled_pass):
            lib1 = relay.build(mod, target, params=params)
        with tvm.transform.PassContext(opt_level=10):
            lib5 = relay.build(mod, target, params=params)
        return lib1, lib5

def random_bsr_matrix(M, N, BS_R, BS_C, density, dtype="float32"):
    Y = np.zeros((M, N), dtype=dtype)
    assert M % BS_R == 0
    assert N % BS_C == 0
    nnz = int(density * M * N)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, M, BS_R), range(0, N, BS_C))))
    assert candidate_blocks.shape[0] == M // BS_R * N // BS_C
    chosen_blocks = candidate_blocks[
        np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)
    ]
    for i in range(len(chosen_blocks)):
        r, c = chosen_blocks[i]
        Y[r : r + BS_R, c : c + BS_C] = np.random.normal(0, 0.2, (BS_R, BS_C))
    s = sp.bsr_matrix(Y, blocksize=(BS_R, BS_C))
    assert s.data.shape == (num_blocks, BS_R, BS_C)
    assert s.data.size >= nnz
    assert s.indices.shape == (num_blocks,)
    assert s.indptr.shape == (M // BS_R + 1,)
    return s.todense().astype('float32')

def random_sparse_bert_params(func, params, density, BS_R, BS_C):
    def deepcopy(param_dic):
        ret = {}
        for k, v in param_dic.items():
            ret[k] = tvm.nd.array(v.astype('float32'))
        return ret

    new_params = deepcopy(params)
    dense_weight_names = relay.analysis.sparse_dense._search_dense_op_weight(func)
    for item in dense_weight_names:
        name = str(item)
        shape = new_params[name].shape
        if shape[0] % BS_R == 0 and shape[1] % BS_C == 0:
            new_w = random_bsr_matrix(shape[0], shape[1], BS_R, BS_C, density)
            new_params[name] = tvm.nd.array(new_w)
    return new_params

def t32(arr):
    for k, v in arr.items():
        arr[k] = v.astype('float32')
def save_model( gmod1,gmod5,case_path):
    gmod1.export_library(case_path+"/compiled_lib1.tar")
    gmod5.export_library(case_path+"/compiled_lib5.tar")
def run_prune(mod, params, target, dev, bs_r, sparsity, gen_weights):
    mod, params = ddo.simplify_fc_transpose.convert(mod["main"], params)
    mod1 = mod
    if gen_weights:
        params = random_sparse_bert_params(mod, params, BS_R=bs_r, BS_C=1, density=1 - sparsity)
        print(list(params.values())[0].dtype)
        params0 = params.copy()
    with tvm.transform.PassContext(opt_level=1, disabled_pass=Disabled_pass):
            lib0 = relay.build(mod, target, params=params0)
            lib0.export_library(case_path+"/compiled_lib0.tar")
    unoptprim_mod = get_primfunc(mod, 1)
    with open(f'{case_path}/unoptirmod.txt', 'w') as f: # str less #[version = "0.0.5"]\n
            f.write(unoptprim_mod)
    mod, params = ddo.bsr_dense.convert(mod, params, (bs_r, 1), sparsity_threshold=0.8)
    with tvm.transform.PassContext(opt_level=1, disabled_pass=Disabled_pass):
            lib1 = relay.build(mod, target, params=params)

    with tvm.transform.PassContext(opt_level=5, required_pass=rpass):
            lib5 = relay.build(mod, target, params=params)
    save_model(lib1,lib5,case_path)
    print("Block Sparse Model with {blocksize}x1 blocks:".format(blocksize=bs_r))
    # path_params = os.path.join(case_path, 'params5.npz')
    # inputarr = dict()
    # for k, v in params.items():
    #     inputarr[k]=v.numpy()
    # np.savez(path_params, **inputarr)
    # print(list(inputarr.values())[0].shape)
    # path_params = os.path.join(case_path, 'inputs.npz')
    # inputarr = dict()
    # for k, v in params0.items():
    #     inputarr[k]=v.numpy()
    # np.savez(path_params, **inputarr)
    # print(list(inputarr.values())[0].dtype)
    optprim_mod = get_primfunc(mod, 5)
    with open(f'{case_path}/optirmod.txt', 'w') as f: # str less #[version = "0.0.5"]\n
            f.write( optprim_mod)
    pruneprim_mod = get_primfunc(mod, 1)
    with open(f'{case_path}/prunetirmod.txt', 'w') as f: # str less #[version = "0.0.5"]\n
            f.write( pruneprim_mod )
    # return run_relay_graph(mod, params,  target, dev)

# load model and compile with prune-opt/opt  options.
def build_pruneopt():
    # load mod
    with open(f'{case_path}/code.txt', 'r') as f:
            mod = relay.parse(f.read())

    path_params = os.path.join(case_path, 'inputs.npz')
    with np.load(path_params) as f:
            loaded_params = dict(f.items())
    # prune save split
    run_prune(mod, loaded_params, target, dev, bs_r, sparsity, True)


build_pruneopt()

# load torch model and summary.
def summmary():
    baseline_model = torch.load(case_path+'/model_scripted.pt')
    baseline_model.eval()
    src = torch.rand((10, 32, 64))
    tgt = torch.rand((20, 32, 64))
    input_data=[src, tgt]
    y = baseline_model(src, tgt)
    from torchsummary import summary
    for k,v in baseline_model.named_parameters():
         print(k,v.detach().numpy().shape)
    # print(summary(baseline_model,(10, 32, 64),(20, 32, 64)))
    print(y.detach().numpy().shape)
    print(np.testing.assert_allclose(y.detach().numpy(),tgt.detach().numpy()))
    assert(np.equal(y.detach().numpy(),tgt.detach().numpy()).all())

# summmary()


'''
newfunc = relay.data_dep_optimization.utils._run_opt_pass(
   mod,
   # layout, kernel_size, bsr_height, bsr_width, sparsity_threshold
   relay.transform._ffi_api.Conv2dToSparse2("NCHW", 3, 16, 1, 0)
)
sp_mod = tvm.ir.IRModule.from_expr(newfunc)
'''
