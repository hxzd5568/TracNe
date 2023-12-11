import tensorflow as tf
import tvm
from tvm import relay,runtime
import numpy as np
import queue
import shutil
import os.path
import random
from tvm.contrib.debugger.debug_executor import GraphModuleDebug
from enum import IntEnum, auto
from tvm import transform, relay, parser, cpu, TVMError, IRModule
from tvm.contrib.graph_executor import GraphModule
from argparse import Namespace, ArgumentParser
from typing import Iterable, List, cast, Optional, Dict
from .mutate_utils import generaltactics, simpleDE
from .base_utils import Checkor
from multiprocessing import Process, Queue
from scipy.optimize import basinhopping, differential_evolution, Bounds, dual_annealing
from threading import Thread
import re
import os
import platform
import sys
from tvm.relay.transform import InferType, ToMixedPrecision, mixed_precision
from packaging import version as package_version
import pytest
import numpy as np
import tvm.testing
from tvm.contrib import graph_executor
from tvm.contrib.nvcc import have_fp16
from tvm.contrib import cudnn, utils
# from relay.utils.tag_span import _create_span, _set_span, _verify_structural_equal_with_span
import torch
from torch.nn import Module
from torch.nn import functional as F
import torchvision

from .fuzzer import Fuzzer
import trace
from .mutate_utils import generaltactics, simpleDE
# import pdb
# pdb.set_trace()
case_path = ''
TensorDict = Dict[str, np.ndarray]
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)
low, high = -5, 5
sys.setrecursionlimit(10000)
import sys
sys.path.append('../')
class DecisionGate(torch.nn.Module):
        def forward(self, x):
            if x.sum() > 0:
                return x
            else:
                return -x

class Cell(torch.nn.Module):
        def __init__(self, dg):
            super().__init__()
            self.dg = dg
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x, h):
            new_h = torch.tanh(self.dg(self.linear(x)) + h)
            return new_h, new_h

class RNNLoop(torch.nn.Module):
        """Pytorch RNNLoop module"""

        def __init__(self):
            super().__init__()
            x = torch.rand(10, 4, dtype=torch.float)
            h = torch.rand(10, 4, dtype=torch.float)
            self.cell = torch.jit.trace(Cell(DecisionGate()), (x, h))

        def forward(self, xs):
            h = torch.zeros(10, 4, dtype=torch.float)
            y = torch.zeros(10, 4, dtype=torch.float)
            for i in range(xs.size(0)):
                y, h = self.cell(xs[i], h)
            return y

def save_arr( params,case_path):
                    inputarr = params
                    path_params = os.path.join(case_path, 'oinputs.npz')
                    np.savez(path_params, **inputarr)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def MSE(y_true, y_pred,):  #precision along with  tf.keras.metrics.MeanRelativeError
    if np.isinf(y_true).any()==1 or np.isnan(y_true).any()==1:
        print('y_true have inf\\nan:locating...')
        #locate_naninf('1')
        # return 0

    if np.isinf(y_pred).any()==1 or np.isnan(y_pred).any()==1:
        print('y_pred have inf\\nan:locating...')
        # return 0
        #locate_naninf('5')
    else:
        pass
    relative_error = np.average(np.abs(y_true - y_pred).astype(np.float64)
                                    / (np.abs(y_true).astype(np.float64) + 1e-8))
    return relative_error

def list_ops(expr):
    """list_ops"""

    class OpLister(tvm.relay.ExprVisitor):
        """OpLister inherits from ExprVisitor"""

        def visit_op(self, op):
            if op not in self.node_set:
                self.node_list.append(op)
            return super().visit_op(op)

        def list_nodes(self, expr):
            self.node_set = {}
            self.node_list = []
            self.visit(expr)
            return self.node_list

    return OpLister().list_nodes(expr)

def assert_shapes_match(tru, est):
    """Verfiy whether the shapes are equal"""
    if tru.shape != est.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(tru.shape, est.shape))

def load_torchvision(model_name):
    """Given a model name, returns a Torchvision model in eval mode as well
    as an example input."""
    with torch.no_grad():
        if model_name.startswith("inception"):
            height = width = 299
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            height = width = 224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        input_shape = [1, 3, height, width]
        input_data = torch.randn(input_shape).float()
        for channel in range(3):
            input_data[:, channel] -= mean[channel]
            input_data[:, channel] /= std[channel]
        # tracer = trace.Trace(count=False, trace=True, ignoredirs=['/usr/lib/python3.7','/venv/apache-tvm-py3.7'])
        # fname  = getattr(torchvision.models, model_name)
        # print('fname', fname)
        # print(tracer.runfunc(getattr,torchvision.models, model_name))
        # print(torchvision.models.__dict__)
        # print(sys.getsizeof(torchvision.models))
        if model_name.startswith("googlenet"):
            model = getattr(torchvision.models, model_name)(pretrained=True, aux_logits=True)
        else:
            model = getattr(torchvision.models, model_name)(pretrained=True)
        model = model.float().eval()
        return model, [input_data]

def load_pretrainedmodels(model_name):
    """Given a model name, returns a pretrainedmodels.pytorch model in eval
    mode as well as an example input."""
    # pylint: disable=import-outside-toplevel
    import pretrainedmodels  # https://github.com/Cadene/pretrained-models.pytorch

    model = getattr(pretrainedmodels, model_name)().float().eval()
    input_shape = [1, *model.input_size]
    input_data = torch.rand(input_shape).float() * 256
    for channel in range(3):
        input_data[:, channel] -= model.mean[channel]
        input_data[:, channel] /= model.std[channel]
    return model, [input_data]

def normalkeys(strs):
    strs= strs.replace('::','')
    strs= strs.replace('_','')
    strs= strs.replace('.','')
    print(strs)
    return strs

def MRE( y_true, y_pred,): # signal exposing numerical bugs
    l_true = np.argmax(y_true[0])
    l_pred = np.argmax(y_pred[0])
    flag = 0
    aexception = np.isinf(y_true).any()==1 or np.isnan(y_true).any()==1
    bexception = np.isinf(y_pred).any()==1 or np.isnan(y_pred).any()==1
    if (aexception and not bexception) :
        return 0,0
    elif not aexception and bexception:
        return 0,0
    elif aexception and bexception:
        return 0,0
    else:
        pass
    relative_error = np.average( np.abs(y_true - y_pred).astype(np.float64)\
                                / (np.abs(y_true).astype(np.float64) + 1e-8) * np.not_equal(y_true, 0) )
    if relative_error > 1.0 or l_true!=l_pred: #!!!
        print("[torchfuzz] unacc relative error is:", relative_error)# y_true,y_pred
        flag = 1
    return relative_error,flag

def normalname(mod):  # return new_mod and if changed flag
    changeflag = []
    mod = re.sub('::','',mod,count=0,flags=re.S|re.M)
    pat = '(?P<value>%[a-zA-Z_]+[.a-zA-Z_0-9]+)'
    def update_internal(matched):
        changeflag.append(1)
        return matched.group('value').replace('_','').replace('.','')
    mod = re.sub(pat, update_internal, mod,count=0, flags=re.M|re.S)
    # pat2 = '(?P<value>%p)'
    # def changep(matched):
    #     changeflag.append(1)
    #     return matched.group('value').replace('p','n')
    # mod = re.sub(pat2, changep, mod,count=0, flags=re.M|re.S)
    return mod

def load_model(model_name):
    """Given a model name, returns a model as well as an example input."""
    if hasattr(torchvision.models, model_name):
        return load_torchvision(model_name)
    # pylint: disable=import-outside-toplevel
    try:
        import pretrainedmodels

        if hasattr(pretrainedmodels, model_name):
            return load_pretrainedmodels(model_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Please install pretrainedmodels.pytorch") from e
    raise RuntimeError("Model not supported")

def run_gmod(gmod: GraphModule, inputs: Dict[str, np.ndarray]=None) -> List[np.ndarray]:
    if not isinstance(gmod, GraphModuleDebug):
        gmod = GraphModule(gmod["default"](dev))
    if inputs is not None:
        gmod.run(**inputs)
    else:
        gmod.run()
    return [gmod.get_output(i).numpy() for i in range(gmod.get_num_outputs())]


def parse_model(
    model_name,
    input_data=None,
    custom_convert_map=None,
    rtol=1e-5,
    atol=1e-5,
    expected_ops=None,
    kind="graph",
    check_correctness=True,
    cpu_only=False,
    validate_structural_equal=True,
):
    """Assert that the output of a compiled model matches with that of its
    baseline."""
    input_data = [] if input_data is None else input_data
    custom_convert_map = custom_convert_map or {}
    expected_ops = expected_ops or []
    baseline_model, baseline_input = load_model(model_name)
    if torch.cuda.is_available():
        if isinstance(baseline_model, torch.nn.Module):
            baseline_model = baseline_model.cuda()
        baseline_input = [inp.cuda() for inp in baseline_input]

    with torch.no_grad():
        baseline_outputs = baseline_model(*[input.clone() for input in baseline_input])

    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.cpu().numpy(),)
    torch.save(baseline_model, case_path+'/model_scripted.pt')

    # Model class must be defined somewhere
    # baseline_model = torch.load(case_path+'/model_scripted.pt')
    baseline_model.eval()
    return baseline_model, baseline_input

def build_tvm(baseline_model,baseline_input):
    trace = torch.jit.trace(baseline_model, [input.clone() for input in baseline_input],strict=False)
    if isinstance(baseline_model, torch.nn.Module):
        trace = trace.float().eval()
        if torch.cuda.is_available():
            trace = trace.cuda()
        else:
            trace = trace.cpu()

    input_names = [f"input{idx}" for idx, _ in enumerate(baseline_input)]
    input_shapes = list(zip(input_names, [inp.shape for inp in baseline_input]))
    with tvm.testing.disable_span_filling():
        mod, params = relay.frontend.from_pytorch(trace, input_shapes)
    return baseline_model,mod, params

def run_tmod(
    model,
    input_data=None,
):
    """Assert that the output of a compiled model matches with that of its
    baseline."""
    global case_path
    with torch.no_grad():
        baseline_input = [input_data]
        baseline_outputs = model(*[input.clone() for input in baseline_input])

    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.cpu().numpy(),)
    return baseline_outputs

def verify_model_with_input(
    test_func,
    input_data,
    *,
    input_dict=None,
    custom_convert_map=None,
    rtol=1e-5,
    atol=1e-5,
    assert_shape_only=False,
    validate_structural_equal=True,
):
    """Generic function to generate and compare Pytorch and TVM output"""
    input_dict = input_dict or {}
    custom_convert_map = custom_convert_map or {}
    baseline_outputs = test_func(*input_data)
    trace = torch.jit.trace(test_func, [input.clone() for input in input_data])
    input_names = [f"input{idx}" for idx, _ in enumerate(input_data)]
    input_shapes = list(zip(input_names, [inp.shape for inp in input_data]))
    with tvm.testing.disable_span_filling():
        mod, params = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
    if validate_structural_equal:
        with tvm.testing.enable_span_filling():
            mod_with_span, _ = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
        assert tvm.ir.structural_equal(mod, mod_with_span, map_free_vars=True)

    with tvm.transform.PassContext(opt_level=3):
        for target in ["llvm", "cuda"]:
            if not tvm.runtime.enabled(target):
                continue
            dev = tvm.device(target, 0)
            lib = relay.build(mod, target=target, params=params)
            relay_model = graph_executor.GraphModule(lib["default"](dev))
            for name, value in input_dict.items():
                relay_model.set_input(name, value)
            relay_model.run()

            compiled_output = relay_model.get_output(0).numpy()
            assert_shapes_match(baseline_outputs, compiled_output)
            if assert_shape_only is False:
                # tvm.testing.assert_allclose(baseline_outputs, compiled_output, rtol=rtol, atol=atol)
                MSE(baseline_outputs, compiled_output)

def savetorch(model):
    global case_path
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(case_path+'/model_scripted.pt')

def gen_ir_module(model, inputs, use_parser_friendly_name=False):
    """Helper function to generate IRModule with meaningful source information"""

    trace = torch.jit.trace(model, inputs)
    input_names = ["input{}".format(idx) for idx, _ in enumerate(inputs)]
    input_shapes = list(zip(input_names, [inp.shape for inp in inputs]))
    mod, _ = relay.frontend.from_pytorch(
        trace,
        input_shapes,
        use_parser_friendly_name=use_parser_friendly_name,
    )
    return mod

def renewmodel(mod:tvm.IRModule, case_path:str)-> tvm.IRModule:
    modstr = mod.astext()
    modstr = normalname(modstr)
    if not os.path.exists(case_path):
        os.mkdir(case_path)
    with open(f'{case_path}/code.txt','w') as fp:
        fp.write(modstr)
    with open(f'{case_path}/code.txt', 'r') as f:
        return relay.parse(f.read())

def distinguish_check(arr :np.array):
    if np.std(arr)>0.01:
        return True
    else:
        return False

def test_forward_add():
    """test_forward_add"""
    torch.set_grad_enabled(False)
    input_shape = [10]
    class TinyModel(torch.nn.Module):

        def __init__(self):
            super(TinyModel, self).__init__()

            self.linear1 = torch.nn.Linear(10, 20)
            self.activation = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(20, 10)
            self.softmax = torch.nn.Softmax()

        def forward(self, x):
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            x = self.softmax(x)
            return x

    tinymodel = TinyModel()
    class Add1(Module):
        def forward(self, *args):
            return args[0] + args[0]

    class Add2(Module):
        def forward(self, x):
            return x + 1

    class Add3(Module):
        def forward(self, *args):
            ones = torch.ones(input_shape, dtype=torch.float)
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] + ones

    class Add4(Module):
        def forward(self, *args):
            ones = torch.ones([], dtype=torch.float)
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] + ones

    input_data = torch.rand(input_shape).float()
    model = tinymodel.float().eval()
    # print(type(torch.jit.script(model)))
    # f: get parameters
    # print(list(model.parameters())[0].size())
    # print(type(list(model.parameters())[0]))
    exit()
    # verify_model(model, input_data=input_data)
def boundarr(x,inputsize):
        arr = np.empty( shape = inputsize).flatten()
        arr.fill(x)
        return arr

def test_dnn(path,caseid, model_name = "inception_v3",):
    # 1. get input_shape range
    # 2. repare tmod, mod1, mod5
    # 3. fuzz
    '''
    def verify_mixed_precision_(
        mod: tvm.runtime.Module,
        mixed_precision_dtype="float16",
        rtol: float = 1e-10,
        atol: float = 0,
        keep_orig_output_dtype=True,
    ) -> tvm.runtime.Module:
        #result_fp32 = run_module(mod)
        mod = InferType()(mod)
        if not keep_orig_output_dtype:
            amp_mod = ToMixedPrecision(mixed_precision_dtype)(mod)
            # result_amp = run_module(amp_mod, mod_params)
        else:   # this path
            with tvm.transform.PassContext(
                config={"relay.ToMixedPrecision.keep_orig_output_dtype": True}
            ):
                amp_mod = ToMixedPrecision(mixed_precision_dtype)(mod)
        return amp_mod
    '''
    import time
    t0 = time.time()
    global case_path
    case_path = path+'/out/'+caseid
    if not os.path.exists(case_path):
        os.mkdir(case_path)
    torch.set_grad_enabled(False)
    global low, high

    if 'transformer'  in case_path or 'rnn' in case_path or 'vgg' in case_path:
        low,high = 0,1
    else :
        import pretrainedmodels
        low, high = pretrainedmodels.pretrained_settings[model_name.rstrip('f16')]['imagenet']['input_range']
        input_size = pretrainedmodels.pretrained_settings[model_name.rstrip('f16')]['imagenet']['input_size']
        input_size = [1]+input_size
        lw = boundarr(0, inputsize=input_size)
        up = boundarr(1, inputsize=input_size)
        bounds = list(zip(lw, up))
    if not os.path.exists(case_path +'/model_scripted.pt') and 'f16' not in case_path:
        if 'transformer0' in case_path:
            """test_transformer"""
            model = torch.nn.Transformer(d_model=64, nhead=4, num_encoder_layers=3, num_decoder_layers=3)
            model = model.eval()
            src = torch.rand((10, 32, 64))
            tgt = torch.rand((20, 32, 64))
            baseline_model = model.eval()
            baseline_input  = [src, tgt]
            torch.save(baseline_model, case_path+'/model_scripted.pt')
        elif 'transformerprune_d2' in case_path:
            import transformers
            # model = transformers.AutoModelForQuestionAnswering.from_pretrained('Intel/distilbert-base-uncased-sparse-90-unstructured-pruneofa')
            from transformers import AutoTokenizer,AutoModel, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
            tokenizer = AutoTokenizer.from_pretrained('/home/user/tvm/models/transformerprune_d2')
            model = AutoModel.from_pretrained("/home/user/tvm/models/transformerprune_d2",from_tf=True)
            transformers.models.distilbert.modeling_distilbert.DistilBertModel

            dummy_input1 = torch.from_numpy(np.random.randint(0,1e4,size=(10, 32)))
            dummy_input2 = torch.from_numpy(np.random.randint(0,1e4,size=(15, 32, 768)))
            dummy_out = model(dummy_input1)
            for k,v in model.named_parameters():
                print(k,v.detach().numpy().shape)
            print(type(dummy_out))
            baseline_model = model.eval()
            baseline_input  = [dummy_input1]

        elif 'rnn' in case_path:
            baseline_model = RNNLoop().eval()# , [(10, 10, 4)],
            torch.save(baseline_model, case_path+'/model_scripted.pt')
            baseline_input =[ torch.rand((10, 10, 4))]
        else:
            baseline_model, baseline_input = parse_model(model_name)
        # baseline_model = test_forward_add
        baseline_model, mod , params = build_tvm( baseline_model, baseline_input)
        nparams =dict()
        for k, v in params.items():
            nparams[normalkeys(k)]= v.numpy()
        mod = renewmodel(mod,case_path=case_path)
        def save_arr( params,case_path):
                    # print('type',type(list(params.values())[0]))
                    inputarr = dict()
                    for k, v in params.items():
                        inputarr[k]=v
                    path_params = os.path.join(case_path, 'torch_inputs.npz')
                    np.savez(path_params, **inputarr)
        save_arr(nparams,case_path=case_path)
        fuzzer = Fuzzer(path =path,case_id=caseid,params=nparams,low=low, high=high)
    else:
        path_params = os.path.join(case_path+'/oinputs.npz')
        with np.load(path_params) as f:
            params = dict(f.items())
        
        nparams =dict()
        for k, v in params.items():
            nparams[normalkeys(k)]= v
        fuzzer = Fuzzer(path =path,case_id=caseid,params=nparams,low=low, high=high)
        # baseline_model = torch.load(case_path +'/model_scripted.pt').float().eval()
        # get float16 model
        #     factorymod1 = tvm.runtime.load_module(case_path+"/compiled_lib1.tar")
        #     factorymod5 = tvm.runtime.load_module(case_path+"/compiled_lib5.tar")
        # factorymod1 = tvm.runtime.load_module(case_path+"/compiled_lib1.tar")
        # factorymod5 = tvm.runtime.load_module(case_path+"/compiled_lib5.tar")
    fuzzer.dnn = 1
    t1= time.time()
    print('load mod',t1-t0)
    # if not os.path.exists(case_path +'f16'):
    #     os.mkdir(case_path+'f16')
    # else:
    #     fuzzer = Fuzzer(path =path,case_id=caseid,low=low, high=high)
    # fuzzer = Fuzzer(path =path,case_id=caseid,low=low, high=high)
    # float16
    # with open(case_path+'/code.txt','r') as fp:
    #     mod= relay.parse(fp.read())
    # path_params = os.path.join(case_path, 'inputs.npz')
    # with np.load(path_params) as f:
    #     params = dict(f.items())
    # mod = verify_mixed_precision_(mod)
    # with open(case_path+'f16/code.txt','w') as fp:
    #     fp.write(mod.astext())
    # fuzzer = Fuzzer(path =path,case_id=caseid+'f16',params=params,low=low, high=high)

    fuzzer.randweight = 0
    fuzzer.dnn= 1

    t2= time.time()
    print('genfuzz',t2-t1)
    print('fuzz mod5 mod1')
    # fuzzer.fastv()

    fuzzer.fuzzps()
    t3= time.time()
    print('fzz',t3-t2)
    print('fuzz torch mod1')
    fuzzer.replay_withdebugger()

    # debug wen ding xing
    # tk,tv =list(baseline_model.named_parameters())[-1]
    # for k,_ in baseline_model.named_parameters():
    #     print(k)
    # print('tk', tk)
    # paras = './tests/dnn/out/resnet18/L1'
    # data = relay.load_param_dict(bytearray(open(paras+'/_tvmdbg_device_CPU_0/output_tensors.params', "rb").read()))
    # path_params = os.path.join(case_path, 'inputs.npz')
    # with np.load(path_params) as f:
    #     loaded_params = dict(f.items())
    # print('-'*50)
    # print(loaded_params.keys())
    # for k, v in loaded_params.items():
    #     if 'atenlinear0bias' in k and v.shape==tv.shape:
    #         print(k,'\n',tk)
    #         print(tvm.testing.assert_allclose(v,tv,rtol=1e-19))
    #         print(np.equal(v,tv)==True)

    # exit()
    # fuzz torch mod1
    # def fuzzfn(arr:np.array):# input:list of np factorymod1: tvm.runtime.Module,factorymod5: tvm.runtime.Module,
    #     inputarr = np.reshape(arr, input_size)
    #     eparams = inputarr
    #     outst = run_tmod(baseline_model,torch.from_numpy(inputarr).float())
    #     outs1 = run_gmod(factorymod1,{'input0':inputarr})
    #     tempdiff = 0
    #     for i, (ro, o) in enumerate(zip(outst, outs1)):
    #         diff , flag =  MRE(ro,o)
    #         if flag>0:
    #             print('enormous diff:', diff)
    #             exit()
    #         tempdiff = max(diff, tempdiff)
    #     return -tempdiff
    # seeds =dict()
    # for i in range(4):
    #     print('another de finding')
    #     for retx, retf in simpleDE(fuzzfn, bounds=bounds, its=25):
    #         print('global minimum: x , f(x) = ',retx,-retf)
    #     retf = -retf
    #     seeds[str(retf)]= retx
    # print('491---')

    # fuzzer.fuzzps()


'''
-----transformer:




/root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth

input_shape = [10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.ReLU().eval(), input_data=input_data)
'''
