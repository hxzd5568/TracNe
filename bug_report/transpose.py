import tvm
from tvm import relay
import torch
import onnxruntime as ort
import onnx
import numpy as np
import pickle
from numpy import testing

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        _args = args
        getitem = _args[0]
        tril = getitem.tril(0)
        tril_1 = tril.tril(0)
        div = torch.div(tril, tril_1)
        argmax = div.argmax(1)
        return (argmax)

model_0 = Model0()
output_names_0 = ['v4_0']
input_data_0 = np.array([[3.346, 4.98 , 6.035, 6.832, 5.625, 6.1  , 4.832, 5.723, 5.83 ],
       [6.625, 6.9  , 6.926, 6.426, 6.41 , 5.695, 4.484, 6.43 , 4.83 ]],
      dtype=np.float16)
input_dict_0 = {'v5_0':input_data_0}
inputs_0 = tuple(torch.from_numpy(v).to('cpu') for _, v in input_dict_0.items())
torch.onnx.export(model_0, inputs_0, '0.onnx', verbose=False, input_names=['v5_0'], output_names=output_names_0, opset_version=14, do_constant_folding=False)

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        _args = args
        getitem = _args[0]
        tril = getitem.tril(0)
        tril_1 = tril.tril(0)
        div = torch.div(tril, tril_1)
        transpose = div.transpose(1, 0)
        argmax = div.argmax(1)
        return (transpose, argmax)

model_1 = Model1()
output_names_1 = ['v14_0', 'v24_0']
input_dict_1 = {'v0_0':input_data_0}
inputs_1 = tuple(torch.from_numpy(v).to('cpu') for _, v in input_dict_1.items())
torch.onnx.export(model_1, inputs_1, '1.onnx', verbose=False, input_names=['v0_0'], output_names=output_names_1, opset_version=14, do_constant_folding=False)

onnx_model_0 = onnx.load('0.onnx')
onnx_model_outputs_0 = [node.name for node in onnx_model_0.graph.output]
shape_dict_0 = {key: val.shape for key, val in input_dict_0.items()}
mod_0, params_0 = relay.frontend.from_onnx(onnx_model_0, shape_dict_0, freeze_params=True)
with tvm.transform.PassContext(opt_level=4):
    executor_0 = relay.build_module.create_executor("graph", mod_0, tvm.cpu(), tvm.target.Target("llvm"), params_0).evaluate()
    executor_res_0 = [executor_0(**input_dict_0).numpy()]
    output_0 = dict(zip(onnx_model_outputs_0, executor_res_0))

onnx_model_1 = onnx.load('1.onnx')
onnx_model_outputs_1 = [node.name for node in onnx_model_1.graph.output]
shape_dict_1 = {key: val.shape for key, val in input_dict_1.items()}
mod_1, params_1 = relay.frontend.from_onnx(onnx_model_1, shape_dict_1, freeze_params=True)
with tvm.transform.PassContext(opt_level=4):
    executor_1 = relay.build_module.create_executor("graph", mod_1, tvm.cpu(), tvm.target.Target("llvm"), params_1).evaluate()
    executor_res_1 = [tensor.numpy() for tensor in executor_1(**input_dict_1)]
    output_1 = dict(zip(onnx_model_outputs_1, executor_res_1))
output_name_dict = {'v4_0': 'v24_0'}

print('=========================')
try:
    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1])
        print(output_0,output_1)
    print("tvm_opt_4 does not trigger assertion")
except AssertionError as e:
    print("tvm_opt_4 triggers assertion")
    print(e)
print('=========================')

shape_dict_0 = {key: val.shape for key, val in input_dict_0.items()}
mod_0, params_0 = relay.frontend.from_onnx(onnx_model_0, shape_dict_0, freeze_params=True)
with tvm.transform.PassContext(opt_level=0):
    executor_0 = relay.build_module.create_executor("graph", mod_0, tvm.cpu(), tvm.target.Target("llvm"), params_0).evaluate()
    executor_res_0 = [executor_0(**input_dict_0).numpy()]
    output_0 = dict(zip(onnx_model_outputs_0, executor_res_0))

shape_dict_1 = {key: val.shape for key, val in input_dict_1.items()}
mod_1, params_1 = relay.frontend.from_onnx(onnx_model_1, shape_dict_1, freeze_params=True)
with tvm.transform.PassContext(opt_level=0):
    executor_1 = relay.build_module.create_executor("graph", mod_1, tvm.cpu(), tvm.target.Target("llvm"), params_1).evaluate()
    executor_res_1 = [tensor.numpy() for tensor in executor_1(**input_dict_1)]
    output_1 = dict(zip(onnx_model_outputs_1, executor_res_1))

print('=========================')
try:
    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1])
        print(output_0,output_1)

    print("tvm_opt_0 does not trigger assertion")
except AssertionError as e:
    print("tvm_opt_0 triggers assertion")
    print(e)
print('=========================')