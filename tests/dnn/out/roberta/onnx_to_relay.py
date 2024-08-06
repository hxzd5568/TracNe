import tvm
from tvm import relay
import numpy as np

from sparsezoo import Model


import os
import onnx
import re
from google.protobuf.json_format import MessageToDict

# !!!change
case_path = "./"

onnx_model = onnx.load("./roberta-base-11.onnx")
# onnx.checker.check_msodel(onnx_model)

graph = onnx_model.graph

for _input in graph.input:
    print(MessageToDict(_input))


input_shape = (2, 3)  # (1,384)

mod, params = relay.frontend.from_onnx(onnx_model, {"input_ids": input_shape})


def normalname(mod):  # return new_mod and if changed flag
    changeflag = []
    mod = re.sub("::", "", mod, count=0, flags=re.S | re.M)
    pat = "(?P<value>%[a-zA-Z_]+[.a-zA-Z_0-9]+)"

    def update_internal(matched):
        changeflag.append(1)
        return matched.group("value").replace("_", "").replace(".", "")

    mod = re.sub(pat, update_internal, mod, count=0, flags=re.M | re.S)
    return mod


def renewmodel(mod: tvm.IRModule, case_path: str) -> tvm.IRModule:
    modstr = mod.astext()
    modstr = normalname(modstr)
    if not os.path.exists(case_path):
        os.mkdir(case_path)
    with open(f"{case_path}/code.txt", "w") as fp:
        fp.write(modstr)
    with open(f"{case_path}/code.txt", "r") as f:
        return relay.parse(f.read())


def normalkeys(strs):
    strs = strs.replace("::", "")
    strs = strs.replace("_", "")
    strs = strs.replace(".", "")
    print(strs)
    return strs


with open(case_path + "./code.txt", "w") as f:
    f.write(mod.astext())

nparams = dict()
for k, v in params.items():
    nparams[normalkeys(k)] = v.numpy()
mod = renewmodel(mod, case_path=case_path)


def save_arr(params, case_path):
    inputarr = dict()
    for k, v in params.items():
        inputarr[k] = v
    path_params = os.path.join(case_path, "torch_inputs.npz")
    np.savez(path_params, **inputarr)


save_arr(nparams, case_path=case_path)

print("success")
