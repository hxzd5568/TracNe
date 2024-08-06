## issue url: https://github.com/apache/tvm/issues/12627



import torch
from tvm import relay, tir
import tvm
import numpy as np
from torch.nn import Module

from onnxutils import MSE

input_data = torch.randn([1, 2048], dtype=torch.float32)
para_1 = torch.randn([1000, 2048], dtype=torch.float32)
para_2 = torch.randn([1000], dtype=torch.float32)


class linear64(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(
            args[0].double(), para_1.double(), para_2.double()
        )


class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1, para_2)


m = linear().float().eval()
m64 = linear64().eval()
torch_outputs = m(input_data)
torch_outputs2 = m64(input_data)

trace = torch.jit.trace(m, input_data)
input_shapes = [("input0", torch.Size([1, 2048]))]

mod, params = relay.frontend.from_pytorch(trace, input_shapes, default_dtype="float32")
# print(mod)
with tvm.transform.PassContext(
    opt_level=1,
    disabled_pass=[
        "SimplifyExpr",
        "FoldConstant",
        "BackwardFoldScaleAxis",
        "ForwardFoldScaleAxis",
        "CanonicalizeCast",
        "CanonicalizeOps",
        "AlterOpLayout",
        "FastMath",
        "SplitArgs",
        "InlineGlobals",
        "LabelOps",
        "AnnotateMemoryScope",
    ],  # ,config={"tir.add_lower_pass": [(3, print_tir)]}
):
    exe = relay.create_executor(
        "graph", mod=mod, params=params, device=tvm.device("llvm", 0), target="llvm"
    ).evaluate()
    # prim_mod, _ = relay.optimize(mod, target='llvm')
    # # tir_mod = relay.transform.ToAnnotatedIR()(mod)

    # tir_mod = tvm.lower(mod, simple_mode=True)
    # print(tir_mod)

    # for func in tir_mod.functions:
    # print(func)
    # print(prim_mod)
# with relay.build_config(opt_level=3):
#     graph, lib, params = relay.build(mod, target='cuda', params=params)
#     lib.export_library("compiled_model.tar")
#     # print(lib.imported_modules[0].get_source())
input_tvm = {"input0": np.array(input_data, dtype="float64")}
tvm_outputs = exe(**input_tvm).asnumpy()

print(MSE(torch_outputs2.numpy(), tvm_outputs))
print(MSE(torch_outputs2.numpy(), torch_outputs.numpy()))
print(MSE(tvm_outputs, torch_outputs.numpy()))
# np.testing.assert_allclose(torch_outputs, tvm_outputs, rtol=1e-5, atol=1e-5)  # the treshold is the same with that in equipped test cases
# np.testing.assert_allclose(torch_outputs, torch_outputs2, rtol=1e-5, atol=1e-5)  # the treshold is the same with that in equipped test cases
# np.testing.assert_allclose(torch_outputs2, tvm_outputs, rtol=1e-5, atol=1e-5)  # the treshold is the same with that in equipped test cases


"""

tvm split the loop
# from tvm.script import tir as T

@T.prim_func
def tvmgen_default_fused_nn_dense_nn_bias_add(p0: T.Buffer((1, 2048), "float32"), p1: T.Buffer((1000, 2048), "float32"), p2: T.Buffer((1000,), "float32"), T_add: T.Buffer((1, 1000), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "hash": "99b287e0bae4ce5f", "target": T.target({"host": {"keys": ["cpu"], "kind": "llvm", "tag": ""}, "keys": ["cpu"], "kind": "llvm", "tag": ""}), "tir.noalias": T.bool(True)})
    packed_weight = T.allocate([2048000], "float32", "global")
    packed_weight_1 = T.Buffer((2048000,), data=packed_weight)
    for z, y, x in T.grid(125, 2048, 8):
        cse_var_1: T.int32 = z * 16384
        p1_1 = T.Buffer((2048000,), data=p1.data)
        packed_weight_1[cse_var_1 + y * 8 + x] = p1_1[cse_var_1 + x * 2048 + y]
    for ax1_outer_ax0_outer_fused in T.parallel(125):
        cse_var_2: T.int32 = ax1_outer_ax0_outer_fused * 8
        compute_global = T.allocate([1], "float32x8", "global")
        compute_global_1 = T.Buffer((1,), "float32x8", data=compute_global, align=32)
        compute_global_1[0] = T.Broadcast(T.float32(0), 8)
        for k_outer in range(2048):
            p0_1 = T.Buffer((2048,), data=p0.data)
            compute_global_1[0] = compute_global_1[0] + T.Broadcast(p0_1[k_outer], 8) * packed_weight_1[ax1_outer_ax0_outer_fused * 16384 + k_outer * 8:ax1_outer_ax0_outer_fused * 16384 + k_outer * 8 + 8]
        T_add_1 = T.Buffer((1000,), data=T_add.data)
        compute_global_2 = T.Buffer((1,), "float32x8", data=compute_global, align=32)
        T_add_1[cse_var_2:cse_var_2 + 8] = compute_global_2[0] + p2[cse_var_2:cse_var_2 + 8]
"""
