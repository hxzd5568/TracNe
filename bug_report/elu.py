import torch
from tvm import relay
import tvm
import numpy as np

# origin question
# https://github.com/apache/tvm/issues/15396

# ---------------solution 1: disable the pass=['FastMath']
m = torch.nn.ELU(
    alpha=-1.8574e38,
)
input_data = torch.tensor(torch.randn([9, 12, 19, 5, 10], dtype=torch.float32))
torch_outputs = m(input_data)
trace = torch.jit.trace(m, input_data)
input_shapes = [("input0", torch.Size([9, 12, 19, 5, 10]))]

mod, params = relay.frontend.from_pytorch(trace, input_shapes)
input_data_np = input_data.numpy()


with tvm.transform.PassContext(opt_level=3, disabled_pass=["FastMath"]):
    exe = relay.create_executor(
        "graph", mod=mod, params=params, device=tvm.device("llvm", 0), target="llvm"
    ).evaluate()
input_tvm = {"input0": tvm.nd.array(input_data_np.astype(np.float32))}
tvm_outputs = exe(**input_tvm).asnumpy()
np.testing.assert_allclose(torch_outputs, tvm_outputs, rtol=1e-5, atol=1e-5)

# ---------------solution 2: enhance the precision


m = torch.nn.ELU(
    alpha=-1.8574e38,
)
input_data = torch.tensor(torch.randn([9, 12, 19, 5, 10], dtype=torch.float64))
torch_outputs = m(input_data)
trace = torch.jit.trace(m, input_data)
input_shapes = [("input0", torch.Size([9, 12, 19, 5, 10]))]

mod, params = relay.frontend.from_pytorch(trace, input_shapes)
input_data_np = input_data.numpy()


with tvm.transform.PassContext(opt_level=3):
    exe = relay.create_executor(
        "graph", mod=mod, params=params, device=tvm.device("llvm", 0), target="llvm"
    ).evaluate()
input_tvm = {"input0": tvm.nd.array(input_data_np.astype(np.float64))}
tvm_outputs = exe(**input_tvm).asnumpy()
np.testing.assert_allclose(torch_outputs, tvm_outputs, rtol=1e-5, atol=1e-5)
