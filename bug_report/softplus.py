# issue report : https://github.com/apache/tvm/issues/12632

import torch
from tvm import relay
import tvm
import numpy as np

m = torch.nn.Softplus(beta=0,)
input_data = torch.tensor(torch.randn([14, 7], dtype=torch.float64))
torch_outputs = m(input_data)
trace = torch.jit.trace(m, input_data)
input_shapes = [('input0', torch.Size([14, 7]))]

mod, params = relay.frontend.from_pytorch(trace, input_shapes)
input_data_np = input_data.numpy()

# --- fix bugs by disable pass: fastmath ---
with tvm.transform.PassContext(opt_level=3,):#disabled_pass=['FastMath']
    exe = relay.create_executor('graph', mod=mod, params=params, device=tvm.device('llvm', 0), target='llvm').evaluate()
input_tvm = {'input0': tvm.nd.array(input_data_np.astype(np.float64))}
tvm_outputs = exe(**input_tvm).asnumpy()
np.testing.assert_allclose(torch_outputs, tvm_outputs, rtol=1e-3, atol=1e-3)
# print('torch-tvm')
# print(torch_outputs, tvm_outputs)
# print('truth')
# print('truth')
