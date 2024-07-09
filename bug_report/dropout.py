# issue url:  https://github.com/apache/tvm/issues/15005

import torch
from tvm import relay
import tvm
import numpy as np
from torch.nn import Module

input_data = torch.randn([5], dtype=torch.float64)
class alpha_dropout(Module):
        def forward(self, *args):
            return torch.nn.functional.alpha_dropout(args[0], 0.2,training=True)
m = alpha_dropout().float().eval()

# TVM is right, but pytorch is wrong.

torch_outputs = m(input_data)
trace = torch.jit.trace(m, input_data)
input_shapes = [('input0', torch.Size([5]))]

mod, params = relay.frontend.from_pytorch(trace, input_shapes)
with tvm.transform.PassContext(opt_level=3):
    exe = relay.create_executor('graph', mod=mod, params=params, device=tvm.device('llvm', 0), target='llvm').evaluate()

input_tvm = {'input0': np.array(input_data, dtype='float64')}
tvm_outputs = exe(**input_tvm).asnumpy()
try:
    np.testing.assert_allclose(input_data, tvm_outputs, rtol=1e-3, atol=1e-3)
    print('tvm result and the truth are the same')
except:
    print('tvm & truth are diff')
     

try:
    np.testing.assert_allclose(torch_outputs, tvm_outputs, rtol=1e-3, atol=1e-3)
    print('tvm result and the torch result are the same')
except:
    print('tvm & torch are diff')




# =================== for functional.dropout =============

class dropout1(Module):
        def forward(self, *args):
            return torch.nn.functional.dropout(args[0], 0.2)
m = dropout1().float().eval()

torch_outputs = m(input_data)
trace = torch.jit.trace(m, input_data)
input_shapes = [('input0', torch.Size([5]))]

mod, params = relay.frontend.from_pytorch(trace, input_shapes)
with tvm.transform.PassContext(opt_level=3):
    exe = relay.create_executor('graph', mod=mod, params=params, device=tvm.device('llvm', 0), target='llvm').evaluate()

input_tvm = {'input0': np.array(input_data, dtype='float64')}
tvm_outputs = exe(**input_tvm).asnumpy()
try:
    np.testing.assert_allclose(input_data, tvm_outputs, rtol=1e-3, atol=1e-3)
    print('tvm result and the truth are the same')
except:
    print('tvm & truth are diff')
     

try:
    np.testing.assert_allclose(torch_outputs, tvm_outputs, rtol=1e-3, atol=1e-3)
    print('tvm result and the torch result are the same')
except:
    print('tvm & torch are diff')


# =================== disabling the 'training' option can solve the issue =============
    
input_data = torch.randn([5], dtype=torch.float64)
class alpha_dropout(Module):
        def forward(self, *args):
            return torch.nn.functional.alpha_dropout(args[0], 0.2,training=False)

m = alpha_dropout().float().eval()
torch_outputs = m(input_data)
trace = torch.jit.trace(m, input_data)
input_shapes = [('input0', torch.Size([5]))]

mod, params = relay.frontend.from_pytorch(trace, input_shapes)
with tvm.transform.PassContext(opt_level=3):
    exe = relay.create_executor('graph', mod=mod, params=params, device=tvm.device('llvm', 0), target='llvm').evaluate()

input_tvm = {'input0': np.array(input_data, dtype='float64')}
tvm_outputs = exe(**input_tvm).asnumpy()
try:
    np.testing.assert_allclose(input_data, tvm_outputs, rtol=1e-3, atol=1e-3)
    print('tvm result and the truth are the same')
except:
    print('tvm & truth are diff')
     

try:
    np.testing.assert_allclose(torch_outputs, tvm_outputs, rtol=1e-3, atol=1e-3)
    print('tvm result and the torch result are the same')
except:
    print('tvm & torch are diff')
