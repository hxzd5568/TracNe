from __future__ import absolute_import, print_function

import numpy as np

import tvm
from tvm import te
from tvm.ir import register_op_attr, register_intrin_lowering


n = te.var("n")
A = te.placeholder((n,), name="A")
# B = te.compute(A.shape, lambda i: tvm.tir.call_pure_extern("float32", "__expf", A[i]), name="B")
B = te.compute(A.shape, lambda i: te.exp(A[i]), name="B")
s = te.create_schedule(B.op)
num_thread = 64
bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
s[B].bind(bx, te.thread_axis("blockIdx.x"))
s[B].bind(tx, te.thread_axis("threadIdx.x"))
f = tvm.build(s, [A, B], "cuda", name="myexp")
print(f.imported_modules[0].get_source())
