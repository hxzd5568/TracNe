# issue url: https://github.com/apache/tvm/issues/6588

import numpy as np
import tvm
from tvm import relay
import tvm.testing
from onnxutils import MSE


@tvm.testing.uses_gpu
def test_dense():
    # ------------ change to float64
    for dtype in ["float64"]:

        x = relay.var("x", shape=(8, 512), dtype=dtype)
        w = relay.var("w", shape=(128, 512), dtype=dtype)
        z = relay.nn.dense(x, w)

        # Check result.
        func = relay.Function([x, w], z)
        x_data = np.random.randn(8, 512).astype(dtype)
        w_data = np.random.randn(128, 512).astype(dtype)
        ref_res = np.dot(x_data, w_data.T)

        target = tvm.target.cuda(arch="sm_89")
        ctx = tvm.cuda(0)
        intrp1 = relay.create_executor("graph", device=ctx, target=target)
        intrp2 = relay.create_executor("debug", device=ctx, target=target)
        op_res1 = intrp1.evaluate(func)(x_data, w_data)
        # tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5)
        op_res2 = intrp2.evaluate(func)(x_data, w_data)
        # tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-5)
        print(MSE(op_res2.asnumpy(), ref_res))
        # ----------error disappear, and the mean relative error is around 1e-15 -----
        tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-13)

        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(func, target="cuda")
            lib.export_library("compiled_model.tar")
            print(lib.imported_modules[0].get_source())


if __name__ == "__main__":
    test_dense()
