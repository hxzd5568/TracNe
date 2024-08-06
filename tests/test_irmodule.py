import pytest

import tvm
import tvm.testing
from tvm import relay
from tvm.target.target import Target
from tvm.relay import testing
from tvm.relay.backend import Runtime, Executor, graph_executor_codegen


@pytest.mark.parametrize(
    "test_target,unsupported_config",
    [
        ["c", "-runtime=c"],
        ["c", "-system-lib=1"],
        ["c", "-executor=aot"],
        ["c", "-interface-api=c"],
        ["c", "-unpacked-api=1"],
        ["c", "-link-params=1"],
    ],
)
def test_build_relay_graph_():
    """Test to build a simple relay graph by using APIs directly"""

    def build_graph(mod, target):
        target, target_host = tvm.target.Target.canon_target_and_host(target)
        mod, _ = relay.optimize(mod, target)
        grc = graph_executor_codegen.GraphExecutorCodegen(None, target)

        _, lowered_funcs, _ = grc.codegen(mod, mod["main"])
        _ = relay.backend._backend.build(lowered_funcs, target)
        print(grc._get_irmodule())
        print(grc._list_params_name())

    def add(shape, dtype):
        lhs = relay.var("A", shape=shape, dtype=dtype)
        rhs = relay.var("B", shape=shape, dtype=dtype)
        out = relay.add(lhs, rhs)
        expr = relay.Function((lhs, rhs), out)
        mod = tvm.IRModule.from_expr(expr)
        return mod

    build_graph(add((1, 8), "float32"), tvm.target.Target("llvm"))


test_build_relay_graph_()
