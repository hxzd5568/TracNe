import os
import re
from argparse import Namespace, ArgumentParser

import numpy as np
import tvm
from tvm import parser
import sys

sys.path.append("..")

from src.gencog.debug import ErrorKind, CompileReducer, RunReducer, ComputeReducer
from src.gencog.graph import visualize
from src.gencog.graph.relay import build_graph

case_path = os.getcwd()
args = sys.argv
if "-" in args[1]:
    l, r = int(args[1].split("-")[0]), int(args[1].split("-")[1]) + 1
    caseids = [str(i) for i in range(l, r, 1)]
else:
    caseids = args[1:]

for caseid in caseids:
    dump_path = case_path + "/out/" + caseid
    print(tvm.__version__)

    opt_level = 10
    with open(os.path.join(dump_path, "code.txt"), "r") as f:
        code = f.read()

    print(f"Reducing case {caseid}:")

    # Possibly load inputs and parameters
    inputs_path = os.path.join(dump_path, "inputs.npz")
    inputs = None
    if os.path.exists(inputs_path):
        with np.load(inputs_path) as f:
            inputs = dict(f.items())
    params = inputs

    reduce_cls = ComputeReducer
    reducer = reduce_cls(code, "err", opt_level, inputs=inputs, params=params)
    reduced_code, extra = reducer.reduce()
    with open(os.path.join(dump_path, "code-reduced.txt"), "w") as f:
        f.write(reduced_code)
    if len(extra) > 0:
        with open(os.path.join(dump_path, "extra.txt"), "w") as f:
            f.write(extra)

    # Visualize reduced code
    graph = build_graph(parser.parse(reduced_code), {} if params is None else params)
    visualize(graph, "graph", dump_path)
    print("passed")

    # Reduce code
    # try:
    #     reduce_cls = ComputeReducer
    #     reducer = reduce_cls(code, 'err', opt_level, inputs=inputs, params=params)
    #     reduced_code, extra = reducer.reduce()
    #     with open(os.path.join(dump_path, 'code-reduced.txt'), 'w') as f:
    #         f.write(reduced_code)
    #     if len(extra) > 0:
    #         with open(os.path.join(dump_path, 'extra.txt'), 'w') as f:
    #             f.write(extra)

    #     # Visualize reduced code
    #     graph = build_graph(parser.parse(reduced_code), {} if params is None else params)
    #     visualize(graph, 'graph', dump_path)
    #     print('passed')
    # except Exception as e:
    #     print(e)
