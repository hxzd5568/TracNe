from statistics import mode
import torch
import numpy as np
import os
import time
import tvm
from tvm import tir
from collections import namedtuple
import onnx
from scipy.special import softmax
from tqdm import tqdm

from tvm.contrib.debugger import debug_executor
from tvm.contrib import graph_executor

from tvm.contrib import relay_viz
from tvm.contrib.relay_viz.dot import DotPlotter
from tvm.contrib.relay_viz.interface import DefaultVizParser

# import pdb
# pdb.set_trace()
# /home/user/tvm/python/tvm/contrib/relay_viz/interface.py
# /home/user/tvm/python/tvm/contrib/relay_viz/dot.py
# /home/user/tvm/python/tvm/contrib/relay_viz/__init__.py(113)render()-


def save_irmod_viz(irmod, basename):
    viz = relay_viz.RelayVisualizer(
        irmod, plotter=DotPlotter(), parser=DefaultVizParser()
    )
    viz.render(basename)
