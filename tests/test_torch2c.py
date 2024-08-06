import tvm
from tvm import relay, runtime
import os
import numpy as np
import queue
import shutil
import os.path
import random
from enum import IntEnum, auto
from tvm import transform, relay, parser, cpu, TVMError, IRModule
from tvm.contrib.graph_executor import GraphModule
from argparse import Namespace, ArgumentParser
from typing import Iterable, List, cast, Optional, Dict
import sys

sys.path.append("..")
from src.base_utils import Checkor
from src.fuzzer import Fuzzer
from multiprocessing import Process
from src.traceerror import Trace_error
import psutil
from src.torch2c import test_dnn
import re

case_path = "./dnn/"

model_name = "log2"
# print(pretrainedmodels.pretrained_settings[model_name]['imagenet']['input_range'])
test_dnn(path=case_path, caseid=model_name)  # resnet [0,1]
