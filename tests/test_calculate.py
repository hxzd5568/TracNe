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
from src.calculator import Calculate_error
from src.base_utils import Checkor

case_path = os.getcwd()
case_id = os.path.basename(__file__).strip(".py")
print(case_path)
checkor = Checkor(path=case_path, case_id="22")

mod = checkor.mod
print(mod)
cal = Calculate_error(mod=mod)
print(cal.test_consistent())
cal.random_difference_test()
