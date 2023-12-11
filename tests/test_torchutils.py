import tvm
from tvm import relay,runtime
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
sys.path.append('..')
from src.base_utils import Checkor
from src.fuzzer import Fuzzer
from multiprocessing import Process
from src.traceerror import Trace_error
import psutil
from src.torch_utils import  test_dnn
import re

case_path = './dnn/'

args = sys.argv
model_name = args[1]
test_dnn(model_name=model_name,path=case_path,caseid= model_name )
# try:
#     test_dnn(model_name=model_name,path=case_path,caseid= model_name )    # resnet [0,1]
# except Exception as e:
#     print(e.__class__.__name__, e)

# test_forward_add()
# for model_name in ['resnet34']:
#     # model_name = 'resnet18','resnet34','resnet50','vgg11','vgg13','vgg16','inceptionv3','dpn68'
#     # print(pretrainedmodels.pretrained_settings[model_name]['imagenet']['input_range'])
#     try:
#         test_dnn(model_name=model_name,path=case_path,caseid= model_name )    # resnet [0,1]
#     except Exception as e:
#         print(e.__class__.__name__, e)

# case_id = os.path.basename(__file__).strip('.py')
# args = sys.argv
# if '-' in args[1]:
#     l,r = int(args[1].split('-')[0]),\
#             int(args[1].split('-')[1])+1
#     caseids = [str(i) for i in range(l,r,1)]
# else:
#     caseids = args[1:]
# for caseid in caseids:
#     dump_path = case_path+'/out/'+caseid
#     print(dump_path)
#     fuzzer = Fuzzer(path =case_path,case_id=caseid,fuzzmode='DEMC2')
#     fuzzer.bigflag =1
#     pfuzzer = Process(target=fuzzer.fuzzps)
#     try:
#         pfuzzer.start()
#         pfuzzer.join(timeout=300)
#         pfuzzer.kill()
#         fuzzer.replay_withdebugger()
#     except Exception as e:
#         print(e.__class__.__name__, e)
#     del fuzzer
#     del pfuzzer

    # trace_error = Trace_error(dump_path)
    # # trace_error.get_trace_message()
    # try:
    #     trace_error.get_node_name()
    # except Exception as e:
    #     print(e.__class__.__name__, e)
    # del trace_error
    # try:

    #     continue
    # except Exception as e:
    #     print(e.__class__.__name__,':',e)
    #     # print (traceback.format_exc ())
    #     continue
'''
resnet18    global minimum: x , f(x) =  0.09812591429267264
0.09812591429267264 me using time 1209.5658423900604


'''
