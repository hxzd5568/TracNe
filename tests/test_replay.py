import sys
import psutil
sys.path.append('..')
from src.base_utils import Checkor
from src.fuzzer import Fuzzer
from src.viz import save_irmod_viz
from tvm import relay
import os
args = sys.argv
# for id in range(20,31,1):
#     fuzzer = Fuzzer(path =case_path,case_id=str(id),fuzzmode='DEMC2')
#     fuzzer.replay_withdebugger()
#     del fuzzer
case_path = './'
args = sys.argv

def draw_torchviz(path):
    with open(path+'/unoptirmod.txt','r') as fp:
        mod1 = relay.parse(fp.read())
    with open(path+'/optirmod.txt','r') as fp:
        mod5 = relay.parse(fp.read())
    save_irmod_viz(mod1, path+'/relayviz1')
    save_irmod_viz(mod5, path+'/relayviz5')

    print('tvmviz done')

if '/' in args[1]:
    dump_path = args[1]
    case_path = dump_path.split('out')[0]
    caseid = dump_path.split('out')[1][1:]
    caseids = [caseid]
elif '-' in args[1]:
    l,r = int(args[1].split('-')[0]),\
            int(args[1].split('-')[1])+1
    caseids = [str(i) for i in range(l,r,1)]
elif ',' in args[1]:
    caseids = args[1].split(',')
else:
    caseids = args[1:]
import time
t0 = time.time()
for caseid in caseids:
    if caseid.isdigit():
        dump_path = case_path+'/out/'+caseid
    else:
        dump_path = case_path+'/dnn/out/'+caseid
        case_path = case_path+'/dnn'
    print(dump_path)
    fuzzer = Fuzzer(path =case_path,case_id=caseid,fuzzmode='MEGA')
    # save_irmod_viz(fuzzer.mod, dump_path+'/relayviz0')
    fuzzer.bigflag = 1
    # fuzzer.dnn = 1
    fuzzer.replay_withdebugger()
    # print(fuzzer.mod)
    # draw_torchviz(dump_path)
    # fuzzer.replay_withsample()
    # fuzzer.replay(fuzzer.fuzzmode)
    # fuzzer.replay_withnewgmod()
    del fuzzer
print(time.time()-t0)
