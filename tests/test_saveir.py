import sys
import psutil

sys.path.append("..")
from src.diagnose import diagnose_mod
from src.base_utils import Checkor
from src.fuzzer import Fuzzer
import os

args = sys.argv
# for id in range(20,31,1):
#     fuzzer = Fuzzer(path =case_path,case_id=str(id),fuzzmode='DEMC2')
#     fuzzer.replay_withdebugger()
#     del fuzzer
case_path = "./"
args = sys.argv

if "/" in args[1]:
    dump_path = args[1]
    case_path = dump_path.split("out")[0]
    caseid = dump_path.split("out")[1][1:]
    caseids = [caseid]
elif "-" in args[1]:
    l, r = int(args[1].split("-")[0]), int(args[1].split("-")[1]) + 1
    caseids = [str(i) for i in range(l, r, 1)]
elif "," in args[1]:
    caseids = args[1].split(",")
else:
    caseids = args[1:]

for caseid in caseids:
    dump_path = case_path + "/out/" + caseid
    print(dump_path)
    fuzzer = Fuzzer(path=case_path, case_id=caseid, fuzzmode="DEMC2")
    fuzzer.bigflag = 1
    fuzzer.save_files()
    # fuzzer.replay_withnewgmod()
    del fuzzer
