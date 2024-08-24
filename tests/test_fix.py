import sys

sys.path.append("..")
import os
from src.base_utils import Checkor
from src.fix_utils import Corrector
import re


def rewrelay(ocode):
    # match mod funciton with f2
    hash = "a9b2bc05398c6062"
    pattern1 = hash + ".*?\}"
    pattern2 = hash[::-1] + "(.*?nf)"
    matched = re.search(pattern1, ocode, flags=re.M | re.S)
    a = matched.group()
    matched = re.search(pattern2, ocode[::-1], flags=re.M | re.S)
    b = matched.group(1)[::-1]
    return b + a


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
else:
    caseids = args[1:]
import time

t0 = time.time()
for caseid in caseids:
    dump_path = case_path + "/out/" + caseid
    print(dump_path)
    corrector = Corrector(dump_path)
    corrector.pipeline()
print("fix", time.time() - t0)
