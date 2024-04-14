import sys
sys.path.append('..')
import os
from src.base_utils import Checkor
from src.propainfo import Corrector


import re
code ="""
fn (%p01: Tensor[(3), float16] /* ty=Tensor[(3), float16] */, %p11: float16 /* ty=float16 */, %p21: Tensor[(3), float16] /* ty=Tensor[(3), float16] */, Primitive=1, hash="d6b94b42f0d777a8") -> Tensor[(3), float16] {
    %8 = add(%p01, %p11) /* ty=Tensor[(3), float16] */;
    %9 = rsqrt(%8) /* ty=Tensor[(3), float16] */;
    multiply(%9, %p21) /* ty=Tensor[(3), float16] */
  } /* ty=fn (Tensor[(3), float16], float16, Tensor[(3), float16]) -> Tensor[(3), float16] */;
  %11 = %10(%graphbnmovingvar, 1.00136e-05f16 /* ty=float16 */, %graphbngamma) /* ty=Tensor[(3), float16] */;
  %12 = fn (%p0: Tensor[(8, 3, 224, 224), float16] /* ty=Tensor[(8, 3, 224, 224), float16] */, %p1: Tensor[(3, 3, 3, 3), float16] /* ty=Tensor[(3, 3, 3, 3), float16] */, %p2: Tensor[(3), float16] /* ty=Tensor[(3), float16] */, %p3: Tensor[(3), float16] /* ty=Tensor[(3), float16] */, %p4: Tensor[(3), float16] /* ty=Tensor[(3), float16] */, Primitive=1, hash="a9b2bc05398c6062", kernel_layout="OIHW", data_layout="NCHW", out_layout="") -> Tensor[(8, 3, 112, 112), float16] {
    %0 = nn.conv2d(%p0, %p1, strides=[2, 2], padding=[1, 1, 1, 1], channels=3, kernel_size=[3, 3]) /* ty=Tensor[(8, 3, 112, 112), float16] */;
    %1 = expand_dims(%p2, axis=1, num_newaxis=2) /* ty=Tensor[(3, 1, 1), float16] */;
    %2 = negative(%p3) /* ty=Tensor[(3), float16] */;
    %3 = multiply(%2, %p2) /* ty=Tensor[(3), float16] */;
    %4 = add(%3, %p4) /* ty=Tensor[(3), float16] */;
    %5 = multiply(%0, %1) /* ty=Tensor[(8, 3, 112, 112), float16] */;
    %6 = expand_dims(%4, axis=1, num_newaxis=2) /* ty=Tensor[(3, 1, 1), float16] */;
    %7 = add(%5, %6) /* ty=Tensor[(8, 3, 112, 112), float16] */;
    nn.relu(%7) /* ty=Tensor[(8, 3, 112, 112), float16] */
  }
"""
def rewrelay(ocode):
    # match mod funciton with f2
    hash = 'a9b2bc05398c6062'
    pattern1 =hash+ '.*?\}'
    pattern2= hash[::-1]+ '(.*?nf)'
    matched  =  re.search(pattern1, ocode, flags=re.M|re.S)
    a = matched.group()
    matched  =  re.search(pattern2, ocode[::-1], flags=re.M|re.S)
    b = matched.group(1)[::-1]
    return b+a
# print(rewrelay(code))


case_path = './'
args = sys.argv

if '/' in args[1]:
    dump_path = args[1]
    case_path = dump_path.split('out')[0]
    caseid = dump_path.split('out')[1][1:]
    caseids = [caseid]
elif '-' in args[1]:
    l,r = int(args[1].split('-')[0]),\
            int(args[1].split('-')[1])+1
    caseids = [str(i) for i in range(l,r,1)]
else:
    caseids = args[1:]
import time
t0 = time.time()
for caseid in caseids:
    if caseid.isdigit():
        dump_path = case_path+'/out/'+caseid
    else:
        dump_path = case_path+'/dnn/out/'+caseid
    print(dump_path)
    corrector = Corrector(dump_path)
    corrector.pipeline()
    # try:
    #     dump_path = case_path+'/out/'+caseid
    #     print(dump_path)
    #     corrector = Corrector(dump_path)
    #     corrector.pipeline()
    # except Exception as e:
    #     print(e)
