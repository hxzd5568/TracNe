import sys
import psutil
sys.path.append('..')
from src.diagnose import diagnose_mod
from src.expand_graph import expand
from src.generate_newseed import ge_seed
from src.rate import ge_rate
ge_rate()
# expand()
# import trace
# # tracer = trace.Trace(count=False, trace=True, ignoredirs=['/usr/lib/python3.7','/venv/apache-tvm-py3.7'])
# # tracer.runfunc(diagnose_mod)

# for i in range(1,2,1):
#     try:
#         print("-"*10,'begin file:',"-"*10,i,"-"*10,'\n\n')
#         diagnose_mod(i)
#         print("#"*10,'finish file:',"#"*10,i,"#"*10,'\n\n\n')
#     except Exception as e:
#         print(e.__class__.__name__,e)
