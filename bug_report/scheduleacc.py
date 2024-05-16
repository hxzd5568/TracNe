import tvm
import json
import random
import numpy as np
from tvm import te
from tvm import tir
from tvm import testing
from tvm import auto_scheduler
from tvm.auto_scheduler.workload_registry import register_workload_tensors

POLICY_PARAMS = {
    "eps_greedy": 0.05,
    "retry_search_one_round_on_empty": 1,
    "sample_init_min_population": 3,
    "sample_init_use_measured_ratio": 0.2,
    "evolutionary_search_population": 5,
    "evolutionary_search_num_iters": 4,
    "evolutionary_search_mutation_prob": 0.85,
    "cpu_multi_level_tiling_structure": "SSRSRS",
    "gpu_multi_level_tiling_structure": "SSSRRSRS",
    # Notice: the default thread bind policy of GPU assumes the tiling structure to have at
    # least 3 spatial tiling levels in outermost
    "max_innermost_split_factor": 64,
    "max_vectorize_size": 16,
    "disable_change_compute_location": 0,
} 

def te_test():
    data_pad = te.placeholder([100], name='data_pad')
    ic = te.reduce_axis([0, 100], name='ic')
    conv2d_NCHWc = te.compute([1], lambda oc_chunk: te.sum(data_pad[ic], axis=[ic]), name='conv2d_NCHWc')
    inline_tensor = te.compute([1], lambda ax1: tir.sin(conv2d_NCHWc[ax1]), name='inline_tensor')
    return [data_pad, conv2d_NCHWc, inline_tensor]

# def te_test():
#     A_1 = te.placeholder([1, 256, 256], name='A')
#     i = te.reduce_axis([0, 256], name='i')
#     j = te.reduce_axis([0, 256], name='j')
#     C = te.compute([1], lambda b : te.sum((A_1[b, i, j]), axis=[i, j]), name='C')
#     inline_tensor = te.compute([1], lambda ax0 : tir.cos(C[ax0]), name='inline_tensor')
#     return [A_1, inline_tensor]

# Get dag and print it.

tensors = te_test()
dag = auto_scheduler.ComputeDAG(tensors)
dict = json.loads(tvm.ir.save_json(tensors))
with open("./saved_json.txt", "w") as file:
    file.write(tvm.ir.save_json(tensors))
print(dag)

# Get inputs.

inputs = []
for tensor in dag.tensors:
    shape = [x.value if 'value' in dir(x) and isinstance(x.value, int) else 1 for x in tensor.shape]
    inputs.append((2 * np.random.uniform(size=shape)+1).astype(tensor.dtype))

# Get program with no schedule.

results = []
mod_list = []
pre_schedule_list = dag.apply_steps_from_state(dag.get_init_state())
pre_mod = tvm.lower(pre_schedule_list[0], pre_schedule_list[1], simple_mode=True)
mod_list.append(pre_mod)
with tvm.transform.PassContext(opt_level=0):
    mod_exec = tvm.build(pre_mod)
    print(pre_mod)

new_inputs = [tvm.nd.array(x, tvm.cpu()) for x in inputs.copy()]
mod_exec(*new_inputs)
result = []
for x in new_inputs:
    try:
        result.append(x.numpy() if isinstance(
            x, tvm.runtime.NDArray) else x)
    except (ValueError, tvm.TVMError):
        result.append(None)
results.append(result)

# Get program with schedule.

register_workload_tensors(dag.workload_key(), tensors)
task = auto_scheduler.SearchTask(workload_key=dag.workload_key(), target=tvm.target.Target("llvm"))
policy = auto_scheduler.SketchPolicy(task, verbose=0, params=POLICY_PARAMS)
states = policy.sample_initial_population()

for state in states:
    schedule_list = dag.apply_steps_from_state(state)
    mod = tvm.lower(schedule_list[0], schedule_list[1], simple_mode=True)
    mod_list.append(mod)
    with tvm.transform.PassContext(opt_level=0):
        mod_exec = tvm.build(mod)
    
    new_inputs = [tvm.nd.array(x, tvm.cpu()) for x in inputs.copy()]
    mod_exec(*new_inputs)
    result = []
    for x in new_inputs:
        try:
            result.append(x.numpy() if isinstance(
                x, tvm.runtime.NDArray) else x)
        except (ValueError, tvm.TVMError):
            result.append(None)
    results.append(result)

for i in range(1, len(results)):
    result = results[i]
    for compare_idex in [-1]:
        try:
            tvm.testing.assert_allclose(results[0][compare_idex], result[compare_idex])
        except AssertionError as e:
            print(e)
            print(mod_list[i])
            break