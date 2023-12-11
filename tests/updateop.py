# update operator register
import os
import re


ops = dict()

# '/home/user/tvm/python/tvm/relay/op/nn' '/home/user/tvm/python/tvm/relay/op/' '/home/user/tvm/python/tvm/relay/op/random
path = '/home/user/tvm/python/tvm/relay/op/' # path-to-tvm
def getkeys(subdirs):
    keys = []#['complex','simple']
    subdirs.remove('strategy')
    subdirs.remove('__pycache__')
    for i in subdirs:
       keys.append(i)
    print('keys',keys)
    return keys

import os


def gci(filepath):
    # Recursively traverse the entire folder
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi)
        if os.path.isdir(fi_d):
            gci(fi_d)
        else:
            if fi[0] == '_' and fi[1] != '_' and fi[-2:]=='py':
                print('handle', fi_d)
                with open(fi_d, 'r') as fp:
                    text = fp.read()
                pat = 'register.*?\"(.*?)\"'
                ops = re.findall(pat,text,re.S|re.M)
                ops = [i.replace('.', '_') for i in ops]
                global opsk
                opsk = opsk + ops
                    # for line in text:
                    #     if 'register' in line:
                    #         matched = re.search('(".+?")', line.replace('.', '_'))
                    #         if matched:
                    #             global opsk
                    #             opsk.append(matched.group().strip('"'))


# traverse subdirs recursively
# gci(path)
opsk= []
subdirs =[i for i in  os.listdir(path) if os.path.isdir(os.path.join(path,i))  ]
files =[i for i in  os.listdir(path) if os.path.isfile(os.path.join(path,i)) and i[0]=='_' and i[1]!='_' ]
subdirs = getkeys(subdirs)
print(files)
for dir in subdirs:
   subpath =  os.path.join(path,dir)
   gci(subpath)
   opsk = list(set(opsk))
   opsk = sorted(opsk,reverse=True)
   ops[dir] = opsk
   opsk= []


# op/_tensor.py etc direct files in origin dir
for file in files:
    filepath = os.path.join(path,file)
    if file[0]=='_' and file[1]!='_' and file[-2:]=='py':
        # print(dir+ '\\'+  filename)
        with open(filepath) as fp:
            text = fp.read()
        pat = 'register.*?\"(.*?)\"'
        op = re.findall(pat,text,re.S|re.M)
        op = [i.replace('.', '_') for i in op]
        file = file[1:-3]
        ops[file] =  op
            # for line in text:
            #     if 'register' in line:
            #         matched = re.search('(".+?")',line.replace('.','_'))
            #         if matched:
            #             ops['simple'].append(matched.group().strip('"'))
allops= []
opscoding = dict()
for key in ops.keys():
    ops[key] = list(set(ops[key]))
    ops[key] = sorted(ops[key],reverse=True)
    allops+= ops[key]
    print(key,len(ops[key]),ops[key])
allops = list(set(allops))
print(len(allops))


#------------------------------dict generation
for key in ops.keys():
    for op in ops[key]:
        # subdirs
        if key =='dyn' :
            opscoding[op]=500
        elif key =='image':
            opscoding[op]=300
        elif key =='vision'or key =='random':
            opscoding[op]=200
        elif key =='nn':
            if 'without_weight' in op:
                opscoding[op]=100
            elif 'weight_transform' in op or 'flatten' in op or 'nn_sparse_transpose'==op or \
                'sparse_fill_empty_rows' in op or 'sparse_to_dense' in op or 'sparse_reshape'in op or\
                  'batch_to' in op or 'space_to' in op or 'depth_to' in op or 'pad' in op  :
                opscoding[op]=0
            elif 'conv3d' in op or 'conv2d'in op or 'dense'in op or 'conv1d'in op or 'matmul'in op:
                opscoding[op]=100
            elif 'pool' in op:
                opscoding[op]=50
            elif 'softmax' in op or 'relu' in op or 'norm' in op or 'lrn' in op:
                opscoding[op]=20
            elif 'add' in op:
                opscoding[op]=2
            else:
                opscoding[op]=30
        # files
        elif key =='reduce' or key== 'algorithm' or key == 'math':
            opscoding[op]=20
        elif key =='transform':
            opscoding[op]=0
        elif key =='tensor_grad':
            continue
        elif key == 'tensor':
            if 'log' in op or 'exp'in op or 'power'in op or 'sigmoid' in op:
                opscoding[op]=10
            elif 'rsqrt' == op or 'divide'==op:#!!!
                opscoding[op]=4
            elif 'isinf'==op:  # special collision
                opscoding[op]=2
            elif 'sin' in op or 'cos' in op or 'tan' in op:
                opscoding[op]=5
            elif 'cast' in op or 'to_like' in op or 'transpose'==op:
                opscoding[op]=0
            elif 'bit'in op or  'shift' in op or 'less' in op or 'greater' in op or \
                'equal' in op or 'logic' in op or 'like' in op:
                opscoding[op]=1
            else:
                opscoding[op]=2
        else:
            print(key,op,'can not be handled') # reshape_nop

import json
dpath = './src/op'
dumpath = os.path.join(dpath,'opcoding.json')
dumpath2 = os.path.join(dpath,'opname.json')
# special ops handle

opscoding['reshape_nop'] = 0

opname = sorted(opscoding,key=lambda item: item[0],reverse=True)
print(opscoding)
with open(dumpath,'w') as fp:
    json.dump(opscoding,fp)
with open(dumpath2,'w') as fp:
    json.dump(opname,fp)
"""
keys ['random', 'memory', 'image', 'dyn', 'contrib', 'nn', 'vm', 'vision', 'annotation']
['_reduce.py', '_algorithm.py', '_math.py', '_make.py', '_transform.py', '_tensor_grad.py', '_tensor.py']
handle /home/user/tvm/python/tvm/relay/op/random/_make.py
handle /home/user/tvm/python/tvm/relay/op/random/_kernel.py
handle /home/user/tvm/python/tvm/relay/op/memory/_make.py
handle /home/user/tvm/python/tvm/relay/op/image/_make.py
handle /home/user/tvm/python/tvm/relay/op/image/_image.py
handle /home/user/tvm/python/tvm/relay/op/dyn/_algorithm.py
handle /home/user/tvm/python/tvm/relay/op/dyn/image/_make.py
handle /home/user/tvm/python/tvm/relay/op/dyn/image/_image.py
handle /home/user/tvm/python/tvm/relay/op/dyn/_make.py
handle /home/user/tvm/python/tvm/relay/op/dyn/nn/_nn.py
handle /home/user/tvm/python/tvm/relay/op/dyn/nn/_make.py
handle /home/user/tvm/python/tvm/relay/op/dyn/_transform.py
handle /home/user/tvm/python/tvm/relay/op/dyn/_tensor.py
handle /home/user/tvm/python/tvm/relay/op/contrib/_ethosn.py
handle /home/user/tvm/python/tvm/relay/op/nn/_nn.py
handle /home/user/tvm/python/tvm/relay/op/nn/_make.py
handle /home/user/tvm/python/tvm/relay/op/vm/_ffi_api.py
handle /home/user/tvm/python/tvm/relay/op/vision/_rcnn.py
handle /home/user/tvm/python/tvm/relay/op/vision/_make.py
handle /home/user/tvm/python/tvm/relay/op/vision/_vision.py
handle /home/user/tvm/python/tvm/relay/op/vision/_yolo.py
handle /home/user/tvm/python/tvm/relay/op/annotation/_make.py
random 5
memory 0
image 7
dyn 16
contrib 0
nn 66
vm 0
vision 9
annotation 0
complex 73
simple 84
260 {'random': ['random_uniform', 'random_threefry_split', 'random_threefry_generate', 'random_normal', 'random_multinomial'], 'memory': [], 'image': ['image_resize3d', 'image_resize2d', 'image_resize1d', 'image_grid_sample', 'image_dilation2d', 'image_crop_and_resize', 'image_affine_grid'], 'dyn': ['dyn_zeros', 'dyn_topk', 'dyn_tile', 'dyn_strided_slice', 'dyn_squeeze', 'dyn_sparse_to_dense', 'dyn_reshape', 'dyn_ones', 'dyn_one_hot', 'dyn_nn_upsampling3d', 'dyn_nn_upsampling', 'dyn_nn_pad', 'dyn_image_resize2d', 'dyn_full', 'dyn_expand_dims', 'dyn_broadcast_to'], 'contrib': [], 'nn': ['nn_upsampling3d', 'nn_upsampling', 'nn_sparse_transpose', 'nn_sparse_dense', 'nn_sparse_conv2d', 'nn_sparse_add', 'nn_space_to_depth', 'nn_space_to_batch_nd', 'nn_softmax', 'nn_relu', 'nn_prelu', 'nn_pad', 'nn_nll_loss', 'nn_mirror_pad', 'nn_max_pool3d', 'nn_max_pool2d_grad', 'nn_max_pool2d', 'nn_max_pool1d', 'nn_matmul', 'nn_lrn', 'nn_log_softmax', 'nn_leaky_relu', 'nn_internal_sparse_dense_padded', 'nn_global_max_pool2d', 'nn_global_avg_pool2d', 'nn_fifo_buffer', 'nn_fast_softmax', 'nn_dilate', 'nn_depth_to_space', 'nn_dense', 'nn_deformable_conv2d', 'nn_cross_entropy_with_logits', 'nn_cross_entropy', 'nn_correlation', 'nn_conv3d_transpose', 'nn_conv3d', 'nn_conv2d_transpose', 'nn_conv2d_backward_weight', 'nn_conv2d', 'nn_conv1d_transpose', 'nn_conv1d', 'nn_contrib_depthwise_conv2d_NCHWc', 'nn_contrib_dense_pack', 'nn_contrib_conv3d_winograd_weight_transform', 'nn_contrib_conv2d_winograd_weight_transform', 'nn_contrib_conv2d_winograd_nnpack_weight_transform', 'nn_contrib_conv2d_gemm_weight_transform', 'nn_contrib_conv2d_NCHWc', 'nn_bitserial_dense', 'nn_bitserial_conv2d', 'nn_bitpack', 'nn_bias_add', 'nn_batch_to_space_nd', 'nn_batch_norm', 'nn_batch_matmul', 'nn_batch_flatten', 'nn_avg_pool3d', 'nn_avg_pool2d_grad', 'nn_avg_pool2d', 'nn_avg_pool1d', 'nn_adaptive_max_pool3d', 'nn_adaptive_max_pool2d', 'nn_adaptive_max_pool1d', 'nn_adaptive_avg_pool3d', 'nn_adaptive_avg_pool2d', 'nn_adaptive_avg_pool1d'], 'vm': [], 'vision': ['vision_yolo_reorg', 'vision_roi_pool', 'vision_roi_align', 'vision_proposal', 'vision_non_max_suppression', 'vision_multibox_transform_loc', 'vision_multibox_prior', 'vision_get_valid_counts', 'vision_all_class_non_max_suppression'], 'annotation': [], 'complex': ['zeros_like', 'unravel_index', 'trunc_mod', 'trunc_divide', 'strided_slice', 'strided_set', 'sparse_to_dense', 'sparse_reshape', 'sparse_fill_empty_rows', 'sliding_window', 'slice_like', 'shape_of', 'sequence_mask', 'scatter_nd', 'scatter_elements', 'right_shift', 'reverse_sequence', 'reshape_like', 'ones_like', 'one_hot', 'not_equal', 'nn_softmax', 'nn_relu', 'nn_max_pool2d', 'nn_matmul', 'nn_log_softmax', 'nn_global_avg_pool2d', 'nn_dense', 'nn_cross_entropy_with_logits', 'nn_cross_entropy', 'nn_conv2d', 'nn_bias_add', 'nn_batch_matmul', 'nn_batch_flatten', 'nn_avg_pool2d', 'ndarray_size', 'meta_schedule_layout_transform', 'matrix_set_diag', 'logical_xor', 'logical_or', 'logical_not', 'logical_and', 'less_equal', 'left_shift', 'layout_transform', 'invert_permutation', 'greater_equal', 'gather_nd', 'full_like', 'floor_mod', 'floor_divide', 'fixed_point_multiply_per_axis', 'fixed_point_multiply', 'fast_tanh', 'fast_exp', 'fast_erf', 'expand_dims', 'dyn_zeros', 'dyn_reshape', 'dyn_ones', 'device_copy', 'contrib_reverse_reshape', 'collapse_sum_to', 'collapse_sum_like', 'cast_like', 'broadcast_to_like', 'broadcast_to', 'bitwise_xor', 'bitwise_or', 'bitwise_not', 'bitwise_and', 'auto_scheduler_layout_transform', 'adv_index'], 'simple': ['zeros', 'where', 'variance', 'unique', 'trunc', 'trilu', 'transpose', 'topk', 'tile', 'tanh', 'tan', 'take', 'sum', 'subtract', 'stft', 'stack', 'squeeze', 'sqrt', 'split', 'sort', 'sinh', 'sin', 'sign', 'sigmoid', 'searchsorted', 'rsqrt', 'round', 'reverse', 'reshape', 'repeat', 'reinterpret', 'prod', 'power', 'ones', 'negative', 'multiply', 'mod', 'minimum', 'min', 'meshgrid', 'mean', 'maximum', 'max', 'log2', 'log10', 'log', 'less', 'isnan', 'isinf', 'isfinite', 'greater', 'gather', 'full', 'floor', 'exp', 'erf', 'equal', 'einsum', 'divide', 'dft', 'cumsum', 'cumprod', 'cosh', 'cos', 'copy', 'concatenate', 'clip', 'ceil', 'cast', 'atanh', 'atan', 'asinh', 'asin', 'argwhere', 'argsort', 'argmin', 'argmax', 'arange', 'any', 'all', 'add', 'acosh', 'acos', 'abs']}
{'nn_internal.sparse_dense_padded', 'reshape_nop,reshape_like', 'nn_re'}
"""
