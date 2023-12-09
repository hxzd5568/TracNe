import re
pat= '[a-zA-Z]+'
matched = re.match(pat,'abs_rsqrt_multiply_sum')
current_name = 'tan_nn_relu'
actualindex = 2
index = 3
special_patterns = ['unravel_index', 'trunc_mod', 'trunc_divide', 'strided_slice', 'sparse_to_dense',
     'slice_like', 'shape_of', 'sequence_mask', 'right_shift', 'reverse_sequence',
     'reshape_like', 'one_hot', 'not_equal', 'nn_upsampling3d', 'nn_upsampling',
     'nn_sparse_transpose', 'nn_sparse_dense', 'nn_sparse_dense', 'nn_sparse_conv2d',
     'nn_space_to_depth', 'nn_pad', 'nn_nll_loss', 'nn_mirror_pad', 'nn_mirror_pad',
     'nn_max_pool2d', 'nn_matmul', 'nn_matmul', 'nn_lrn','nn_relu'
     'nn_internal.sparse_dense_padded', 'nn_global_max_pool2d',
     'nn_global_avg_pool2d', 'nn_fifo_buffer', 'nn_dilate', 'nn_dilate',
     'nn_depth_to_space', 'nn_dense', 'nn_dense', 'nn_dense', 'nn_deformable_conv2d',
     'nn_deformable_conv2d', 'nn_deformable_conv2d', 'nn_cross_entropy_with_logits',
     'nn_cross_entropy', 'nn_conv3d_transpose', 'nn_conv3d', 'nn_conv3d',
     'nn_conv2d_transpose', 'nn_conv2d_transpose', 'nn_conv2d_backward_weight',
     'nn_conv2d_backward_weight', 'nn_conv2d', 'nn_contrib_dense_pack',
     'nn_contrib_conv3d_winograd_weight_transform', 'nn_contrib_conv2d_winograd_weight_transform',
     'nn_contrib_conv2d_winograd_nnpack_weight_transform', 'nn_contrib_conv2d_gemm_weight_transform',
     'nn_contrib_conv2d_NCHWc', 'nn_bitserial_conv2d', 'nn_bitpack', 'nn_batch_matmul',
     'nn_batch_matmul', 'nn_batch_flatten', 'nn_avg_pool2d', 'ndarray_size', 'matrix_set_diag',
     'logical_xor', 'logical_or', 'logical_not', 'logical_and', 'less_equal', 'left_shift',
     'greater_equal', 'gather_nd', 'full_like', 'floor_mod', 'floor_divide', 'fast_tanh',
     'fast_exp', 'fast_erf', 'expand_dims', 'device_copy', 'contrib_reverse_reshape',
     'collapse_sum_to', 'collapse_sum_like', 'cast_like', 'broadcast_to_like', 'broadcast_to',
     'bitwise_xor', 'bitwise_or', 'bitwise_not', 'bitwise_and', 'adv_index']

while(current_name!=''):
    # special name
    for pattern in special_patterns:
        matched = re.match(pattern,current_name)
        if matched is not None:
            actualindex+=1
            l,r = matched.span()
            if len(current_name) > r-l:
                current_name= current_name[r+1:]
            else:
                print(actualindex,index)
                exit() 
    # regular name
    pat= '[a-zA-Z]+'
    matched = re.match(pat,current_name)
    if matched is not None:
            actualindex+=1
            l,r = matched.span()
            if len(current_name) > r-l:
                current_name= current_name[r+1:]
            else:
                current_name = ''
                print(actualindex,index)
                exit() 
    else:
        exit()
print(actualindex,index) 
