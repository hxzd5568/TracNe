'''

only need a tempdir to analyse
config={"relay.FuseOps.max_depth": max_fused_ops

# cast don't impact the fuse. So we can insert it to enhance precision.
Example of trace two different executing graphs:
L1:

new_x____topo-index:0____output-num:0 (4, 3)
new_y____topo-index:1____output-num:0 (4, 3)
tvmgen_default_fused_abs____topo-index:2____output-num:0 (4, 3)
tvmgen_default_fused_sqrt____topo-index:3____output-num:0 (4, 3)
tvmgen_default_fused_divide____topo-index:4____output-num:0 (4, 3)
tvmgen_default_fused_sum____topo-index:5____output-num:0 (4,)
tvmgen_default_fused_tan____topo-index:6____output-num:0 (4,)
tvmgen_default_fused_nn_relu____topo-index:7____output-num:0 (4,)

L5:

new_x____topo-index:0____output-num:0 (4, 3)
new_y____topo-index:1____output-num:0 (4, 3)
tvmgen_default_fused_abs_rsqrt_multiply_sum____topo-index:2____output-num:0 (4,)
tvmgen_default_fused_tan_nn_relu____topo-index:3____output-num:0 (4,)

match op: topo-index:x+node_ops-1 = topo-index:y
x from L5  y from L1

{topo-index:x,{next-node-topo-index:y, errormessage}}
struct errormessage:{error,[output1,output5],node-name(of x),fix-method}

'''
# we can get a rough compare by comparing the same actually topoindex.
# But becuase of fuseops it is too rough.

# But it is benefical because error always immese from one pattern not a op.
# then we enhance the precision of subgraph and rerun.

# question solved.
# ops:g

import re
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
import json
import string
from multiprocessing import Process, Queue
import tvm.testing
import json

comparebaseorder = 0

def remove_virtarget(ncode):
    rncode = re.sub('({virtual_.*?}:)', ':', ncode,count=0, flags=re.M|re.S)
    rncode = re.sub('(virtual_.*?->)', ') ->', rncode,count=0, flags=re.M|re.S)
    return rncode
def remove_primary(code):
    return  re.sub('(, Primitive\=.*?->)', ') ->', code,count=0, flags=re.M|re.S)

def get_primfunc(opt_level,mod,target='llvm'):# i.e. get tvm.ir.module.IRModule
    target, target_host = tvm.target.Target.canon_target_and_host(target)
    with tvm.transform.PassContext(opt_level=opt_level):# config={"relay.FuseOps.max_depth": 10}
        prim_mod, _ = relay.optimize(mod, target)
        code = remove_virtarget(str(prim_mod))
    return code
def read_primfunc(path):
    with open(path,'r') as fp:
        return fp.read()

class Matchinfo():
    def __init__(self,  opt_order:int, unopt_order: int, ):
        self.opt_order =opt_order
        self.unopt_order = unopt_order

class Errormessage:
    def __init__(self, error: np = -1,hash='', opindex: int = -1, node_name_hint: str = ''):
        self.error = error
        self.params_keys = {'unopt_params_keys': [''], 'opt_params_keys': ['']}
        self.params = {'unopt_params': [''], 'opt_params': ['']}
        self.node_name_hint = node_name_hint
        self.opindex = opindex
        self.hash = hash


class Trace_item(Errormessage):
    def __init__(self,  topoindex: int, pre_topoindex: List[int],  errormessage=Errormessage()):
        self.topoindex = topoindex
        self.pre_topoindex = pre_topoindex.copy()
        self.errormessage = errormessage

class Actualnodeinfo:
    def __init__(self, nodeindex: int, t_order:int, coding:np.uint64, shape:List[int],tpnum:int,\
                   params_keys: List[str],):#params: List[np.ndarray]
        self.nodeindex = nodeindex
        self.torder = t_order
        self.coding = coding
        self.shape = shape
        self.tpnum = tpnum
        self.params_keys = params_keys.copy()  # many outputs

class Trace_error():
    def __init__(self, case_path) :
        self.case_path = case_path
        self.opt_root = self.case_path+'/L5/'
        self.unopt_root = self.case_path+'/L1/'
        with open(f'{self.case_path}/code.txt', 'r') as f:
            self.mod = relay.parse(f.read())
        self.optprim_mod= read_primfunc(f'{self.case_path}/optirmod.txt')
        self.unoptprim_mod= read_primfunc(f'{self.case_path}/unoptirmod.txt')
        with open('../src/op/opname.json','r')as fp:
            self.opname = json.load(fp)
        with open('../src/op/opcoding.json','r')as fp:
            self.opcoding = json.load(fp)
        self.qnn = False
        if len(re.findall(r'qnn\.',str(self.mod),flags=re.S|re.M))>0:
            self.qnn = True

    # add a new fix attribute
    # #if prenode is reshape, and there are more than one reshape node, then it may find incorrect reshape
    # because the debug info is not clear for reshape_nop
    def get_topo_structure(self,graph_json_path: str, topo_name_index: Dict[str, int]) -> List[Trace_item]:
        # graph node order is consistent with nature number order
        trace_messages: Dict[int:Trace_item] = dict()
        with open(graph_json_path, 'r') as gfile:
            gdata = json.load(gfile)
        self.kerneldict = gdata
        length = len(gdata['nodes'])
        for i in range(length):
            node = gdata['nodes'][i]
            index = i #topo_name_index[node['name'].rstrip('_')]
            if node['op'] == 'param':  # local params also does not have inputs
                trace_messages[index] = Trace_item(index,[-1],
                                                 Errormessage().__dict__).__dict__
            else:
                hash = node["attrs"]['hash']
                trace_messages[index] = Trace_item(index, [topo_name_index[i.rstrip('_')] for i in
                                                node['inputs']],# ! may need reshape_nop handler
                                                 Errormessage(hash=hash).__dict__).__dict__
        return trace_messages


    def get_topo_name_index(self,keys: List[str]):
        topo_name = dict()
        for key in keys:
            name, index = key.split('____')[0], int(
                key.split('____')[1].split(':')[1])
            topo_name[name] = index
        return topo_name

    def rm_invalid(self,str):
        pat = '(_e[a-f0-9]+e)'
        #print(str)
        str = re.sub(pat,'',str,count=0, flags=re.M|re.S)
        #print(str)
        return str
    def handle_strings(self, funcname_all: str):
        print(':'*10)
        # special name + regular name + from_kernel_name(when ops_number>9)
        def get_hash_from_name(funcname):
            nodes = self.kerneldict['nodes']
            for node in nodes:
                if funcname == node['name'].rstrip('_'):
                    return node["attrs"]['hash']
        def get_modstr_from_hash(hash):
            #pattern = 'fn.*'+hash+'.*?}'
            pattern = hash+'.*?}'
            matched  =  re.search(pattern,self.unoptprim_mod,flags=re.M|re.S)
            if matched is not None:
                l,r = matched.span()
                ocode = self.unoptprim_mod
            else:
                matched  =  re.search(pattern,self.optprim_mod,flags=re.M|re.S)
                l,r = matched.span()
                ocode = self.optprim_mod
            origin_func = ocode[l:r]
            return remove_primary(origin_func)
        def getops_from_kernel(text):
            pat = r'(?P<value>\/\*.*?\*\/)'
            text = re.sub(pat,'',text,count=0,flags=re.S|re.M)
            pat2 = r'[a-zA-Z_]+[.a-zA-Z0-9_]+\('
            ops = re.findall(pat2,text,flags=re.S|re.M)
            ops = [i.rstrip('(').replace('.','_')  for i in ops]
            return ops
        def getconst_from_kernel(text):
            pat2 = r'f\ \/\*\ ty\='
            ops = re.findall(pat2,text,flags=re.S|re.M)
            ops = len(ops)
            return ops
        funcname = funcname_all.split('____')[0]
        print(funcname)
        lendefault = len('tvmgen_default_fused_')
        current_name = funcname[lendefault:].rstrip(string.digits).rstrip('_')
        while(current_name.split('_')[-1].isnumeric()):
            current_name = current_name.rstrip(string.digits).rstrip('_')
        current_name = self.rm_invalid(current_name)
        index = int(funcname_all.split('____')[1].split(':')[1])
        op_numbers = 0
        coding = np.uint64(0)
        ops = []        # record ops along kernel order
        def call_kernel_analysis():
            hash = get_hash_from_name(funcname)
            print(hash)
            kernelstr = get_modstr_from_hash(hash)
            ops = getops_from_kernel(kernelstr)
            constnum = getconst_from_kernel(kernelstr)
            return ops,constnum
        ops, currtpnum = call_kernel_analysis()
        op_numbers = len(ops)
        for op in ops:
            if op in self.opname:
                coding += np.uint64(self.opcoding[op])
            else:
                print('can not handle op:',op)
                exit()
        print(op_numbers, index, coding)
        print(ops)
        return op_numbers, index, coding, currtpnum


    def reorderfunc_withindex(self, funcname_alls) -> List[str]:
        def fitness(item):
            ten = int(item.split('____')[1].split(':')[1])
            one = int(item.split('____')[2].split(':')[1])
            return ten*10+one
        return sorted(funcname_alls,key=lambda item: fitness(item))

    def get_shape(self,key):
        gdata = self.kerneldict
        for node in gdata['nodes']:
            index = node['name'].rstrip('_')
            if key == index:
                shape = node['shape'].copy()
                return shape
        print(key,' no shape')
        exit()

    def get_actual_index_param(self, params: Dict[str, np.ndarray]) -> Dict[int,Actualnodeinfo]:
        actualparams = dict()
        keys = self.reorderfunc_withindex(params.keys())
        nodeindex = 0
        opindex = 0  # topo order
        baseindex = 0
        order = 0
        basecoding = np.uint64(0)
        tpnum = 0  # temp time node has how many global params
        for key in keys:
        # distinguish placehold and function node
            if ('tvmgen_default' in key or 'reshape_nop' in key):
                addops, nodeindex , coding, currtpnum = self.handle_strings(key)
                opindex = baseindex + addops - 1
                baseindex += addops
                basecoding += coding
                tpnum += currtpnum

            else:
                nodeindex = int(key.split('____')[1].split(':')[1])
                opindex = baseindex + 1 - 1
                baseindex +=1
                tpnum += 1
            # get shape
            shape = self.get_shape(key.split('____')[0].split(':')[0])
            # populate
            if order not in actualparams.keys():
                actualparams[order] = Actualnodeinfo(nodeindex,opindex,basecoding,shape,tpnum, [key])
            else:
                # actualparams[order].params.append(params[key])
                actualparams[order].params_keys.append(key)
            order += 1
        return actualparams

    # helper function
    def get_node_name(self):
        params_path = os.path.join(self.opt_root,
                                '_tvmdbg_device_CPU_0/output_tensors.params')
        params: Dict[str, np.array] = relay.load_param_dict(bytearray(open(
                params_path, "rb").read()))
        keys = self.reorderfunc_withindex(params.keys())
        unnoptparams_path = os.path.join(self.unopt_root,
                                        '_tvmdbg_device_CPU_0/output_tensors.params')
        params1: Dict[str, int] = relay.load_param_dict(bytearray(open(
            unnoptparams_path, "rb").read()))
        keys2 = self.reorderfunc_withindex(params1.keys())
        fixportpath = os.path.join(self.case_path, 'node_names')
        with open(fixportpath,'w') as fp:
                fp.write('\n\n********opt5'+'*'*50+'\n\n')
                print(self.optprim_mod,file=fp)
                fp.write('\n\n********opt1'+'*'*50+'\n\n')
                print(self.unoptprim_mod,file=fp)
        graph_json_path = os.path.join(self.opt_root,
                            '_tvmdbg_device_CPU_0/_tvmdbg_graph_dump.json')
        with open(graph_json_path, 'r') as gfile:
            gdata = json.load(gfile)
        self.kerneldict = gdata
        optparams = self.get_actual_index_param(params)

        names = [(i,key.split('____')[0].split(':')[0]) for i,key in enumerate(keys)]
        graph_json_path = os.path.join(self.unopt_root,
                    '_tvmdbg_device_CPU_0/_tvmdbg_graph_dump.json')
        with open(graph_json_path, 'r') as gfile:
            gdata = json.load(gfile)
        self.kerneldict = gdata
        unoptparams = self.get_actual_index_param(params1)
        names2 = [(i,key.split('____')[0].split(':')[0]) for i,key in enumerate(keys2)]
        with open(fixportpath,'a') as fp:
            fp.write('\n\n********opt5'+'*'*50+'\n\n')
            for name in names:
                print(name,file=fp)
            print('op topo:', [(key, value.nodeindex)
                for key, value in optparams.items()],file = fp)
            fp.write('\n\n********opt1'+'*'*50+'\n\n')
            for name in names2:
                print(name,file=fp)
            print('op unopt_topo:', [(key, value.nodeindex)
                for key, value in unoptparams.items()],file = fp)
        del params
        del params1

        return

    def locate_naninf(self,modstr:str):
        print('enter locating')
        if modstr=='1':
            dump_root= self.case_path+'/L1/'
        else:
           dump_root= self.case_path+'/L5/'
        # binary find  using a list [nodeindex: key]
        params_path = os.path.join(dump_root,
                                '_tvmdbg_device_CPU_0/output_tensors.params')
        params: Dict[str, np.array] = relay.load_param_dict(bytearray(open(
                    params_path, "rb").read()))
        keys  = self.reorderfunc_withindex(list(params.keys()))
        lens = len(keys)
        # locate last nonnan
        def isnan(y_true):
            if np.isinf(y_true).any()==1 or np.isnan(y_true).any()==1:
                return True
            else:
                return False

        def binary_search(l,r):
            if(l>=r):
                return l
            m = int((l+r)/2)
            if((not isnan(params[keys[m]].numpy())) and isnan(params[keys[m+1]].numpy())):
                return m
            if(isnan(params[keys[m]].numpy()) ):
                binary_search(l,m-1)
            else:
                binary_search(m+1, r)
        lastindex = binary_search(0,int(lens-1))
        fixportpath = os.path.join(dump_root, 'Locate_NAN_Report')
        with open(fixportpath,'a') as fp:
                fp.write('Located')
                fp.write('\n\n\n'+'*'*50+'\n')
                fp.write('The first nan/inf incurs in pattern:',keys[lastindex+1])
                fp.write('\n\n\n'+'*'*50+'\n')
                fp.write('The input arr is:\n\n')
                fp.write(params[keys[lastindex]])
                fp.write('\n\n\n'+'*'*50+'\n')
                fp.write('The output arr is:\n\n')
                fp.write(params[keys[lastindex+1]])

    def MSE(self,y_true, y_pred,):  #precision along with  tf.keras.metrics.MeanRelativeError
        if np.isinf(y_true).any()==1 or np.isnan(y_true).any()==1:
            print('y_true have inf\\nan:locating...')
            #self.locate_naninf('1')
            return 0
        if np.isinf(y_pred).any()==1 or np.isnan(y_pred).any()==1:
            print('y_pred have inf\\nan:locating...')
            return 0
            #self.locate_naninf('5')
        else:
            pass
        d = np.abs(y_true.astype(np.float64) - y_pred)
        relative_error = np.average( d \
                / (np.abs(y_true).astype(np.float64) + 1e-8) * np.not_equal(y_true, 0)\
                 + np.equal(y_true, 0)* d )
        return relative_error

    def encode_obj(self,obj):
        obj_dict = {
        }
        obj_dict.update(obj.__dict__)
        return obj_dict

    # look op params' key


    def lookup_keys(self,path):
        data = relay.load_param_dict(bytearray(open(path, "rb").read()))
        for k, v in data.items():
            tvm_array = v.numpy()
            print(k, tvm_array.shape)


    def get_trace_message(self,report:Queue=None) -> str:
        params_path = os.path.join(self.opt_root,
                                '_tvmdbg_device_CPU_0/output_tensors.params')
        graph_json_path = os.path.join(self.opt_root,
                                    '_tvmdbg_device_CPU_0/_tvmdbg_graph_dump.json')
        params: Dict[str, np.array] = relay.load_param_dict(bytearray(open(
                params_path, "rb").read()))
        keys = self.reorderfunc_withindex(params.keys())
        names = [(i,key.split('____')[0].split(':')[0]) for i,key in enumerate(keys)]
        print('opt5 k:',names)
        # params check
        # path_params = os.path.join(self.case_path, 'inputs.npz')
        # with np.load(path_params) as f:
        #     loaded_params = dict(f.items())
        # self.eparams = loaded_params
        # for k, v in self.eparams.items():
        #     for ck in params.keys():
        #         if k in ck:
        #             print(k, ck)
        #             tvm.testing.assert_allclose(v,params[ck].numpy() , rtol=1e-8, atol=1e-8)
        topo_name_index = self.get_topo_name_index(params.keys())
        index_topo_name = dict([val, key] for key, val in topo_name_index.items())
        trace_messages: Dict[int:Trace_item] = self.get_topo_structure(
            graph_json_path, topo_name_index)
        # populate_errormessages(trace_messages, params)
        optparams = self.get_actual_index_param(params)
        print('op topo:', [(value.torder, value.nodeindex)
            for key, value in optparams.items()])
        for key, value in optparams.items():
            print('op opt_topo:', 'key:',key,value.__dict__)
        graph_json_path = os.path.join(self.unopt_root,
                            '_tvmdbg_device_CPU_0/_tvmdbg_graph_dump.json')
        with open(graph_json_path, 'r') as gfile:
            self.kerneldict = json.load(gfile)
        unnoptparams_path = os.path.join(self.unopt_root,
                            '_tvmdbg_device_CPU_0/output_tensors.params')
        params1: Dict[str, int] = relay.load_param_dict(bytearray(open(
            unnoptparams_path, "rb").read()))
        unoptparams = self.get_actual_index_param(params1)

        keys = self.reorderfunc_withindex(params1.keys())
        names = [(i,key.split('____')[0].split(':')[0]) for i,key in enumerate(keys)]
        print('opt1 k:',names)
        self.pararms5 = params
        del params
        self.pararms1 = params1
        del params1
        for key, value in unoptparams.items():
            print('op unopt_topo:','key:',key, value.__dict__)
        diff = 0
        self.matchinfos = []
        def imprecsionmatch(optorder,optnode, unoptparams):
            global comparebaseorder
            for uk, unoptnode in list(unoptparams.items())[comparebaseorder:]:
                unoptcoding =  unoptnode.coding
                optcoding = optnode.coding
                utpnums = unoptnode.tpnum
                tpnums = optnode.tpnum
                uleng = unoptnode.torder
                leng = optnode.torder
                # modify coding amendment
                if self.qnn:
                    pass
                else:
                    if  tpnums <utpnums:
                        optcoding += np.uint64((utpnums-tpnums)*2)
                    else:
                        unoptcoding += np.uint64((tpnums-utpnums)*2)
                # get diff
                diff = float(optcoding-unoptcoding) if optcoding> unoptcoding \
                    else float(unoptcoding-optcoding)
                # get tolerance
                if self.qnn:
                    tolerance = 1 + 5* abs(leng)/10.0  # may modify  #10
                else:
                    tolerance = 1 + abs(leng)/10.0  # may modify  #10
                # print('tolerance',tolerance)
                if diff < tolerance:
                    # shape compare
                    ushape = unoptnode.shape
                    shape =optnode.shape
                    if  np.prod(ushape)!=np.prod(shape):# np.cumprod(ushape)
                        continue
                    if uk+1== len(list(unoptparams.items())):
                        self.matchinfos.append(Matchinfo(optorder, uk))
                        comparebaseorder = uk+1
                        return uk, True
                    # next diff check:
                    unoptnode2 = unoptparams[uk+1]
                    unoptcoding =  unoptnode2.coding
                    optcoding = optnode.coding
                    utpnums = unoptnode2.tpnum
                    tpnums = optnode.tpnum
                    uleng = unoptnode2.torder
                    leng = optnode.torder
                    # modify coding amendment
                    if self.qnn:
                        pass
                    else:
                        if  tpnums <utpnums:
                            optcoding += np.uint64((utpnums-tpnums)*2)
                        else:
                            unoptcoding += np.uint64((tpnums-utpnums)*2)
                    # get diff
                    diff2 = float(optcoding-unoptcoding) if optcoding> unoptcoding \
                        else float(unoptcoding-optcoding)
                    ushape2 = unoptnode2.shape
                    if diff2<diff and (ushape2== ushape or np.prod(ushape)==np.prod(ushape2)):
                        self.matchinfos.append(Matchinfo(optorder, uk+1))
                        comparebaseorder = uk+2
                        return uk+1,True
                    else:
                        self.matchinfos.append(Matchinfo(optorder, uk))
                        comparebaseorder = uk+1
                        return uk, True
            return -1, False


        for key, indexparams in optparams.items():
            outs5 = [self.pararms5[i] for i in indexparams.params_keys]
            if key+1<len(optparams.keys()):
                if indexparams.nodeindex+1 in index_topo_name.keys() and \
                    'fused_layout_transform' in index_topo_name[indexparams.nodeindex+1] \
                    and optparams[key].coding==optparams[key+1].coding:#optparams[key+1].:
                    continue
            unopt_ordernum, flag = imprecsionmatch(key,optparams[key], unoptparams)  # fuzzy match
            if flag==False:
                continue
            print('match',key,unopt_ordernum,flag)
            outs1 = [self.pararms1[i] for i in unoptparams[unopt_ordernum].params_keys]
            diff = 0
            # # if layout_transform make shape diff, then back opt out once.
            # if outs1[0].numpy().shape != outs5[0].numpy().shape and 'layout_transform' in index_topo_name[indexparams.nodeindex] and\
            #     optparams[key].coding==optparams[key-1].coding:
            #     outs5 = [self.pararms5[i] for i in optparams[key-1].params_keys]
            #     assert(outs1[0].numpy().shape != outs5[0].numpy().shape)
            for ro, o in zip(outs1, outs5):
                diff = max(diff, self.MSE(ro.numpy().flatten(), o.numpy().flatten()))
            # print(trace_messages.keys())
            trace_messages[indexparams.nodeindex]['errormessage']['error'] = float(diff)
            trace_messages[indexparams.nodeindex]['errormessage']['opindex'] = key
            trace_messages[indexparams.nodeindex]['errormessage']['params_keys']['unopt_params_keys']\
                = unoptparams[unopt_ordernum].params_keys
            trace_messages[indexparams.nodeindex]['errormessage']['params_keys']['opt_params_keys']\
                = indexparams.params_keys
            if diff > 1e-10:
                trace_messages[indexparams.nodeindex]['errormessage']['params']['unopt_params']\
                    = [np.array2string(i.numpy()) for i in outs1]
                trace_messages[indexparams.nodeindex]['errormessage']['params']['opt_params']\
                    = [np.array2string(i.numpy()) for i in outs5]
            trace_messages[indexparams.nodeindex]['errormessage']['node_name_hint'] = \
                index_topo_name[indexparams.nodeindex] if indexparams.nodeindex in index_topo_name.keys() else 'reshape_nop'
        global comparebaseorder
        comparebaseorder = 0
        dump_path = os.path.join(self.case_path, 'trace.json')
        with open(dump_path, 'w') as fp:
            json.dump(trace_messages, fp, default=self.encode_obj, indent=4,
                    sort_keys=True,)
        if report is not None:
            report.put({'trace stored in file:',dump_path})
        else:
            print('trace stored in file:',dump_path)
        return dump_path
