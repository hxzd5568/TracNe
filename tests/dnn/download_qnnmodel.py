from sparsezoo import Model
import os 
import onnx
import re
from google.protobuf.json_format import MessageToDict
# !!!change
stub = 'zoo:cv/segmentation/yolov8-s/pytorch/ultralytics/coco/base_quant-none'
case_path = './out/obertamqatransformer'



model = Model(stub)
model.download()
locpath = model.path 
locpath = '/root/.cache/sparsezoo/9273788e-f3a8-41f8-8e93-e3249621866c'
onnx_model = onnx.load(locpath+"/deployment/model.onnx")
onnx.checker.check_model(onnx_model)
import tvm
from tvm import relay
import numpy as np

graph = onnx_model.graph

for _input in graph.input:
    print(MessageToDict(_input))

for _input in graph.output:
    print(MessageToDict(_input))
input_shape = (1,384)#(1,3,640,640)
input_shape2 = (1,384)
input_shape3 = (1,384)
# {'name': 'input_ids', 'type': {'tensorType': {'elemType': 7, 'shape': {'dim': [{'dimParam': 'batch'}, {'dimValue': '384'}]}}}}
# {'name': 'attention_mask', 'type': {'tensorType': {'elemType': 7, 'shape': {'dim': [{'dimParam': 'batch'}, {'dimValue': '384'}]}}}}

# maybe 'token_type_ids' 'token_type_ids':input_shape2
# graph inputs
# for _input in onnx_model.graph.input:
#     m_dict = MessageToDict(_input)
#     dim_info = m_dict.get("type").get("tensorType").get("shape").get("dim")  # ugly but we have to live with this when using dict
#     input_shape = [int(d.get("dimValue")) for d in dim_info]  # [4,3,384,640]
#     print(input_shape)
#     break
mod, params = relay.frontend.from_onnx(onnx_model, {"input_ids": input_shape,'attention_mask':input_shape2})#input (general)/images #{"input": input_shape}
# for mask model 'token_type_ids':(1,512)

if not os.path.exists(case_path):
    os.mkdir(case_path)

def normalname(mod):  # return new_mod and if changed flag
    changeflag = []
    mod = re.sub('::','',mod,count=0,flags=re.S|re.M)
    pat = '(?P<value>%[a-zA-Z_]+[.a-zA-Z_0-9]+)'
    def update_internal(matched):
        changeflag.append(1)
        return matched.group('value').replace('_','').replace('.','')
    mod = re.sub(pat, update_internal, mod,count=0, flags=re.M|re.S)
    # pat2 = '(?P<value>%p)'
    # def changep(matched):
    #     changeflag.append(1)
    #     return matched.group('value').replace('p','n')
    # mod = re.sub(pat2, changep, mod,count=0, flags=re.M|re.S)
    return mod

def renewmodel(mod:tvm.IRModule, case_path:str)-> tvm.IRModule:
    modstr = mod.astext()
    modstr = normalname(modstr)
    if not os.path.exists(case_path):
        os.mkdir(case_path)
    with open(f'{case_path}/code.txt','w') as fp:
        fp.write(modstr)
    with open(f'{case_path}/code.txt', 'r') as f:
        return relay.parse(f.read())

def normalkeys(strs):
    strs= strs.replace('::','')
    strs= strs.replace('_','')
    strs= strs.replace('.','')
    print(strs)
    return strs

with open(case_path+'/code.txt','w') as f:
    f.write(mod.astext())

nparams =dict()
for k, v in params.items():
    nparams[normalkeys(k)]= v.numpy()
mod = renewmodel(mod,case_path=case_path)
def save_arr( params,case_path):
            # print('type',type(list(params.values())[0]))
            inputarr = dict()
            for k, v in params.items():
                inputarr[k]=v
            path_params = os.path.join(case_path, 'torch_inputs.npz')
            np.savez(path_params, **inputarr)
save_arr(nparams,case_path=case_path)

print('success')



# ******************download model
# stub = 'zoo:cv/classification/resnet_v1-34/pytorch/sparseml/imagenet/pruned-conservative'
# # stub = "zoo:nlp/question_answering/bert-base_cased/pytorch/huggingface/squad/pruned90-none"
# # path = '/root/.cache/sparsezoo/31912a93-e8f0-4648-9a7d-3b116661ac56'
# model = Model(stub)#download_path = path
# model.download()
# print(model.path)



# print(type(model))
# print(model.onnx_model)

# from sparseml.pytorch.utils import ModuleExporter

# exporter = ModuleExporter(model, output_dir=os.path.join(".", "onnx-export"))
# exporter.export_onnx(sample_batch=torch.randn(1, 1, 28, 28))
# for k,v in model.named_parameters():
#     print(k,v.detach().numpy().shape)

# import os
# import torch
# from sparseml.pytorch.models import mnist_net
# from sparseml.pytorch.utils import ModuleExporter,TrainingMode

# model = mnist_net()
# exporter = ModuleExporter(model, output_dir=os.path.join(".", "onnx-export"))
# exporter.export_onnx(sample_batch=torch.randn(1, 1, 28, 28),training=TrainingMode.EVAL)

# from sparsezoo import Zoo
# from sparsezoo.models.classification import resnet_50

# search_model = resnet_50()
# sparse_models = Zoo.search_sparse_models(search_model)

# print(sparse_models)
# print(type(sparse_models))

'''
nlp
zoo:nlp/question_answering/oberta-small/pytorch/huggingface/squad_v2/pruned90_quant-none  * download  \/
            '/root/.cache/sparsezoo/321dac91-0d0a-4815-bc95-9ec13ef997aa'
zoo:nlp/masked_language_modeling/obert-small/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni * download x frame error too large
            /root/.cache/sparsezoo/9887ee48-6cd3-4885-8105-51213ac2ff6f
zoo:nlp/text_classification/biobert-base_cased/pytorch/huggingface/pubmedqa/pruned80_quant-none-vnni    can't be downloaded

zoo:nlp/masked_language_modeling/oberta-base/pytorch/huggingface/wikipedia_bookcorpus/base_quant-none       * x frame error too large
            /root/.cache/sparsezoo/0dd6c83d-7ef2-4d03-9e1d-9ba04e4ed70e

zoo:nlp/question_answering/oberta-medium/pytorch/huggingface/squad_v2/pruned90_quant-none
            /root/.cache/sparsezoo/9273788e-f3a8-41f8-8e93-e3249621866c


zoo:nlp/masked_language_modeling/oberta-base/pytorch/huggingface/wikipedia_bookcorpus/pruned95_quant-none       

zoo:nlp/text_classification/biobert-base_cased/pytorch/huggingface/pubmedqa/pruned90-none

zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni

cv-classify:
zoo:cv/classification/resnet_v1-18/pytorch/sparseml/imagenet/pruned85_quant-none-vnni   * pass 


cv-detection:
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_96        * pass  x
    '/root/.cache/sparsezoo/e6b94348-7bb1-4fc2-bd41-f2f70cfd264b'
zoo:cv/detection/yolov5-n6/pytorch/ultralytics/coco/pruned40_quant-none-vnni           * x

zoo:cv/detection/yolov5-s/pytorch/ultralytics/voc/pruned_quant-aggressive_96           x


quant 
zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned_quant-moderate   * dui not pass x
zoo:cv/segmentation/yolov8-l/pytorch/ultralytics/coco/base_quant-none
zoo:cv/segmentation/yolov8-n/pytorch/ultralytics/coco/base_quant-none
zoo:cv/segmentation/yolov8-s/pytorch/ultralytics/coco/base_quant-none           \/
     /root/.cache/sparsezoo/e37ce71a-d72f-42ec-b87f-21994e8fb6df
zoo:cv/segmentation/yolov8-x/pytorch/ultralytics/coco/base_quant-none

6 outputs
def @main(%images: Tensor[(2, 3, 640, 640), uint8] /* ty=Tensor[(2, 3, 640, 640), uint8] span=/model.0/conv/module/Conv_quant.images:0:0 */) -> (Tensor[(2, 116, 8400), float32], Tensor[(2, 144, 80, 80), float32], Tensor[(2, 144, 40, 40), float32], Tensor[(2, 144, 20, 20), float32], Tensor[(2, 32, 8400), float32], Tensor[(2, 32, 160, 160), float32]) {
zoo:cv/detection/yolov8-m/pytorch/ultralytics/coco/pruned75_quant-none
zoo:nlp/text_classification/biobert-base_cased/pytorch/huggingface/bioasq/pruned90_quant-none

prune:


zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned90-none      




'''
'''
stub = 'zoo:nlp/question_answering/oberta-small/pytorch/huggingface/squad_v2/pruned90_quant-none'
case_path = './out/transformeroberts'



# model = Model(stub)
# model.download()
# locpath = model.path
locpath = '/root/.cache/sparsezoo/321dac91-0d0a-4815-bc95-9ec13ef997aa'
'''
