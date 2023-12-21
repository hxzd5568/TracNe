# Test industrial models

Validating industrial models is slightly different from testing normal Relay models. This handbook provides instructions for users.

## Prepare models

### PyTorch models
TracNe supports PyTorch models in $pt$ format which is widely used for restoring trained models. Users can use tests/test_torchutils.py to download models.
```shell
python test_torchutils.py  Model_name
```
$Model\_name$ is the name of the known models such as vgg, resnet, mobilenet, and inception.
This process will download the model to the workspace dnn/out/ and compile model automatically.

### Onnx models

TracNe provides a script to download pretrained and quantized ONNX models.
```shell
python download_qnnmodel.py  
```
Users can choose the model index from the [website](https://sparsezoo.neuralmagic.com/), and then assign parameter $stub$ in the script using the index.

## Detect numerical errors in models 

```shell
python test_fuzztorch.py ./dnn/out/model  
```
This script supports searching numerical errors caused by framework differences and optimizations.
Users can choose to detect framework errors by setting $fuzzframe$ True. The default targets optimization errors. 