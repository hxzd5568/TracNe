# TracNe

## Introduction

To diagnose compiler-introduced numerical errors in NN models, we introduce a automated approach TracNe, which consists of two tasks: detecting and tracing DLC-related numerical errors in a given NN model, a valid input range and specific compilation options. The results on two benchmarks show that TracNe is helpful in locating erroneous code and passes for both DL compiler users and developers. It can serve as a unit test for ensuring the model's robustness.

## Supported Table
| Frontend Models |     ONNX |PyTorch   |   TensorFlow2    |
| ------------ | ------------------------------------ | ----------------------------------------------- | ---------------------------------------------- |
| [`TVM`](https://github.com/apache/tvm)      | ✅                                    | ✅                                               | ✅                                              |
| [`XLA`](https://www.tensorflow.org/xla)  |                                   |                                                 |     🔨                                             | 
| [`GLOW`](https://pytorch.org/docs/stable/jit.html)      |                                    | 🔨                                               

✅: Supported; 🔨: Developing;
## Contents



* [`src`](src): Our approach implementation;
  * [`op`](src/op): Operators supperted by TracNe that can be extended.
  * [`gencog`](src/gencog): Scripts for generating benchmark.
  of [Wang et al.](https://ieeexplore.ieee.org/document/9401995/)
* [`tests`](tests): The interfaces of our approach;
  * [`out`](tests/out): Benchmark and diagnostic reports for the general NN models.
  * [`dnn`](tests/dnn): Handbook and recommended workspace for industrial models.
* [`bug`](bug): [Bug list](bug/pr02.py), TVM bug reports;

## Dependency

###  Python enviroment
TracNe is written in Python. Run `pip install -r requirements.txt` to get python dependencies. 
###  Compiler configuration
* TVM should be built in debugging mode. 
  * Download tvm code of version 0.12.dev0 at least.
  * In config.cmake set the USE_PROFILER flag to ON.
  * [Install tvm from source](https://tvm.apache.org/docs/install/from_source.html#developers-get-source-from-github)
  

## Usage

### Find error-triggering inputs given a model

```shell
cd tests
python test_fuzzer.py  model_dir --low 0 --high 1 --optlevel 5
```

The tested model should be placed in `out/model_dir`. After running the python script, the erroneous input triggering maximal errors will be stored in this directory. Augments low and high are the constraints for the input range. Users can control isolation granularity by $granularity$. Default number 64 is enough for error localization. It should be better than 4. 

If the model is secure under the selected compilation option and input range, the errors found by the process are zero or less than the tolerance. Otherwise, the model are suspectable to the compilers' optimization.


### Error tracing and isolating

```shell
python test_replay.py  model_dir
```

It reproduces the errors by running optimized and un-optimized executable models under searched input. Meanwhile, the process stores concrete results of each funciton of the models.

```shell
python test_traceerror.py  model_dir
```

It matches corresponding functions between symbolic optimized and up-optimized models and compares the results of each equivalent and paired function. The matching and comparison information are saved in the model_dir/trace.json.

```shell
python test_propainfo.py  model_dir
```

It backtrack the error-accumulation changes along the calculation graph. For each discrepancy output, it generates a error-accumulation graph from which the generation and amplification of the errors can be clearly understood. If an error arise in function A, then developers can know how A are optimized and transformed when compliation from trace.json.

```shell
python test_pass.py  model_dir
```

This process isolates optimization pass that incurs the numerical errors. Users can diable it to ensure the security and robustness of the model.

### Pipeline

```shell
python test_batch.py  model_dir1-model_dir9
```

Above scripts are integrated to a single file which detects and diagnoses numerical errors in a batch of models.
## Evaluation

We have provided comparison methods to evaluate the performance and efficiency of the TracNe.

### Searching algorithm


```shell
python relay_fuzzer.py model_dir --method MCMC/DEMC/MEGA 
```

MEGA is our detection algorithm, which MCMC is from [Yu et al.](https://ieeexplore.ieee.org/document/10123640/) and DEMC is devised by [Yi et al.](https://doi.org/10.1145/3290369)

### Error localization algorithm

```shell
python reduce_error.py model_dir
```

This method is implemented following [Guo et al.](https://ieeexplore.ieee.org/document/9355325)




## Support New DL Compilers

The utilities for searching and tracing method can be reused, e.g., mutate_utils.py and fuzzer.py. 

What is required for new DL compilers are to update:
* $build\_workload$ : function in base_utils.py to compile models and build executable files.
* $run\_mod$ : function in base_utils.py to run executable files.
* src/pass : passes' name in the DL compiler.
* src/op : unique operators of the DL compiler.