# Learning Optimal Inter-class Margin Adaptively for Few-Shot Class-Incremental Learning via Neural Collapse-based Meta-learning

### Hang Ran, Weijun Li, Lusi Li, Songsong Tian, Xin Ning, Prayag Tiwari


## Requirements

The `conda` software is required for running the code. Generate a new environment with

```
$ conda create --name metanc_env python=3.6
$ conda activate metanc_env
```

We need PyTorch 1.3 and CUDA. 

```
$ (metanc_env) conda install pytorch=1.3 torchvision cudatoolkit=10.1 -c pytorch
$ (metanc_env) pip install -r requirements.txt
```
## Datasets

We provide the code for running experiments on miniImageNet, CIFAR100 and CUB200. For CIFAR100, the dataset will be download automatically. For miniImageNet, you can download the dataset [here](https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN?usp=sharing). Please put the downloaded file under `src/data/` folder and unzip it. 
```    
$ (metanc_env) cd src/data/
$ (metanc_env) gdown 1_x4o0iFetEv-T3PeIxdSbBPUG4hFfT8U
$ (metanc_env) tar -xvf miniimagenet.tar 
```
## Usage

The whole simulator is runnable from the command line via the `src/main.py` script which serves as a command parser. Everything should be run from the `src` directory. 

The structure of any command looks like
```
$ (metanc_env) python main.py command [subcommand(s)] [-option(s)] [argument(s)]
```
The `main.py` file also contains all default parameters used for simulations.

### Simulation

To run a single simulation of the model (incl. training, validation, testing), use the `simulation` command. A logging directory should be specified, in case the default path is not wanted. Any simulation parameter that should be different from the default found in `main.py` can be specified by chaining `-p parameter value` pairs.
```bash
$ (metanc_env) python main.py simulation --logdir path/to/logdir -p parameter_1 value_1 -p parameter_2 value_2
```
All parameters are interpreted as strings and translated by the parser, so no `"`s are needed. Boolean parameters' value can be specified as `t`, `true`, `f` or `false`.



Run main experiments on CIFAR100
```bash
# Pretraining
$ (metanc_env) python -u main.py simulation -v -ld "log/test_cifar100/pretrain_basetrain" -p max_train_iter 120 -p data_folder "data" -p trainstage pretrain_baseFSCIL -p dataset cifar100 -p learning_rate 0.01 -p batch_size 32 -p optimizer SGD -p SGDnesterov True -p lr_step_size 30 -p dim_features 512 -p block_architecture mini_resnet12 -p num_workers 8
# Metatraining
$ (metanc_env) python -u main.py simulation -v -ld "log/test_cifar100/meta_basetrain" -p max_train_iter 70000 -p data_folder "data" -p resume "log/test_cifar100/pretrain_basetrain"  -p trainstage metatrain_baseFSCIL -p dataset cifar100 -p average_support_vector_inference True -p learning_rate 0.01 -p batch_size_training 10 -p batch_size_inference 128 -p optimizer SGD -p SGDnesterov True -p lr_step_size 30000 -p dim_features 512 -p num_ways 60 -p num_shots 5 -p block_architecture mini_resnet12
# Evaluation(num_shots relates only to number of shots in base session, on novel there are always 5)
$ (metanc_env) python -u main.py simulation -v -ld "log/test_cifar100/eval/mode1"  -p data_folder "data"  -p resume "log/test_cifar100/meta_basetrain" -p dim_features 512  -p trainstage train_FSCIL -p dataset cifar100 -p learning_rate 0.01 -p batch_size_training 128 -p batch_size_inference 128 -p num_query_training 0 -p optimizer SGD -p SGDnesterov True -p retrain_act tanh -p num_ways 60 -p num_shots 200 -p block_architecture mini_resnet12
```


Run main experiments on miniImageNet
```bash
# Pretraining
$ (metanc_env) python -u main.py simulation -v -ld "log/test_miniImageNet/pretrain_basetrain" -p max_train_iter 120 -p data_folder "data" -p trainstage pretrain_baseFSCIL -p dataset miniImageNet -p learning_rate 0.01 -p batch_size 32 -p optimizer SGD -p SGDnesterov True -p lr_step_size 30 -p dim_features 512 -p block_architecture mini_resnet12 -p num_workers 8
# Metatraining
$ (metanc_env) python -u main.py simulation -v -ld "log/test_miniImageNet/meta_basetrain" -p max_train_iter 70000 -p data_folder "data" -p resume "log/test_miniImageNet/pretrain_basetrain"  -p trainstage metatrain_baseFSCIL -p dataset miniImageNet -p average_support_vector_inference True -p learning_rate 0.01 -p batch_size_training 10 -p batch_size_inference 128 -p optimizer SGD -p SGDnesterov True -p lr_step_size 30000 -p dim_features 512 -p num_ways 60 -p num_shots 5 -p block_architecture mini_resnet12
# Evaluation(num_shots relates only to number of shots in base session, on novel there are always 5)
$ (metanc_env) python -u main.py simulation -v -ld "log/test_miniImageNet/eval/mode1"  -p data_folder "data"  -p resume "log/test_miniImageNet/meta_asetrain" -p dim_features 512  -p trainstage train_FSCIL -p dataset miniImageNet -p learning_rate 0.01 -p batch_size_training 128 -p batch_size_inference 128 -p num_query_training 0 -p optimizer SGD -p SGDnesterov True -p retrain_act tanh -p num_ways 60 -p num_shots 200 -p block_architecture mini_resnet12
```

Run main experiments on cub200
```bash
# Pretraining
$ (metanc_env) python -u main.py simulation -v -ld "log/test_cub200/pretrain_basetrain" -p max_train_iter 120 -p data_folder "data" -p trainstage pretrain_baseFSCIL -p dataset miniImageNet -p learning_rate 0.01 -p batch_size 32 -p optimizer SGD -p SGDnesterov True -p lr_step_size 30 -p dim_features 512 -p block_architecture mini_resnet18 -p num_workers 8
# Metatraining
$ (metanc_env) python -u main.py simulation -v -ld "log/test_cub200/meta_basetrain" -p max_train_iter 70000 -p data_folder "data" -p resume "log/test_cub200/pretrain_basetrain"  -p trainstage metatrain_baseFSCIL -p dataset cub200 -p average_support_vector_inference True -p learning_rate 0.01 -p batch_size_training 10 -p batch_size_inference 128 -p optimizer SGD -p SGDnesterov True -p lr_step_size 30000 -p dim_features 512 -p num_ways 60 -p num_shots 5 -p block_architecture mini_resnet18
# Evaluation(num_shots relates only to number of shots in base session, on novel there are always 5)
$ (metanc_env) python -u main.py simulation -v -ld "log/test_cub200/eval/mode1"  -p data_folder "data"  -p resume "log/test_cub200/meta_asetrain" -p dim_features 512  -p trainstage train_FSCIL -p dataset cub200 -p learning_rate 0.01 -p batch_size_training 128 -p batch_size_inference 128 -p num_query_training 0 -p optimizer SGD -p SGDnesterov True -p retrain_act tanh -p num_ways 60 -p num_shots 200 -p block_architecture mini_resnet18
```


### Inspection with TensorBoard

For a detailed inspection of the simulation, the TensorBoard tool can be used. During simulations, data is collected which can be illustrated by the tool in the browser. 

## Acknowledgment

Our code is based on 
- [CFSCIL](https://github.com/IBM/constrained-FSCIL) (Main framework)
- [ALICE](https://github.com/CanPeng123/FSCIL_ALICE)(Data augmentation)


