# Structured Network Pruning

This is the repository for the structured pruning results for ResNet.

## Getting Started

You will need [Python 3.7](https://www.python.org/downloads) and the packages specified in _requirements.txt_.
We recommend setting up a [virtual environment with pip](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
and installing the packages there.

Install packages with:

```
pip install -r requirements.txt 
```


## Reproduce the Paper Results

```
python train.py [dataset] [model_name] --depth [depth] --epochs [number of epochs] --cold_start [cold_start] --lamda [lambda] --decay [weight decay] --batch_size [batch size] --lr [learning rate] --scheduler_type {1, 2} --save [save_directory]
```
[dataset] = c10 or c100

[model name] = resnet


**Example: CIFAR 10 with Lambda = 0.2**:
```
python train.py c10 resnet --depth 164 --lr 0.05 --scheduler_type 2 --epochs 200 --cuda_id 1 --lamda 0.2 --save ./results/c10/ --cold_start 20
```


