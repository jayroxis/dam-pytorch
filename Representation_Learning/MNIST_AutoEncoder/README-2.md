IGMC-DAM -- Inductive Graph-based Matrix Completion
===============================================================================

Requirements
------------

Latest tested combination: Python 3.8.1 + PyTorch 1.4.0 + PyTorch_Geometric 1.4.2.

Install [PyTorch](https://pytorch.org/)

Install [PyTorch_Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)

Other required python libraries: numpy, scipy, pandas, h5py, networkx, tqdm etc.

Usages
------

### Flixster, Douban and YahooMusic

To train on Flixster, type:

    python Main.py --data-name douban --epochs 100 --testing --ensemble

The results will be saved in "results/flixster\_testmode/". The processed enclosing subgraphs will be saved in "data/flixster/testmode/". Change flixster to douban or yahoo\_music to do the same experiments on Douban and YahooMusic datasets, respectively. The lambda\_reg value has been set to 0.1 for Flixter and Douban dataset. For YahooMusic, we set the lambda\_reg value 50.0.

