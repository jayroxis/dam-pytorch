# Learning Compact Representations of NeuralNetworks using DiscriminAtive Masking (DAM)

This repository offers the codes for reproducing experiment results for the paper: https://arxiv.org/abs/2110.00684

```
@misc{bu2021learning,
      title={Learning Compact Representations of Neural Networks using DiscriminAtive Masking (DAM)}, 
      author={Jie Bu and Arka Daw and M. Maruf and Anuj Karpatne},
      year={2021},
      eprint={2110.00684},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

<p align="center">
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/oPbcQNrhwaw/0.jpg)](https://www.youtube.com/watch?v=oPbcQNrhwaw)
<\p>

*For any issues regarding the code, please contact us via the [issue](https://github.com/jayroxis/dam-pytorch/issues) page!*

-------------

# DAM Modules

The 1-D DAM module takes vectors as input (e.g., used in MLP), which can be implemented as the following:
```python
class DAM(nn.Module):
    """ Discriminative Masking Layer (1-D) """
    def __init__(self, in_dim):
        super(DAM, self).__init__()
        self.in_dim = in_dim
        
        self.mu = torch.arange(self.in_dim).float() / self.in_dim * 5.0
        self.mu = nn.Parameter(self.mu, requires_grad=False)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=False)
        self.register_parameter('mu', self.mu)
        self.register_parameter('beta', self.beta)
        self.register_parameter('alpha', self.alpha)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return x * self.mask()
    
    def mask(self):
        return self.relu(self.tanh((self.alpha ** 2) * (self.mu + self.beta)))
```
The 2-D DAM module takes feature maps (2D matrices) as input (e.g., used in CNN), which can be implemented as the following:
```python
class DAM_2d(nn.Module):
    """ Discriminative Amplitude Modulator Layer (2-D) """
    def __init__(self, in_channel):
        super(DAM_2d, self).__init__()
        self.in_channel = in_channel
        
        self.mu = torch.arange(self.in_channel).float() / self.in_channel * 5
        self.mu = nn.Parameter(self.mu.reshape(-1, self.in_channel, 1, 1), requires_grad=False)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=False)
        self.register_parameter('beta', self.beta)
        self.register_parameter('alpha', self.alpha)
        self.register_parameter('mu', self.mu)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return x * self.mask()
    
    def mask(self):
        return self.relu(self.tanh((self.alpha ** 2) * (self.mu + self.beta)))
```

-------------

# Minimal Example: Pruning MLP using 1-D DAM Module
```python
import torch
import torch as nn
```
The following example shows how to use a 1-D DAM module, it only takes one argument: the input dimension to DAM. The DAM only scales the inputs in the forward propagation.
Assume we have a simple neural network (MLP) defined as:
```python
class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.layers(x)
```
If we want to prune all the layers, simply add the DAM module to the end of every layer as demonstrated in the following example:
```python
class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            DAM(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            DAM(128),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.layers(x)
```
