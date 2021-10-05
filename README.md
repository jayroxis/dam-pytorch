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

*Code will be available soon, thanks!*


# Minimal Example
```python
import torch
import torch as nn

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
