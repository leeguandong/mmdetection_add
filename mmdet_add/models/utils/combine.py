'''
@Time    : 2022/4/5 13:39
@Author  : leeguandon@gmail.com
'''
import torch
import torch.nn as nn


class WeightedAdd(nn.Module):
    def __init__(self, num_inputs, eps=1e-4):
        super(WeightedAdd, self).__init__()
        self.num_inputs = num_inputs
        self.eps = eps
        self.relu = nn.ReLU()

        init_val = 1.0 / self.num_inputs
        self.weights = nn.Parameter(torch.Tensor(self.num_inputs, ).fill_(init_val))

    def forward(self, x):
        assert len(x) == self.num_inputs
        weights = self.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        outs = torch.stack([x[i] * weights[i] for i in range(len(weights))])
        return torch.sum(outs, 0)
