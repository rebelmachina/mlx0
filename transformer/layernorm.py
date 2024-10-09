import torch.nn as nn
from typing import Union, List
import torch

class MyLayerNorm(nn.Module):
    def __init__(self, normalizing_shape: Union[int, List[int]], eps=1e-5):
        super(MyLayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(normalizing_shape))
        self.beta  = nn.Parameter(torch.zeros(normalizing_shape))
        self.eps = eps

    def forward(self, x):

        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        out = x_normalized * self.gamma + self.beta

        return out


if __name__ == "__main__":
    
    feature_dim = 10
    layer_norm = MyLayerNorm(normalizing_shape=feature_dim)

    x = torch.randn(2,3,feature_dim)
    output = layer_norm(x)

    print(f"input:\n{x}\nmean: {x.mean(dim=-1)}\nvar: {x.var(dim=-1)}")
    print(f"output after layer norm:\n{output}\nmean: {output.mean(dim=-1)}\nvar: {output.var(dim=-1)}")

