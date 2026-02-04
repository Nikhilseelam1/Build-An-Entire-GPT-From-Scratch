import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(n_embd))
        self.shift = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_hat + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device)) *
                (x + 0.044715 * torch.pow(x, 3))
            )
        )
