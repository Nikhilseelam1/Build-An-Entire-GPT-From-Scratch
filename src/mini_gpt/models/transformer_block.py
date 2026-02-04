import torch.nn as nn
from mini_gpt.models.attention import MultiHeadAttention
from mini_gpt.models.feed_forward import FeedForward
from mini_gpt.models.layers import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            dropout=dropout,
            block_size=block_size,
        )
        self.ln2 = LayerNorm(n_embd)
        self.ffwd = FeedForward(
            n_embd=n_embd,
            dropout=dropout,
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
