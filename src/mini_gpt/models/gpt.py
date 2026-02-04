import torch
import torch.nn as nn
import yaml
from pathlib import Path

from mini_gpt.models.transformer_block import TransformerBlock

class GPTModel(nn.Module):
    def __init__(self, cfg):
    
        super().__init__()
        if isinstance(cfg, (str, Path)):
            with open(cfg, "r") as f:
                cfg = yaml.safe_load(f)

        self.cfg = cfg
        self.vocab_size = cfg["vocab_size"]
        self.block_size = cfg["block_size"]
        self.n_embd = cfg["n_embd"]
        self.n_layer = cfg["n_layer"]
        self.n_head = cfg["n_head"]
        self.dropout = cfg["dropout"]
        self.token_embedding = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding = nn.Embedding(self.block_size, self.n_embd)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_embd=self.n_embd,
                    n_head=self.n_head,
                    dropout=self.dropout,
                    block_size=self.block_size,
                )
                for _ in range(self.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.block_size
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(
            torch.arange(T, device=idx.device)
        )
        x = tok_emb + pos_emb
        x = self.dropout_layer(x)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
