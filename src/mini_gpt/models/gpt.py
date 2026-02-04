import torch
import torch.nn as nn
import yaml
from pathlib import Path

from mini_gpt.models.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, config_path: str):
        super().__init__()

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.vocab_size = config["vocab_size"]
        self.block_size = config["block_size"]
        self.n_embd = config["n_embd"]
        self.n_layer = config["n_layer"]
        self.n_head = config["n_head"]
        self.dropout = config["dropout"]

        self.token_embedding = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding = nn.Embedding(self.block_size, self.n_embd)

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

        self.dropout_layer = nn.Dropout(self.dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """
        idx: (B, T)
        """
        B, T = idx.size()
        assert T <= self.block_size, "Sequence length exceeds block size"

        token_emb = self.token_embedding(idx)              
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.position_embedding(pos)            

        x = token_emb + pos_emb
        x = self.dropout_layer(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)                               

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Autoregressive generation
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx
