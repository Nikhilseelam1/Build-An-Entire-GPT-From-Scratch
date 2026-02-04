import torch
import tiktoken


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def text_to_token_ids(text, tokenizer, device):
    token_ids = tokenizer.encode(
        text, allowed_special={"<|endoftext|>"}
    )
    return torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.tolist())
