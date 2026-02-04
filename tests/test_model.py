import torch
import yaml
from mini_gpt.models.gpt import GPTModel
from mini_gpt.inference.generate import generate
from mini_gpt.utils.tokenizer import get_tokenizer

def load_test_config():
    with open("configs/model_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg["block_size"] = 32
    return cfg

def test_model_forward_shape():
    cfg = load_test_config()
    model = GPTModel(cfg)

    batch_size = 2
    seq_len = 16
    x = torch.randint(
        0, cfg["vocab_size"], (batch_size, seq_len)
    )
    logits = model(x)
    assert logits.shape == (
        batch_size,
        seq_len,
        cfg["vocab_size"],
    )

def test_text_generation_runs():
    cfg = load_test_config()
    model = GPTModel(cfg)
    tokenizer = get_tokenizer()
    idx = torch.tensor(
        tokenizer.encode("Hello"),
        dtype=torch.long
    ).unsqueeze(0)
    out = generate(
        model=model,
        idx=idx,
        max_new_tokens=5,
        context_size=cfg["block_size"],
    )
    assert out.shape[1] > idx.shape[1]
