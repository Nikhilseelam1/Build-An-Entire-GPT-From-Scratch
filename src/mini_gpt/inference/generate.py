import torch

from mini_gpt.utils.tokenizer import (
    get_tokenizer,
    text_to_token_ids,
    token_ids_to_text,
)


@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None,
):
    """
    Autoregressive generation with optional temperature and top-k sampling
    (exactly from your notebook)
    """

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Top-k filtering
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )

        # Sampling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Optional EOS stopping
        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


@torch.no_grad()
def generate_text(
    model,
    device,
    start_text="",
    max_new_tokens=100,
    temperature=0.0,
    top_k=None,
):
    """
    Convenience wrapper for text generation
    """

    model.eval()
    tokenizer = get_tokenizer()

    if start_text == "":
        start_text = "<|endoftext|>"

    idx = text_to_token_ids(start_text, tokenizer, device)

    idx = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=model.block_size,
        temperature=temperature,
        top_k=top_k,
    )

    return token_ids_to_text(idx[0].cpu(), tokenizer)
