import torch

from mini_gpt.utils.tokenizer import (
    get_tokenizer,
    text_to_token_ids,
    token_ids_to_text,
)


@torch.no_grad()
def generate_text(
    model,
    device,
    start_text="",
    max_new_tokens=100,
):
    """
    Generates text using the GPT model
    """
    model.eval()

    tokenizer = get_tokenizer()

    if start_text == "":
        start_text = "<|endoftext|>"

    idx = text_to_token_ids(start_text, tokenizer, device)

    idx = model.generate(idx, max_new_tokens)

    generated_text = token_ids_to_text(
        idx[0].cpu(), tokenizer
    )

    return generated_text
