import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for input_batch, target_batch in data_loader:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        logits = model(input_batch)
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.view(B * T, V),
            target_batch.view(B * T)
        )
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(1, num_batches)


@torch.no_grad()
def generate_and_print_sample(
    model,
    tokenizer,
    device,
    start_text="",
    max_new_tokens=100,
):
    model.eval()
    if start_text == "":
        start_text = "<|endoftext|>"
    idx = torch.tensor(
        tokenizer.encode(start_text),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)
    idx = model.generate(idx, max_new_tokens)
    text = tokenizer.decode(idx[0].tolist())
    print("Generated Sample")
    print(text)
    print("\n")
