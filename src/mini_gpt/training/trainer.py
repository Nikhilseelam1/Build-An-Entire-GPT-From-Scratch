import torch
import torch.nn.functional as F
from tqdm import tqdm


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Computes cross-entropy loss for one batch
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    B, T, V = logits.shape

    loss = F.cross_entropy(
        logits.view(B * T, V),
        target_batch.view(B * T)
    )

    return loss


def calc_loss_loader(data_loader, model, device):
    """
    Computes average loss over an entire dataloader
    """
    total_loss = 0.0
    num_batches = 0

    for input_batch, target_batch in data_loader:
        loss = calc_loss_batch(
            input_batch,
            target_batch,
            model,
            device
        )
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_interval=100,
):
    """
    Simple training loop exactly as in the notebook
    """

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for step, (input_batch, target_batch) in enumerate(loop):
            optimizer.zero_grad()

            loss = calc_loss_batch(
                input_batch,
                target_batch,
                model,
                device
            )

            loss.backward()
            optimizer.step()

            loop.set_postfix(train_loss=loss.item())

            if step % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = calc_loss_loader(
                        val_loader,
                        model,
                        device
                    )
                model.train()

        val_loss = calc_loss_loader(val_loader, model, device)

        if len(train_loader) > 0:
            print(
                f"Epoch {epoch+1} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )
        else:
            print(
                f"Epoch {epoch+1} | "
                f"Train Loss: N/A | "
                f"Val Loss: {val_loss:.4f}"
            )

