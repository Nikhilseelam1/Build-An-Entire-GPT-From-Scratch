import os
import urllib.request
import torch
import yaml

from mini_gpt.models.gpt import GPTModel
from mini_gpt.data.dataloader import create_dataloader_v1
from mini_gpt.training.trainer import train_model_simple
from mini_gpt.training.evaluator import generate_and_print_sample
from mini_gpt.utils.tokenizer import get_tokenizer

DATA_FILE = "the-verdict.txt"
DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
    "ch02/01_main-chapter-code/the-verdict.txt"
)

if not os.path.exists(DATA_FILE):
    print("Downloading dataset...")
    with urllib.request.urlopen(DATA_URL) as response:
        text_data = response.read().decode("utf-8")
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        f.write(text_data)
else:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text_data = f.read()



train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))

train_text = text_data[:split_idx]
val_text = text_data[split_idx:]



with open("configs/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

context_length = config["block_size"]
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
num_epochs = config["max_epochs"]



torch.manual_seed(123)


train_loader = create_dataloader_v1(
    train_text,
    batch_size=batch_size,
    max_length=context_length,
    stride=context_length//2,
    shuffle=True,
    drop_last=True,
)

val_loader = create_dataloader_v1(
    val_text,
    batch_size=batch_size,
    max_length=context_length,
    stride=context_length//2,
    shuffle=False,
    drop_last=False,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPTModel(config)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)



train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs,
)


tokenizer = get_tokenizer()

generate_and_print_sample(
    model=model,
    tokenizer=tokenizer,
    device=device,
    start_text="The",
    max_new_tokens=200,
)
