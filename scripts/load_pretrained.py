import torch
import yaml

from mini_gpt.models.gpt import GPTModel
from mini_gpt.utils.weights import load_weights_into_gpt
from mini_gpt.utils.tokenizer import get_tokenizer, text_to_token_ids, token_ids_to_text
from gpt_download3 import download_and_load_gpt2

from mini_gpt.inference.generate import generate

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Download GPT-2 weights
# ---------------------------
print("Downloading GPT-2 pretrained weights...")
settings, params = download_and_load_gpt2(
    model_size="124M",
    models_dir="gpt2"
)


# ---------------------------
# Load base config
# ---------------------------
with open("configs/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)


# ---------------------------
# Override config to match GPT-2
# ---------------------------
# IMPORTANT: must match GPT-2 architecture exactly
config.update({
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "block_size": 1024,
})


# ---------------------------
# Build model
# ---------------------------
model = GPTModel(config)
model.eval()


# ---------------------------
# Load pretrained weights
# ---------------------------
load_weights_into_gpt(model, params)
model.to(device)

print("Pretrained GPT-2 weights loaded successfully.")


# ---------------------------
# Text generation
# ---------------------------
tokenizer = get_tokenizer()

prompt = "Every effort moves you"

idx = text_to_token_ids(prompt, tokenizer, device)

with torch.no_grad():
    idx = generate(
        model=model,
        idx=idx,
        max_new_tokens=25,
        context_size=model.block_size,
        temperature=1.5,
        top_k=50,
    )


output_text = token_ids_to_text(
    idx[0].cpu(),
    tokenizer
)

print("\n--- Output text ---")
print(output_text)
print("-------------------")
