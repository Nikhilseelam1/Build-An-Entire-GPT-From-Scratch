import torch
import yaml

from mini_gpt.models.gpt import GPTModel
from mini_gpt.utils.weights import load_weights_into_gpt
from mini_gpt.utils.tokenizer import get_tokenizer, text_to_token_ids, token_ids_to_text
from gpt_download3 import download_and_load_gpt2
from mini_gpt.inference.generate import generate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Downloading GPT-2 pretrained weights...")
settings, params = download_and_load_gpt2(
    model_size="124M",
    models_dir="gpt2"
)

with open("configs/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

config.update({
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "block_size": 1024,
})

model = GPTModel(config)
model.eval()

load_weights_into_gpt(model, params)
model.to(device)
print("pretrained GPT-2 weights loaded successfully.")
tokenizer = get_tokenizer()
prompt = "every effort moves you"
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
print("\nOutputtext")
print(output_text)
