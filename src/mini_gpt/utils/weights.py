import torch
import numpy as np


def assign(left, right):
    """
    Assigns pretrained weights safely
    """
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. Left: {left.shape}, Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """
    Loads GPT-2 pretrained weights into our GPT model
    Assumes architecture compatibility (GPT-2 style)
    """

    # Token + position embeddings
    gpt.token_embedding.weight = assign(
        gpt.token_embedding.weight,
        params["wte"]
    )

    gpt.position_embedding.weight = assign(
        gpt.position_embedding.weight,
        params["wpe"]
    )

    # Transformer blocks
    for b in range(len(params["blocks"])):
        block = gpt.blocks[b]
        attn_params = params["blocks"][b]["attn"]
        mlp_params = params["blocks"][b]["mlp"]

        # ---- Attention QKV ----
        q_w, k_w, v_w = np.split(
            attn_params["c_attn"]["w"], 3, axis=-1
        )
        q_b, k_b, v_b = np.split(
            attn_params["c_attn"]["b"], 3, axis=-1
        )

        block.attn.query.weight = assign(
            block.attn.query.weight, q_w.T
        )
        block.attn.key.weight = assign(
            block.attn.key.weight, k_w.T
        )
        block.attn.value.weight = assign(
            block.attn.value.weight, v_w.T
        )

        block.attn.query.bias = assign(
            block.attn.query.bias, q_b
        )
        block.attn.key.bias = assign(
            block.attn.key.bias, k_b
        )
        block.attn.value.bias = assign(
            block.attn.value.bias, v_b
        )

        # ---- Attention output projection ----
        block.attn.proj.weight = assign(
            block.attn.proj.weight,
            attn_params["c_proj"]["w"].T
        )
        block.attn.proj.bias = assign(
            block.attn.proj.bias,
            attn_params["c_proj"]["b"]
        )

        # ---- Feed-forward ----
        block.ffwd.net[0].weight = assign(
            block.ffwd.net[0].weight,
            mlp_params["c_fc"]["w"].T
        )
        block.ffwd.net[0].bias = assign(
            block.ffwd.net[0].bias,
            mlp_params["c_fc"]["b"]
        )

        block.ffwd.net[2].weight = assign(
            block.ffwd.net[2].weight,
            mlp_params["c_proj"]["w"].T
        )
        block.ffwd.net[2].bias = assign(
            block.ffwd.net[2].bias,
            mlp_params["c_proj"]["b"]
        )

        # ---- LayerNorms ----
        block.ln1.scale = assign(
            block.ln1.scale,
            params["blocks"][b]["ln_1"]["g"]
        )
        block.ln1.shift = assign(
            block.ln1.shift,
            params["blocks"][b]["ln_1"]["b"]
        )

        block.ln2.scale = assign(
            block.ln2.scale,
            params["blocks"][b]["ln_2"]["g"]
        )
        block.ln2.shift = assign(
            block.ln2.shift,
            params["blocks"][b]["ln_2"]["b"]
        )

    gpt.ln_f.scale = assign(
        gpt.ln_f.scale,
        params["g"]
    )
    gpt.ln_f.shift = assign(
        gpt.ln_f.shift,
        params["b"]
    )

    gpt.head.weight = assign(
        gpt.head.weight,
        params["wte"]
    )
