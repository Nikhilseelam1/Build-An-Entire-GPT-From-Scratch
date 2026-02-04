#  Build an Entire GPT From Scratch (With GPT-2 Weight Loading)

>  **from-scratch implementation of a GPT-style Transformer** in PyTorch, including **training**, **advanced decoding**, and **manual GPT-2 pretrained weight loading**, built end-to-end **without using HuggingFace Transformers**.

This project focuses on **deep understanding and implementation of Large Language Models**, not just usage.

---

##  Project Motivation

Most modern GPT projects rely on high-level libraries that hide the internal mechanics of Transformers.  
This project was built with a different objective:

> **To understand, implement, and control every internal component of GPT — from embeddings and attention to training loops and pretrained weight mapping.**

The goal is to move beyond “using GPT” and instead **rebuild GPT from first principles**.

---

##  What This Project Demonstrates

- How **decoder-only Transformer (GPT)** models are constructed internally  
- How **causal self-attention** enforces autoregressive behavior  
- How **training pipelines** for language models are written manually  
- How **GPT-2 pretrained weights** can be loaded without Transformers libraries  
- How **top-k and temperature sampling** influence text generation  
- How to design a **clean, modular, production-style ML codebase**

---

##  Model Architecture Overview

This implementation follows the **GPT (decoder-only Transformer)** architecture:

- Token Embeddings + Positional Embeddings  
- Stack of **Pre-LayerNorm Transformer Blocks**  
- Causal Multi-Head Self-Attention  
- Feed-Forward Networks with **custom GELU activation**  
- Residual connections throughout the network  
- Final LayerNorm followed by a Linear output projection  
- **Weight tying** between token embeddings and output head  

Every component is implemented **manually in PyTorch**.

---

##  From-Scratch Implementations

The following components are implemented without relying on high-level abstractions:

- Multi-Head Causal Self-Attention  
- Transformer Block (Pre-LayerNorm design)  
- Layer Normalization (custom implementation)  
- GELU activation using the original GPT approximation  
- Autoregressive token generation  
- Cross-entropy loss computation  
- Manual training loop with backpropagation  

No HuggingFace Transformers are used.

---

## Advanced Text Generation

The project supports **production-grade decoding strategies**, implemented manually:

- Context window cropping for long sequences  
- Temperature-based sampling  
- Top-k filtering to control randomness  
- Greedy decoding fallback  
- Optional EOS-aware stopping  

These techniques allow fine-grained control over model creativity and stability.

---

##  Dataset & Data Pipeline

- Dataset: **`the-verdict.txt`** (public literary text)  
- Tokenization: **GPT-2 tokenizer (`tiktoken`)**  
- Data preparation:
  - Sliding-window chunking over token sequences  
  - Autoregressive `(input, target)` token pairs  
  - Train / validation split  

The dataset pipeline closely mirrors how real GPT models are trained.

---

##  GPT-2 Pretrained Weight Loading

A major highlight of this project is **manual GPT-2 weight loading**:

- GPT-2 weights are downloaded externally  
- QKV matrices are **manually split and mapped**  
- Feed-Forward and LayerNorm weights are assigned explicitly  
- Token embeddings are tied to the output head  
- Architecture is adapted dynamically to match GPT-2 configurations  

This demonstrates a deep understanding of **GPT-2 internals and parameter layout**.

---

##  Future Enhancements

- Key-Value (KV) caching for faster decoding  
- Model checkpointing and resume training  
- Mixed-precision training (FP16 / BF16)  
- Flash Attention integration  
- Multi-GPU and distributed training support  

---

##  Author

**Nikhil Seelam**  
Aspiring **AI / GenAI Engineer**  
Focused on **Machine Learning, Deep Learning, Transformers, and LLM internals**

GitHub: https://github.com/Nikhilseelam1

