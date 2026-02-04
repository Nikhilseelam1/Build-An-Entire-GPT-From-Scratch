#  Build an Entire GPT From Scratch (With GPT-2 Weight Loading)

> A **from-scratch implementation of a GPT-style Transformer** in PyTorch, including **training**, **advanced decoding**, and **manual GPT-2 pretrained weight loading** — built end-to-end without using HuggingFace Transformers.

---

##  Project Motivation

Most GPT projects today rely heavily on high-level libraries that abstract away the core ideas.  
This project was built with a different goal:

> **To understand, implement, and control every internal component of GPT — from embeddings to attention, from training loops to pretrained weight mapping.**

This repository demonstrates **deep understanding of Transformer internals**, not just usage.

---

##  What This Project Demonstrates

- How GPT models are built **from first principles**
- How **causal self-attention** works internally
- How **training pipelines** for language models are written
- How **GPT-2 pretrained weights** can be loaded manually
- How **top-k and temperature sampling** affect generation
- How to structure a **production-quality ML codebase**

---

##  Model Architecture Overview

This implementation follows the **GPT (decoder-only Transformer)** design:

- Token Embeddings + Positional Embeddings  
- Stack of **Pre-LayerNorm Transformer Blocks**  
- Causal Multi-Head Self-Attention  
- Feed-Forward Networks with **custom GELU**  
- Residual Connections throughout  
- Final LayerNorm + Linear Output Head  
- Weight tying between embeddings and output head  

All components are implemented **manually**.

---

##  Key Components (From Scratch)

### Custom Implementations
- Multi-Head Causal Self-Attention  
- Transformer Block (Pre-LN)  
- LayerNorm (no `nn.LayerNorm`)  
- GELU activation (GPT approximation)  
- Autoregressive decoding  
- Top-k sampling  
- Temperature sampling  

No HuggingFace, no shortcuts.

---



##  Execution Flow

### Training From Scratch
main.py → train.py
→ dataset & dataloader
→ GPTModel (from config)
→ training loop
→ evaluation + generation


### GPT-2 Pretrained Inference
main.py → load_pretrained.py
→ GPT-2 weight download
→ config override (GPT-2)
→ manual weight loading
→ advanced text generation

##  Advanced Text Generation

This project supports **production-grade decoding techniques**:

- Context window cropping  
- Temperature scaling  
- Top-k sampling  
- Greedy decoding fallback  
- EOS-aware stopping  

Implemented manually.

##  Dataset

- Dataset: `the-verdict.txt`  
- Source: Public literary text  
- Pipeline:
  - GPT-2 tokenizer (`tiktoken`)
  - Sliding-window chunking
  - Autoregressive `(x, y)` token pairs
  - Train / validation split

## Author
Nikhilseelam
AI/GEN-AI-Engineer

