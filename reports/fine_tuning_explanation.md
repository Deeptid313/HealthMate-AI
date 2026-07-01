# Fine-Tuning Explanation — LoRA, QLoRA, Non-Instruction FT, SFT, and DPO

**Project:** Healthcare FAQ Assistant  
**Date:** 2026-06-30

---

## 1. Why Full Fine-Tuning Is Expensive

Full fine-tuning updates every single parameter in a large language model. A model like LLaMA-3 (8B parameters) stores each parameter as a 16-bit float, meaning the model weights alone consume around 16 GB of GPU memory. During training, the optimizer (e.g., AdamW) needs to store two momentum vectors per parameter, adding another ~32 GB. Combined with activations and gradients, training a 7B+ model end-to-end can require 80–160 GB of GPU VRAM — far beyond the capacity of a single consumer or even mid-range cloud GPU.

Beyond memory, full fine-tuning is slow (days per training run on large datasets), expensive in cloud compute costs, and risks catastrophic forgetting — where the model loses general knowledge it learned during pre-training while learning the new domain.

---

## 2. What LoRA Does

LoRA (Low-Rank Adaptation) avoids updating all model weights. Instead, it freezes the original pre-trained weights completely and adds two small trainable matrices — A and B — alongside specific weight matrices (typically the attention projection layers: Q, K, V, and O).

The key insight is that the weight update ΔW needed for adaptation is low-rank — it can be approximated by the product of two smaller matrices: ΔW ≈ A × B, where A has shape (d × r) and B has shape (r × d), with rank r ≪ d. For a typical 1024×1024 weight matrix, updating it fully requires training 1,048,576 parameters. With LoRA at rank 16, you only train (1024×16) + (16×1024) = 32,768 parameters — a 97% reduction.

The frozen base weights are loaded in 4-bit or 8-bit quantized form to save memory, while only the small LoRA adapters are trained in full precision.

---

## 3. What QLoRA Does

QLoRA (Quantized LoRA) extends LoRA by aggressively quantizing the frozen base model weights to **4-bit NormalFloat (NF4)** precision using a technique called bitsandbytes quantization. The LoRA adapter matrices (A and B) are still trained in 16-bit (BF16) precision.

QLoRA also introduces:
- **Double Quantization** — quantizes the quantization constants themselves, saving an additional ~0.3 bits per parameter.
- **Paged Optimizers** — uses NVIDIA's unified memory to page optimizer states between GPU and CPU RAM, avoiding out-of-memory errors during gradient spikes.

Without QLoRA, a 7B model requires ~14 GB just to load in FP16. With QLoRA 4-bit, it loads in ~4 GB — enabling fine-tuning on a single 8–12 GB consumer GPU (e.g., RTX 3080 or free Google Colab T4).

---

## 4. Why QLoRA Is Useful on Limited GPU

QLoRA makes fine-tuning accessible on free or low-cost hardware:

| Setup | GPU VRAM Needed | Feasible For |
|-------|-----------------|--------------|
| Full fine-tuning (FP16) | 80–160 GB | Multi-GPU clusters only |
| LoRA (FP16 base) | 14–40 GB | A100, H100 |
| QLoRA (4-bit base + LoRA) | 4–8 GB | T4, RTX 3060, free Colab |

In this project, using QLoRA on TinyLlama-1.1B allowed training on a free Google Colab T4 GPU (15 GB VRAM) within the session time limit. Without QLoRA, fine-tuning would have required a paid instance or local high-end hardware.

---

## 5. What Is Non-Instruction Fine-Tuning?

Non-instruction fine-tuning (also called domain-adaptive pre-training or continued pre-training) trains the model on raw, unformatted domain text — the kind you might find in a medical textbook, FAQ page, or clinical document. There are no instruction-response pairs; the model simply learns to predict the next token in a sequence.

The goal is to expose the model to domain vocabulary, writing style, and factual content before teaching it how to answer questions. For example, after training on raw healthcare text, the model better understands terms like "DASH diet," "systolic pressure," "HbA1c," and "FAST acronym" — making the later instruction fine-tuning more effective.

In this project, 50+ paragraphs covering general health, common illnesses, mental health, medications, and preventive care were used for this stage.

---

## 6. What Is Instruction Fine-Tuning?

Instruction fine-tuning (Supervised Fine-Tuning / SFT) trains the model on paired examples of (instruction, response) — teaching it not just domain knowledge, but how to answer questions in that domain.

Each example is formatted with a prompt template that tells the model: "given this question, produce this answer." After SFT, the model learns to follow the question-answering format and produces domain-specific, structured responses rather than continuing random text.

In this project, 100+ healthcare Q&A pairs (covering symptoms, treatments, prevention, medications, and emergencies) were used. The Alpaca prompt format was applied:

```
Below is a question about health and medicine. Write a clear, accurate, and helpful response.

### Question:
{instruction}

### Answer:
{response}
```

---

## 7. What Is DPO?

DPO (Direct Preference Optimization) is a preference alignment technique that teaches the model to prefer certain responses over others — without requiring a separate reward model (as in RLHF).

DPO works on a preference dataset where each example contains:
- A **prompt** (the question)
- A **chosen** response (the good answer)
- A **rejected** response (the bad answer)

DPO reformulates the RLHF objective directly in terms of the language model's probability distribution — increasing the likelihood of chosen responses and decreasing the likelihood of rejected ones in a single training pass. This is simpler, more stable, and more compute-efficient than the full RLHF pipeline (which requires training a reward model separately, then running PPO).

In this project, 50+ preference examples were created with medically accurate chosen responses and unsafe/incorrect/vague rejected responses (e.g., "just take painkillers" as a rejected response to a heart attack question).

---

## 8. Difference Between SFT and DPO

| Aspect | SFT (Instruction Fine-Tuning) | DPO (Preference Alignment) |
|--------|-------------------------------|----------------------------|
| **Goal** | Teach the model to follow instructions and answer questions | Teach the model to prefer good answers over bad ones |
| **Input data** | (instruction, response) pairs | (prompt, chosen response, rejected response) triples |
| **Training signal** | Maximize likelihood of the correct response | Increase P(chosen) and decrease P(rejected) |
| **Stage** | Stage 2 — after domain adaptation | Stage 3 — after SFT |
| **Addresses** | Format following, domain knowledge | Safety, tone, quality, refusal of harmful content |
| **Result** | Model answers questions correctly | Model answers questions in a safer, higher-quality, more aligned way |

SFT teaches the model *what* to say. DPO teaches it *how* to say it better — avoiding unsafe, vague, or unhelpful patterns even when the model technically knows the right content.

---

## 9. Hyperparameter Values Used

### LoRA / QLoRA Configuration

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| **Rank (r)** | 16 | Controls the size of the LoRA adapter matrices. Higher rank = more expressiveness but more parameters. Rank 16 is a good balance for a 1.1B model on a small dataset. |
| **Alpha (lora_alpha)** | 32 | Scaling factor for LoRA updates (effective lr scale = alpha/rank = 2.0). Setting alpha = 2×r is a common convention that keeps the adapter contribution stable. |
| **Dropout** | 0.05 | Regularization applied to the LoRA layers. Small dropout (0.05) reduces overfitting slightly on the small 100-example dataset without hurting convergence. |
| **Target modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All attention and MLP layers targeted for maximum domain adaptation. |
| **Bias** | none | Not adapting bias parameters — common practice for LoRA. |
| **4-bit quantization** | True (NF4) | Base model loaded in 4-bit to minimize VRAM usage. |

### Training Configuration

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| **Learning rate** | 2e-4 | Standard LoRA learning rate. Higher than full fine-tuning (1e-5) because only the small adapter layers are being updated. |
| **Batch size** | 2 (per device) | Small batch size due to GPU memory constraints on T4. |
| **Gradient accumulation** | 4 steps | Effective batch size = 2 × 4 = 8, simulating a larger batch without additional memory. |
| **Epochs** | 3 | Three passes over the 100-example dataset. Enough to learn the format without overfitting. |
| **LR scheduler** | Cosine | Gradually reduces learning rate following a cosine curve — better convergence than linear decay for small datasets. |
| **Optimizer** | AdamW 8-bit | Memory-efficient AdamW from bitsandbytes. |
| **Max sequence length** | 512 tokens | Sufficient for the instruction-response pairs in this dataset. |
| **Weight decay** | 0.01 | Mild L2 regularization. |

These values were chosen based on community best practices for LoRA fine-tuning of sub-2B models on datasets of 50–200 examples, and adjusted for the constraints of a free Colab T4 GPU.
