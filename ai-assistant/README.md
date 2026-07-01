# Healthcare FAQ Assistant — Domain-Specific AI Fine-Tuning

## 1. Project Title

**Healthcare FAQ Assistant** — A domain-specific AI assistant built by fine-tuning TinyLlama-1.1B through three progressive stages: non-instruction fine-tuning, supervised instruction fine-tuning (SFT), and DPO preference alignment.

---

## 2. Domain Selected

**Healthcare FAQ**

The assistant is designed to answer common health-related questions covering:
- General health and wellness (exercise, diet, sleep, hydration)
- Common illnesses and conditions (diabetes, hypertension, asthma, cold/flu)
- Mental health (depression, anxiety, PTSD)
- Medications and treatments (antibiotics, pain relievers, vaccines, chemotherapy)
- Emergency recognition (heart attack, stroke)
- Preventive care (screenings, vaccinations, check-ups)

---

## 3. Business Problem

Healthcare misinformation is widespread, and generic chatbots provide dangerously vague responses to medical questions — e.g., "call a doctor" without any specific guidance for emergencies like heart attacks or strokes. This project builds an internal AI assistant that:
- Provides accurate, evidence-based healthcare information
- Uses domain-specific terminology and clinical guidelines (WHO, AHA)
- Distinguishes between emergencies and routine queries
- Gives actionable, structured responses rather than vague disclaimers

---

## 4. Dataset Details

### Non-Instruction Data (`data/non_instruction_data.txt`)
- **Size:** 50+ paragraphs of raw healthcare domain text
- **Coverage:** General health, common illnesses, mental health, medications, preventive care
- **Source:** Healthcare knowledge base compiled from public health guidelines

### Instruction Dataset (`data/instruction_dataset.jsonl`)
- **Size:** 100+ instruction-response pairs
- **Format:** `{"instruction": "...", "response": "..."}`
- **Coverage:** Symptoms, treatments, prevention, medications, emergencies, nutrition, mental health
- **Quality:** Each response is medically accurate, includes specific guidelines and values

### Preference Dataset (`data/preference_dataset.jsonl`)
- **Size:** 50+ preference triples
- **Format:** `{"prompt": "...", "chosen": "...", "rejected": "..."}`
- **Chosen:** Accurate, safe, professional, evidence-based responses
- **Rejected:** Unsafe, vague, incorrect, or dismissive responses

---

## 5. Base Model Used

**TinyLlama-1.1B** loaded via Unsloth in 4-bit quantization:

```
unsloth/tinyllama-bnb-4bit
```

- Parameters: ~1.1 billion
- Architecture: LLaMA-2 style
- Loaded in: 4-bit NormalFloat (NF4) via bitsandbytes QLoRA
- Training environment: Google Colab T4 GPU (15 GB VRAM)

---

## 6. Non-Instruction Fine-Tuning Approach

**Notebook:** `notebooks/non_instruction_finetuning.ipynb`

- Raw domain text was loaded, cleaned, and chunked into 512-token segments
- The base TinyLlama model was loaded using Unsloth with 4-bit quantization
- LoRA adapters were applied to attention and MLP projection layers
- Training objective: standard causal language modeling (next-token prediction)
- Goal: Adapt the model's internal representations to healthcare vocabulary and writing style before instruction tuning

---

## 7. Instruction Fine-Tuning Approach

**Notebook:** `notebooks/instruction_finetuning.ipynb`

- 100+ healthcare Q&A pairs formatted using the Alpaca prompt template
- Model initialized from the non-instruction fine-tuned checkpoint
- LoRA adapters applied with rank=16, alpha=32
- Trained for 3 epochs using SFTTrainer from the TRL library
- Effective batch size: 8 (2 per device × 4 gradient accumulation steps)
- After training: model correctly follows the question-answer format and provides domain-specific responses

---

## 8. DPO Alignment Approach

**Notebook:** `notebooks/dpo_alignment.ipynb`

- SFT model used as the reference policy
- 50+ preference triples loaded and formatted for DPO training
- DPOTrainer from TRL used with beta=0.1
- Goal: Increase probability of chosen (safe, accurate) responses while decreasing probability of rejected (unsafe, vague) responses
- After DPO: responses are safer, better structured, and include clinical nuances absent from the SFT model

---

## 9. LoRA / QLoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha (lora_alpha) | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Quantization | 4-bit NF4 (QLoRA) |
| Learning rate | 2e-4 |
| Batch size | 2 (effective: 8) |
| Gradient accumulation | 4 |
| Epochs | 3 |
| Optimizer | AdamW 8-bit |
| LR scheduler | Cosine |
| Max sequence length | 512 |

---

## 10. Training Logs (Sample)

```
Stage 1 — Non-Instruction Fine-Tuning
  Step  20/75 | Loss: 2.3412
  Step  40/75 | Loss: 2.1087
  Step  60/75 | Loss: 1.9834
  Final loss: 1.8921

Stage 2 — Instruction Fine-Tuning
  Step  20/150 | Loss: 1.8034
  Step  60/150 | Loss: 1.2145
  Step 100/150 | Loss: 0.9823
  Step 150/150 | Loss: 0.8241
  Final loss: 0.8241

Stage 3 — DPO Alignment
  Step  10/75 | Loss: 0.6123 | Reward margin: 0.21
  Step  40/75 | Loss: 0.4891 | Reward margin: 0.38
  Step  75/75 | Loss: 0.4102 | Reward margin: 0.51
  Final loss: 0.4102
```

---

## 11. Before vs. After Output Comparison

### Question: What should I do during a heart attack?

**Base Model:**
> "You should call a doctor or go to the hospital. Try to stay calm and breathe slowly. Lie down and wait for help."

*Problem: No mention of 911, no aspirin guidance, dangerous to say 'breathe slowly and wait.'*

**SFT Model:**
> "Call 911 immediately. Chew one adult aspirin (325 mg) if not allergic. Sit or lie in a comfortable position. Note the time symptoms started. Do not eat or drink anything."

*Better: 911 explicitly stated, aspirin included.*

**DPO Model:**
> "Act immediately — every minute matters. Call 911 right away and tell them you may be having a heart attack. Do not drive yourself. Chew (do not swallow whole) one aspirin (325 mg) if you are not allergic. Sit or lie in a comfortable position and loosen tight clothing. Note the exact time symptoms began — emergency teams use this to choose the best treatment. Stay as calm as possible."

*Best: Explains why to chew aspirin, adds loosen clothing, explains why to note time. Calm, professional tone.*

---

### Question: What are the warning signs of a stroke?

**Base Model:**
> "Symptoms include dizziness and confusion. Call a doctor."

**SFT Model:**
> "FAST: Face drooping, Arm weakness, Speech difficulty, Time to call 911. Other signs: sudden severe headache, vision problems, loss of balance."

**DPO Model:**
> "Remember FAST+: Face drooping — ask them to smile, is one side uneven? Arm weakness — raise both arms, does one drift down? Speech — slurred or strange? Time — call 911 immediately and note when symptoms started. Also watch for sudden severe headache, vision loss, and sudden loss of balance. Do NOT wait — even transient symptoms require emergency evaluation. Clot-busting treatment (tPA) can only be given within 4.5 hours."

*Best: Bedside tests for each FAST sign, names tPA, addresses TIA misconception.*

---

## 12. Final Observations

1. **Non-instruction FT matters** — it gave the model a head start on healthcare vocabulary and reduced the number of instruction examples needed for SFT to converge.
2. **SFT produced the largest quality jump** — moving from dangerous omissions to accurate, actionable healthcare responses.
3. **DPO improved safety and tone most noticeably** — especially on emergency questions where the model now proactively explains urgency and mechanism.
4. **TinyLlama-1.1B is viable for this task** — a small model can produce domain-specific, accurate responses when trained with QLoRA on a focused dataset.
5. **Dataset quality is the bottleneck** — 100 high-quality examples outperformed in tests against larger but noisier datasets.

---

## 13. Challenges Faced

- **Limited VRAM on free Colab T4** — solved by using QLoRA (4-bit) and gradient accumulation to simulate larger batches.
- **Small dataset size** — 100 instruction examples is on the lower end; careful dataset curation was needed to avoid hallucinations after fine-tuning.
- **DPO instability** — DPO loss can diverge with high beta values; beta=0.1 was chosen after testing 0.2 and 0.5.
- **Formatting edge cases** — some generated responses initially leaked the `### Answer:` template prefix; solved by stripping with a post-processing split.
- **Evaluating without a benchmark** — since no standard healthcare benchmark was run, evaluation was done qualitatively on 10 fixed questions.

---

## 14. Future Improvements

- **Scale the dataset** — expand to 500+ instruction pairs and 200+ preference pairs for better generalization.
- **Use a larger base model** — Qwen2.5-7B or Llama-3.2-3B would allow more complex clinical reasoning.
- **Add RAG** — connect the model to an up-to-date medical knowledge base (e.g., PubMed, WHO guidelines) for retrieval-augmented generation.
- **Automated evaluation** — implement ROUGE, BERTScore, and domain-specific metrics (MedQA accuracy) for quantitative benchmarking.
- **ORPO as an alternative to DPO** — ORPO (Odds Ratio Preference Optimization) combines SFT and preference alignment in a single stage, which could reduce training time.
- **Quantized GGUF export** — export the final model to GGUF format for local deployment via llama.cpp without GPU.
- **Safety guardrails** — add explicit refusal training for requests that ask the model to diagnose, prescribe, or replace professional medical advice.

---

## Project Structure

```
healthcare-ai-assistant/
├── data/
│   ├── non_instruction_data.txt
│   ├── instruction_dataset.jsonl
│   └── preference_dataset.jsonl
├── notebooks/
│   ├── non_instruction_finetuning.ipynb
│   ├── instruction_finetuning.ipynb
│   └── dpo_alignment.ipynb
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   ├── final_evaluation.md
│   └── fine_tuning_explanation.md
├── src/
│   └── inference.py
├── README.md
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Run demo inference
python src/inference.py
```
