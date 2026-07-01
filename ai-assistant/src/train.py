from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer, DPOTrainer, DPOConfig
from peft import PeftModel

# =============================
# CONFIG
# =============================
MODEL_NAME = "unsloth/tinyllama-bnb-4bit"
MAX_SEQ_LENGTH = 512

SFT_OUTPUT = "/content/ai-assistant/models/sft_model"
DPO_OUTPUT = "/content/ai-assistant/models/dpo_model"

SFT_DATA_PATH = "/content/ai-assistant/data/instruction_dataset.jsonl"
DPO_DATA_PATH = "/content/ai-assistant/data/preference_dataset.jsonl"

# =============================
# PROMPT TEMPLATE
# =============================
def format_prompt(example):
    return {
        "text": f"""Below is a question about health and medicine.

### Question:
{example['instruction']}

### Answer:
{example['response']}"""
    }

# =============================
# LOAD DATA
# =============================
def load_sft_data():
    dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")
    return dataset.map(format_prompt)

def load_dpo_data():
    return load_dataset("json", data_files=DPO_DATA_PATH, split="train")

# =============================
# SFT TRAINING
# =============================
def train_sft():
    print("🚀 Loading base model for SFT...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # ✅ Apply LoRA ONLY here
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False,
    )

    dataset = load_sft_data()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=10,
            output_dir=SFT_OUTPUT,
            save_strategy="no",
            logging_strategy="steps",
            fp16=True,
        ),
    )

    print("🚀 Training SFT...")
    trainer.train()

    model.save_pretrained(SFT_OUTPUT)
    tokenizer.save_pretrained(SFT_OUTPUT)

    print("✅ SFT complete!")

# =============================
# DPO TRAINING
# =============================
def train_dpo():
    print("🚀 Loading base + SFT adapter for DPO...")

    # ✅ Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # ✅ Load SFT adapter (DO NOT recreate LoRA)
    model = PeftModel.from_pretrained(model, SFT_OUTPUT)

    # ✅ Enable training mode
    model.train()

    # ✅ ONLY enable gradients for LoRA weights
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False

    # ✅ Disable gradient checkpointing (critical)
    model.gradient_checkpointing_disable()

    dataset = load_dpo_data()

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=DPOConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=5e-5,
            logging_steps=10,
            output_dir=DPO_OUTPUT,
            save_strategy="no",
            fp16=True,
        ),
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("🚀 Training DPO...")
    trainer.train()

    model.save_pretrained(DPO_OUTPUT)
    tokenizer.save_pretrained(DPO_OUTPUT)

    print("✅ DPO complete!")

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    train_sft()
    train_dpo()

    print("\n🎉 Training pipeline finished successfully!")