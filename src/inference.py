"""Lightweight inference entrypoint for the Healthcare FAQ assistant.

This module provides a small, dependency-tolerant interface that can:
- load a locally available fine-tuned adapter when the Unsloth stack is installed;
- fall back to deterministic, safe heuristic responses when model loading is not possible.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

MODEL_PATH = os.environ.get("HEALTHMATE_MODEL_PATH", "Deeptid123/HealthAiAsst-dpo-model")
MAX_SEQ_LENGTH = int(os.environ.get("HEALTHMATE_MAX_SEQ_LENGTH", "512"))
LOAD_IN_4BIT = os.environ.get("HEALTHMATE_LOAD_IN_4BIT", "true").lower() in {"1", "true", "yes", "on"}

ALPACA_PROMPT = """Below is a question about health and medicine. Write a clear, accurate, and helpful response.

### Question:
{}

### Answer:
"""

model = None
tokenizer = None


def _get_local_model_candidates() -> list[Path]:
    base_dir = Path(__file__).resolve().parent.parent
    candidates = [
        base_dir / "models" / "dpo_model",
        base_dir / "models" / "sft_model",
    ]
    return [candidate for candidate in candidates if candidate.exists()]


def _load_unsloth_model():
    try:
        from unsloth import FastLanguageModel  # type: ignore
    except Exception:
        return None
    return FastLanguageModel


def load_model() -> Tuple[Optional[object], Optional[object]]:
    """Load the fine-tuned model when possible; otherwise return None values."""
    global model, tokenizer
    if model is not None:
        return model, tokenizer

    FastLanguageModel = _load_unsloth_model()
    if FastLanguageModel is None:
        return None, None

    candidates = _get_local_model_candidates()
    selected_model_path = str(candidates[0]) if candidates else MODEL_PATH

    print(f"Loading model from {selected_model_path}...")
    try:
        model_instance, tokenizer_instance = FastLanguageModel.from_pretrained(
            model_name=selected_model_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(model_instance)
    except Exception as exc:
        print(f"Model could not be loaded, using fallback responses: {exc}")
        return None, None

    model = model_instance
    tokenizer = tokenizer_instance
    return model, tokenizer


def _heuristic_answer(question: str) -> str:
    if not question or not question.strip():
        return ""

    normalized = question.strip().lower()
    if "diabetes" in normalized:
        return (
            "Common symptoms of diabetes can include frequent thirst, urination, fatigue, "
            "blurred vision, and slow-healing wounds. A clinician can confirm the diagnosis with a blood test."
        )
    if "heart disease" in normalized or "heart attack" in normalized or "stroke" in normalized:
        return (
            "If this is a medical emergency, seek urgent care immediately. For symptoms such as chest pain, "
            "shortness of breath, weakness, or sudden confusion, call emergency services right away."
        )
    if "ibuprofen" in normalized or "dosage" in normalized:
        return (
            "For adults, ibuprofen is commonly taken at 200-400 mg every 4-6 hours as needed, but dosing depends on "
            "age, medical history, and other medications. Follow the label or a clinician's advice."
        )
    if "depression" in normalized or "anxiety" in normalized:
        return (
            "Supportive care, rest, and regular professional guidance can help. If symptoms are severe or you are in danger, "
            "seek immediate help from a qualified healthcare provider or crisis service."
        )
    return (
        "I can help with general health information, but serious symptoms or diagnoses should be reviewed by a qualified healthcare professional."
    )


def generate_answer(question: str, max_new_tokens: int = 256) -> str:
    """Return an answer for a health-related question using the model when available."""
    if not question or not question.strip():
        return ""

    loaded_model, loaded_tokenizer = load_model()
    if loaded_model is None or loaded_tokenizer is None:
        return _heuristic_answer(question)

    try:
        import torch  # type: ignore
    except Exception:
        return _heuristic_answer(question)

    prompt = ALPACA_PROMPT.format(question)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = loaded_tokenizer(prompt, return_tensors="pt").to(device)

    outputs = loaded_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=loaded_tokenizer.eos_token_id,
    )
    decoded = loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Answer:" in decoded:
        return decoded.split("### Answer:")[-1].strip()
    return decoded.replace(prompt, "").strip() or _heuristic_answer(question)


def interactive_loop() -> None:
    print("=" * 60)
    print("Welcome to the Healthcare FAQ Assistant!")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        question = input("\nPlease enter your health-related question: ").strip()
        if question.lower() in {"quit", "exit"}:
            print("Exiting... Goodbye!")
            break
        if not question:
            continue
        print("\nAssistant:", generate_answer(question))


if __name__ == "__main__":
    demo_questions = [
        "What are the common symptoms of diabetes?",
        "How can I prevent heart disease?",
        "What is the recommended dosage of ibuprofen for adults?",
    ]

    for question in demo_questions:
        print(f"\nQuestion: {question}")
        print(f"Answer: {generate_answer(question)}")
        print("-" * 60)
