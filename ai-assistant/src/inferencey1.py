### Healthcare FAQ assistant - inference script
## Uses the DPO-aligned TinyLlama-1.1B model fine tuned with Unsloth.

from unsloth import FastLanguageModel

# ✅ Use Hugging Face model instead of local path
MODEL_PATH = "Deeptid123/HealthAiAsst-dpo-model"
MAX_SEQ_LENGTH = 512
LOAD_IN_4BIT = True

ALPACA_PROMPT = """Below is a question about health and medicine. Write a clear, accurate, and helpful response.

### Question:
{}

### Answer:
"""

model = None
tokenizer = None


def load_model():
    global model, tokenizer
    if model is not None:
        return

    print(f"Loading model from {MODEL_PATH}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    FastLanguageModel.for_inference(model)

    print("✅ Model loaded.\n")


def generate_answer(question: str, max_new_tokens: int = 256) -> str:
    load_model()

    prompt = ALPACA_PROMPT.format(question)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ✅ clean extraction
    if "### Answer:" in decoded:
        answer = decoded.split("### Answer:")[-1].strip()
    else:
        answer = decoded.replace(prompt, "").strip()

    return answer


def interactive_loop():
    print("=" * 60)
    print("Welcome to the Healthcare FAQ Assistant!")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    load_model()

    while True:
        question = input("\nPlease enter your health-related question: ").strip()

        if question.lower() in ["quit", "exit"]:
            print("Exiting... Goodbye!")
            break

        if not question:
            continue

        print("\nAssistant:", generate_answer(question))


if __name__ == "__main__":
    # Quick demo
    demo_questions = [
        "What are the common symptoms of diabetes?",
        "How can I prevent heart disease?",
        "What is the recommended dosage of ibuprofen for adults?",
    ]

    load_model()

    for q in demo_questions:
        print(f"\nQuestion: {q}")
        print(f"Answer: {generate_answer(q)}")
        print("-" * 60)

    # Uncomment to enable chat mode
    # interactive_loop()