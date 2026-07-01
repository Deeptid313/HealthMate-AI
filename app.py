import gradio as gr
from unsloth import FastLanguageModel

MODEL_PATH = "Deeptid313/HealthMate-AI"  # your HF model

# Load model once
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_PATH,
    max_seq_length=512,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

PROMPT_TEMPLATE = """Below is a question about health and medicine.

### Question:
{}

### Answer:
"""

def generate_response(message, history):
    prompt = PROMPT_TEMPLATE.format(message)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Answer:" in decoded:
        answer = decoded.split("### Answer:")[-1].strip()
    else:
        answer = decoded.replace(prompt, "").strip()

    return answer


# ✅ Chat UI
demo = gr.ChatInterface(
    fn=generate_response,
    title="🏥 HealthMate AI",
    description="Ask health-related questions and get AI-powered answers.",
)

demo.launch()