### Healthcare FAQ assistant - inference script
## Uses the DPO-aligned TinyLlama-1.1B model fine tuned with Unsloth.

try:
    from inference import generate_answer, interactive_loop, load_model
except ModuleNotFoundError:  # pragma: no cover - fallback for direct script execution
    from src.inference import generate_answer, interactive_loop, load_model

__all__ = ["generate_answer", "interactive_loop", "load_model"]