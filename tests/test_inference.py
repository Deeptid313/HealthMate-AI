import importlib


def test_inference_module_is_importable():
    module = importlib.import_module("src.inference")
    assert hasattr(module, "generate_answer")
    assert hasattr(module, "load_model")


def test_generate_answer_returns_text_for_simple_question():
    module = importlib.import_module("src.inference")
    answer = module.generate_answer("What are common symptoms of diabetes?")
    assert isinstance(answer, str)
    assert answer.strip()


def test_generate_answer_handles_empty_question():
    module = importlib.import_module("src.inference")
    assert module.generate_answer("") == ""
