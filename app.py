import base64
import os
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# Optional provider imports
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openrouter import ChatOpenRouter

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_TEXT_MODEL = os.getenv("GOOGLE_TEXT_MODEL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HUGGINGFACE_TEXT_MODEL = os.getenv("HUGGINGFACE_TEXT_MODEL")
GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
if HUGGINGFACEHUB_API_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


def init_gemini_text_model():
    # Use Gemini for text generation instead of OpenAI.
    model_name = GOOGLE_TEXT_MODEL or "gemini-2.5-pro"
    return ChatGoogleGenerativeAI(model=model_name, api_key=GOOGLE_API_KEY, temperature=0.2)


def init_openrouter_image_model():
    return ChatOpenRouter(model="gpt-image-1")


def init_gemini_tts_model():
    return ChatGoogleGenerativeAI(model="gemini-2.5-pro", api_key=GOOGLE_API_KEY)


def init_hf_image_caption_model():
    return HuggingFaceEndpoint(repo_id="Salesforce/blip-image-captioning-base", task="image-to-text")


def init_hf_audio_transcription_model():
    return HuggingFaceEndpoint(repo_id="openai/whisper-large-v2", task="audio-transcription")


def init_hf_text_model():
    if not HUGGINGFACE_TEXT_MODEL:
        raise RuntimeError("HUGGINGFACE_TEXT_MODEL is not set")
    return HuggingFaceEndpoint(repo_id=HUGGINGFACE_TEXT_MODEL, task="text-generation")


def init_groq_text_model():
    if not GROQ_TEXT_MODEL:
        raise RuntimeError("GROQ_API_KEY is set but GROQ_TEXT_MODEL is not configured.")
    return ChatGroq(model=GROQ_TEXT_MODEL)


def init_text_model():
    """Select a text model (non-OpenAI) by preference: Groq -> Anthropic -> Gemini."""
    if GROQ_API_KEY and GROQ_TEXT_MODEL:
        return init_groq_text_model()
    if ANTHROPIC_API_KEY:
        try:
            return ChatAnthropic()
        except Exception:
            pass
    if HUGGINGFACE_TEXT_MODEL and HUGGINGFACEHUB_API_TOKEN:
        try:
            return init_hf_text_model()
        except Exception:
            pass
    if GOOGLE_API_KEY:
        return init_gemini_text_model()
    raise RuntimeError(
        "No supported text model API key configured. Set GROQ_API_KEY and GROQ_TEXT_MODEL, ANTHROPIC_API_KEY, or GOOGLE_API_KEY."
    )


def text_to_text(prompt: str) -> tuple[str, str]:
    """Try available text providers in order and return first successful result.

    Returns (text, provider_name).
    """
    candidates: list[tuple[object, str]] = []
    errors: list[tuple[str, Exception]] = []

    # Build candidate model instances in preferred order
    if GROQ_API_KEY and GROQ_TEXT_MODEL:
        try:
            candidates.append((init_groq_text_model(), "Groq"))
        except Exception as e:
            errors.append(("Groq-init", e))
    elif GROQ_API_KEY:
        errors.append(("Groq-config", RuntimeError("GROQ_API_KEY is set but GROQ_TEXT_MODEL is missing. Set GROQ_TEXT_MODEL to a valid Groq text model.")))
    if ANTHROPIC_API_KEY:
        try:
            candidates.append((ChatAnthropic(), "Anthropic"))
        except Exception as e:
            errors.append(("Anthropic-init", e))
    if HUGGINGFACE_TEXT_MODEL and HUGGINGFACEHUB_API_TOKEN:
        try:
            candidates.append((init_hf_text_model(), "HuggingFace"))
        except Exception as e:
            errors.append(("HuggingFace-init", e))
    if GOOGLE_API_KEY:
        try:
            candidates.append((init_gemini_text_model(), "Gemini"))
        except Exception as e:
            errors.append(("Gemini-init", e))

    if not candidates:
        raise RuntimeError("No configured text providers available. Set GROQ_API_KEY, ANTHROPIC_API_KEY, HUGGINGFACE_TEXT_MODEL, or GOOGLE_API_KEY.")

    # Try each provider until one succeeds
    for model, provider in candidates:
        try:
            if provider == "HuggingFace":
                response = model.invoke({"inputs": prompt})
            else:
                response = model.invoke([
                    SystemMessage(content="You are a concise assistant."),
                    HumanMessage(content=prompt),
                ])

            if hasattr(response, "content") and isinstance(response.content, str):
                return response.content, provider
            if getattr(response, "additional_kwargs", None) and "text" in response.additional_kwargs:
                return response.additional_kwargs["text"], provider
            return str(getattr(response, "content", "")), provider
        except ChatGoogleGenerativeAIError as e:
            # If Gemini quota error, record and continue to next candidate
            errors.append((f"{provider}-invoke", e))
            continue
        except Exception as e:
            errors.append((f"{provider}-invoke", e))
            continue

    # All candidates failed — format provider-specific guidance
    def _is_model_not_found_error(exc: Exception) -> bool:
        s = str(exc).lower()
        return ("model_not_found" in s) or ("does not exist" in s) or ("not found" in s)

    def _is_resource_exhausted_error(exc: Exception) -> bool:
        s = str(exc).lower()
        return ("resource_exhausted" in s) or ("quota" in s) or ("exceeded" in s)

    def _format_provider_error(name: str, exc: Exception) -> str:
        s = str(exc)
        lname = name.lower()
        if "groq" in lname and _is_model_not_found_error(exc):
            return f"{name}: Groq model not found or inaccessible. Set GROQ_TEXT_MODEL to a model you have access to, or check your Groq account. Original error: {s}"
        if "gemini" in lname and _is_resource_exhausted_error(exc):
            return f"{name}: Gemini quota/billing issue (RESOURCE_EXHAUSTED). Check your Google Cloud billing and Gemini quota, or use a different model. Original error: {s}"
        return f"{name}: {type(exc).__name__}: {s}"

    msg_lines = [_format_provider_error(name, err) for name, err in errors]
    raise RuntimeError("All text providers failed:\n" + "\n".join(msg_lines))


def text_to_image(prompt: str) -> str:
    llm = init_openrouter_image_model()
    response = llm.invoke(prompt)
    return response.content


def text_to_audio(prompt: str) -> bytes:
    tts = init_gemini_tts_model()
    try:
        response = tts.invoke([
            SystemMessage(content="Convert this text into spoken English audio."),
            HumanMessage(content=prompt),
        ])
        if hasattr(response, "content"):
            return response.content.encode("utf-8")
    except Exception:
        pass
    raise RuntimeError("Text-to-audio generation failed. Check Gemini API and model availability.")


def image_to_text(image_data: bytes) -> str:
    hf = init_hf_image_caption_model()
    b64 = base64.b64encode(image_data).decode("utf-8")
    prompt = {"data": f"data:image/png;base64,{b64}"}
    response = hf.invoke(prompt)
    return response.content


def audio_to_text(audio_path: str) -> str:
    hf = init_hf_audio_transcription_model()
    prompt = {"file": audio_path}
    response = hf.invoke(prompt)
    return response.content


def extract_audio(video_path: str) -> str:
    audio_fd, audio_path = tempfile.mkstemp(suffix=".wav")
    os.close(audio_fd)
    subprocess.run([
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
    ], check=True, capture_output=True)
    return audio_path


def run_ui():
    st.set_page_config(page_title="Multimodal AI Web App", layout="wide")
    st.title("Multimodal AI Web App")

    st.sidebar.header("Provider status")
    st.sidebar.markdown(f"- OpenAI: {'✅' if OPENAI_API_KEY else '⚠️ missing'}")
    st.sidebar.markdown(f"- Anthropic: {'✅' if ANTHROPIC_API_KEY else '⚠️ missing'}")
    st.sidebar.markdown(f"- Gemini: {'✅' if GOOGLE_API_KEY else '⚠️ missing'}")
    if GOOGLE_API_KEY and GOOGLE_TEXT_MODEL:
        st.sidebar.caption(f"Gemini model: {GOOGLE_TEXT_MODEL}")
    st.sidebar.markdown(f"- Groq: {'✅' if GROQ_API_KEY and GROQ_TEXT_MODEL else '⚠️ missing or incomplete'}")
    if GROQ_API_KEY and not GROQ_TEXT_MODEL:
        st.sidebar.caption("Set GROQ_TEXT_MODEL for Groq text generation.")
    st.sidebar.markdown(f"- OpenRouter: {'✅' if OPENROUTER_API_KEY else '⚠️ missing'}")
    st.sidebar.markdown(f"- HuggingFace: {'✅' if HUGGINGFACEHUB_API_TOKEN else '⚠️ missing'}")

    mode = st.sidebar.selectbox("Choose input modality", ["Text", "Image", "Audio", "Video"])

    if mode == "Text":
        st.header("Text input")
        prompt = st.text_area("Enter text prompt", height=180)
        output_format = st.selectbox("Generate output as", ["Text", "Image", "Audio", "Video storyboard"])
        if st.button("Generate"):
            if not prompt.strip():
                st.warning("Please enter a prompt.")
            else:
                if output_format == "Text":
                    try:
                        result, provider = text_to_text(prompt)
                        st.subheader("Text Output")
                        st.caption(f"Provider: {provider}")
                        st.write(result)
                    except Exception as exc:
                        st.error(str(exc))
                elif output_format == "Image":
                    result = text_to_image(prompt)
                    st.subheader("Image Output")
                    st.write("Generated image response from OpenRouter model:")
                    st.write(result)
                elif output_format == "Audio":
                    with st.spinner("Generating audio..."):
                        try:
                            audio_bytes = text_to_audio(prompt)
                            st.audio(audio_bytes, format="audio/wav")
                        except Exception as exc:
                            st.error(str(exc))
                else:
                    try:
                        storyboard, provider = text_to_text(
                            f"Write a short video script and shot list for the following idea: {prompt}"
                        )
                        st.subheader("Video Storyboard")
                        st.caption(f"Provider: {provider}")
                        st.write(storyboard)
                    except Exception as exc:
                        st.error(str(exc))

    elif mode == "Image":
        st.header("Image input")
        image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if image_file is not None:
            data = image_file.read()
            st.image(data, caption="Uploaded image", use_column_width=True)
            if st.button("Describe Image"):
                with st.spinner("Analyzing image..."):
                    caption = image_to_text(data)
                    st.subheader("Image caption")
                    st.write(caption)

    elif mode == "Audio":
        st.header("Audio input")
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "flac"])
        if audio_file is not None:
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix)
            temp_audio.write(audio_file.read())
            temp_audio.flush()
            st.audio(temp_audio.name)
            if st.button("Transcribe Audio"):
                with st.spinner("Transcribing audio..."):
                    transcript = audio_to_text(temp_audio.name)
                    st.subheader("Transcript")
                    st.write(transcript)

    else:
        st.header("Video input")
        video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "mkv", "webm"])
        if video_file is not None:
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix)
            temp_video.write(video_file.read())
            temp_video.flush()
            st.video(temp_video.name)
            if st.button("Transcribe Video"):
                with st.spinner("Extracting audio and transcribing..."):
                    audio_path = extract_audio(temp_video.name)
                    transcript = audio_to_text(audio_path)
                    st.subheader("Video Transcript")
                    st.write(transcript)
                    Path(audio_path).unlink(missing_ok=True)

    st.markdown("---")
    st.info(
        "This app uses different providers and models for each modality. "
        "Set the required API keys in your environment or `.env` file before running."
    )


if __name__ == "__main__":
    run_ui()
