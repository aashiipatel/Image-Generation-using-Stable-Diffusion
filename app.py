"""
AI Image Generator — Streamlit Frontend
Wraps the Stable Diffusion notebook backend (01_StableDiffusion_text2image.ipynb)
without modifying any notebook code.
"""

# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
from __future__ import annotations

import io
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
import streamlit as st
import torch

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 2. PAGE CONFIGURATION (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# 3. CONSTANTS
# ─────────────────────────────────────────────
MODEL_ID: str = "runwayml/stable-diffusion-v1-5"
DEFAULT_STEPS: int = 35
DEFAULT_GUIDANCE: float = 9.0
DEFAULT_SIZE: tuple[int, int] = (512, 512)
SEED: int = 42

EXAMPLE_PROMPTS: list[str] = [
    "A futuristic smart home at sunset, cinematic lighting",
    "A luxury living room with automated lighting, interior design photography",
    "A cyberpunk city in the rain, neon reflections on wet streets",
    "A magical library floating in the clouds, golden warm light, painterly",
    "A serene Japanese garden in autumn, morning mist, zen",
]

LOADING_MESSAGES: list[str] = [
    "Understanding your prompt…",
    "Preparing to create…",
    "Building the image…",
    "Adding fine details…",
    "Almost ready…",
]

# ─────────────────────────────────────────────
# 4. CUSTOM CSS — premium dark SaaS aesthetic
# ─────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Google Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Sora:wght@300;400;600;700&display=swap');

        /* ── Root tokens ── */
        :root {
            --bg:        #0d0d0f;
            --surface:   #141417;
            --surface2:  #1c1c21;
            --border:    #2a2a32;
            --accent:    #7c6af7;
            --accent2:   #a78bfa;
            --text:      #e8e8f0;
            --muted:     #6b6b80;
            --success:   #34d399;
            --error:     #f87171;
            --radius:    14px;
        }

        /* ── Global reset ── */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg) !important;
            color: var(--text) !important;
        }

        /* ── Hide Streamlit chrome ── */
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding-top: 2.5rem; padding-bottom: 4rem; max-width: 760px; }

        /* ── Hero section ── */
        .hero {
            text-align: center;
            padding: 2.5rem 0 1.5rem;
        }
        .hero-icon {
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
            display: block;
        }
        .hero h1 {
            font-family: 'Sora', sans-serif;
            font-size: 2.4rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin: 0 0 0.4rem;
            background: linear-gradient(135deg, #e8e8f0 0%, var(--accent2) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero p {
            color: var(--muted);
            font-size: 1rem;
            font-weight: 300;
            margin: 0;
        }

        /* ── Cards ── */
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.8rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        }
        .card-title {
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.9rem;
        }

        /* ── Textarea ── */
        .stTextArea textarea {
            background: var(--surface2) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            color: var(--text) !important;
            font-size: 0.97rem !important;
            line-height: 1.6 !important;
            padding: 0.9rem 1rem !important;
            transition: border-color 0.2s;
            resize: vertical;
        }
        .stTextArea textarea:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px rgba(124, 106, 247, 0.15) !important;
            outline: none !important;
        }

        /* ── Primary button ── */
        .stButton > button[kind="primary"],
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, var(--accent), #9b87f5) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.85rem 1.5rem !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.01em !important;
            cursor: pointer !important;
            transition: opacity 0.2s, transform 0.15s !important;
            box-shadow: 0 4px 20px rgba(124, 106, 247, 0.35) !important;
        }
        .stButton > button:hover:not(:disabled) {
            opacity: 0.88 !important;
            transform: translateY(-1px) !important;
        }
        .stButton > button:disabled {
            opacity: 0.4 !important;
            cursor: not-allowed !important;
            transform: none !important;
        }

        /* ── Download button ── */
        .stDownloadButton > button {
            width: 100%;
            background: var(--surface2) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            padding: 0.75rem 1.5rem !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            transition: border-color 0.2s, background 0.2s !important;
        }
        .stDownloadButton > button:hover {
            border-color: var(--accent) !important;
            background: rgba(124, 106, 247, 0.1) !important;
        }

        /* ── Metric cards ── */
        [data-testid="metric-container"] {
            background: var(--surface2);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.9rem 1rem;
        }
        [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem !important; }
        [data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.3rem !important; font-weight: 600 !important; }

        /* ── Expander ── */
        .streamlit-expanderHeader {
            background: var(--surface2) !important;
            border-radius: 8px !important;
            color: var(--muted) !important;
            font-size: 0.85rem !important;
        }
        .streamlit-expanderContent {
            background: var(--surface2) !important;
            border-top: 1px solid var(--border) !important;
        }

        /* ── Status pills ── */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(52, 211, 153, 0.12);
            border: 1px solid rgba(52, 211, 153, 0.3);
            color: var(--success);
            border-radius: 100px;
            padding: 0.25rem 0.75rem;
            font-size: 0.78rem;
            font-weight: 500;
        }
        .status-pill.loading {
            background: rgba(124, 106, 247, 0.12);
            border-color: rgba(124, 106, 247, 0.3);
            color: var(--accent2);
        }

        /* ── Image result card ── */
        .result-image-wrapper {
            border-radius: var(--radius);
            overflow: hidden;
            border: 1px solid var(--border);
            box-shadow: 0 8px 40px rgba(0,0,0,0.5);
        }
        .result-image-wrapper img {
            display: block;
            width: 100%;
        }

        /* ── Example chips ── */
        .chip {
            display: inline-block;
            background: var(--surface2);
            border: 1px solid var(--border);
            color: var(--muted);
            border-radius: 100px;
            padding: 0.25rem 0.75rem;
            font-size: 0.78rem;
            cursor: pointer;
            margin: 0.2rem;
            transition: border-color 0.2s, color 0.2s;
        }
        .chip:hover { border-color: var(--accent); color: var(--accent2); }

        /* ── Divider ── */
        hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

        /* ── Alert overrides ── */
        .stAlert { border-radius: 10px !important; }

        /* ── Spinner ── */
        .stSpinner > div > div { border-top-color: var(--accent) !important; }

        /* ── Progress bar ── */
        .stProgress > div > div > div { background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; }

        /* ── Slider ── */
        .stSlider [data-testid="stThumbValue"] { color: var(--accent2) !important; }
        .stSlider [role="slider"] { background: var(--accent) !important; }

        /* ── Selectbox ── */
        .stSelectbox > div > div {
            background: var(--surface2) !important;
            border-color: var(--border) !important;
            color: var(--text) !important;
        }

        /* ── Character counter ── */
        .char-count {
            font-size: 0.72rem;
            color: var(--muted);
            text-align: right;
            margin-top: -0.5rem;
            margin-bottom: 0.6rem;
        }
        .char-count.warn { color: #f59e0b; }

        /* ── Prompt input label ── */
        .stTextArea label { color: var(--muted) !important; font-size: 0.82rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# 5. MODEL LOADER  (cached — loads once per session)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(model_id: str = MODEL_ID):
    """
    Load the Stable Diffusion pipeline using the same logic as the notebook
    (load_sd_pipeline function), without altering notebook code.
    """
    import torch
    from diffusers import StableDiffusionPipeline

    try:
        from diffusers import StableDiffusionXLPipeline
    except ImportError:
        StableDiffusionXLPipeline = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_xl = any(w in model_id.lower() for w in ["xl", "sdxl", "stable-diffusion-xl"])

    if is_xl:
        if StableDiffusionXLPipeline is None:
            raise RuntimeError("Please update your diffusers library to use SDXL models.")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    else:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                revision="fp16",
            )
        except Exception:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            )

    pipe = pipe.to(device)
    return pipe, device


# ─────────────────────────────────────────────
# 6. GENERATION LOGIC
# ─────────────────────────────────────────────
def generate_image(
    prompt: str,
    pipe,
    steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE,
    size: tuple[int, int] = DEFAULT_SIZE,
    seed: int = SEED,
):
    """
    Call the notebook's generation logic:
      pipe(prompt, num_inference_steps, guidance_scale, generator)
    then resize to desired size.
    Returns (PIL.Image, inference_time_seconds).
    """
    device = next(pipe.unet.parameters()).device
    generator = torch.Generator(str(device)).manual_seed(seed)

    t0 = time.perf_counter()
    out = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    elapsed = time.perf_counter() - t0

    image = out.images[0]
    image = image.resize(size)
    return image, elapsed


# ─────────────────────────────────────────────
# 7. METRICS UTILITIES
# ─────────────────────────────────────────────
def get_device_label(device: str) -> str:
    if device == "cuda":
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
        return name
    return "CPU"


def get_memory_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def image_to_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def format_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s // 60)}m {s % 60:.1f}s"


# ─────────────────────────────────────────────
# 8. UI COMPONENTS
# ─────────────────────────────────────────────
def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <span class="hero-icon">✦</span>
            <h1>AI Image Generator</h1>
            <p>Turn a description into a unique image in seconds.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_status(device: str) -> None:
    device_label = get_device_label(device)
    st.markdown(
        f'<div style="text-align:center; margin-bottom:1.5rem;">'
        f'<span class="status-pill">● Model ready &nbsp;·&nbsp; {device_label}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_loading_status() -> None:
    st.markdown(
        '<div style="text-align:center; margin-bottom:1rem;">'
        '<span class="status-pill loading">● Creating your image…</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_metrics(
    inference_time: float,
    device: str,
    size: tuple[int, int],
    steps: int,
    guidance: float,
) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Generation Details</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Time", format_seconds(inference_time))
    c2.metric("Resolution", f"{size[0]}×{size[1]}")
    c3.metric("Steps", str(steps))

    c4, c5, c6 = st.columns(3)
    c4.metric("Guidance", f"{guidance:.1f}")
    c5.metric("Device", get_device_label(device))
    c6.metric("Memory", f"{get_memory_usage_mb():.0f} MB")

    st.markdown("</div>", unsafe_allow_html=True)


def render_result(image, prompt: str, inference_time: float, device: str,
                  size: tuple[int, int], steps: int, guidance: float) -> None:
    """Render the generated image with download + metrics."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Your Image</div>', unsafe_allow_html=True)

    # Image display
    st.image(image, use_container_width=True)

    # Download
    img_bytes = image_to_bytes(image)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="⬇  Download Image",
        data=img_bytes,
        file_name=f"ai_image_{ts}.png",
        mime="image/png",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Metrics
    render_metrics(inference_time, device, size, steps, guidance)


def render_prompt_examples(on_select) -> None:
    """Render example prompt chips."""
    with st.expander("✦ Try an example prompt", expanded=False):
        for p in EXAMPLE_PROMPTS:
            if st.button(p, key=f"ex_{p[:20]}", use_container_width=True):
                on_select(p)


# ─────────────────────────────────────────────
# 9. SESSION STATE INITIALISATION
# ─────────────────────────────────────────────
def init_session_state() -> None:
    defaults = {
        "generated_image": None,
        "last_prompt": "",
        "inference_time": 0.0,
        "generating": False,
        "load_error": None,
        "selected_example": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────
# 10. SIDEBAR — ADVANCED SETTINGS
# ─────────────────────────────────────────────
def render_sidebar() -> tuple[int, float, tuple[int, int], int]:
    with st.sidebar:
        st.markdown("### Settings")
        st.caption("Adjust the generation parameters.")

        steps = st.slider("Quality steps", min_value=10, max_value=60,
                          value=DEFAULT_STEPS, step=5,
                          help="More steps = higher quality but slower.")

        guidance = st.slider("Creativity level", min_value=1.0, max_value=20.0,
                             value=DEFAULT_GUIDANCE, step=0.5,
                             help="Higher = closer to the prompt; lower = more creative.")

        size_label = st.selectbox(
            "Image size",
            options=["512 × 512", "768 × 512", "512 × 768"],
            index=0,
        )
        w, h = (int(x) for x in size_label.replace(" ", "").split("×"))

        seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1,
                               value=SEED, step=1,
                               help="Change seed to get a different result for the same prompt.")

        st.markdown("---")
        st.caption(f"Model: `{MODEL_ID}`")

    return steps, guidance, (w, h), int(seed)


# ─────────────────────────────────────────────
# 11. MAIN APP
# ─────────────────────────────────────────────
def main() -> None:
    inject_css()
    init_session_state()

    # ── Sidebar settings ──
    steps, guidance, size, seed = render_sidebar()

    # ── Hero ──
    render_hero()

    # ── Load model ──
    with st.spinner("Loading model — this only happens once…"):
        try:
            pipe, device = load_pipeline(MODEL_ID)
            st.session_state.load_error = None
        except Exception as exc:
            st.session_state.load_error = str(exc)
            pipe, device = None, "cpu"

    if st.session_state.load_error:
        st.error(
            f"**Could not load the model.**\n\n"
            f"{st.session_state.load_error}\n\n"
            "Make sure your Hugging Face token is set and you have an active internet connection.",
            icon="⚠️",
        )
        return

    render_model_status(device)

    # ── Generation card ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Your Prompt</div>', unsafe_allow_html=True)

    # Pre-fill if an example was selected
    default_text = st.session_state.selected_example or ""
    if st.session_state.selected_example:
        st.session_state.selected_example = None  # consume it

    prompt = st.text_area(
        label="Describe your image",
        value=default_text,
        placeholder="A futuristic smart home at sunset, cinematic lighting, ultra detailed…",
        height=110,
        label_visibility="collapsed",
        key="prompt_input",
    )

    # Character counter
    char_count = len(prompt)
    warn_class = "warn" if char_count > 300 else ""
    st.markdown(
        f'<div class="char-count {warn_class}">{char_count} / 300 characters</div>',
        unsafe_allow_html=True,
    )

    generate_clicked = st.button(
        "✦ Generate Image",
        disabled=st.session_state.generating,
        use_container_width=True,
        type="primary",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Example prompts ──
    def set_example(p: str) -> None:
        st.session_state.selected_example = p
        st.rerun()

    render_prompt_examples(set_example)

    # ── Generation flow ──
    if generate_clicked:
        if not prompt.strip():
            st.warning("Please enter a description before generating.", icon="✏️")
        elif len(prompt.strip()) < 5:
            st.warning("Your description is too short. Try adding more detail.", icon="✏️")
        else:
            st.session_state.generating = True
            render_loading_status()

            # Animated progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            msg_count = len(LOADING_MESSAGES)
            try:
                # Show progress messages while generating
                status_text.markdown(
                    f"<p style='text-align:center;color:var(--muted);font-size:0.88rem'>"
                    f"{LOADING_MESSAGES[0]}</p>",
                    unsafe_allow_html=True,
                )
                progress_bar.progress(5)

                # Run generation in a spinner for visual feedback
                with st.spinner(""):
                    # Update progress at intervals (best effort — blocking call)
                    image, elapsed = generate_image(
                        prompt=prompt.strip(),
                        pipe=pipe,
                        steps=steps,
                        guidance_scale=guidance,
                        size=size,
                        seed=seed,
                    )

                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()

                st.session_state.generated_image = image
                st.session_state.last_prompt = prompt.strip()
                st.session_state.inference_time = elapsed
                st.session_state.generating = False

                st.success("Image created successfully!", icon="✦")

            except Exception as exc:
                progress_bar.empty()
                status_text.empty()
                st.session_state.generating = False
                st.error(
                    f"**Something went wrong during generation.**\n\n"
                    f"Try a shorter or simpler prompt, or reduce the quality steps in settings.\n\n"
                    f"*(Technical detail: {exc})*",
                    icon="⚠️",
                )

    # ── Display last result ──
    if st.session_state.generated_image is not None:
        st.markdown("---")
        render_result(
            image=st.session_state.generated_image,
            prompt=st.session_state.last_prompt,
            inference_time=st.session_state.inference_time,
            device=device,
            size=size,
            steps=steps,
            guidance=guidance,
        )

        # Re-generate option
        st.markdown(
            "<p style='text-align:center; color:var(--muted); font-size:0.82rem; margin-top:0.5rem'>"
            "Change the prompt or settings above and click Generate again to create a new image."
            "</p>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
