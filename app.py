import streamlit as st
import torch
import librosa
import numpy as np
from transformers import AutoModel
import io




# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="IndicConformer Speech-to-Text",
    page_icon="🎤",
)

st.title("🎤 IndicConformer Speech-to-Text Demo")

# -----------------------------------
# LOAD MODEL (cached so it loads once)
# -----------------------------------

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    HF_TOKEN = st.secrets["HF_TOKEN"]

    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True,
        use_auth_token=HF_TOKEN
    )


    model = model.to(device)

    # Warmup (optional but recommended)
    dummy_audio = torch.randn(1, 16000).to(device)
    _ = model(dummy_audio, "hi", "rnnt")

    return model, device


model, DEVICE = load_model()

st.success("Model loaded successfully!")

# -----------------------------------
# UI
# -----------------------------------

language = st.selectbox(
    "Select language code",
    ["hi", "kn", "ta", "te", "ml", "bn", "mr", "gu", "pa", "or"]
)

uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "flac"]
)

# -----------------------------------
# INFERENCE
# -----------------------------------

if uploaded_file:

    st.audio(uploaded_file)

    if st.button("Transcribe"):

        with st.spinner("Transcribing..."):

            # Load audio
            audio_bytes = uploaded_file.read()

            waveform, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

            waveform = torch.tensor(waveform).unsqueeze(0).to(DEVICE)

            # Run model
            result = model(waveform, language, "rnnt")

        st.subheader("Transcription:")
        st.write(result)
