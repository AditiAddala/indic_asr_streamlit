import streamlit as st
import torch
import librosa
import numpy as np
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
# LOAD MODEL (cached)
# -----------------------------------


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    HF_TOKEN = st.secrets["HF_TOKEN"]

    try:
        # Try to import custom model class
        from model_onnx import Speech2TextModel

        model = Speech2TextModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            use_auth_token=HF_TOKEN
        )
        model_type = "custom"
    
    except Exception:
        # Fallback to AutoModel
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True,
            use_auth_token=HF_TOKEN
        )
        model_type = "auto"
    
    model = model.to(device)
    return model, device, model_type

model, DEVICE, MODEL_TYPE = load_model()
st.success(f"✅ Model loaded successfully! (using {MODEL_TYPE} model)")

# -----------------------------------
# UI: Language selection and file upload
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

def transcribe_audio(model, waveform, language, model_type):
    """
    Handles both custom and AutoModel forward calls.
    """
    if model_type == "custom":
        return model(waveform, language, "rnnt")  # custom model API
    else:
        # AutoModel fallback
        return model(waveform)  # simple forward; adjust as needed

if uploaded_file:
    st.audio(uploaded_file)
    
    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            try:
                # Load audio
                audio_bytes = uploaded_file.read()
                waveform, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
                waveform = torch.tensor(waveform).unsqueeze(0).to(DEVICE)
                
                # Run inference
                result = transcribe_audio(model, waveform, language, MODEL_TYPE)
                
                st.subheader("Transcription:")
                st.write(result)
            
            except Exception as e:
                st.error(f"Error during transcription: {e}")
