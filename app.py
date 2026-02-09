import streamlit as st
import torch
import librosa
import numpy as np
import io
from transformers import AutoModel

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

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True,
            use_auth_token=st.secrets["HF_TOKEN"]  # uses token from Streamlit secrets
        )
        model = model.to(device)
        return model, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()  # stop app if model fails

model, DEVICE = load_model()
st.success("✅ Model loaded successfully!")

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

if uploaded_file:
    st.audio(uploaded_file)
    
    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            try:
                # Load audio properly
                audio_bytes = uploaded_file.read()
                waveform, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
                
                # Convert to torch tensor
                waveform = torch.tensor(waveform).unsqueeze(0).to(DEVICE)
                
                # Run model - note: adjust the following if your model requires a custom call
                # Some Indic Conformer repos provide their own inference function, e.g. model.predict()
                # If AutoModel doesn’t support direct forward call, replace with correct method
                result = model(waveform, language, "rnnt")
                
                st.subheader("Transcription:")
                st.write(result)
            
            except Exception as e:
                st.error(f"Error during transcription: {e}")
