import os
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", APP_DIR / "person2_resnet18_best.pth"))
CLASS_NAMES = {
    "c0": "Safe driving",
    "c1": "Texting — right hand",
    "c2": "Talking on phone — right hand",
    "c3": "Texting — left hand",
    "c4": "Talking on phone — left hand",
    "c5": "Operating the radio",
    "c6": "Drinking",
    "c7": "Reaching behind",
    "c8": "Hair and makeup",
    "c9": "Talking to passenger",
}

st.set_page_config(page_title="DriveAware | Driver Activity Classifier", page_icon="🚘", layout="centered")
st.markdown("""
<style>
  .stApp { background: #0b1120; color: #eef4ff; }
  [data-testid="stHeader"] { background: rgba(11,17,32,0); }
  .block-container { max-width: 850px; padding-top: 2.5rem; }
  .hero { padding: 1.5rem 1.6rem; border: 1px solid #263851; border-radius: 18px; background: linear-gradient(135deg,#14243a,#101827); margin-bottom: 1.2rem; }
  .eyebrow { color: #73e0b2; font-size: .75rem; letter-spacing: .14em; font-weight: 700; }
  .muted { color: #a0aec3; }
  .prediction { border: 1px solid #2b6658; background: #122a2b; padding: 1rem 1.2rem; border-radius: 14px; }
  div[data-testid="stMetric"] { background: #111c2d; border: 1px solid #263851; padding: 1rem; border-radius: 12px; }
</style>
<div class="hero">
  <div class="eyebrow">TEAM 21 · MACHINE LEARNING MINI PROJECT</div>
  <h1 style="margin-bottom:.3rem">DriveAware</h1>
  <p class="muted" style="margin-bottom:0">Upload a driver image and let your trained ResNet-18 model predict one of the ten behavior classes.</p>
</div>
""", unsafe_allow_html=True)


def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}. Copy person2_resnet18_best.pth into this app folder."
        )
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    # Support common checkpoint formats used in PyTorch projects.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
    else:
        state = checkpoint

    # Remove a DataParallel prefix if one was saved.
    if isinstance(state, dict) and state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading trained ResNet-18 model…")
def load_model():
    return build_model()


def preprocess_image(image: Image.Image):
    # IMPORTANT: These transforms must match the preprocessing used during training.
    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
    return transform(image.convert("RGB")).unsqueeze(0)


uploaded = st.file_uploader("Choose a driver image", type=["jpg", "jpeg", "png", "webp"])
st.caption("Best results come from images similar to the State Farm training dataset.")

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Predict driver activity", type="primary", use_container_width=True):
        try:
            model = load_model()
            tensor = preprocess_image(image)
            with torch.inference_mode():
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0]
            confidence, index = torch.max(probabilities, dim=0)
            class_id = f"c{int(index)}"
            st.markdown('<div class="prediction">', unsafe_allow_html=True)
            st.markdown(f"### Prediction: **{CLASS_NAMES[class_id]}**")
            st.markdown(f"Class label: `{class_id}`")
            st.progress(float(confidence))
            st.metric("Model confidence", f"{float(confidence) * 100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            with st.expander("View scores for all classes"):
                rows = sorted(
                    [(CLASS_NAMES[f"c{i}"], f"c{i}", float(probabilities[i]) * 100) for i in range(10)],
                    key=lambda item: item[2], reverse=True
                )
                st.dataframe(
                    [{"Class": label, "Label": code, "Score (%)": round(score, 2)} for label, code, score in rows],
                    use_container_width=True, hide_index=True
                )
            st.warning("This is a project demonstration, not a safety-critical driver-monitoring system. Confidence is the model's score, not a guarantee that the prediction is correct.")
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error("The checkpoint could not be loaded or its architecture/preprocessing does not match this app.")
            st.code(str(e))
            st.info("Check that this is the ResNet-18 checkpoint with a 10-class output layer. Also make sure the normalization and image size match the training notebook.")
else:
    st.info("Upload an image above to enable prediction.")

with st.expander("About this project"):
    st.write("The project compares a custom CNN baseline (98.75% validation accuracy) with an ImageNet-pretrained ResNet-18 (99.75% validation accuracy) on the ten-class State Farm Distracted Driver Detection task. These are reported validation results; real-world performance may differ.")
    st.write("Classes: c0 Safe driving · c1 Texting right · c2 Phone right · c3 Texting left · c4 Phone left · c5 Operating radio · c6 Drinking · c7 Reaching behind · c8 Hair and makeup · c9 Talking to passenger.")
