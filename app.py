"""Streamlit frontend for the breast cancer predictor.

Run:  streamlit run app.py
"""
import os
import subprocess
import sys

import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

from breast_cancer_predictor import MODEL_PATH, predict

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺", layout="wide")


@st.cache_data
def load_reference():
    d = load_breast_cancer(as_frame=True)
    return d.data, d.target


@st.cache_resource
def ensure_model():
    """Train the model on first launch if no saved model exists."""
    if not os.path.exists(MODEL_PATH):
        subprocess.run([sys.executable, "breast_cancer_predictor.py"], check=True)
    return True


X, y = load_reference()
with st.spinner("Preparing model..."):
    ensure_model()

FEATURES = list(X.columns)
GROUPS = {
    "Mean values": [f for f in FEATURES if f.startswith("mean ")],
    "Standard error": [f for f in FEATURES if f.endswith(" error")],
    "Worst values": [f for f in FEATURES if f.startswith("worst ")],
}


def load_preset(label):
    """label 0 = malignant, 1 = benign. Fills sliders with that class's median."""
    for name, val in X[y == label].median().items():
        st.session_state[name] = float(val)


for name, val in X[y == 1].median().items():
    st.session_state.setdefault(name, float(val))

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Quick start")
    st.button("Load typical benign case", on_click=load_preset, args=(1,), use_container_width=True)
    st.button("Load typical malignant case", on_click=load_preset, args=(0,), use_container_width=True)
    st.divider()
    st.header("Batch prediction")
    upload = st.file_uploader("Upload CSV with the 30 feature columns", type="csv")
    st.caption("Tip: sklearn's column names, e.g. `mean radius`, `worst texture`.")

# ---------------- Main ----------------
st.title("🩺 Breast Cancer Predictor")
st.warning("Educational demo only. Not a medical device and not for real diagnosis.")

st.subheader("Tumor measurements")
tabs = st.tabs(list(GROUPS))
values = {}
for tab, feats in zip(tabs, GROUPS.values()):
    with tab:
        cols = st.columns(2)
        for i, f in enumerate(feats):
            lo, hi = float(X[f].min()), float(X[f].max())
            with cols[i % 2]:
                values[f] = st.slider(f, lo, hi, key=f, step=(hi - lo) / 500)

result = predict(values)[0]
p_mal = result["probability_malignant"]

st.subheader("Prediction")
c1, c2 = st.columns([1, 2])
c1.metric("Result", result["prediction"])
c1.metric("Probability malignant", f"{p_mal:.1%}")
c2.write("Malignancy probability")
c2.progress(p_mal)
if result["prediction"] == "Malignant":
    c2.error("Model predicts MALIGNANT.")
else:
    c2.success("Model predicts BENIGN.")

if upload is not None:
    st.subheader("Batch results")
    try:
        df = pd.read_csv(upload)
        out = pd.DataFrame(predict(df))
        st.dataframe(pd.concat([df.reset_index(drop=True), out], axis=1), use_container_width=True)
        st.download_button("Download results", out.to_csv(index=False), "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Could not process file: {e}")
