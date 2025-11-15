import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from src.gradcam import make_gradcam_heatmap, overlay_heatmap


IMG_SIZE = 224
LAST_CONV_LAYER_NAME = "conv5_block3_out"


@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
    return tf.keras.models.load_model(model_path)


def load_and_preprocess_image(file_bytes, img_size=IMG_SIZE):
    img = tf.io.decode_image(file_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32)
    pre = resnet_preprocess(img[None, ...])
    return img.numpy().astype("uint8"), pre


def predict_and_explain(model, preprocessed, display_img):
    prob = float(model.predict(preprocessed, verbose=0)[0, 0])
    label = "Parasitized" if prob >= 0.5 else "Uninfected"

    heatmap = make_gradcam_heatmap(preprocessed, model, LAST_CONV_LAYER_NAME, pred_index=0)
    overlay = overlay_heatmap(heatmap, display_img)
    return label, prob, overlay


def main():
    st.title("Malaria Blood Cell Classification (ResNet-50)")
    st.write("Upload a single-cell blood smear image to classify it as Parasitized or Uninfected.")

    models_dir = Path("models")
    model_path = models_dir / "best_resnet50.h5"

    if not model_path.exists():
        st.error(f"Model file not found at {model_path}. Train the model first using the notebook.")
        return

    model = load_model(model_path)

    uploaded = st.file_uploader("Upload cell image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        file_bytes = uploaded.read()
        display_img, pre = load_and_preprocess_image(file_bytes)

        st.subheader("Input image")
        st.image(display_img, channels="RGB")

        with st.spinner("Running prediction and Grad-CAM..."):
            label, prob, overlay = predict_and_explain(model, pre, display_img)

        st.subheader("Prediction")
        st.write(f"**{label}** with probability **{prob:.4f}**")

        st.subheader("Grad-CAM heatmap")
        st.image(overlay, channels="RGB")


if __name__ == "__main__":
    main()
