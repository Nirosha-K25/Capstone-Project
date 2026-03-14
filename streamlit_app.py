"""
Multi-class COVID-19 Detection from Chest X-ray Images
Streamlit Web Application for Model Deployment

Classes: COVID-19 | Normal | Viral Pneumonia
"""

import streamlit as st
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# TensorFlow import with error handling
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ─── Page Configuration ───
st.set_page_config(
    page_title="COVID-19 X-ray Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .covid {background-color: #ffcccc; border: 2px solid #ff4444;}
    .normal {background-color: #ccffcc; border: 2px solid #44ff44;}
    .pneumonia {background-color: #ccccff; border: 2px solid #4444ff;}
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───
IMG_SIZE = (128, 128)
MODEL_PATH = "covid19_xray_model.keras"
CLASS_NAMES_PATH = "class_names.json"

# Default class names if JSON not found
DEFAULT_CLASS_NAMES = {0: "Covid", 1: "Normal", 2: "Viral Pneumonia"}

# Class colors for visualization
CLASS_COLORS = {
    "Covid": "#FF6B6B",
    "Normal": "#4ECDC4",
    "Viral Pneumonia": "#45B7D1"
}


@st.cache_resource
def load_model():
    """Load the trained model."""
    if not TF_AVAILABLE:
        return None
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    return None


def load_class_names():
    """Load class names mapping."""
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        # Convert string keys to int
        return {int(k): v for k, v in class_names.items()}
    return DEFAULT_CLASS_NAMES


def preprocess_image(image):
    """Preprocess uploaded image for model prediction."""
    img = image.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def generate_gradcam(model, img_array):
    """Generate Grad-CAM for Sequential VGG16 model.
    Accesses the base model's functional API to avoid Keras 3 Sequential .input issues."""
    if not TF_AVAILABLE:
        return None

    try:
        # Get the VGG16 base model (first layer of Sequential)
        base = model.layers[0]

        # Find last conv layer in base model
        last_conv = None
        for layer in base.layers:
            if 'conv' in layer.name:
                last_conv = layer.name

        if last_conv is None:
            return None

        # Build grad model using the base model's functional API
        inputs = base.input
        conv_output = base.get_layer(last_conv).output
        x = base.output
        for layer in model.layers[1:]:
            x = layer(x)

        grad_model = tf.keras.models.Model(
            inputs=inputs,
            outputs=[conv_output, x]
        )

        img_tensor = tf.constant(img_array)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            pred_idx = tf.argmax(predictions[0])
            class_output = predictions[:, pred_idx]

        grads = tape.gradient(class_output, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Resize heatmap
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_resized = np.array(Image.fromarray(heatmap_uint8).resize(IMG_SIZE))
        heatmap_colored = cm.jet(heatmap_resized / 255.0)[:, :, :3]

        # Overlay on original image
        original = img_array[0]
        superimposed = np.clip(original * 0.6 + heatmap_colored * 0.4, 0, 1)

        return superimposed

    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {str(e)}")
        return None


# ─── Sidebar ───
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lungs.png", width=80)
    st.title("About")
    st.markdown("""
    This application uses **Deep Learning** to classify chest X-ray images into:

    - **COVID-19** - Coronavirus infection
    - **Normal** - Healthy lungs
    - **Viral Pneumonia** - Non-COVID pneumonia

    **Model:** VGG16 Transfer Learning (fine-tuned on ImageNet)

    **Dataset:** COVID-19 Image Dataset (Kaggle)
    """)

    st.divider()
    st.markdown("### How to Use")
    st.markdown("""
    1. Upload a chest X-ray image
    2. Click **Analyze X-ray**
    3. View the prediction and Grad-CAM explanation
    """)

    st.divider()
    st.warning("""
    **Disclaimer:** This tool is for educational purposes only.
    It should NOT be used for actual medical diagnosis.
    Always consult a qualified healthcare professional.
    """)


# ─── Main Content ───
st.markdown("<h1 class='main-header'>Multi-class COVID-19 Detection from Chest X-rays</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Upload a chest X-ray image to classify it as COVID-19, Normal, or Viral Pneumonia</p>",
            unsafe_allow_html=True)

# Load model
model = load_model()
class_names = load_class_names()

if model is None:
    st.error("""
    **Model not found!** Please ensure you have:
    1. Trained the model using the Jupyter notebook (`covid19_xray_classification.ipynb`)
    2. The model file `covid19_xray_model.keras` exists in the same directory

    Run the notebook first to generate the trained model.
    """)
    if not TF_AVAILABLE:
        st.error("TensorFlow is not installed. Install it with: `pip install tensorflow`")
else:
    st.success("Model loaded successfully!")

st.divider()

# ─── File Upload ───
col_upload, col_preview = st.columns([1, 1])

with col_upload:
    st.subheader("Upload X-ray Image")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a chest X-ray image in JPG, PNG, or BMP format"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

with col_preview:
    if uploaded_file is not None and model is not None:
        st.subheader("Analysis")

        if st.button("Analyze X-ray", type="primary", use_container_width=True):
            with st.spinner("Analyzing X-ray image..."):
                # Preprocess
                img_array = preprocess_image(image)

                # Predict
                predictions = model.predict(img_array, verbose=0)
                pred_class_idx = np.argmax(predictions[0])
                pred_class_name = class_names[pred_class_idx]
                confidence = predictions[0][pred_class_idx]

                # Display prediction
                css_class = "covid" if "Covid" in pred_class_name else \
                            "normal" if "Normal" in pred_class_name else "pneumonia"

                st.markdown(
                    f"<div class='prediction-box {css_class}'>"
                    f"Prediction: {pred_class_name}<br>"
                    f"Confidence: {confidence:.1%}</div>",
                    unsafe_allow_html=True
                )

                # Show all class probabilities
                st.markdown("#### Class Probabilities")
                for i in range(len(class_names)):
                    cls_name = class_names[i]
                    prob = predictions[0][i]
                    color = CLASS_COLORS.get(cls_name, "#666666")
                    st.markdown(f"**{cls_name}**")
                    st.progress(float(prob), text=f"{prob:.1%}")

                # Grad-CAM
                st.markdown("#### Grad-CAM Visualization")
                st.caption("Highlights regions the model focuses on for its prediction")

                gradcam_result = generate_gradcam(model, img_array)

                if gradcam_result is not None:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

                    axes[0].imshow(img_array[0])
                    axes[0].set_title("Original X-ray")
                    axes[0].axis('off')

                    axes[1].imshow(gradcam_result)
                    axes[1].set_title(f"Grad-CAM: {pred_class_name} ({confidence:.1%})")
                    axes[1].axis('off')

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("Grad-CAM visualization not available for this model architecture.")

    elif uploaded_file is not None and model is None:
        st.warning("Model is not loaded. Please train the model first.")
    else:
        st.info("Upload an X-ray image to get started.")

# ─── Footer ───
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
    <p>Multi-class COVID-19 Detection | Deep Learning Project</p>
    <p>Built with Streamlit | Model: VGG16 Transfer Learning</p>
</div>
""", unsafe_allow_html=True)
