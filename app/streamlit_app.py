"""
Streamlit Web Application for Plant Leaf Disease Classification.
Provides an interactive, aesthetic interface for real-time plant disease detection,
sample leaf testing, confidence metrics, and prediction session history.
"""

from datetime import datetime
import os
import sys
import time
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import CLASS_NAMES, parse_label, resolve_dataset_dir
from src.monitoring import PredictionLogger
from src.predict import PlantDiseasePredictor
from src.registry import get_current_production_model

# Page configuration
st.set_page_config(
    page_title="Plant Leaf Disease AI Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #2E7D32;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .status-card {
        padding: 1.2rem;
        border-radius: 10px;
        background-color: #F1F8E9;
        border-left: 6px solid #4CAF50;
        margin-bottom: 1rem;
    }
    .warning-card {
        padding: 1.2rem;
        border-radius: 10px;
        background-color: #FFFDE7;
        border-left: 6px solid #FBC02D;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_inference_engine():
    """Initializes and caches the prediction engine and logger."""
    prod_info = get_current_production_model()
    checkpoint_path = prod_info.get("checkpoint_path", "plant_disease_resnet50.pth")
    if not os.path.isabs(checkpoint_path) and not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", checkpoint_path))
    
    predictor = PlantDiseasePredictor(checkpoint_path=checkpoint_path)
    logger = PredictionLogger()
    return predictor, logger, prod_info


try:
    predictor, logger, prod_info = load_inference_engine()
except Exception as e:
    st.error(f"Failed to load model checkpoint: {str(e)}")
    st.stop()

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?w=500&auto=format&fit=crop&q=60", use_container_width=True)
    st.title("🌱 Model Information")
    st.markdown(f"**Architecture:** ResNet-50 (Transfer Learning)")
    st.markdown(f"**Active Model Version:** `{prod_info.get('version', '1.0.0-baseline')}`")
    st.markdown(f"**Baseline Validation Accuracy:** `95.01%`")
    st.markdown(f"**Execution Device:** `{predictor.target_device}`")
    st.markdown(f"**Supported Classes:** `15`")

    st.divider()
    st.subheader("⚙️ Settings")
    uncertainty_thresh = st.slider("Uncertainty Warning Threshold", min_value=0.30, max_value=0.90, value=0.60, step=0.05)
    predictor.uncertainty_threshold = uncertainty_thresh

    st.divider()
    st.markdown("""
    **Supported Crops:**
    - 🫑 Bell Pepper
    - 🥔 Potato
    - 🍅 Tomato

    ⚠️ **Disclaimer:** This tool is an AI diagnostic aid intended for research and educational demonstrations. It is not an authoritative replacement for certified agronomist inspection or lab testing.
    """)

# Main Content
st.markdown('<div class="main-header">🌿 Plant Disease Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated leaf diagnosis across Pepper, Potato, and Tomato crops using Deep Learning</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Provide Leaf Image")
    input_source = st.radio("Choose image source:", ["Upload Leaf Image", "Select a Sample from Dataset"], horizontal=True)

    selected_image = None
    image_name = "uploaded_leaf.jpg"

    if input_source == "Upload Leaf Image":
        uploaded_file = st.file_uploader("Upload leaf photograph (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file)
                image_name = uploaded_file.name
            except Exception as e:
                st.error("Uploaded file is not a valid image.")
    else:
        # Sample selector
        try:
            dataset_dir = resolve_dataset_dir()
            available_classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")]
            selected_class = st.selectbox("Select sample disease category:", available_classes)
            class_folder = os.path.join(dataset_dir, selected_class)
            sample_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:10]
            if sample_files:
                selected_sample_file = st.selectbox("Select image:", sample_files)
                sample_path = os.path.join(class_folder, selected_sample_file)
                selected_image = Image.open(sample_path)
                image_name = f"Sample: {selected_class}/{selected_sample_file}"
        except Exception:
            st.info("Dataset samples folder not available for interactive browsing.")

    if selected_image is not None:
        st.image(selected_image, caption=image_name, use_container_width=True)

with col2:
    st.subheader("🔬 Diagnosis Results")

    if selected_image is not None:
        if st.button("Run AI Diagnosis", type="primary", use_container_width=True):
            with st.spinner("Analyzing leaf pathology..."):
                try:
                    result = predictor.predict(selected_image)

                    # Log prediction to monitoring
                    logger.log_prediction(
                        model_version=result["model_version"],
                        predicted_class=result["top_prediction"],
                        confidence=result["confidence"],
                        latency_ms=result["inference_time_ms"],
                        image_width=selected_image.width,
                        image_height=selected_image.height,
                        status="success",
                        is_uncertain=result["is_uncertain"]
                    )

                    # Store in session history
                    st.session_state.history.append({
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Image": image_name,
                        "Crop": result["crop"],
                        "Diagnosis": result["disease"],
                        "Confidence": f"{result['confidence']:.1%}",
                        "Status": "Healthy" if result["is_healthy"] else "Diseased",
                        "Latency": f"{result['inference_time_ms']} ms"
                    })

                    # Display Top Result Card
                    status_class = "warning-card" if result["is_uncertain"] else "status-card"
                    badge = "✅ HEALTHY" if result["is_healthy"] else "⚠️ DISEASE DETECTED"
                    
                    st.markdown(f"""
                    <div class="{status_class}">
                        <h4 style="margin:0; color:#333;">{badge}</h4>
                        <h2 style="margin: 0.2rem 0; color:#1B5E20;">{result['display_name']}</h2>
                        <p style="margin:0; color:#555;">Crop: <b>{result['crop']}</b> | Condition: <b>{result['disease']}</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                    if result["is_uncertain"]:
                        st.warning(result["warning"])

                    # Metrics Row
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Confidence Score", f"{result['confidence']:.1%}")
                    with m2:
                        st.metric("Inference Latency", f"{result['inference_time_ms']} ms")
                    with m3:
                        st.metric("Model Architecture", result["model_version"])

                    st.markdown("---")
                    st.markdown("#### Top 3 Differential Predictions")
                    for idx, candidate in enumerate(result["top_k"]):
                        p_col1, p_col2 = st.columns([3, 1])
                        with p_col1:
                            st.write(f"**{idx+1}. {candidate['display_name']}**")
                            st.progress(candidate["confidence"])
                        with p_col2:
                            st.write(f"**{candidate['confidence']:.1%}**")

                except Exception as e:
                    st.error(f"Inference Error: {str(e)}")
    else:
        st.info("👈 Upload an image or select a sample from the left panel to run plant leaf classification.")

# Session Prediction History
if st.session_state.history:
    st.divider()
    st.subheader("📋 Current Session Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
