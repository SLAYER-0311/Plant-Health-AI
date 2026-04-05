"""
PlantHealth AI — Streamlit Demo Application
=============================================
A polished web interface for plant disease classification.
Upload a leaf image and get instant disease diagnosis.

Usage:
    streamlit run streamlit_app.py
"""

import sys
import os
import json
from pathlib import Path
from io import BytesIO

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet_transfer import PlantDiseaseResNet
from src.utils.ood_detection import OODDetector, create_default_detector


# ============================
# Configuration
# ============================
MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "plant_disease_model.pth"
CLASS_NAMES_PATH = PROJECT_ROOT / "backend" / "models" / "class_names.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "resnet50_best.pth"
IMAGE_SIZE = 224

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Treatment recommendations
TREATMENT_TIPS = {
    "healthy": "✅ Your plant looks healthy! Continue with regular care and monitoring.",
    "Apple_scab": "🍎 Apply fungicide (myclobutanil). Remove fallen infected leaves. Improve air circulation.",
    "Black_rot": "🍇 Remove infected plant parts. Apply copper-based fungicide. Ensure good drainage.",
    "Cedar_apple_rust": "🍎 Remove nearby juniper hosts. Apply fungicide in spring. Use resistant varieties.",
    "Powdery_mildew": "🌿 Apply neem oil or sulfur-based fungicide. Improve air circulation. Avoid overhead watering.",
    "Cercospora_leaf_spot": "🌽 Rotate crops. Apply fungicide. Remove crop debris after harvest.",
    "Common_rust": "🌽 Plant resistant hybrids. Apply fungicide at early signs. Remove infected leaves.",
    "Northern_Leaf_Blight": "🌽 Use resistant varieties. Apply foliar fungicide. Practice crop rotation.",
    "Esca": "🍇 No cure; manage by pruning affected areas. Avoid trunk wounds. Use resistant rootstocks.",
    "Leaf_blight": "🍇 Apply copper-based fungicide. Remove affected leaves. Improve air flow.",
    "Haunglongbing": "🍊 No cure; remove infected trees to prevent spread. Control psyllid vectors.",
    "Bacterial_spot": "🫑 Apply copper sprays. Use disease-free seeds. Practice crop rotation.",
    "Early_blight": "🍅 Apply chlorothalonil fungicide. Mulch around plants. Water at soil level.",
    "Late_blight": "🍅 Apply mancozeb or chlorothalonil. Remove infected plants immediately. Avoid wetting leaves.",
    "Leaf_Mold": "🍅 Improve ventilation. Reduce humidity. Apply fungicide preventatively.",
    "Septoria_leaf_spot": "🍅 Remove infected leaves. Apply fungicide. Avoid overhead irrigation.",
    "Spider_mites": "🍅 Spray with insecticidal soap or neem oil. Increase humidity. Introduce predatory mites.",
    "Target_Spot": "🍅 Apply fungicide. Improve air circulation. Remove lower infected leaves.",
    "Yellow_Leaf_Curl_Virus": "🍅 Control whitefly vectors. Use resistant varieties. Remove infected plants.",
    "Tomato_mosaic_virus": "🍅 No cure; remove infected plants. Disinfect tools. Use resistant varieties.",
    "Leaf_scorch": "🍓 Ensure adequate watering. Apply fungicide. Mulch to retain moisture.",
}


# ============================
# Model Loading
# ============================
@st.cache_resource
def load_model():
    """Load the trained model and class names."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load class names
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, 'r') as f:
            data = json.load(f)
            class_names = data if isinstance(data, list) else data.get('class_names', [])
    else:
        class_names = get_default_class_names()
    
    num_classes = len(class_names)
    
    # Create model
    model = PlantDiseaseResNet(
        num_classes=num_classes,
        dropout_rate=0.5,
        pretrained=False,
        freeze_backbone=False,
    )
    
    # Try loading model weights from multiple locations
    model_loaded = False
    for path in [MODEL_PATH, CHECKPOINT_PATH]:
        if path.exists():
            try:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                model_loaded = True
                break
            except Exception as e:
                continue
    
    if not model_loaded:
        st.warning("⚠️ No trained model found. Predictions will be random. Train a model first.")
    
    model.to(device)
    model.eval()
    
    # OOD detector
    ood_detector = create_default_detector(strict=True)
    
    return model, class_names, device, ood_detector


def get_default_class_names():
    return [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
        "Apple___healthy", "Blueberry___healthy",
        "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy", "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot", "Peach___healthy",
        "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
        "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
        "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch", "Strawberry___healthy",
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
        "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess an image for model inference."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def parse_class_name(class_name: str):
    """Parse class name into plant and condition."""
    if "___" in class_name:
        parts = class_name.split("___")
        plant = parts[0].replace("_", " ").replace(",", ", ")
        condition = parts[1].replace("_", " ")
    else:
        plant = class_name
        condition = "Unknown"
    return plant, condition


def get_treatment(condition: str) -> str:
    """Get treatment recommendation for a condition."""
    for key, tip in TREATMENT_TIPS.items():
        if key.lower() in condition.lower():
            return tip
    return "📖 Consult a local agricultural extension service for specific treatment recommendations."


# ============================
# Streamlit App
# ============================
def main():
    # Page config
    st.set_page_config(
        page_title="PlantHealth AI — Plant Disease Classifier",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
        }
        .prediction-card {
            background: linear-gradient(135deg, #f0f9f0, #e8f5e9);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #c8e6c9;
            margin: 0.5rem 0;
        }
        .warning-card {
            background: linear-gradient(135deg, #fff3e0, #ffe0b2);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #ffcc80;
            margin: 0.5rem 0;
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2e7d32;
        }
        .stProgress > div > div > div > div {
            background-color: #4caf50;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<div class='main-header'>", unsafe_allow_html=True)
    st.title("🌿 PlantHealth AI")
    st.markdown("### *Instant Plant Disease Diagnosis Powered by Deep Learning*")
    st.markdown("Upload a photo of a plant leaf and get an AI-powered diagnosis in seconds.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Load model
    model, class_names, device, ood_detector = load_model()
    
    # ---- Sidebar ----
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **PlantHealth AI** uses a **ResNet50** deep learning model 
        trained on the PlantVillage dataset to classify plant diseases 
        from leaf images.
        """)
        
        st.divider()
        
        st.subheader("🌱 Supported Plants")
        plants = sorted(set(
            name.split("___")[0].replace("_", " ").replace(",", ", ")
            for name in class_names
        ))
        for plant in plants:
            st.markdown(f"• {plant}")
        
        st.divider()
        
        st.subheader("📊 Model Info")
        st.markdown(f"""
        - **Architecture**: ResNet50
        - **Classes**: {len(class_names)}
        - **Input Size**: {IMAGE_SIZE}×{IMAGE_SIZE}
        - **Device**: {device}
        """)
        
        st.divider()
        st.caption("Built with ❤️ by PlantHealth AI Team")
    
    # ---- Main Content ----
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📸 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a leaf image...",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Upload a clear, well-lit photo of a plant leaf.",
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width="stretch")
            
            # Image info
            st.caption(f"📐 {image.size[0]}×{image.size[1]} px | {uploaded_file.size / 1024:.1f} KB")
    
    with col2:
        st.subheader("🔬 Diagnosis Results")
        
        if uploaded_file is not None:
            with st.spinner("🧠 Analyzing leaf..."):
                # Preprocess
                tensor = preprocess_image(image).to(device)
                
                # Inference
                import time
                start = time.perf_counter()
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1)
                inference_time = (time.perf_counter() - start) * 1000
                
                # OOD Detection
                is_ood, ood_scores = ood_detector.detect(outputs, return_scores=True)
                
                if is_ood:
                    st.markdown("<div class='warning-card'>", unsafe_allow_html=True)
                    st.warning("⚠️ **This image does not appear to be a plant leaf!**")
                    st.markdown(ood_detector.get_recommendation(ood_scores))
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.info("Please upload a clear photo of a plant leaf for accurate diagnosis.")
                else:
                    # Top prediction
                    probs_np = probs[0].cpu().numpy()
                    top_idx = np.argsort(probs_np)[::-1]
                    
                    top_class = class_names[top_idx[0]]
                    top_conf = float(probs_np[top_idx[0]] * 100)
                    plant, condition = parse_class_name(top_class)
                    
                    # Display results
                    is_healthy = "healthy" in condition.lower()
                    
                    if is_healthy:
                        st.success(f"### ✅ {plant} — Healthy!")
                    else:
                        st.error(f"### ⚠️ {plant} — {condition}")
                    
                    # Confidence
                    st.metric("Confidence", f"{top_conf:.1f}%")
                    st.progress(float(top_conf / 100))
                    
                    # Treatment
                    st.markdown("---")
                    st.markdown("#### 💊 Treatment Recommendation")
                    treatment = get_treatment(condition)
                    st.info(treatment)
                    
                    # Top-5 predictions
                    st.markdown("---")
                    with st.expander("📊 Top 5 Predictions", expanded=False):
                        for i in range(min(5, len(top_idx))):
                            idx = top_idx[i]
                            name = class_names[idx]
                            conf = float(probs_np[idx] * 100)
                            p, c = parse_class_name(name)
                            
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.markdown(f"**{i+1}.** {p} — {c}")
                                st.progress(float(conf / 100))
                            with col_b:
                                st.markdown(f"**{conf:.1f}%**")
                    
                    # OOD details
                    with st.expander("🛡️ Image Quality Check", expanded=False):
                        st.markdown(f"- Max probability: {ood_scores['max_probability']:.3f}")
                        st.markdown(f"- Prediction entropy: {ood_scores['entropy']:.3f}")
                        st.markdown(f"- In-distribution votes: {ood_scores['in_distribution_votes']}/{ood_scores['total_votes']}")
                        st.markdown(f"- ✅ Image appears to be a valid plant leaf")
                
                # Inference time
                st.caption(f"⚡ Inference time: {inference_time:.1f} ms")
        
        else:
            st.info("👈 Upload a leaf image to get started!")
            
            st.markdown("### How it works")
            st.markdown("""
            1. 📸 **Upload** a photo of a plant leaf
            2. 🧠 **AI analyzes** the image using deep learning
            3. 🔬 **Get diagnosis** with disease name and confidence
            4. 💊 **Treatment tips** for identified diseases
            """)


if __name__ == "__main__":
    main()
