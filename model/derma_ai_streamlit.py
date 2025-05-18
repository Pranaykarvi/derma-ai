import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

from captum.attr import LayerGradCam, visualize_image_attr
from torchvision.models import swin_t, Swin_T_Weights

# Load model
@st.cache_resource
def load_model():
    model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    model.head = torch.nn.Linear(model.head.in_features, 7)  # For HAM10000 classes
    model.load_state_dict(torch.load("swin_skin_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Class names
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return transform(image).unsqueeze(0)

# Generate Grad-CAM heatmap
def generate_heatmap(model, input_tensor, target_layer):
    gradcam = LayerGradCam(model, target_layer)
    attr = gradcam.attribute(input_tensor, target=torch.argmax(model(input_tensor)))
    upsampled_attr = torch.nn.functional.interpolate(attr, size=(224, 224), mode='bilinear')[0, 0].detach().numpy()
    
    # Normalize
    upsampled_attr = (upsampled_attr - upsampled_attr.min()) / (upsampled_attr.max() - upsampled_attr.min() + 1e-8)
    return upsampled_attr

# Heatmap overlay
def overlay_heatmap(image, heatmap):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_MAGMA)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    image = np.array(image.resize((224, 224))).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    return overlay

# Streamlit UI
st.title("🧠 Derma-AI: Skin Lesion Classification with Explainability")

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    input_tensor = preprocess_image(image)

    with st.spinner("Classifying and generating explanation..."):
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        class_name = class_names[prediction]

        heatmap = generate_heatmap(model, input_tensor, model.features[-1].blocks[-1].norm1)
        overlay = overlay_heatmap(image, heatmap)

    st.subheader(f"🔍 Predicted: {class_name.upper()}")
    st.image(overlay, caption="Explainability Heatmap (Grad-CAM)", use_column_width=True)
