import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import timm
from captum.attr import IntegratedGradients
from timm.models import swin_tiny_patch4_window7_224
import cv2

# Streamlit page config
st.set_page_config(page_title="Derma-AI", layout="wide")

# Load model
@st.cache_resource
def load_model():
    model = swin_tiny_patch4_window7_224(pretrained=False, num_classes=7)
    model.load_state_dict(torch.load(r"D:\target\ml\derma-ai\notebooks\vit_normal\best_swin_model.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Class info
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
class_descriptions = {
    'akiec': 'Actinic keratoses and intraepithelial carcinoma',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}
suggestions = {
    'akiec': "⚠️ Precancerous: Consult a dermatologist soon for biopsy or treatment options.",
    'bcc': "🧬 Low-risk skin cancer: Usually treated with surgery. Follow up with a skin specialist.",
    'bkl': "✔️ Benign: No immediate concern. Still, monitor for any changes in size or color.",
    'df': "✔️ Benign: Dermatofibromas are harmless but can be removed if bothersome.",
    'mel': "🚨 Urgent: High-risk melanoma detected. Seek immediate medical evaluation.",
    'nv': "✔️ Likely benign nevus (mole). Regular skin checks recommended.",
    'vasc': "✔️ Vascular lesion: Typically harmless but may require cosmetic treatment if needed."
}

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB normalization
])

def preprocess_image(image):
    img = image.convert("RGB")
    return transform(img)

# Overlay heatmap on image
def overlay_heatmap(orig_img, attr_map):
    attr_map_resized = cv2.resize(attr_map, orig_img.size)
    attr_map_resized = (attr_map_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(attr_map_resized, cv2.COLORMAP_PLASMA)
    heatmap = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    overlay = Image.blend(orig_img, heatmap, alpha=0.5)
    return overlay

# App UI
st.title("🧴 Derma-AI: Skin Lesion Classification using Vision Transformers")
st.write("Upload a skin lesion image and receive an AI-based diagnosis with interpretability.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = preprocess_image(image)

    # 🔍 Prediction
    with torch.no_grad():
        outputs = model(img_tensor.unsqueeze(0))
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    pred_label = class_names[pred_class]
    st.success(f"✅ **Prediction**: `{pred_label}` ({class_descriptions[pred_label]})")
    st.info(f"🧪 **Confidence**: `{confidence:.2f}`")
    st.warning(suggestions[pred_label])

    # Explainability Map
    if st.checkbox("🧠 Show Explainability Map"):
        ig = IntegratedGradients(model)
        img_tensor.requires_grad_()
        attr, delta = ig.attribute(img_tensor.unsqueeze(0), target=pred_class, return_convergence_delta=True)
        attr = attr.squeeze().detach().numpy()  # shape: (3, 224, 224)

        # Improved attribution map (sum of absolute values across channels)
        attr_map = np.sum(np.abs(attr), axis=0)

        # Normalize to 0-1
        attr_map = (attr_map - np.min(attr_map)) / (np.max(attr_map) - np.min(attr_map) + 1e-8)

        result_img = overlay_heatmap(image, attr_map)
        st.image(result_img, caption="🧠 Explainability Map", use_column_width=True)
        st.caption("This visualization highlights regions that most influenced the model’s decision.")
