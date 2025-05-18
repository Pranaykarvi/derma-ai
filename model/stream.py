# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
from gradcam import get_gradcam_heatmap
import json

# Load models
model1 = load_model(r"D:\target\ml\derma-ai\notebooks\3-model-ka-sangam..)\model\model_1.h5")  # EfficientNetB3
model2 = load_model(r"D:\target\ml\derma-ai\notebooks\3-model-ka-sangam..)\model\model_2.h5")  # DenseNet201
models = [model1, model2]

# Load treatment info
with open("treatment_info.json", "r") as f:
    treatment_info = json.load(f)

labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def ensemble_predict(img):
    preds = [model.predict(img)[0] for model in models]
    avg_pred = np.mean(preds, axis=0)
    return avg_pred

def show_heatmap(image, model, last_conv_name='top_conv'):
    img_array = preprocess_image(image)
    heatmap = get_gradcam_heatmap(model, img_array, last_conv_name)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)
    return superimposed

# --- Streamlit UI ---
st.set_page_config(page_title="Derma-AI", layout="centered")
st.title("🩺 Derma-AI: Skin Lesion Classifier")
st.write("Upload a skin lesion image to classify its type and get treatment info.")

file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess_image(image)
    prediction = ensemble_predict(img)
    class_index = np.argmax(prediction)
    label = labels[class_index]
    confidence = prediction[class_index] * 100

    st.subheader("Prediction")
    st.write(f"**Condition:** {label.upper()} ({confidence:.2f}% confidence)")
    st.info(treatment_info.get(label, "No treatment info available."))

    # Show Grad-CAM
    if st.checkbox("Show Grad-CAM Heatmap"):
        heatmap_img = show_heatmap(image, model1)  # EfficientNetB3
        st.image(heatmap_img, caption="Grad-CAM Heatmap", use_column_width=True)
