# Derma-AI

An end-to-end deep learning project that detects and classifies seven types of common skin lesions using the HAM10000 dataset and a CNN-based model. The application also provides medical insights and treatment suggestions based on the prediction.

---

## 📌 Project Overview

Skin diseases, especially malignant ones like melanoma, can be life-threatening if not detected early. This project leverages the power of **Convolutional Neural Networks (CNNs)** and **Transfer Learning** to classify dermatological images and guide users with basic treatment advice.

### 🔍 Key Features
- Classifies skin lesions into **7 categories**
- Uses **EfficientNet/ResNet** for high accuracy
- Web interface for uploading images
- Provides **confidence scores**, medical names, and **first-aid/treatment suggestions**
- Optional Grad-CAM visualization to highlight lesion areas

---

## 📂 Dataset

**Name:** HAM10000 - Human Against Machine with 10000 training images  
**Source:** [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

### Classes:
| Label  | Condition                   |
|--------|-----------------------------|
| akiec  | Actinic keratoses           |
| bcc    | Basal cell carcinoma        |
| bkl    | Benign keratosis-like lesions |
| df     | Dermatofibroma              |
| mel    | Melanoma                    |
| nv     | Melanocytic nevi            |
| vasc   | Vascular lesions            |

---

## 🧠 Model Architecture

We used **Transfer Learning** with fine-tuned CNN models such as:

- `EfficientNetB0`
- `ResNet50`

### Layers:
- Image Input Layer (`224x224`)
- Convolutional Base from pretrained model
- Global Average Pooling
- Dense + Dropout Layers
- Softmax Output (7 classes)

### Loss & Metrics:
- `Categorical Crossentropy`
- `Accuracy`, `Precision`, `Recall`, `F1-score`

---

## 🛠️ Tech Stack

| Tool            | Purpose                           |
|-----------------|-----------------------------------|
| Python          | Core language                     |
| TensorFlow / Keras | Deep learning library            |
| Pandas, NumPy   | Data preprocessing                |
| OpenCV / PIL    | Image handling                    |
| Matplotlib / Seaborn | Visualization                  |
| Streamlit       | Web app frontend                  |
| Grad-CAM        | Visual explanation of predictions |

---

## 🧪 Model Training

### 🔄 Preprocessing:
- Resizing images to `224x224`
- Normalizing pixel values (0–1)
- One-hot encoding labels
- Train-Validation-Test Split (70/15/15)
- Class balancing using **data augmentation**

### 📈 Results:
- Accuracy: ~`90+%` (Varies with model)
- Confusion matrix and classification report used for performance analysis

---

## 🌐 Web Application

### 📸 Features:
- Upload skin image
- Get:
  - Predicted condition
  - Confidence level
  - Medical name and short description
  - First-aid and treatment guidance
- View highlighted region (optional Grad-CAM)

### 🖥️ Launch App Locally:
```bash
git clone https://github.com/your-username/skin-disease-classifier.git
cd skin-disease-classifier
pip install -r requirements.txt
streamlit run app.py
```
## 📁 Project Structure
```
Derma-AI/
├── data/
│   ├── HAM10000_images/
│   └── metadata.csv
├── model/
│   └── skin_disease_model.h5
├── app.py
├── utils/
│   ├── preprocess.py
│   ├── gradcam.py
├── treatment_info.json
├── requirements.txt
└── README.md
```
---
## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

  
