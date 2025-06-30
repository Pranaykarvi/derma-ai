import torch
import torch.nn.functional as F
from model.loader import load_model
import uuid

model = load_model()
model.eval()
model.to("cpu")

class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def predict_image(input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor.unsqueeze(0))
        probs = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    return {
        "predictedClass": class_names[pred_idx],
        "confidence": round(confidence, 4)
    }
