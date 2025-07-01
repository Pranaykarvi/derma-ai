from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import torch

from utils.preprocess import preprocess_image
from model.predictor import get_model, predict_image

app = FastAPI()

# ✅ Mount the static folder (for Grad-CAM or debug images if any)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domain in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Lazy-load model at startup
model = None

@app.on_event("startup")
def load_model_once():
    global model
    model = get_model()  # Defined in model/predictor.py


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = preprocess_image(image)
    prediction = predict_image(model, input_tensor)  # Pass model explicitly
    return {"prediction": prediction}
