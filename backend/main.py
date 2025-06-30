from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # ✅ Add this
from model.predictor import predict_image
from utils.preprocess import preprocess_image
from PIL import Image
import io

app = FastAPI()

# ✅ Mount the static folder to serve Grad-CAM images
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = preprocess_image(image)
    prediction = predict_image(input_tensor)
    return {"prediction": prediction}
