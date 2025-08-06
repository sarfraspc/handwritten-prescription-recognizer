
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import FastAPI, File, UploadFile
from src.inference.predictor import Predictor
import shutil
import os

# --- Configuration ---
MODEL_PATH = 'models/crnn_model.h5'
CHAR_MAP_PATH = 'data/processed/char_map.json'
IMG_HEIGHT = 64
IMG_WIDTH = 256

# Initialize the FastAPI app and the predictor
app = FastAPI()
predictor = Predictor(MODEL_PATH, CHAR_MAP_PATH, IMG_HEIGHT, IMG_WIDTH)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """
    Receives an image file, saves it temporarily, and returns the predicted text.
    """
    # Create a temporary path to save the uploaded file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Get the prediction
    try:
        prediction = predictor.predict(temp_path)
        return {"prediction": prediction}
    finally:
        # Clean up the temporary file
        os.remove(temp_path)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Handwritten Prescription OCR API"}

