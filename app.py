from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI()

# Mount static folder (css, images, js)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# ---------------- Load ML Model ----------------
crop_model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ---------------- Home Page ----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------------- Crop Recommendation ----------------
@app.post("/crop-recommendation", response_class=HTMLResponse)
async def crop_recommendation(
    request: Request,
    ph: float = Form(...),
    ec: float = Form(...),
    n: float = Form(...),
    p: float = Form(...),
    k: float = Form(...),
    texture: str = Form(...)
):
    # Encode texture into numerical features
    texture_map = {
        "sandy": [1, 0, 0, 0, 0, 0, 0],
        "loamy": [0, 1, 0, 0, 0, 0, 0],
        "clayey": [0, 0, 1, 0, 0, 0, 0],
        "silty": [0, 0, 0, 1, 0, 0, 0],
        "sandy-loam": [0, 0, 0, 0, 1, 0, 0],
        "clay-loam": [0, 0, 0, 0, 0, 1, 0],
        "silt-loam": [0, 0, 0, 0, 0, 0, 1],
    }
    texture_encoded = texture_map.get(texture.lower(), [0, 0, 0, 0, 0, 0, 0])

    # Prepare input features (adjust order as per your training dataset)
    input_features = np.array([[ph, ec, n, p, k] + texture_encoded])
    input_scaled = scaler.transform(input_features)

    # Predict crop
    prediction = crop_model.predict(input_scaled)
    predicted_crop = label_encoder.inverse_transform(prediction)[0]

    # Prepare results
    crops = [{"name": predicted_crop, "score": "Best Match (ML Prediction)"}]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "crop_results": crops
    })


