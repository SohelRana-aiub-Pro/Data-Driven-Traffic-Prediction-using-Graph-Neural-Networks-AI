#import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates





import numpy as np
import torch

from models.stgcn_model import STGCN  # Best performer

# ✅ Project title
app = FastAPI(title="Data Driven Traffic Prediction API (Synthetic Dataset)")

# Templates only (no static folder)
templates = Jinja2Templates(directory="deployment/templates")

# -----------------------------
# Load synthetic dataset metadata
# -----------------------------
data = np.load("data/synthetic_pems_bay.npz", allow_pickle=True)
X, y = data["x"], data["y"]

num_samples, timesteps, num_nodes, features = X.shape
input_size = num_nodes * features  # usually 325

# -----------------------------
# Load trained ST-GCN model
# -----------------------------
model = STGCN(input_size=input_size, hidden_size=64, output_size=input_size)
model.load_state_dict(torch.load("models/stgcn_weights.pth"))
model.eval()

# -----------------------------
# Homepage
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Data Driven Traffic Prediction API (Synthetic Dataset)",
            "description": "This API predicts future traffic speeds using a synthetic dataset and ST-GCN model.",
            "usage": [
                {"endpoint": "/predict", "method": "POST", "details": "Send traffic sequence data and get predictions"},
                {"endpoint": "/docs", "method": "GET", "details": "Interactive API documentation"},
                {"endpoint": "/redoc", "method": "GET", "details": "Alternative API documentation"}
            ],
            "expected_size": input_size
        }
    )

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(input_data: dict):
    try:
        arr = np.array(input_data["data"], dtype=np.float32)

        # ✅ Validate shape
        if arr.shape[1] != input_size:
            return {
                "error": f"Each timestep must have {input_size} values "
                         f"(traffic speeds for {num_nodes} locations), but got {arr.shape[1]}"
            }

        arr = arr.reshape(1, arr.shape[0], arr.shape[1])  # (1, timesteps, input_size)
        tensor = torch.tensor(arr, dtype=torch.float32)

        with torch.no_grad():
            preds = model(tensor)
            preds = preds[:, -y.shape[1]:, :]

        return {"prediction": preds.squeeze().tolist()}
    except Exception as e:
        return {"error": str(e)}