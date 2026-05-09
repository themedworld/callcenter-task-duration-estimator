from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# =========================
# Charger le modèle
# =========================
MODEL_PATH = "./best_estimatedHours_task_callcenter_lgbm.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Estimated Hours API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ en prod tu peux mettre ton domaine frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Schéma input (Pydantic)
# =========================
class TaskInput(BaseModel):
    type: str
    priority: str
    status: str
    complexityScore: int
    riskLevel: int
    targetAgentCount: int
    expectedCallsPerAgent: int
    targetConversionRate: float
    qualityScoreTarget: int
    dependenciesCount: int
    reopenCount: int
    delayHours: float


# =========================
# Health check
# =========================
@app.get("/")
def home():
    return {"message": "API is running 🚀"}

# =========================
# Prediction endpoint
# =========================
@app.post("/predict")
def predict(data: TaskInput):
    # Convertir input en DataFrame
    df = pd.DataFrame([data.dict()])

    # Prédiction (pipeline = preprocess + model)
    prediction = model.predict(df)

    return {
        "estimatedHours": float(prediction[0])
    }