"""
Ship Fuel Prediction — FastAPI Backend
Deploy on Railway. All models pre-loaded at startup.
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Suppress version-mismatch warnings from loading pickled models
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="Ship Fuel Prediction API",
    description="Predicts main engine fuel consumption (L/hr) using Linear Regression, Random Forest, and XGBoost.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Load models at startup
# ─────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
lr_model = joblib.load(os.path.join(MODELS_DIR, "linear_regression.pkl"))
rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
FEATURES  = joblib.load(os.path.join(MODELS_DIR, "features.pkl"))
metrics_data = joblib.load(os.path.join(MODELS_DIR, "metrics.pkl"))
fi_data      = joblib.load(os.path.join(MODELS_DIR, "feature_importance.pkl"))
stats_data   = joblib.load(os.path.join(MODELS_DIR, "stats.pkl"))

# Load augmentation comparison data (static JSON from analysis)
aug_comp_path = os.path.join(MODELS_DIR, "augmentation_comparison.json")
if os.path.exists(aug_comp_path):
    with open(aug_comp_path, "r") as f:
        aug_comparison_data = json.load(f)
else:
    aug_comparison_data = None

# ─────────────────────────────────────────────
# Request/Response schemas
# ─────────────────────────────────────────────
class PredictInput(BaseModel):
    rpm: float              = Field(..., description="Main engine shaft RPM", ge=0, le=200)
    power: float            = Field(..., description="Main engine shaft power (KW)", ge=0, le=20000)
    mean_draft: float       = Field(..., description="Mean draft (m)", ge=0, le=25)
    wave_height: float      = Field(..., description="Wave height (m)", ge=0, le=15)
    wind_speed: float       = Field(..., description="Wind speed (m/s)", ge=0, le=60)
    current_speed: float    = Field(..., description="Current speed (m/s)", ge=0, le=10)
    speed_through_water: float = Field(..., description="Speed through water (kn)", ge=0, le=30)
    swell_height: float     = Field(..., description="Swell height (m)", ge=0, le=15)
    trim: float             = Field(..., description="Trim (m, positive = stern deeper)", ge=-5, le=5)


class PredictResponse(BaseModel):
    linear_regression: float
    random_forest: float
    xgboost: float
    ensemble_avg: float
    unit: str = "L/hr"
    inputs: dict

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Ship Fuel Prediction API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "models_loaded": True}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(data: PredictInput):
    """
    Predict main engine fuel consumption (L/hr) given ship operating parameters.
    Returns predictions from Linear Regression, Random Forest, and XGBoost.
    """
    input_values = [
        data.rpm,
        data.power,
        data.mean_draft,
        data.wave_height,
        data.wind_speed,
        data.current_speed,
        data.speed_through_water,
        data.swell_height,
        data.trim,
    ]

    df_input = pd.DataFrame([input_values], columns=FEATURES)
    scaled = scaler.transform(df_input)

    lr_pred  = float(lr_model.predict(scaled)[0])
    rf_pred  = float(rf_model.predict(scaled)[0])
    xgb_pred = float(xgb_model.predict(scaled)[0])
    avg_pred = round((lr_pred + rf_pred + xgb_pred) / 3, 2)

    return PredictResponse(
        linear_regression=round(lr_pred, 2),
        random_forest=round(rf_pred, 2),
        xgboost=round(xgb_pred, 2),
        ensemble_avg=avg_pred,
        inputs=data.model_dump(),
    )


@app.get("/metrics", tags=["Model Info"])
def get_metrics():
    """
    Returns training/test metrics (R², MAE, RMSE, MAPE) for all 3 models.
    """
    return metrics_data


@app.get("/feature-importance", tags=["Model Info"])
def get_feature_importance():
    """
    Returns feature importance scores from Linear Regression (abs coef),
    Random Forest, and XGBoost.
    """
    result = []
    for i, feat in enumerate(fi_data["features"]):
        result.append({
            "feature": feat,
            "linear_regression": round(fi_data["linear_regression"][i], 4),
            "random_forest": round(fi_data["random_forest"][i], 4),
            "xgboost": round(fi_data["xgboost"][i], 4),
        })
    # Sort by XGBoost importance descending
    result.sort(key=lambda x: x["xgboost"], reverse=True)
    return result


@app.get("/stats", tags=["Dataset"])
def get_stats():
    """
    Returns dataset statistics: row counts, real vs synthetic split, target distribution.
    """
    return {
        "total_rows": stats_data["total_rows"],
        "real_rows": stats_data["real_rows"],
        "synthetic_rows": stats_data["synthetic_rows"],
        "target": {
            "mean": round(stats_data["target_mean"], 2),
            "std": round(stats_data["target_std"], 2),
            "min": round(stats_data["target_min"], 2),
            "max": round(stats_data["target_max"], 2),
        },
        "features": [
            {
                "name": feat,
                "mean": round(stats_data["feature_stats"][feat]["mean"], 3),
                "std": round(stats_data["feature_stats"][feat]["std"], 3),
                "min": round(stats_data["feature_stats"][feat]["min"], 3),
                "max": round(stats_data["feature_stats"][feat]["max"], 3),
            }
            for feat in FEATURES
        ],
    }


@app.get("/augmentation-comparison", tags=["Dataset"])
def get_augmentation_comparison():
    """
    Returns the effect of synthetic data augmentation on model accuracy.
    Compares models trained on real-only vs real+synthetic data,
    tested on a held-out real-only test set.
    """
    if aug_comparison_data is None:
        raise HTTPException(status_code=404, detail="Augmentation comparison data not available")
    return aug_comparison_data

