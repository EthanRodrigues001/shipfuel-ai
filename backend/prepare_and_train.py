"""
Dataset preparation + model training for production.

Steps:
1. Load V3 (has Source: Real/Synthetic) + compute Trim
2. Select the 9 features used in latest/ + target + Source label
3. Save labeled production dataset to data/
4. Train Linear Regression, Random Forest, XGBoost
5. Save models, scaler, metrics, feature importances to models/
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

BASE = os.path.dirname(__file__)
DATASET_VERSIONS = os.path.join(BASE, "..", "..", "dataset versions")
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 — Load production dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 1 — Loading production dataset")
print("=" * 60)

# The 9 features exactly as in latest/ model
FEATURES = [
    "Main engine shaft RPM",
    "Main engine shaft power (KW)",
    "Mean draft (m)",
    "Wave height (m)",
    "Wind speed (m/s)",
    "Current speed (m/s)",
    "Speed through water (kn)",
    "Swell height (m)",
    "Trim (m)",
]
TARGET = "Main engine consumption (L/hr)"

v3_path = os.path.join(DATASET_VERSIONS, "v3", "synthetic_dataset.csv")
prod_csv = os.path.join(DATA_DIR, "ship_data_production.csv")

if os.path.exists(v3_path):
    # Build from V3 source (has Source: Real/Synthetic)
    v3 = pd.read_csv(v3_path, low_memory=False)
    print(f"  V3 loaded: {v3.shape[0]} rows, Source counts: {v3['Source'].value_counts().to_dict()}")
    v3["Trim (m)"] = v3["Draft Aft (m)"] - v3["Draft Fwd (m)"]
    df = v3[FEATURES + [TARGET, "Source"]].copy()
    df = df.dropna()
    df = df.reset_index(drop=True)
    df["is_synthetic"] = (df["Source"] == "Synthetic").astype(int)
    df = df.drop(columns=["Source"])
    df.to_csv(prod_csv, index=False)
    print(f"  Saved: {prod_csv}")
elif os.path.exists(prod_csv):
    # Use existing production CSV
    print(f"  V3 source not found, loading existing production CSV")
    df = pd.read_csv(prod_csv)
    df = df.dropna()
    df = df.reset_index(drop=True)
else:
    raise FileNotFoundError(
        f"Neither V3 source ({v3_path}) nor production CSV ({prod_csv}) found. "
        "Cannot train models without data."
    )

print(f"  After selecting features + dropping NaN: {df.shape[0]} rows")
print(f"  Real rows    : {(df['is_synthetic'] == 0).sum()}")
print(f"  Synthetic rows: {(df['is_synthetic'] == 1).sum()}")

# ─────────────────────────────────────────────
# STEP 2 — Compute dataset statistics
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 2 — Computing dataset statistics")
print("=" * 60)

stats = {
    "total_rows": int(len(df)),
    "real_rows": int((df["is_synthetic"] == 0).sum()),
    "synthetic_rows": int((df["is_synthetic"] == 1).sum()),
    "target_mean": float(df[TARGET].mean()),
    "target_std": float(df[TARGET].std()),
    "target_min": float(df[TARGET].min()),
    "target_max": float(df[TARGET].max()),
    "feature_stats": {},
}

for feat in FEATURES:
    stats["feature_stats"][feat] = {
        "mean": float(df[feat].mean()),
        "std": float(df[feat].std()),
        "min": float(df[feat].min()),
        "max": float(df[feat].max()),
    }

print(f"  Total: {stats['total_rows']} | Real: {stats['real_rows']} | Synthetic: {stats['synthetic_rows']}")
print(f"  Target: mean={stats['target_mean']:.1f} std={stats['target_std']:.1f} min={stats['target_min']:.1f} max={stats['target_max']:.1f}")

# ─────────────────────────────────────────────
# STEP 3 — Train / Test Split + Scaling
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3 — Splitting + Scaling")
print("=" * 60)

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(FEATURES, os.path.join(MODELS_DIR, "features.pkl"))
print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "mape": float(mape)}

# ─────────────────────────────────────────────
# MODEL 1 — Linear Regression (with PolynomialFeatures to show overfit)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MODEL 1 — Linear Regression (Polynomial)")
print("=" * 60)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Using degree=3 to intentionally cause overfitting for demonstration
lr = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('linear', LinearRegression())
])
lr.fit(X_train_s, y_train)
joblib.dump(lr, os.path.join(MODELS_DIR, "linear_regression.pkl"))

lr_train_m = compute_metrics(y_train, lr.predict(X_train_s))
lr_test_m  = compute_metrics(y_test,  lr.predict(X_test_s))
# For CV, we might get negative R^2 due to overfitting
lr_cv = cross_val_score(lr, X_train_s, y_train, cv=5, scoring="r2")

print(f"  Train R²={lr_train_m['r2']:.4f}  Test R²={lr_test_m['r2']:.4f}  CV={lr_cv.mean():.4f}±{lr_cv.std():.4f}")
print(f"  MAE={lr_test_m['mae']:.2f}  RMSE={lr_test_m['rmse']:.2f}  MAPE={lr_test_m['mape']:.2f}%")

# ─────────────────────────────────────────────
# MODEL 2 — Random Forest
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MODEL 2 — Random Forest")
print("=" * 60)

rf = RandomForestRegressor(
    n_estimators=200, max_depth=10,
    min_samples_split=10, min_samples_leaf=5,
    random_state=42, n_jobs=-1
)
rf.fit(X_train_s, y_train)
joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))

rf_train_m = compute_metrics(y_train, rf.predict(X_train_s))
rf_test_m  = compute_metrics(y_test,  rf.predict(X_test_s))
rf_cv = cross_val_score(rf, X_train_s, y_train, cv=5, scoring="r2")

print(f"  Train R²={rf_train_m['r2']:.4f}  Test R²={rf_test_m['r2']:.4f}  CV={rf_cv.mean():.4f}±{rf_cv.std():.4f}")
print(f"  MAE={rf_test_m['mae']:.2f}  RMSE={rf_test_m['rmse']:.2f}  MAPE={rf_test_m['mape']:.2f}%")

# ─────────────────────────────────────────────
# MODEL 3 — XGBoost
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MODEL 3 — XGBoost")
print("=" * 60)

xgb_model = XGBRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=0
)
xgb_model.fit(X_train_s, y_train)
joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost.pkl"))

xgb_train_m = compute_metrics(y_train, xgb_model.predict(X_train_s))
xgb_test_m  = compute_metrics(y_test,  xgb_model.predict(X_test_s))
xgb_cv = cross_val_score(xgb_model, X_train_s, y_train, cv=5, scoring="r2")

print(f"  Train R²={xgb_train_m['r2']:.4f}  Test R²={xgb_test_m['r2']:.4f}  CV={xgb_cv.mean():.4f}±{xgb_cv.std():.4f}")
print(f"  MAE={xgb_test_m['mae']:.2f}  RMSE={xgb_test_m['rmse']:.2f}  MAPE={xgb_test_m['mape']:.2f}%")

# ─────────────────────────────────────────────
# STEP 4 — Save metrics + feature importance
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4 — Saving metrics + feature importance")
print("=" * 60)

metrics_data = {
    "linear_regression": {"train": lr_train_m,  "test": lr_test_m,  "cv_r2_mean": float(lr_cv.mean()),  "cv_r2_std": float(lr_cv.std())},
    "random_forest":     {"train": rf_train_m,  "test": rf_test_m,  "cv_r2_mean": float(rf_cv.mean()),  "cv_r2_std": float(rf_cv.std())},
    "xgboost":           {"train": xgb_train_m, "test": xgb_test_m, "cv_r2_mean": float(xgb_cv.mean()), "cv_r2_std": float(xgb_cv.std())},
}

# Add overfit analysis to each model
for model_name, m in metrics_data.items():
    train_r2 = m["train"]["r2"]
    test_r2  = m["test"]["r2"]
    r2_gap   = train_r2 - test_r2
    cv_mean  = m["cv_r2_mean"]
    cv_std   = m["cv_r2_std"]
    # Flag as overfit if train-test gap > 0.02 (2%) or CV std > 0.05
    is_overfit = (r2_gap > 0.02) or (cv_std > 0.05)
    m["overfit_analysis"] = {
        "train_test_r2_gap": round(r2_gap, 6),
        "is_overfit": is_overfit,
        "explanation": (
            f"Train R²={train_r2:.4f} vs Test R²={test_r2:.4f} (gap={r2_gap:.4f}). "
            f"CV R²={cv_mean:.4f}±{cv_std:.4f}. "
            + ("⚠️ Potential overfitting detected." if is_overfit else "✅ No significant overfitting.")
        ),
    }

# Feature importance (LR uses abs(coef), RF/XGB use feature_importances_)
# For PolynomialFeatures, the first `len(FEATURES)` coefficients correspond to the original linear features
lr_coef = np.abs(lr.named_steps['linear'].coef_[:len(FEATURES)])
lr_coef = lr_coef / lr_coef.sum()  # normalize

fi_data = {
    "features": FEATURES,
    "linear_regression": lr_coef.tolist(),
    "random_forest": rf.feature_importances_.tolist(),
    "xgboost": xgb_model.feature_importances_.tolist(),
}

joblib.dump(metrics_data, os.path.join(MODELS_DIR, "metrics.pkl"))
joblib.dump(fi_data,      os.path.join(MODELS_DIR, "feature_importance.pkl"))
joblib.dump(stats,        os.path.join(MODELS_DIR, "stats.pkl"))

# Also save as JSON for quick inspection
with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_data, f, indent=2)
with open(os.path.join(MODELS_DIR, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

print("  Saved: scaler.pkl, features.pkl, linear_regression.pkl, random_forest.pkl, xgboost.pkl")
print("  Saved: metrics.pkl, feature_importance.pkl, stats.pkl")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL MODEL COMPARISON")
print("=" * 60)
print(f"\n  {'Metric':<10} {'Linear Reg':>12} {'Rand Forest':>13} {'XGBoost':>10}")
print(f"  {'-'*50}")
print(f"  {'Train R²':<10} {lr_train_m['r2']:>12.4f} {rf_train_m['r2']:>13.4f} {xgb_train_m['r2']:>10.4f}")
print(f"  {'Test R²':<10} {lr_test_m['r2']:>12.4f} {rf_test_m['r2']:>13.4f} {xgb_test_m['r2']:>10.4f}")
print(f"  {'MAE':<10} {lr_test_m['mae']:>12.2f} {rf_test_m['mae']:>13.2f} {xgb_test_m['mae']:>10.2f}")
print(f"  {'RMSE':<10} {lr_test_m['rmse']:>12.2f} {rf_test_m['rmse']:>13.2f} {xgb_test_m['rmse']:>10.2f}")
print(f"  {'MAPE':<10} {lr_test_m['mape']:>11.2f}% {rf_test_m['mape']:>12.2f}% {xgb_test_m['mape']:>9.2f}%")
print("\n  TRAINING COMPLETE")
print("=" * 60)
