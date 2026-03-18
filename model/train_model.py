"""
train_model.py
--------------
Trains a Logistic Regression model for the
ASD Meltdown Probability Predictor project.
 
Steps:
    1. Load dataset from data/dataset.csv
    2. Split into features (X) and target (y)
    3. Train / Test split
    4. Scale features with StandardScaler
    5. Train Logistic Regression
    6. Evaluate and print metrics
    7. Save model + scaler to model/model.pkl
 
Run from project root:
    python model/train_model.py
"""
 
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
 
# ── Paths (always relative to project root) ───────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT_DIR, "data",  "dataset.csv")
MODEL_PATH  = os.path.join(ROOT_DIR, "model", "model.pkl")
 
# ── Feature columns & target ──────────────────────────────────────────────────
FEATURES = [
    "age",
    "sleep_hours",
    "stress_level",
    "social_interaction",
    "sensory_sensitivity",
    "anxiety_level",
    "routine_change",
    "noise_tolerance",
]
TARGET = "meltdown"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset and validate required columns."""
    print(f"[INFO] Loading dataset from → {path}")
 
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] Dataset not found at: {path}\n"
            f"        Run  data/generate_data.py  first."
        )
 
    df = pd.read_csv(path)
 
    required = FEATURES + [TARGET]
    missing  = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing columns in dataset: {missing}")
 
    print(f"[INFO] Dataset loaded  → {df.shape[0]} rows, {df.shape[1]} columns")
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. Prepare X and y
# ─────────────────────────────────────────────────────────────────────────────
def prepare_xy(df: pd.DataFrame):
    """Split DataFrame into feature matrix X and target vector y."""
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
 
    print(f"[INFO] Features : {FEATURES}")
    print(f"[INFO] Target   : '{TARGET}'  |  Classes → {sorted(y.unique())}")
    print(f"[INFO] Class distribution:\n{y.value_counts().to_string()}\n")
 
    return X, y
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. Train / Test Split
# ─────────────────────────────────────────────────────────────────────────────
def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, seed: int = 42):
    """Stratified 80/20 train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,          # preserves class ratio in both splits
    )
    print(f"[INFO] Train samples : {len(X_train)}  |  Test samples : {len(X_test)}")
    return X_train, X_test, y_train, y_test
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. Scale Features
# ─────────────────────────────────────────────────────────────────────────────
def scale_features(X_train, X_test):
    """
    Fit StandardScaler on training data only.
    Transform both train and test to prevent data leakage.
    """
    scaler   = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
 
    print("[INFO] Features scaled with StandardScaler (fit on train only)")
    return X_train_scaled, X_test_scaled, scaler
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. Train Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────
def train_model(X_train_scaled, y_train) -> LogisticRegression:
    """Train Logistic Regression with L2 regularisation."""
    model = LogisticRegression(
        C=1.0,              # inverse regularisation strength
        max_iter=1000,      # enough iterations to converge
        solver="lbfgs",     # efficient for small datasets
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    print("[INFO] Logistic Regression model trained successfully")
    return model
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluate Model
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model: LogisticRegression, X_test_scaled, y_test) -> float:
    """Print accuracy, classification report, and confusion matrix."""
    y_pred    = model.predict(X_test_scaled)
    accuracy  = accuracy_score(y_test, y_pred)
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(y_test, y_pred,
                                      target_names=["No Meltdown", "Meltdown"])
 
    print("\n" + "─" * 50)
    print(f"  ACCURACY SCORE   : {accuracy * 100:.2f}%")
    print("─" * 50)
    print("\n  CLASSIFICATION REPORT:")
    print(report)
    print("  CONFUSION MATRIX:")
    print(f"  {cm}\n")
    print("─" * 50)
 
    return accuracy
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 7. Save Model + Scaler
# ─────────────────────────────────────────────────────────────────────────────
def save_model(model: LogisticRegression, scaler: StandardScaler, path: str) -> None:
    """
    Persist model and scaler together as a single pickle file.
    Bundling both ensures the scaler used at training is always
    paired with the model at inference — no mismatch possible.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
 
    bundle = {
        "model":    model,
        "scaler":   scaler,
        "features": FEATURES,
    }
 
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
 
    print(f"[INFO] Model + Scaler saved → {path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "═" * 50)
    print("  ASD Meltdown Predictor — Model Training")
    print("═" * 50 + "\n")
 
    # Step 1 — Load
    df = load_data(DATA_PATH)
 
    # Step 2 — Prepare
    X, y = prepare_xy(df)
 
    # Step 3 — Split
    X_train, X_test, y_train, y_test = split_data(X, y)
 
    # Step 4 — Scale
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
 
    # Step 5 — Train
    model = train_model(X_train_sc, y_train)
 
    # Step 6 — Evaluate
    evaluate_model(model, X_test_sc, y_test)
 
    # Step 7 — Save
    save_model(model, scaler, MODEL_PATH)
 
    print("\n[✓] Training pipeline complete.\n")
 
 
if __name__ == "__main__":
    main()
 