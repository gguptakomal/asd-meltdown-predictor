"""
project/predictor.py
--------------------
Handles model loading and inference logic for the Flask application.
"""

import os
import pickle
import numpy as np

# Resolve model path relative to this file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

def load_model_bundle():
    """Load model + scaler + features from the pickle bundle."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# Global model instance, loaded once on app/module import
model_bundle = load_model_bundle()

def predict(inputs: dict) -> tuple[float, int]:
    """
    Scale inputs and run logistic regression prediction.
    
    Parameters
    ----------
    inputs : dict
        Keys matching model features (age, sleep_hours, stress_level, etc.)
        
    Returns
    -------
    tuple (probability_percentage, predicted_class_label)
    """
    if model_bundle is None:
        raise FileNotFoundError(
            f"❌ Predictive model not found at '{MODEL_PATH}'. "
            "Please make sure model.pkl has been relocated successfully."
        )

    model    = model_bundle["model"]
    scaler   = model_bundle["scaler"]
    features = model_bundle["features"]

    # Order inputs matching feature list
    X_raw  = np.array([[inputs[f] for f in features]])
    X_sc   = scaler.transform(X_raw)

    prob   = model.predict_proba(X_sc)[0][1]   # P(meltdown = 1)
    label  = model.predict(X_sc)[0]

    return round(prob * 100, 2), int(label)
