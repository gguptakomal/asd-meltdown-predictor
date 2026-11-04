"""
generate_data.py
----------------
Generates a realistic synthetic dataset for the
ASD Meltdown Probability Predictor project.

Features:
    age, sleep_hours, stress_level, social_interaction,
    sensory_sensitivity, anxiety_level, routine_change, noise_tolerance

Target:
    meltdown (0 = No, 1 = Yes)

Logic:
    Meltdown probability is driven by a weighted risk score
    computed from clinically-inspired feature relationships.
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
N_SAMPLES   = 400
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")


# ── Feature Generators ────────────────────────────────────────────────────────

def generate_features(n: int) -> pd.DataFrame:
    """
    Generate raw feature columns with realistic distributions.
    Younger children and those with ASD traits tend to cluster
    toward higher sensitivity / lower tolerance ranges.
    """

    age = np.random.randint(3, 16, size=n)                          # 3 – 15 years

    # Younger children sleep more; mild negative correlation with age
    sleep_hours = np.clip(
        np.random.normal(loc=7.5, scale=1.2, size=n) - 0.08 * (age - 3),
        4, 10
    ).round(1)

    stress_level         = np.random.randint(1, 11, size=n)         # 1 – 10
    social_interaction   = np.random.randint(1, 11, size=n)         # 1 – 10
    sensory_sensitivity  = np.random.randint(1, 11, size=n)         # 1 – 10
    anxiety_level        = np.random.randint(1, 11, size=n)         # 1 – 10
    routine_change       = np.random.choice([0, 1], size=n,
                                            p=[0.55, 0.45])         # slight imbalance
    noise_tolerance      = np.random.randint(1, 11, size=n)         # 1 – 10

    return pd.DataFrame({
        "age":                age,
        "sleep_hours":        sleep_hours,
        "stress_level":       stress_level,
        "social_interaction": social_interaction,
        "sensory_sensitivity": sensory_sensitivity,
        "anxiety_level":      anxiety_level,
        "routine_change":     routine_change,
        "noise_tolerance":    noise_tolerance,
    })


# ── Risk Score & Target ───────────────────────────────────────────────────────

def compute_risk_score(df: pd.DataFrame) -> np.ndarray:
    """
    Compute a continuous risk score (0 – 1) using a weighted linear
    combination of features, passed through a sigmoid function.

    Positive contributors  → raise meltdown risk
    Negative contributors  → lower meltdown risk
    """

    # Normalise each feature to [0, 1] range before weighting
    stress       = (df["stress_level"]        - 1) / 9   # high  → bad
    anxiety      = (df["anxiety_level"]       - 1) / 9   # high  → bad
    sensory      = (df["sensory_sensitivity"] - 1) / 9   # high  → bad
    sleep_inv    = 1 - (df["sleep_hours"]     - 4) / 6   # low sleep → bad
    noise_inv    = 1 - (df["noise_tolerance"] - 1) / 9   # low tolerance → bad
    social_inv   = 1 - (df["social_interaction"] - 1) / 9  # low interaction → bad
    routine      = df["routine_change"].astype(float)    # 1 = routine broken → bad
    age_norm     = (df["age"] - 3) / 12                  # younger → slightly higher risk

    # ── Weights (sum ≈ 10 for clean scaling) ──────────────────────────────
    raw_score = (
        2.2 * stress        +   # strongest predictor
        2.0 * anxiety       +
        1.8 * sensory       +
        1.4 * sleep_inv     +
        1.2 * noise_inv     +
        0.8 * routine       +
        0.4 * social_inv    +
        0.2 * (1 - age_norm)    # younger children slightly more vulnerable
    )

    # Normalise to [0, 1]
    raw_score = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min())

    # Sigmoid squeeze: keeps probability realistic (avoids 0/1 extremes)
    def sigmoid(x, k=8, x0=0.5):
        return 1 / (1 + np.exp(-k * (x - x0)))

    probability = sigmoid(raw_score)

    return probability


def assign_target(probability: np.ndarray, noise_std: float = 0.06) -> np.ndarray:
    """
    Convert risk probability to binary label (0/1).
    Small Gaussian noise keeps the boundary realistic (avoids perfect separation).
    """
    noisy_prob = np.clip(probability + np.random.normal(0, noise_std, len(probability)), 0, 1)
    return (noisy_prob >= 0.50).astype(int)


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_dataset(n: int = N_SAMPLES, save_path: str = OUTPUT_PATH) -> pd.DataFrame:
    """
    Full pipeline: generate features → compute risk → assign label → save CSV.

    Parameters
    ----------
    n         : number of samples to generate
    save_path : output CSV path

    Returns
    -------
    pd.DataFrame with all features + target column
    """
    print(f"[INFO] Generating {n} samples …")

    df          = generate_features(n)
    probability = compute_risk_score(df)
    df["meltdown_probability"] = probability.round(4)   # keep for EDA / notebook
    df["meltdown"]             = assign_target(probability)

    # ── Class-balance report ──────────────────────────────────────────────
    counts = df["meltdown"].value_counts()
    print(f"[INFO] Class distribution → No Meltdown (0): {counts.get(0, 0)} | "
          f"Meltdown (1): {counts.get(1, 0)}")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[INFO] Dataset saved → {save_path}")

    return df


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_dataset()
    print("\n── Sample Preview ──────────────────────────────────────────────")
    print(df.head(10).to_string(index=False))
    print(f"\n── Shape: {df.shape} | Columns: {list(df.columns)}")
