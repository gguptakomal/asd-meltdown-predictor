"""
app.py
------
Streamlit dashboard for the
ASD Meltdown Probability Predictor.

Run from project root:
    streamlit run app/app.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ── Resolve project root so imports work from any CWD ─────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.suggestions   import get_suggestions
from utils.visualization import plot_probability, plot_features, plot_distribution

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT, "model", "model.pkl")
DATA_PATH  = os.path.join(ROOT, "data",  "dataset.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "ASD Meltdown Predictor",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    .main { background-color: #0f1117; }

    /* ── Header banner ── */
    .header-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 14px;
        padding: 28px 36px;
        margin-bottom: 24px;
        border: 1px solid #1e3a5f;
    }
    .header-banner h1 {
        color: #e0e6ff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .header-banner p {
        color: #7a9cc4;
        font-size: 0.95rem;
        margin: 0;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #2a2a4a;
        text-align: center;
    }
    .metric-card .label {
        color: #7a9cc4;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
    }
    .metric-card .sub {
        color: #7a9cc4;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* ── Risk badge ── */
    .risk-badge {
        display: inline-block;
        border-radius: 20px;
        padding: 6px 18px;
        font-size: 1rem;
        font-weight: 700;
        margin-top: 8px;
    }
    .risk-low    { background: #0d3320; color: #2ecc71; border: 1px solid #2ecc71; }
    .risk-medium { background: #3b2800; color: #f39c12; border: 1px solid #f39c12; }
    .risk-high   { background: #3b0a0a; color: #e74c3c; border: 1px solid #e74c3c; }

    /* ── Section title ── */
    .section-title {
        color: #c8d8f0;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        border-left: 3px solid #3d7dd6;
        padding-left: 10px;
        margin: 24px 0 14px 0;
    }

    /* ── Suggestion card ── */
    .suggestion-card {
        background: #151525;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 4px solid #3d7dd6;
    }
    .suggestion-card .topic {
        color: #7ab4f5;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .suggestion-card .text {
        color: #ccd6e8;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #10101e;
        border-right: 1px solid #1e1e3a;
    }
    [data-testid="stSidebar"] label {
        color: #a0b4cc !important;
        font-size: 0.87rem !important;
    }

    /* ── Divider ── */
    .custom-divider {
        border: none;
        border-top: 1px solid #1e2a3a;
        margin: 20px 0;
    }

    /* ── Info box ── */
    .info-box {
        background: #0d1b2a;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 14px 18px;
        color: #7a9cc4;
        font-size: 0.88rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load Model Bundle
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_bundle():
    """Load model + scaler + features from pickle bundle."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_dataset():
    """Load dataset CSV for distribution charts."""
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Inputs
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("### 🧒 Child Profile")
        st.markdown("<hr style='border-color:#1e1e3a;margin:8px 0 16px'>",
                    unsafe_allow_html=True)

        age = st.slider("Age (years)", min_value=3, max_value=15,
                        value=8, step=1)

        st.markdown("---")
        st.markdown("### 😴 Sleep & Stress")

        sleep_hours = st.slider("Sleep Hours (last night)",
                                min_value=4.0, max_value=10.0,
                                value=7.0, step=0.5)

        stress_level = st.slider("Stress Level", min_value=1, max_value=10,
                                 value=5, step=1,
                                 help="1 = Very calm  |  10 = Extremely stressed")

        st.markdown("---")
        st.markdown("### 🧩 Sensory & Social")

        sensory_sensitivity = st.slider("Sensory Sensitivity",
                                        min_value=1, max_value=10,
                                        value=5, step=1,
                                        help="1 = Low sensitivity  |  10 = Very high")

        noise_tolerance = st.slider("Noise Tolerance",
                                    min_value=1, max_value=10,
                                    value=5, step=1,
                                    help="1 = Cannot tolerate noise  |  10 = Very tolerant")

        social_interaction = st.slider("Social Interaction",
                                       min_value=1, max_value=10,
                                       value=5, step=1,
                                       help="1 = Very withdrawn  |  10 = Very engaged")

        st.markdown("---")
        st.markdown("### 😰 Anxiety & Routine")

        anxiety_level = st.slider("Anxiety Level",
                                  min_value=1, max_value=10,
                                  value=5, step=1,
                                  help="1 = Very calm  |  10 = Severe anxiety")

        routine_change = st.selectbox(
            "Routine Change Today?",
            options=[0, 1],
            format_func=lambda x: "✅ No — Routine was stable" if x == 0
                                   else "⚠️ Yes — Routine was disrupted",
        )

        st.markdown("---")
        predict_btn = st.button("🔍 Predict Meltdown Risk",
                                use_container_width=True,
                                type="primary")

    return {
        "inputs": {
            "age":                age,
            "sleep_hours":        sleep_hours,
            "stress_level":       stress_level,
            "social_interaction": social_interaction,
            "sensory_sensitivity":sensory_sensitivity,
            "anxiety_level":      anxiety_level,
            "routine_change":     routine_change,
            "noise_tolerance":    noise_tolerance,
        },
        "predict": predict_btn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────────────
def predict(bundle: dict, inputs: dict) -> tuple[float, int]:
    """
    Scale inputs and run logistic regression prediction.
    Returns (probability_percent, predicted_class).
    """
    model    = bundle["model"]
    scaler   = bundle["scaler"]
    features = bundle["features"]

    X_raw  = np.array([[inputs[f] for f in features]])
    X_sc   = scaler.transform(X_raw)

    prob   = model.predict_proba(X_sc)[0][1]   # P(meltdown = 1)
    label  = model.predict(X_sc)[0]

    return round(prob * 100, 2), int(label)


# ─────────────────────────────────────────────────────────────────────────────
# Render Results
# ─────────────────────────────────────────────────────────────────────────────
def render_results(probability: float, predicted: int, inputs: dict):

    result = get_suggestions(probability)
    level  = result["risk_level"]

    # ── Risk colour ───────────────────────────────────────────────────────
    if probability < 40:
        prob_color  = "#2ecc71"
        badge_class = "risk-low"
    elif probability <= 70:
        prob_color  = "#f39c12"
        badge_class = "risk-medium"
    else:
        prob_color  = "#e74c3c"
        badge_class = "risk-high"

    # ── Metric cards ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Meltdown Probability</div>
            <div class="value" style="color:{prob_color}">{probability:.1f}%</div>
            <div class="sub">Logistic Regression output</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Risk Level</div>
            <div style="margin-top:8px">
                <span class="risk-badge {badge_class}">{level}</span>
            </div>
            <div class="sub">Rule-based classification</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        pred_text  = "Meltdown Likely" if predicted == 1 else "No Meltdown"
        pred_color = "#e74c3c" if predicted == 1 else "#2ecc71"
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Model Prediction</div>
            <div class="value" style="color:{pred_color};font-size:1.4rem">
                {pred_text}
            </div>
            <div class="sub">Binary class output</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Overall summary ───────────────────────────────────────────────────
    st.markdown(f"""
    <div class="info-box">
        📋 <strong>Summary:</strong> {result['overall']}
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📊 Visual Analysis</div>",
                unsafe_allow_html=True)

    gcol1, gcol2 = st.columns([1, 1])

    with gcol1:
        fig1 = plot_probability(probability)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with gcol2:
        fig2 = plot_features(inputs)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Suggestions ───────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>💡 Actionable Suggestions</div>",
                unsafe_allow_html=True)

    topic_icons = {
        "Sleep":            "😴",
        "Stress":           "😤",
        "Sensory Overload": "👂",
        "Routine Change":   "📅",
        "Anxiety":          "😰",
    }

    col_a, col_b = st.columns(2)
    items = list(result["suggestions"].items())

    for i, (topic, text) in enumerate(items):
        icon = topic_icons.get(topic, "•")
        card = f"""
        <div class="suggestion-card">
            <div class="topic">{icon} {topic}</div>
            <div class="text">{text}</div>
        </div>"""
        if i < 3:
            col_a.markdown(card, unsafe_allow_html=True)
        else:
            col_b.markdown(card, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Tab
# ─────────────────────────────────────────────────────────────────────────────
def render_dataset_tab():
    df = load_dataset()
    if df is None:
        st.warning("⚠️ dataset.csv not found. Run `data/generate_data.py` first.")
        return

    st.markdown("<div class='section-title'>📂 Dataset Overview</div>",
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows",      len(df))
    m2.metric("Total Features",  8)
    m3.metric("Meltdown Cases",  int(df["meltdown"].sum()))
    m4.metric("Stable Cases",    int((df["meltdown"] == 0).sum()))

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    fig3 = plot_distribution(df)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🗃️ Raw Data Sample</div>",
                unsafe_allow_html=True)
    st.dataframe(
        df.sample(10, random_state=1).reset_index(drop=True),
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="header-banner">
        <h1>🧠 ASD Meltdown Probability Predictor</h1>
        <p>
            Enter a child's profile in the sidebar and click <strong>Predict</strong>
            to estimate meltdown risk using Logistic Regression.
            Includes probability score, risk level, feature analysis,
            and actionable suggestions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ────────────────────────────────────────────────────────
    bundle = load_model_bundle()
    if bundle is None:
        st.error(
            "❌ Model not found at `model/model.pkl`.\n\n"
            "Run the following commands first:\n"
            "```\npython data/generate_data.py\n"
            "python model/train_model.py\n```"
        )
        st.stop()

    # ── Sidebar inputs ────────────────────────────────────────────────────
    sidebar = render_sidebar()
    inputs  = sidebar["inputs"]

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Dataset Explorer"])

    with tab1:
        if sidebar["predict"]:
            with st.spinner("Running prediction …"):
                probability, predicted = predict(bundle, inputs)
            render_results(probability, predicted, inputs)

        else:
            st.markdown("""
            <div class="info-box" style="text-align:center;padding:40px 24px">
                👈 &nbsp; Fill in the child's profile in the sidebar
                and press <strong>Predict Meltdown Risk</strong> to get results.
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        render_dataset_tab()


if __name__ == "__main__":
    main()