"""
visualization.py
----------------
All chart functions for the
ASD Meltdown Probability Predictor project.
 
Functions:
    plot_probability(probability)   → gauge-style probability bar chart
    plot_features(values_dict)      → horizontal bar chart of input features
    plot_distribution(df)           → dataset distribution overview (3-panel)
 
Rules:
    - Every function returns a matplotlib Figure object
    - plt.show() is NEVER called (Streamlit handles rendering)
    - All figures are self-contained and closed after creation
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
 
# ── Global Style ──────────────────────────────────────────────────────────────
 
sns.set_theme(style="darkgrid", palette="muted")
 
PALETTE = {
    "low":      "#2ecc71",   # green
    "medium":   "#f39c12",   # amber
    "high":     "#e74c3c",   # red
    "neutral":  "#3498db",   # blue
    "bg":       "#1e1e2e",   # dark background
    "text":     "#e0e0e0",   # light text
    "bar_bg":   "#2c2c3e",   # bar background
}
 
FEATURES_LABELS = {
    "age":                "Age",
    "sleep_hours":        "Sleep Hours",
    "stress_level":       "Stress Level",
    "social_interaction": "Social Interaction",
    "sensory_sensitivity":"Sensory Sensitivity",
    "anxiety_level":      "Anxiety Level",
    "routine_change":     "Routine Change",
    "noise_tolerance":    "Noise Tolerance",
}
 
 
def _risk_color(probability: float) -> str:
    """Return color based on probability threshold."""
    if probability < 40:
        return PALETTE["low"]
    elif probability <= 70:
        return PALETTE["medium"]
    else:
        return PALETTE["high"]
 
 
def _risk_label(probability: float) -> str:
    if probability < 40:
        return "Low Risk"
    elif probability <= 70:
        return "Medium Risk"
    else:
        return "High Risk"
 
 
def _dark_fig(figsize: tuple) -> tuple:
    """Create a dark-themed figure and axes."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bar_bg"])
    return fig, ax
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. plot_probability
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_probability(probability: float) -> plt.Figure:
    """
    Horizontal gauge-style bar showing meltdown probability %.
 
    Parameters
    ----------
    probability : float  (0 – 100)
 
    Returns
    -------
    matplotlib Figure
    """
    fig, ax = _dark_fig(figsize=(7, 2.2))
 
    color = _risk_color(probability)
    label = _risk_label(probability)
 
    # Background track
    ax.barh(0, 100, color=PALETTE["bar_bg"], height=0.5,
            edgecolor="#444", linewidth=1.2)
 
    # Filled portion
    ax.barh(0, probability, color=color, height=0.5,
            edgecolor=color, linewidth=1.2)
 
    # Zone separators
    for xv, lbl in [(40, "40%"), (70, "70%")]:
        ax.axvline(x=xv, color="#888", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(xv + 0.8, 0.33, lbl, color="#aaa",
                fontsize=8, va="center")
 
    # Probability label inside bar
    ax.text(
        min(probability - 2, 95), 0,
        f"{probability:.1f}%",
        va="center", ha="right",
        color="white", fontsize=14, fontweight="bold",
    )
 
    # Risk label on right
    ax.text(
        101, 0, label,
        va="center", ha="left",
        color=color, fontsize=11, fontweight="bold",
    )
 
    ax.set_xlim(0, 130)
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.set_xlabel("Meltdown Probability (%)",
                  color=PALETTE["text"], fontsize=10)
    ax.set_title("Meltdown Risk Gauge",
                 color=PALETTE["text"], fontsize=12, fontweight="bold", pad=10)
 
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
 
    fig.tight_layout()
    return fig
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. plot_features
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_features(values_dict: dict) -> plt.Figure:
    """
    Horizontal bar chart showing the user's input feature values
    with colour-coded risk contribution.
 
    Parameters
    ----------
    values_dict : dict
        Keys = feature names, Values = numeric input values
 
    Returns
    -------
    matplotlib Figure
    """
    # ── Define max scale per feature ──────────────────────────────────────
    MAX_VALUES = {
        "age":                 15,
        "sleep_hours":         10,
        "stress_level":        10,
        "social_interaction":  10,
        "sensory_sensitivity": 10,
        "anxiety_level":       10,
        "routine_change":       1,
        "noise_tolerance":     10,
    }
 
    # ── Features where HIGH value = HIGH risk ─────────────────────────────
    HIGH_IS_BAD = {
        "stress_level", "sensory_sensitivity",
        "anxiety_level", "routine_change",
    }
    # Features where LOW value = HIGH risk
    LOW_IS_BAD = {
        "sleep_hours", "noise_tolerance", "social_interaction",
    }
 
    labels, values, colors, risk_scores = [], [], [], []
 
    for key, val in values_dict.items():
        label = FEATURES_LABELS.get(key, key)
        max_v = MAX_VALUES.get(key, 10)
        norm  = val / max_v                       # 0–1 normalised
 
        if key in HIGH_IS_BAD:
            risk = norm                           # higher value → more risk
        elif key in LOW_IS_BAD:
            risk = 1 - norm                       # lower value → more risk
        else:
            risk = 0.3                            # age → neutral colour
 
        # Colour gradient: green → amber → red
        if risk < 0.4:
            c = PALETTE["low"]
        elif risk <= 0.70:
            c = PALETTE["medium"]
        else:
            c = PALETTE["high"]
 
        labels.append(label)
        values.append(val)
        colors.append(c)
        risk_scores.append(risk)
 
    fig, ax = _dark_fig(figsize=(7, 5))
 
    y_pos = np.arange(len(labels))
    bars  = ax.barh(y_pos, values, color=colors,
                    edgecolor="#333", linewidth=0.8, height=0.6)
 
    # Value annotations
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center", ha="left",
            color=PALETTE["text"], fontsize=9,
        )
 
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color=PALETTE["text"], fontsize=10)
    ax.set_xlabel("Feature Value", color=PALETTE["text"], fontsize=10)
    ax.set_title("Input Feature Profile",
                 color=PALETTE["text"], fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="x", colors=PALETTE["text"], labelsize=9)
    ax.set_xlim(0, 14)
 
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
 
    # Legend
    legend_patches = [
        mpatches.Patch(color=PALETTE["low"],    label="Low Contribution"),
        mpatches.Patch(color=PALETTE["medium"], label="Medium Contribution"),
        mpatches.Patch(color=PALETTE["high"],   label="High Contribution"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=8,
        facecolor=PALETTE["bg"],
        edgecolor="#555",
        labelcolor=PALETTE["text"],
    )
 
    fig.tight_layout()
    return fig
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. plot_distribution
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    3-panel dataset distribution overview:
      Panel 1 → Meltdown class count (bar)
      Panel 2 → Stress level vs Anxiety level (scatter, coloured by meltdown)
      Panel 3 → Feature correlation heatmap
 
    Parameters
    ----------
    df : pd.DataFrame
        Must contain feature columns + 'meltdown' target column.
 
    Returns
    -------
    matplotlib Figure
    """
    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor(PALETTE["bg"])
 
    # ── Grid layout: 2 rows × 2 cols, last cell spans ────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)
 
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(PALETTE["bar_bg"])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.tick_params(colors=PALETTE["text"])
 
    # ── Panel 1: Class Distribution ───────────────────────────────────────
    counts  = df["meltdown"].value_counts().sort_index()
    labels  = ["No Meltdown (0)", "Meltdown (1)"]
    bar_colors = [PALETTE["low"], PALETTE["high"]]
 
    bars = ax1.bar(labels, counts.values, color=bar_colors,
                   edgecolor="#333", linewidth=0.8, width=0.5)
    for bar in bars:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            str(int(bar.get_height())),
            ha="center", va="bottom",
            color=PALETTE["text"], fontsize=11, fontweight="bold",
        )
    ax1.set_title("Class Distribution",
                  color=PALETTE["text"], fontsize=11, fontweight="bold")
    ax1.set_ylabel("Count", color=PALETTE["text"], fontsize=10)
    ax1.tick_params(axis="x", colors=PALETTE["text"])
    ax1.tick_params(axis="y", colors=PALETTE["text"])
    ax1.set_ylim(0, counts.max() + 40)
 
    # ── Panel 2: Stress vs Anxiety scatter ────────────────────────────────
    melt_0 = df[df["meltdown"] == 0]
    melt_1 = df[df["meltdown"] == 1]
 
    ax2.scatter(
        melt_0["stress_level"], melt_0["anxiety_level"],
        c=PALETTE["low"], alpha=0.65, s=30, label="No Meltdown", edgecolors="none",
    )
    ax2.scatter(
        melt_1["stress_level"], melt_1["anxiety_level"],
        c=PALETTE["high"], alpha=0.65, s=30, label="Meltdown", edgecolors="none",
    )
    ax2.set_xlabel("Stress Level",   color=PALETTE["text"], fontsize=10)
    ax2.set_ylabel("Anxiety Level",  color=PALETTE["text"], fontsize=10)
    ax2.set_title("Stress vs Anxiety\n(by Meltdown Label)",
                  color=PALETTE["text"], fontsize=11, fontweight="bold")
    legend = ax2.legend(fontsize=9, facecolor=PALETTE["bg"],
                        edgecolor="#555", labelcolor=PALETTE["text"])
 
    # ── Panel 3: Correlation Heatmap ──────────────────────────────────────
    feature_cols = list(FEATURES_LABELS.keys()) + ["meltdown"]
    corr = df[[c for c in feature_cols if c in df.columns]].corr()
 
    sns.heatmap(
        corr,
        ax=ax3,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="#333",
        annot_kws={"size": 8, "color": "white"},
        cbar_kws={"shrink": 0.8},
    )
    ax3.set_title("Feature Correlation Heatmap",
                  color=PALETTE["text"], fontsize=11, fontweight="bold", pad=10)
    ax3.tick_params(axis="x", colors=PALETTE["text"], rotation=30, labelsize=8)
    ax3.tick_params(axis="y", colors=PALETTE["text"], rotation=0,  labelsize=8)
 
    # Colorbar text colour
    cbar = ax3.collections[0].colorbar
    cbar.ax.tick_params(colors=PALETTE["text"])
 
    fig.suptitle(
        "ASD Meltdown Dataset — Overview",
        color=PALETTE["text"], fontsize=14, fontweight="bold", y=1.01,
    )
 
    fig.tight_layout()
    return fig
 
 
# ── Quick Test ────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
    # Test plot_probability
    fig1 = plot_probability(78.5)
    fig1.savefig("test_probability.png", bbox_inches="tight",
                 facecolor=PALETTE["bg"])
    print("[✓] plot_probability saved → test_probability.png")
 
    # Test plot_features
    sample_input = {
        "age": 8, "sleep_hours": 5.5, "stress_level": 8,
        "social_interaction": 3, "sensory_sensitivity": 9,
        "anxiety_level": 7, "routine_change": 1, "noise_tolerance": 2,
    }
    fig2 = plot_features(sample_input)
    fig2.savefig("test_features.png", bbox_inches="tight",
                 facecolor=PALETTE["bg"])
    print("[✓] plot_features saved → test_features.png")
 
    # Test plot_distribution
    try:
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df   = pd.read_csv(os.path.join(ROOT, "data", "dataset.csv"))
        fig3 = plot_distribution(df)
        fig3.savefig("test_distribution.png", bbox_inches="tight",
                     facecolor=PALETTE["bg"])
        print("[✓] plot_distribution saved → test_distribution.png")
    except FileNotFoundError:
        print("[!] dataset.csv not found — skipping plot_distribution test")
 
    plt.close("all")
    print("\n[✓] All visualization tests complete.")