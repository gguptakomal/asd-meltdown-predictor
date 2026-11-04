"""
suggestions.py
--------------
Rule-based suggestion engine for the
ASD Meltdown Probability Predictor project.
 
Function:
    get_suggestions(probability) → dict
 
Risk Levels:
    Low     →  probability < 40
    Medium  →  probability 40 – 70
    High    →  probability > 70
"""
 
 
# ── Suggestion Bank ───────────────────────────────────────────────────────────
 
_SUGGESTIONS = {
 
    "Low": {
        "risk_level": "🟢 Low Risk",
        "sleep": (
            "Sleep schedule looks healthy. "
            "Keep maintaining a consistent bedtime routine — "
            "aim for 8–10 hours for younger children."
        ),
        "stress": (
            "Stress levels appear manageable. "
            "Continue with calming activities like drawing, "
            "reading, or light outdoor play."
        ),
        "sensory": (
            "Sensory environment seems comfortable. "
            "Keep noise and lighting at familiar, predictable levels "
            "to maintain this balance."
        ),
        "routine": (
            "Routine appears stable. "
            "Consistent daily schedules are a strong protective factor — "
            "keep up the predictability."
        ),
        "anxiety": (
            "Anxiety is within a low range. "
            "Positive reinforcement and praise for small achievements "
            "help sustain this level."
        ),
    },
 
    "Medium": {
        "risk_level": "🟡 Medium Risk",
        "sleep": (
            "Sleep may be slightly disrupted. "
            "Establish a wind-down routine 30 minutes before bedtime — "
            "dim lights, avoid screens, and use a comfort object if helpful."
        ),
        "stress": (
            "Stress levels are elevated. "
            "Introduce short sensory breaks during the day. "
            "Try deep breathing exercises or a calm corner with fidget tools."
        ),
        "sensory": (
            "Sensory sensitivity is moderate. "
            "Reduce background noise where possible — "
            "noise-cancelling headphones or a quiet room can help significantly."
        ),
        "routine": (
            "A routine change may have occurred. "
            "Prepare the child in advance using visual schedules or social stories "
            "to ease transitions before they happen."
        ),
        "anxiety": (
            "Anxiety is noticeable. "
            "Identify specific triggers and create a simple coping plan. "
            "Consider regular check-ins using an emotion chart or feelings card."
        ),
    },
 
    "High": {
        "risk_level": "🔴 High Risk",
        "sleep": (
            "Poor sleep is a major meltdown trigger. "
            "Prioritise sleep hygiene urgently — fixed wake/sleep times, "
            "blackout curtains, white noise machines, and no stimulating "
            "activity at least 1 hour before bed."
        ),
        "stress": (
            "Stress levels are critically high. "
            "Reduce task demands immediately. "
            "Create a structured calm-down kit with preferred sensory items. "
            "Avoid introducing new tasks or changes on high-stress days."
        ),
        "sensory": (
            "Sensory overload risk is high. "
            "Move the child to a low-stimulation environment immediately "
            "if signs of distress appear. "
            "Use weighted blankets, dimmed lighting, and minimal verbal instructions."
        ),
        "routine": (
            "Routine disruption is a strong contributor at this level. "
            "Re-establish structure as quickly as possible. "
            "Use a visual timetable and give advance warnings "
            "for every transition — even small ones."
        ),
        "anxiety": (
            "Anxiety is at a high level. "
            "Consult a behavioural therapist or ASD specialist if this is persistent. "
            "In the moment: use grounding techniques — name 5 things you can see, "
            "4 you can touch, 3 you can hear."
        ),
    },
}
 
 
# ── Core Function ─────────────────────────────────────────────────────────────
 
def get_suggestions(probability: float) -> dict:
    """
    Generate a risk level and actionable suggestions based on
    meltdown probability percentage.
 
    Parameters
    ----------
    probability : float
        Meltdown probability in the range 0 – 100.
 
    Returns
    -------
    dict with keys:
        risk_level       (str)  → coloured risk label
        overall          (str)  → one-line overall summary
        suggestions      (dict) → topic-wise suggestion strings
            ├── sleep
            ├── stress
            ├── sensory
            ├── routine
            └── anxiety
    """
 
    if not (0 <= probability <= 100):
        raise ValueError(
            f"[ERROR] probability must be between 0 and 100, got {probability}"
        )
 
    # ── Determine risk tier ───────────────────────────────────────────────────
    if probability < 40:
        tier = "Low"
        overall = (
            f"Meltdown probability is {probability:.1f}%. "
            "The child is currently in a stable state. "
            "Maintain the current environment and routine."
        )
 
    elif probability <= 70:
        tier = "Medium"
        overall = (
            f"Meltdown probability is {probability:.1f}%. "
            "Some risk factors are present. "
            "Proactive steps now can prevent escalation."
        )
 
    else:
        tier = "High"
        overall = (
            f"Meltdown probability is {probability:.1f}%. "
            "Multiple risk factors are elevated. "
            "Immediate calming strategies are strongly recommended."
        )
 
    bank = _SUGGESTIONS[tier]
 
    return {
        "risk_level": bank["risk_level"],
        "overall":    overall,
        "suggestions": {
            "Sleep":            bank["sleep"],
            "Stress":           bank["stress"],
            "Sensory Overload": bank["sensory"],
            "Routine Change":   bank["routine"],
            "Anxiety":          bank["anxiety"],
        },
    }
 
 
# ── Quick Test ────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
 
    test_cases = [25.0, 55.5, 82.3]
 
    for prob in test_cases:
        result = get_suggestions(prob)
        print("\n" + "═" * 55)
        print(f"  Probability : {prob}%")
        print(f"  Risk Level  : {result['risk_level']}")
        print(f"  Overall     : {result['overall']}")
        print("  Suggestions :")
        for topic, text in result["suggestions"].items():
            print(f"\n  [{topic}]\n  {text}")
    print("\n" + "═" * 55)
 