from typing import List, Dict, Tuple

import numpy as np

POSITIVE_WORDS = [
    "upgrade",
    "upgrades",
    "beat",
    "record",
    "surge",
    "rally",
    "profit",
    "profits",
    "growth",
    "positive",
    "strong",
    "beat estimates",
]

NEGATIVE_WORDS = [
    "downgrade",
    "downgrades",
    "miss",
    "missed",
    "loss",
    "losses",
    "scam",
    "fraud",
    "fall",
    "drop",
    "negative",
    "penalty",
    "fine",
    "probe",
    "investigation",
]

EVENT_KEYWORDS = [
    "results",
    "earnings",
    "q1",
    "q2",
    "q3",
    "q4",
    "quarter",
    "dividend",
    "bonus",
    "split",
    "merger",
    "acquisition",
    "rbi",
    "policy",
    "court",
    "hearing",
    "order",
    "judgment",
    "verdict",
]

BREAKING_KEYWORDS = [
    "breaking",
    "just in",
    "live",
    "alert",
    "urgent",
    "flash",
]


def simple_sentiment_from_headlines(headlines: List[Dict]) -> Tuple[float, str, str]:
    """
    Very simple sentiment scoring based on presence of positive/negative keywords.
    Returns (score, label, summary).
    score in [-1, 1], label in {POSITIVE, NEGATIVE, NEUTRAL}.
    """
    if not headlines:
        return 0.0, "NEUTRAL", "No recent news found for this stock."

    scores = []
    for h in headlines:
        text = (h.get("title", "") or "") + " " + (h.get("description", "") or "")
        text_lower = text.lower()
        score = 0
        for w in POSITIVE_WORDS:
            if w in text_lower:
                score += 1
        for w in NEGATIVE_WORDS:
            if w in text_lower:
                score -= 1
        scores.append(score)

    if not scores:
        return 0.0, "NEUTRAL", "News headlines were not clearly positive or negative."

    avg_raw = np.mean(scores)
    max_abs = max(1, max(abs(s) for s in scores))
    score_norm = float(avg_raw / max_abs)

    if score_norm > 0.2:
        label = "POSITIVE"
    elif score_norm < -0.2:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    raw_titles = [h.get("title", "") for h in headlines if h.get("title")]
    summary = "; ".join(raw_titles[:2]) if raw_titles else "News is available, but not strongly directional."

    return score_norm, label, summary


def determine_news_risk(headlines: List[Dict]) -> str:
    """
    Classify overall news risk:
    - NONE: no or very mild news.
    - EVENT_RISK: important scheduled or structural events (earnings, policy, court, corporate actions).
    - BREAKING: very recent, urgent or unexpected type headlines.
    """
    if not headlines:
        return "NONE"

    has_event = False
    has_breaking = False

    for h in headlines:
        text = ((h.get("title", "") or "") + " " + (h.get("description", "") or "")).lower()
        for w in EVENT_KEYWORDS:
            if w in text:
                has_event = True
                break
        for w in BREAKING_KEYWORDS:
            if w in text:
                has_breaking = True
                break

    if has_breaking:
        return "BREAKING"
    if has_event:
        return "EVENT_RISK"
    return "NONE"
