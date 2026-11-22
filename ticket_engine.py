"""Local ticket preparation, ML predictions, priority rules and SLA calculations."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
TRAINING_FILE = ROOT / "training_examples.csv"

ALIASES = {
    "ticket_id": ["id", "ticket", "case_id", "case", "reference"],
    "created_at": ["created", "date", "created_date", "created_on", "opened_at", "submitted_at"],
    "subject": ["title", "summary", "issue", "topic"],
    "description": ["message", "body", "details", "ticket_description", "customer_message", "request"],
    "status": ["state", "ticket_status", "resolution_status"],
    "customer": ["customer_name", "requester", "requester_name", "client"],
    "resolved_at": ["closed_at", "resolution_date", "completed_at"],
}

SLA_HOURS = {"Urgent": 4, "High": 8, "Medium": 24, "Low": 72}
VALID_STATUSES = {"open", "in progress", "pending", "resolved", "closed"}


def _normalise_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _normalise_date(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce", utc=True)
    return dates.dt.tz_convert(None)


@lru_cache(maxsize=1)
def load_models() -> tuple[Pipeline, Pipeline]:
    """Train small demonstration classifiers from the repository's synthetic examples."""
    if not TRAINING_FILE.exists():
        raise FileNotFoundError("training_examples.csv is missing from the project.")
    training = pd.read_csv(TRAINING_FILE).fillna("")
    required = {"text", "category", "sentiment"}
    if not required.issubset(training.columns):
        raise ValueError("training_examples.csv must include text, category and sentiment columns.")

    def make_model() -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=1,
                max_features=8000,
                sublinear_tf=True,
            )),
            ("classifier", LogisticRegression(
                max_iter=1200,
                class_weight="balanced",
                random_state=42,
            )),
        ])

    category_model = make_model()
    sentiment_model = make_model()
    category_model.fit(training["text"].astype(str), training["category"].astype(str))
    sentiment_model.fit(training["text"].astype(str), training["sentiment"].astype(str))
    return category_model, sentiment_model


def prepare_tickets(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize a support-ticket export and validate the required date/text columns."""
    if raw is None or raw.empty:
        raise ValueError("The uploaded ticket file is empty.")

    frame = raw.copy()
    frame.columns = [_normalise_name(column) for column in frame.columns]

    for canonical, aliases in ALIASES.items():
        if canonical not in frame.columns:
            match = next((alias for alias in aliases if alias in frame.columns), None)
            if match:
                frame = frame.rename(columns={match: canonical})

    if "created_at" not in frame.columns:
        raise ValueError("A creation date column is required (for example: created_at, created or date).")
    if "subject" not in frame.columns and "description" not in frame.columns:
        raise ValueError("At least one text column is required: subject/title or description/message.")

    if "subject" not in frame.columns:
        frame["subject"] = ""
    if "description" not in frame.columns:
        frame["description"] = ""

    frame["created_at"] = _normalise_date(frame["created_at"])
    frame = frame.loc[frame["created_at"].notna()].copy()
    if frame.empty:
        raise ValueError("No valid creation dates were found. Check the date column.")

    frame["subject"] = frame["subject"].fillna("").astype(str).str.strip()
    frame["description"] = frame["description"].fillna("").astype(str).str.strip()
    frame["ticket_text"] = (frame["subject"] + ". " + frame["description"]).str.strip(" .")
    frame = frame.loc[frame["ticket_text"].str.len() > 0].copy()
    if frame.empty:
        raise ValueError("No usable ticket text was found in the subject or description columns.")

    if "ticket_id" not in frame.columns:
        frame["ticket_id"] = [f"TKT-{index:05d}" for index in range(1, len(frame) + 1)]
    else:
        frame["ticket_id"] = frame["ticket_id"].fillna("").astype(str).str.strip()
        empty_ids = frame["ticket_id"].eq("")
        frame.loc[empty_ids, "ticket_id"] = [
            f"TKT-{index:05d}" for index in range(1, int(empty_ids.sum()) + 1)
        ]

    if "customer" not in frame.columns:
        frame["customer"] = "Customer"
    frame["customer"] = frame["customer"].fillna("Customer").astype(str).replace("", "Customer")

    if "status" not in frame.columns:
        frame["status"] = "Open"
    frame["status"] = frame["status"].fillna("Open").astype(str).str.strip()
    frame["status"] = frame["status"].replace("", "Open")
    frame["status_normalized"] = frame["status"].str.lower()
    frame.loc[~frame["status_normalized"].isin(VALID_STATUSES), "status_normalized"] = "open"
    frame["status_normalized"] = frame["status_normalized"].replace({
        "in progress": "in progress", "pending": "pending",
        "resolved": "resolved", "closed": "closed", "open": "open",
    })
    if "resolved_at" in frame.columns:
        frame["resolved_at"] = _normalise_date(frame["resolved_at"])
    else:
        frame["resolved_at"] = pd.NaT

    return frame.sort_values("created_at").reset_index(drop=True)


def _priority_from_text(text: str) -> tuple[str, str]:
    value = text.lower()
    urgent = [
        "account hacked", "data breach", "security breach", "unauthorized access",
        "fraud", "stolen account", "system down", "service outage", "production down",
        "all users", "payment system down",
    ]
    high = [
        "urgent", "asap", "cannot log in", "can't log in", "unable to access",
        "charged twice", "double charged", "payment failed", "data loss",
        "blocked account", "not working", "not received", "missing order",
        "deadline today", "business stopped",
    ]
    low = [
        "feature request", "suggestion", "when will", "how do i", "how can i",
        "thank you", "thanks", "just asking", "general question",
    ]
    hit = next((phrase for phrase in urgent if phrase in value), None)
    if hit:
        return "Urgent", f"Critical signal: “{hit}”"
    hit = next((phrase for phrase in high if phrase in value), None)
    if hit:
        return "High", f"High-impact signal: “{hit}”"
    hit = next((phrase for phrase in low if phrase in value), None)
    if hit:
        return "Low", f"Low urgency signal: “{hit}”"
    return "Medium", "Default priority; no strong urgency phrase detected"


def analyze_tickets(data: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Predict category/sentiment and calculate transparent priority and SLA status."""
    if data.empty:
        return data.copy()
    category_model, sentiment_model = load_models()
    result = data.copy()
    text = result["ticket_text"].astype(str)
    result["predicted_category"] = category_model.predict(text)
    result["predicted_sentiment"] = sentiment_model.predict(text)

    priority_pairs = result["ticket_text"].map(_priority_from_text)
    result["priority"] = priority_pairs.map(lambda pair: pair[0])
    result["priority_reason"] = priority_pairs.map(lambda pair: pair[1])
    result["sla_hours"] = result["priority"].map(SLA_HOURS).astype(int)
    result["sla_due_at"] = result["created_at"] + pd.to_timedelta(result["sla_hours"], unit="h")

    current = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    if current.tzinfo is not None:
        current = current.tz_convert(None)
    result["ticket_age_hours"] = (
        (current - result["created_at"]).dt.total_seconds().div(3600).clip(lower=0)
    ).round(1)

    sla_status = []
    for row in result.itertuples():
        status = row.status_normalized
        due = row.sla_due_at
        if status in {"resolved", "closed"}:
            if pd.isna(row.resolved_at):
                sla_status.append("Completed • no resolution time")
            elif row.resolved_at <= due:
                sla_status.append("Met")
            else:
                sla_status.append("Breached")
            continue

        remaining_hours = (due - current).total_seconds() / 3600
        if remaining_hours < 0:
            sla_status.append("Breached")
        elif remaining_hours <= max(1.0, row.sla_hours * 0.25):
            sla_status.append("At risk")
        else:
            sla_status.append("On track")

    result["sla_status"] = sla_status
    result["sla_remaining_hours"] = (
        (result["sla_due_at"] - current).dt.total_seconds().div(3600)
    ).round(1)
    return result


def ticket_summary(data: pd.DataFrame) -> dict[str, float]:
    """Calculate top-level support queue KPIs."""
    total = len(data)
    active = int((~data["status_normalized"].isin(["resolved", "closed"])).sum())
    breached = int((data["sla_status"] == "Breached").sum())
    at_risk = int((data["sla_status"] == "At risk").sum())
    negative = int((data["predicted_sentiment"] == "Negative").sum())
    return {
        "total": float(total),
        "active": float(active),
        "breached": float(breached),
        "at_risk": float(at_risk),
        "negative_pct": (negative / total * 100.0) if total else 0.0,
    }
