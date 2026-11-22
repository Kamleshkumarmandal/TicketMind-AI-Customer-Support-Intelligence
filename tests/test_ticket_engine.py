import pandas as pd
import pytest

from ticket_engine import analyze_tickets, prepare_tickets, ticket_summary


def test_prepare_tickets_accepts_common_aliases_and_creates_ids():
    raw = pd.DataFrame({
        "Date": ["2026-11-20 10:00:00", "2026-11-21 11:00:00"],
        "Title": ["Payment charged twice", "How do I change my profile?"],
        "Message": ["Please refund the duplicate charge", "I want to update my profile photo"],
        "State": ["Open", "Pending"],
    })
    result = prepare_tickets(raw)
    assert len(result) == 2
    assert result["ticket_id"].str.startswith("TKT-").all()
    assert result["status_normalized"].tolist() == ["open", "pending"]


def test_prepare_tickets_requires_date_and_text():
    with pytest.raises(ValueError, match="creation date"):
        prepare_tickets(pd.DataFrame({"subject": ["Help"]}))
    with pytest.raises(ValueError, match="text column"):
        prepare_tickets(pd.DataFrame({"created_at": ["2026-11-20"]}))


def test_prediction_and_priority_rules():
    raw = pd.DataFrame({
        "ticket_id": ["T1", "T2"],
        "created_at": ["2026-11-22 08:00:00", "2026-11-22 08:00:00"],
        "subject": ["My account was hacked", "Thank you for the quick response"],
        "description": ["There is unauthorized access and fraud", "Everything works perfectly"],
        "status": ["Open", "Open"],
    })
    prepared = prepare_tickets(raw)
    result = analyze_tickets(prepared, now=pd.Timestamp("2026-11-22 09:00:00"))
    assert result.loc[0, "priority"] == "Urgent"
    assert result.loc[1, "priority"] == "Low"
    assert result.loc[0, "predicted_category"]
    assert result.loc[1, "predicted_sentiment"]


def test_sla_is_breached_for_old_open_ticket_and_met_for_timely_resolved_ticket():
    raw = pd.DataFrame({
        "ticket_id": ["T1", "T2"],
        "created_at": ["2026-11-20 08:00:00", "2026-11-22 08:00:00"],
        "subject": ["System down for all users", "Please help with invoice"],
        "description": ["The production service outage affects all users", "Question about invoice date"],
        "status": ["Open", "Resolved"],
        "resolved_at": [None, "2026-11-22 09:00:00"],
    })
    result = analyze_tickets(
        prepare_tickets(raw),
        now=pd.Timestamp("2026-11-22 10:00:00"),
    )
    assert result.loc[0, "sla_status"] == "Breached"
    assert result.loc[1, "sla_status"] == "Met"


def test_summary_counts_active_breached_and_at_risk():
    raw = pd.DataFrame({
        "ticket_id": ["T1", "T2", "T3"],
        "created_at": ["2026-11-22 06:00:00", "2026-11-22 08:30:00", "2026-11-22 09:00:00"],
        "subject": ["Account hacked", "Password reset help", "Feature request"],
        "description": ["Fraud and unauthorized access", "Need help to sign in", "I would like a new feature"],
        "status": ["Open", "Open", "Resolved"],
        "resolved_at": [None, None, "2026-11-22 09:30:00"],
    })
    analyzed = analyze_tickets(
        prepare_tickets(raw),
        now=pd.Timestamp("2026-11-22 10:00:00"),
    )
    summary = ticket_summary(analyzed)
    assert summary["total"] == 3
    assert summary["active"] == 2
    assert summary["breached"] >= 1


def test_csv_training_examples_produce_expected_label_sets():
    from ticket_engine import load_models
    category_model, sentiment_model = load_models()
    assert set(category_model.classes_) == {
        "Account Access", "Billing", "Product & Sales", "Shipping & Returns", "Technical Support"
    }
    assert set(sentiment_model.classes_) == {"Negative", "Neutral", "Positive"}
