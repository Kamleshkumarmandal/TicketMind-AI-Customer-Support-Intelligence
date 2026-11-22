"""TicketMind: customer support ticket intelligence dashboard."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ticket_engine import analyze_tickets, prepare_tickets, ticket_summary


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="TicketMind | Support Intelligence", page_icon="🎫", layout="wide")

st.title("🎫 TicketMind")
st.caption("Customer support intelligence · triage, sentiment and SLA visibility")

with st.sidebar:
    st.header("Ticket workspace")
    uploaded = st.file_uploader("Upload tickets", type=["csv", "xlsx", "xls"])
    st.caption("Ticket analysis runs locally in this app. Avoid uploading private customer data to a public demo.")
    st.divider()
    st.subheader("SLA targets")
    st.caption("Demo targets: Urgent 4h · High 8h · Medium 24h · Low 72h")

try:
    if uploaded is None:
        raw = pd.read_csv(ROOT / "sample_tickets.csv")
        source_name = "Synthetic demo tickets"
    elif uploaded.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded)
        source_name = uploaded.name
    else:
        raw = pd.read_excel(uploaded)
        source_name = uploaded.name

    tickets = prepare_tickets(raw)
    with st.spinner("Analyzing tickets with the local NLP models..."):
        analyzed = analyze_tickets(tickets)
except Exception as exc:
    st.error(f"Unable to analyze this file: {exc}")
    st.info("Upload a CSV/XLSX file containing a creation date and at least a subject or description column.")
    st.stop()

st.caption(f"Source: **{source_name}** · {len(analyzed):,} tickets analyzed")

with st.sidebar:
    st.divider()
    st.subheader("Filters")
    category_options = sorted(analyzed["predicted_category"].unique().tolist())
    sentiment_options = ["Negative", "Neutral", "Positive"]
    priority_options = ["Urgent", "High", "Medium", "Low"]
    status_options = sorted(analyzed["status"].unique().tolist())
    selected_categories = st.multiselect("Predicted category", category_options, default=category_options)
    selected_sentiments = st.multiselect("Predicted sentiment", sentiment_options, default=sentiment_options)
    selected_priorities = st.multiselect("Priority", priority_options, default=priority_options)
    selected_statuses = st.multiselect("Ticket status", status_options, default=status_options)

filtered = analyzed[
    analyzed["predicted_category"].isin(selected_categories)
    & analyzed["predicted_sentiment"].isin(selected_sentiments)
    & analyzed["priority"].isin(selected_priorities)
    & analyzed["status"].isin(selected_statuses)
].copy()

if filtered.empty:
    st.warning("No tickets match these filters. Select more categories, sentiments, priorities or statuses.")
    st.stop()

kpis = ticket_summary(filtered)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tickets", f'{kpis["total"]:,.0f}')
c2.metric("Active queue", f'{kpis["active"]:,.0f}')
c3.metric("SLA breached", f'{kpis["breached"]:,.0f}')
c4.metric("SLA at risk", f'{kpis["at_risk"]:,.0f}')
c5.metric("Negative sentiment", f'{kpis["negative_pct"]:.1f}%')

overview, triage, sla, data_tab = st.tabs(
    ["Overview", "AI Triage", "SLA Monitor", "Ticket Data"]
)

with overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Ticket volume by category")
        category_counts = filtered["predicted_category"].value_counts().rename_axis("category").reset_index(name="tickets")
        fig = px.bar(category_counts, x="category", y="tickets", color="category")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Tickets", margin=dict(l=5, r=5, t=10, b=5))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Customer sentiment")
        sentiment_counts = filtered["predicted_sentiment"].value_counts().rename_axis("sentiment").reset_index(name="tickets")
        fig = px.pie(sentiment_counts, names="sentiment", values="tickets", hole=0.55)
        fig.update_layout(margin=dict(l=5, r=5, t=10, b=5), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Queue priority")
    priority_counts = (
        filtered["priority"].value_counts()
        .reindex(["Urgent", "High", "Medium", "Low"], fill_value=0)
        .rename_axis("priority").reset_index(name="tickets")
    )
    fig = px.bar(priority_counts, x="priority", y="tickets", color="priority")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Tickets", margin=dict(l=5, r=5, t=10, b=5))
    st.plotly_chart(fig, use_container_width=True)

with triage:
    st.subheader("Tickets needing attention")
    attention = filtered[
        filtered["priority"].isin(["Urgent", "High"])
        | filtered["predicted_sentiment"].eq("Negative")
        | filtered["sla_status"].isin(["Breached", "At risk"])
    ].copy()
    attention = attention.sort_values(
        ["priority", "ticket_age_hours"],
        ascending=[True, False],
        key=None,
    )
    if attention.empty:
        st.success("No high-attention tickets match the current filters.")
    else:
        view_cols = [
            "ticket_id", "created_at", "subject", "predicted_category",
            "predicted_sentiment", "priority", "priority_reason",
            "status", "sla_status", "ticket_age_hours",
        ]
        st.dataframe(attention[view_cols], use_container_width=True, hide_index=True)
    st.caption("Category and sentiment are ML predictions. Priority is assigned by explainable keyword rules; review all suggested actions.")

with sla:
    st.subheader("SLA status")
    sla_counts = filtered["sla_status"].value_counts().rename_axis("sla_status").reset_index(name="tickets")
    fig = px.bar(sla_counts, x="sla_status", y="tickets", color="sla_status")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Tickets", margin=dict(l=5, r=5, t=10, b=5))
    st.plotly_chart(fig, use_container_width=True)

    breached = filtered[filtered["sla_status"].eq("Breached")].sort_values("ticket_age_hours", ascending=False)
    st.subheader("Breached tickets")
    if breached.empty:
        st.success("No breached tickets in the current filtered view.")
    else:
        st.dataframe(
            breached[["ticket_id", "subject", "priority", "created_at", "sla_due_at", "status"]],
            use_container_width=True,
            hide_index=True,
        )
    st.caption("SLA deadlines use the demo targets shown in the sidebar. Configure them to match your organization's actual support policy before operational use.")

with data_tab:
    st.subheader("Analyzed ticket records")
    display_columns = [
        "ticket_id", "customer", "created_at", "subject", "description",
        "status", "predicted_category", "predicted_sentiment", "priority",
        "priority_reason", "sla_due_at", "sla_status", "ticket_age_hours",
    ]
    st.dataframe(filtered[display_columns], use_container_width=True, hide_index=True)
    st.download_button(
        "Download analyzed tickets (CSV)",
        data=filtered[display_columns].to_csv(index=False).encode("utf-8"),
        file_name="ticketmind_analyzed_tickets.csv",
        mime="text/csv",
    )

st.divider()
st.caption("TicketMind is a portfolio demo using synthetic training examples. Predictions are not calibrated probabilities and are not independently validated for real support operations.")
