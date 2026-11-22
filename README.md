# TicketMind — AI Customer Support Intelligence

TicketMind is a Streamlit dashboard that helps support teams triage customer tickets. It predicts a ticket category and sentiment with local TF-IDF + Logistic Regression models, assigns a transparent rule-based priority, and checks service-level agreement (SLA) timing.

> **Demo-data notice:** The bundled examples and the model's training examples are synthetic. This is a portfolio demonstration, not a model validated on a real support queue. Review predictions before using them to make customer-impacting decisions.

## Features

- Upload CSV or Excel customer-support ticket exports.
- Predict category and customer sentiment locally on the app host.
- Assign Low, Medium, High or Urgent priority using visible keyword and impact signals.
- Calculate ticket age, SLA due time, breached tickets and tickets approaching their SLA.
- Filter by predicted category, sentiment, priority and ticket status.
- View workload, sentiment, priority, category and SLA charts.
- Download enriched tickets as CSV.
- Includes synthetic demo tickets and unit tests.

## Run locally

Python 3.10 or newer is recommended.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.

## Upload format

The app needs a creation date and at least one text field: subject or description/message. It accepts common column aliases such as `id`, `ticket_id`, `created_at`, `date`, `subject`, `title`, `description`, `message`, `status`, `customer`, and `resolved_at`.

| Canonical column | Required | Purpose |
|---|---|---|
| `ticket_id` | Optional | Ticket reference; generated when missing |
| `created_at` | Yes | Ticket creation date/time |
| `subject` | One of subject/description | Short issue summary |
| `description` | One of subject/description | Ticket text |
| `status` | Optional | Open, in progress, pending, resolved or closed |
| `customer` | Optional | Customer label |
| `resolved_at` | Optional | Resolution timestamp for historical SLA checks |

Uploaded columns are normalized; original rows are not overwritten. The app shows predicted category and sentiment in separate fields.

## How the ticket intelligence works

- **Category:** TF-IDF word/phrase features and Logistic Regression, trained on the included synthetic examples.
- **Sentiment:** TF-IDF and Logistic Regression, trained on the same synthetic example file.
- **Priority:** An explainable keyword/impact ruleset. Urgent/security/outage language receives higher priority.
- **SLA:** Priority maps to a demo target (Urgent 4h, High 8h, Medium 24h, Low 72h). Active tickets are marked On track, At risk or Breached. Resolved tickets are compared with `resolved_at` when supplied.

The training corpus is intentionally small and synthetic. Predictions may be wrong, especially for ambiguous, multilingual, domain-specific or nuanced tickets. The model scores are not calibrated probabilities. For real operations, replace `training_examples.csv` with appropriately reviewed, representative and consented labeled support data, evaluate on a held-out dataset, and configure SLA targets to match the service policy. Do not use predictions as an automatic replacement for human triage.

## Privacy

Ticket text is processed inside the Streamlit app process. This project does not call an external model or send ticket text to a third-party AI API. If deployed to a hosted Streamlit service, data is processed by that hosting provider as part of running the app. Do not upload personal or confidential tickets to a public demo.

## Tests

```powershell
pip install -r requirements-dev.txt
pytest -q
```

## Deploy

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/).
2. Choose this repository, branch `main`, and main file path `app.py`.
3. Deploy the app. Streamlit will provide the public `*.streamlit.app` URL.

## Data and license

`sample_tickets.csv` and `training_examples.csv` contain synthetic text and example records created for this repository. They are not real customer data.

MIT License — see [LICENSE](LICENSE).
