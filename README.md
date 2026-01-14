# CreditWise — Auditable Credit Risk ML System

CreditWise is an end-to-end **credit default risk prediction system** designed with a strong emphasis on **explainability, auditability, and responsible ML practices** rather than raw predictive performance alone.

The system provides probability-based risk scores, local explanations for individual predictions, observational fairness metrics across sensitive attributes, and auto-generated governance artifacts such as PDF reports and model cards.

---

## System Overview

**Core capabilities**
- Probability-based credit risk scoring with interpretable risk bands
- Local explainability using SHAP (TreeExplainer) and LIME
- Batch inference with automated PDF report generation
- Observational fairness monitoring across sensitive groups
- Auto-generated model card documenting model intent, metrics, and limitations

**Primary goal**
> Make model behavior inspectable and reviewable by humans in realistic enterprise workflows.

---

## Architecture

Raw Data
↓
Data Preprocessing
↓
CatBoost Model
↓
Prediction & Risk Banding

---


The Gradio application (`app.py`) acts only as an orchestration and visualization layer over the core pipeline.

---

## Key Engineering Decisions

- **CatBoost** was selected for stable tabular performance and native categorical feature handling.
- Explainability, fairness, and reporting logic are **explicitly separated** to avoid hidden coupling.
- Fairness metrics are **monitoring-only** and are not used for automated decision enforcement.
- Batch workflows were prioritized to reflect real-world audit and governance processes.

Detailed rationale is documented in [`docs/engineering_decisions.md`](docs/engineering_decisions.md).

---

## Explicit Limitations

- Fairness metrics are observational and do not mitigate bias automatically.
- Model performance may not generalize across populations without retraining.
- LIME explanations are local approximations and may exhibit instability.
- This system is intended for educational and demonstration purposes, not live credit decisioning.

Limitations are documented intentionally and transparently.

---

## Repository Structure

creditwise-ml/
├── app.py
├── train_model.py
├── run_pipeline.py
├── src/
│ ├── predict.py
│ ├── explainability.py
│ ├── fairness.py
│ ├── report.py
│ ├── model_card.py
│ └── data_prep.py
├── docs/
├── data/
├── docker/
└── requirements.txt

---


---

## Running the Project

### Local Execution
```bash
pip install -r requirements.txt
python run_pipeline.py
