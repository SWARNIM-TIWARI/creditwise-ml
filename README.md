# 🏦 CreditWise — Auditable Credit Risk ML System

An end-to-end credit default risk prediction system built with a strong emphasis on
explainability, auditability, and responsible ML practices over raw predictive performance.

> **Design Philosophy:** Make model behavior inspectable and reviewable by humans
> in realistic enterprise workflows — not just maximize accuracy.

---

## 🚀 Features

- 📊 Probability-based credit risk scoring with interpretable risk bands
- 🔍 Local explainability using SHAP (TreeExplainer) and LIME, surfaced directly in the UI
- 📄 Batch inference with automated PDF report generation for governance workflows
- ⚖️ Observational fairness monitoring across sensitive groups (not automated enforcement)
- 🗂 Auto-generated model card documenting intent, features, metrics, and known limitations
- 🐳 Fully containerized with Docker for reproducible deployment

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Core ML** | CatBoost, scikit-learn |
| **Explainability** | SHAP, LIME |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **UI** | Gradio |
| **Reporting** | FPDF |
| **Deployment** | Docker, Joblib |

---

## 🧠 Key Engineering Decisions

- **CatBoost** was selected for stable tabular performance and native categorical feature handling —
  no manual encoding required.
- **Explainability, fairness, and reporting logic are explicitly separated** to avoid hidden
  coupling and maintain clean auditability boundaries.
- **Fairness metrics are monitoring-only** and are not used for automated decision enforcement.
- **Batch workflows were prioritized** to reflect real-world audit and governance processes
  over real-time inference.

Full rationale is documented in `docs/engineering_decisions.md`.

---

## 📂 Project Structure
```
creditwise-ml/
├── app.py                  # Gradio UI — orchestration & visualization layer
├── train_model.py          # Model training script
├── run_pipeline.py         # Full pipeline entry point
├── requirements.txt
├── src/
│   ├── predict.py          # Risk scoring & probability output
│   ├── explainability.py   # SHAP & LIME explanations
│   ├── fairness.py         # Group-level fairness metrics
│   ├── report.py           # Automated PDF report generation
│   ├── model_card.py       # Auto-generated model card
│   └── data_prep.py        # Data preprocessing pipeline
├── docs/
│   └── engineering_decisions.md
├── data/
└── docker/
```

---

## ⚙️ Installation & Running

### Option 1 — Local

1. **Clone the repository**
```bash
git clone https://github.com/SWARNIM-TIWARI/creditwise-ml.git
cd creditwise-ml
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
```

3. **Activate the virtual environment**

- Windows (PowerShell):
```bash
  .\venv\Scripts\Activate.ps1
```
- Windows (CMD):
```bash
  .\venv\Scripts\activate.bat
```
- macOS/Linux:
```bash
  source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Run the pipeline**
```bash
python run_pipeline.py
```

6. **Launch the Gradio UI**
```bash
python app.py
```

---

### Option 2 — Docker

1. **Build the image**
```bash
docker build -t creditwise .
```

2. **Run the container**
```bash
docker run -p 7860:7860 creditwise
```

3. Open your browser at `http://localhost:7860`

---

## 🏗 System Architecture
```
Raw Data
   ↓
Data Preprocessing (data_prep.py)
   ↓
CatBoost Model (train_model.py)
   ↓
Prediction & Risk Banding (predict.py)
   ↓
┌─────────────────────────────────────┐
│  Explainability   Fairness   Report │
│  (SHAP & LIME)   Monitoring   PDF  │
└─────────────────────────────────────┘
   ↓
Gradio UI (app.py)
```

---

## ⚠️ Explicit Limitations

These limitations are documented **intentionally and transparently**:

- Fairness metrics are **observational only** — they do not mitigate bias automatically.
- Model performance **may not generalize** across populations without retraining.
- LIME explanations are **local approximations** and may exhibit instability.
- This system is intended for **educational and demonstration purposes**, not live credit decisioning.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

*Built to demonstrate that responsible ML design — transparency, auditability,
and honest failure documentation — matters as much as model performance.*
