import gradio as gr
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from pathlib import Path

from src.predict import predict
from src.explainability import shap_explain, lime_explain
from src.fairness import compute_fairness_metrics
from src.report import generate_pdf_report, generate_batch_shap_plot
from src.model_card import generate_model_card
from src.constants import FEATURE_DESCRIPTIONS

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
Path("assets").mkdir(exist_ok=True)

def launch_app():
    # ----------------------------------------------------------------------
    # Load model + metadata
    # ----------------------------------------------------------------------
    model = CatBoostClassifier()
    model.load_model("models/catboost_model.cbm")
    meta = joblib.load("models/train_meta.pkl")

    feature_names = meta["feature_names"]
    categorical_cols = meta["categorical_cols"]

    # ----------------------------------------------------------------------
    # Core logic
    # ----------------------------------------------------------------------
    def single_prediction(*values):
        raw_df = pd.DataFrame([dict(zip(feature_names, values))])
        preds = predict(raw_df, model, categorical_cols)

        prob = float(preds["Pred_Prob"].iloc[0])
        band = preds["Risk_Band"].iloc[0]

        # XAI explanations
        shap_explain(model, raw_df)
        lime_exp = lime_explain(model, raw_df, raw_df, categorical_cols)
        reasons = [
            f"- {FEATURE_DESCRIPTIONS.get(k, k)} ({v:+.3f})"
            for k, v in sorted(lime_exp.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        ]

        return f"{prob:.2%}", band, "\n".join(reasons), "assets/shap_summary.png"

    def batch_prediction(file):
        raw_df = pd.read_csv(file.name)
        preds = predict(raw_df, model, categorical_cols)
        shap_path = generate_batch_shap_plot(raw_df, model, categorical_cols, max_rows=50)
        return preds, shap_path

    def generate_report(batch_df):
        df = pd.DataFrame(batch_df)
        # Corrected PDF call: only df, model, categorical_cols handled inside if needed
        return generate_pdf_report(df)

    def fairness_view():
        df = pd.read_csv("data/raw/UCL_Credit_Card.csv")
        preds = predict(df, model, categorical_cols)
        return compute_fairness_metrics(preds)

    def model_card_view():
        return generate_model_card(model, meta)

    def explain_single_customer(idx):
        df_all = pd.read_csv("data/raw/UCL_Credit_Card.csv")
        idx = int(idx)
        if idx < 0 or idx >= len(df_all):
            # fallback safe row
            df_row = pd.DataFrame([dict(zip(feature_names, [0]*len(feature_names)))])
        else:
            df_row = df_all.iloc[[idx]]
        return single_prediction(*df_row.values[0])

    # ----------------------------------------------------------------------
    # UI
    # ----------------------------------------------------------------------
    with gr.Blocks(title="Explainable Credit Risk AI") as demo:
        gr.Markdown(
            "# Explainable Credit Risk AI  \n"
            "**Production-grade Responsible ML demo (XAI, Fairness, Governance)**"
        )

        # TAB 1 — Single Prediction
        with gr.Tab("Credit Decision"):
            inputs = [gr.Number(label=f) for f in feature_names]
            run_btn = gr.Button("Run Credit Decision")
            prob = gr.Textbox(label="Default Probability")
            band = gr.Textbox(label="Risk Band")
            reasons = gr.Textbox(label="Top Risk Drivers", lines=6)
            shap_img = gr.Image(label="SHAP Explanation")

            run_btn.click(single_prediction, inputs, [prob, band, reasons, shap_img])

        # TAB 2 — Explainability (XAI)
        with gr.Tab("Explainability (XAI)"):
            gr.Markdown("Run local SHAP & LIME explanations for a single customer")
            customer_idx = gr.Number(label="Customer index (0-based)")
            shap_img2 = gr.Image(label="SHAP Summary")
            lime_out = gr.Textbox(label="Top Drivers", lines=6)
            explain_btn = gr.Button("Run Explanation")

            explain_btn.click(explain_single_customer, customer_idx, [prob, band, lime_out, shap_img2])

        # TAB 3 — Batch Scoring
        with gr.Tab("Batch Scoring"):
            file_input = gr.File(label="Upload CSV")
            batch_out = gr.Dataframe(label="Predictions")
            shap_img_batch = gr.Image(label="Batch SHAP (first 50 rows)")
            pdf_file = gr.File(label="PDF Report")

            score_btn = gr.Button("Score Batch")
            score_btn.click(batch_prediction, file_input, [batch_out, shap_img_batch])

            pdf_btn = gr.Button("Generate PDF Report")
            pdf_btn.click(generate_report, batch_out, pdf_file)

        # TAB 4 — Fairness & Governance
        with gr.Tab("Fairness & Governance"):
            gender_img = gr.Image(label="Risk by Gender")
            edu_img = gr.Image(label="Risk by Education")
            metrics_txt = gr.Textbox(label="Bias Metrics", lines=5)
            fairness_btn = gr.Button("Run Fairness Check")
            fairness_btn.click(fairness_view, inputs=None, outputs=[gender_img, edu_img, metrics_txt])

        # TAB 5 — Model Card
        with gr.Tab("Model Card"):
            model_card_md = gr.Markdown()
            model_card_btn = gr.Button("Generate Model Card")
            model_card_btn.click(model_card_view, inputs=None, outputs=model_card_md)

    demo.launch()


if __name__ == "__main__":
    launch_app()
