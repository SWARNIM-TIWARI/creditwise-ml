import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
from pathlib import Path
import pandas as pd

# Ensure assets folder exists
Path("assets").mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# PDF REPORT GENERATION
# ----------------------------------------------------------------------
def generate_pdf_report(df_preds, output_path="assets/credit_report.pdf"):
    """
    Generates a PDF report from a DataFrame of predictions.
    Works for both single-customer and batch predictions.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Credit Decision Report", ln=True, align="C")
    
    pdf.set_font("Arial", "", 12)
    
    for i, row in df_preds.iterrows():
        pdf.ln(5)
        pdf.cell(0, 8, f"Customer {i+1}", ln=True)
        pdf.cell(0, 8, f"Default Probability: {row['Pred_Prob']:.2%}", ln=True)
        pdf.cell(0, 8, f"Risk Band: {row['Risk_Band']}", ln=True)
        
        # Optionally include top drivers if column exists
        if 'Top_Drivers' in row:
            pdf.multi_cell(0, 8, f"Top Drivers:\n{row['Top_Drivers']}", ln=True)
    
    pdf.output(output_path)
    return output_path

# ----------------------------------------------------------------------
# BATCH SHAP PLOT
# ----------------------------------------------------------------------
def generate_batch_shap_plot(raw_df, model, categorical_cols, max_rows=50):
    """
    Generates SHAP summary plot for first max_rows of batch.
    Returns path to saved plot.
    """
    df = raw_df.copy()

    # Ensure categorical columns are strings
    for c in categorical_cols:
        if c not in df.columns:
            df[c] = "missing"
        df[c] = df[c].fillna("missing").astype(str)

    # Select features used in the model
    feature_df = df[model.feature_names_].head(max_rows)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_df)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, feature_df, show=False)
    path = "assets/batch_shap_summary.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return path

# ----------------------------------------------------------------------
# SINGLE-CUSTOMER SHAP PLOT (optional helper)
# ----------------------------------------------------------------------
def generate_single_shap_plot(df_row, model, categorical_cols):
    """
    Generates SHAP plot for a single customer row.
    Returns path to saved plot.
    """
    df = df_row.copy()
    for c in categorical_cols:
        if c not in df.columns:
            df[c] = "missing"
        df[c] = df[c].fillna("missing").astype(str)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    plt.figure(figsize=(8, 5))
    shap.force_plot(explainer.expected_value, shap_values, df, matplotlib=True, show=False)
    path = "assets/shap_summary.png"
    plt.savefig(path, bbox_inches="tight")
