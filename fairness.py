import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from src.predict import predict

Path("assets").mkdir(exist_ok=True)

def compute_fairness_metrics(df, model=None, categorical_cols=None):
    """
    Returns gender/education plots and bias metrics.
    If model is provided, predict first.
    """
    df = df.copy()

    # If model is provided, generate predictions
    if model is not None and categorical_cols is not None:
        df = predict(df, model, categorical_cols)

    # Ensure columns exist
    for col in ["SEX", "EDUCATION"]:
        if col not in df.columns:
            df[col] = "missing"

    # Gender risk
    gender_avg = df.groupby("SEX")["Pred_Prob"].mean().reset_index()
    sns.barplot(x="SEX", y="Pred_Prob", data=gender_avg)
    plt.title("Average Predicted Risk by Gender")
    gender_path = "assets/fairness_SEX.png"
    plt.savefig(gender_path, bbox_inches="tight")
    plt.close()

    # Education risk
    edu_avg = df.groupby("EDUCATION")["Pred_Prob"].mean().reset_index()
    sns.barplot(x="EDUCATION", y="Pred_Prob", data=edu_avg)
    plt.title("Average Predicted Risk by Education")
    edu_path = "assets/fairness_EDUCATION.png"
    plt.savefig(edu_path, bbox_inches="tight")
    plt.close()

    # Bias metrics
    metrics_text = f"Gender Risk Difference: {gender_avg['Pred_Prob'].max() - gender_avg['Pred_Prob'].min():.3f}\n"
    metrics_text += f"Education Risk Difference: {edu_avg['Pred_Prob'].max() - edu_avg['Pred_Prob'].min():.3f}"

    return gender_path, edu_path, metrics_text
