import pandas as pd

def risk_band(p):
    return "Low" if p < 0.3 else "Medium" if p < 0.6 else "High"

def predict(df, model, categorical_cols):
    df = df.copy()

    # Ensure categorical columns exist & are strings
    for c in categorical_cols:
        if c not in df.columns:
            df[c] = "missing"
        df[c] = df[c].fillna("missing").astype(str)

    # Keep ONLY model features
    feature_df = df[model.feature_names_].copy()

    probs = model.predict_proba(feature_df)[:, 1]

    result = df.copy()
    result["Pred_Prob"] = probs
    result["Risk_Band"] = [risk_band(p) for p in probs]

    return result
