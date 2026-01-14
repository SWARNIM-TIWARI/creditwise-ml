import pandas as pd

def generate_model_card(model, meta):
    feature_names = meta.get("feature_names", [])
    categorical_cols = meta.get("categorical_cols", [])

    model_card_text = f"""
### Model Card: Explainable Credit Risk AI

**Model:** CatBoostClassifier  
**Objective:** Predict probability of customer default  
**Features ({len(feature_names)}):** {', '.join(feature_names[:10])}...  
**Categorical Features:** {', '.join(categorical_cols)}

**Training Data:** Confidential customer credit dataset  
**Evaluation Metrics:** AUC, Logloss, etc. (see training logs)  
**Known Limitations:**  
- Sensitive to missing or unseen categorical values  
- Performance may vary on other populations

**Governance Notes:**  
- Fairness monitored across gender & education groups  
- Bias metrics available in Fairness & Governance tab

"""
    return model_card_text
