import shap
import lime.lime_tabular
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("assets").mkdir(exist_ok=True)

def shap_explain(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("assets/shap_summary.png", bbox_inches="tight")
    plt.close()
    return shap_values

def lime_explain(model, X_train, row, categorical_cols):
    def predict_fn(x):
        df = pd.DataFrame(x, columns=X_train.columns)
        for c in categorical_cols:
            df[c] = df[c].astype(str)
        return model.predict_proba(df)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        categorical_features=[X_train.columns.get_loc(c) for c in categorical_cols],
        mode="classification"
    )

    exp = explainer.explain_instance(row.values[0], predict_fn, num_features=5)
    return dict(exp.as_list())
