from catboost import CatBoostClassifier
import joblib
from pathlib import Path

def train_model(X_train, y_train, categorical_cols, feature_names):
    model = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_seed=42
    )
    model.fit(X_train, y_train, cat_features=categorical_cols)

    Path("models").mkdir(exist_ok=True)
    model.save_model("models/catboost_model.cbm")

    joblib.dump(
        {"feature_names": feature_names, "categorical_cols": categorical_cols},
        "models/train_meta.pkl"
    )

    return model
