import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={"default.payment.next.month": "DEFAULT"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)

    categorical_cols = ["SEX","EDUCATION","MARRIAGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
    numeric_cols = [c for c in df.columns if c not in categorical_cols + ["DEFAULT"]]

    feature_names = numeric_cols + categorical_cols

    X = df[feature_names]
    y = df["DEFAULT"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        {
            "X_train": X_train,
            "y_train": y_train,
            "feature_names": feature_names,
            "categorical_cols": categorical_cols
        },
        "models/preprocessed.pkl"
    )

    return X_train, y_train, feature_names, categorical_cols
