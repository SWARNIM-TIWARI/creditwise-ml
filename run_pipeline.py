from src.data_prep import preprocess_data
from train_model import train_model
from catboost import CatBoostClassifier
import joblib
import os
import app
from pathlib import Path

# ---------------------------
# Create required directories
# ---------------------------
Path("models").mkdir(exist_ok=True)
Path("assets").mkdir(exist_ok=True)
Path("data/raw").mkdir(exist_ok=True)

# ---------------------------
# Train model if not exists
# ---------------------------
if not os.path.exists("models/catboost_model.cbm"):
    print("Training CatBoost model...")
    X_train, y_train, feature_names, categorical_cols = preprocess_data(
        "data/raw/UCL_Credit_Card.csv"
    )
    train_model(X_train, y_train, categorical_cols, feature_names)
    print("Model training complete!")

# ---------------------------
# Launch Gradio app
# ---------------------------
print("Launching Gradio app...")
app.launch_app()
