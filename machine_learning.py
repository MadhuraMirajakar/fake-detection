# machine_learning.py
from pathlib import Path
import joblib
import pandas as pd
from sklearn.naive_bayes import GaussianNB

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "gnb_model.pkl"

def train_and_save_demo_model(save_path=MODEL_PATH):
    """Train a small demo GaussianNB model if no real model exists."""
    data = pd.DataFrame({
        'length': [10, 50, 120, 90, 25, 200, 40, 15, 300, 60],
        'num_dots': [1, 3, 6, 4, 2, 7, 3, 1, 8, 3],
        'digit_count': [0, 2, 10, 3, 1, 12, 2, 0, 20, 4],
        'has_https': [1, 1, 0, 0, 1, 0, 1, 1, 0, 1],
        'host_len': [6, 12, 20, 14, 8, 24, 10, 5, 30, 11],
        'keyword_count': [0, 1, 3, 2, 0, 4, 1, 0, 5, 1],
        'label': [0, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    })

    X = data.drop(columns=['label'])
    y = data['label']

    gnb = GaussianNB()
    gnb.fit(X, y)

    joblib.dump(gnb, save_path)
    return gnb

def load_model():
    """Load model; if missing, train demo model."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_and_save_demo_model()

def predict_from_features(model, features):
    """Return 0 (safe) or 1 (phishing) prediction."""
    return int(model.predict([features])[0])
