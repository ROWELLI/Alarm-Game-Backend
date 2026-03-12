from pathlib import Path
import joblib
import numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "rps_model.pkl"

model = joblib.load(MODEL_PATH)

def predict_rps(features: np.ndarray):
    x = np.array(features, dtype=np.float32).reshape(1, -1)

    pred = model.predict(x)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        confidence = float(np.max(proba))

    label = str(pred)
    return label, confidence