import os
import cv2
import pickle
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_direction_mlp_final.pkl")

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

scaler = saved["scaler"]
mlp = saved["mlp"]
classes = saved["classes"]


def extract_face_features(face_landmarks):
    features = []
    for lm in face_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features, dtype=np.float32).reshape(1, -1)


def analyze_ccc_from_bytes(image_bytes: bytes) -> dict:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return {
            "success": False,
            "face_detected": False,
            "label": None,
            "confidence": None,
            "message": "이미지를 디코딩할 수 없습니다."
        }

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {
            "success": True,
            "face_detected": False,
            "label": None,
            "confidence": None,
        }

    face_landmarks = results.multi_face_landmarks[0]
    x = extract_face_features(face_landmarks)

    try:
        x_scaled = scaler.transform(x)
        pred = mlp.predict(x_scaled)[0]
        probs = mlp.predict_proba(x_scaled)[0]
        confidence = float(np.max(probs))
    except Exception as e:
        return {
            "success": False,
            "face_detected": True,
            "label": None,
            "confidence": None,
            "message": str(e),
        }

    pred = str(pred).lower()

    if pred == "left":
        pred = "right"
    elif pred == "right":
        pred = "left"

    if pred not in ["left", "right", "front"]:
        return {
            "success": False,
            "face_detected": True,
            "label": None,
            "confidence": confidence,
            "message": f"Unexpected label: {pred}",
        }

    return {
        "success": True,
        "face_detected": True,
        "label": pred,
        "confidence": confidence,
    }