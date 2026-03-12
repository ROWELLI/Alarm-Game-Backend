import cv2
import mediapipe as mp
import numpy as np
from pose_model import predict_pose

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
)

def get_angle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return float(angle)

def extract_pose_features(image_bgr):
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks.landmark

    nose = lm[0]
    l_sh = lm[11]
    r_sh = lm[12]
    l_el = lm[13]
    r_el = lm[14]
    l_wr = lm[15]
    r_wr = lm[16]

    required = [nose, l_sh, r_sh, l_el, r_el, l_wr, r_wr]

    # 상체 핵심 포인트 visibility 검사
    if any(p.visibility < 0.5 for p in required):
        return None

    shoulder_cx = (l_sh.x + r_sh.x) / 2.0
    shoulder_cy = (l_sh.y + r_sh.y) / 2.0

    shoulder_width = np.sqrt(
        (r_sh.x - l_sh.x) ** 2 +
        (r_sh.y - l_sh.y) ** 2
    )
    shoulder_width = max(shoulder_width, 1e-6)

    def rel_xy(p):
        return [
            (p.x - shoulder_cx) / shoulder_width,
            (p.y - shoulder_cy) / shoulder_width,
        ]

    def dist(p1, p2):
        return np.sqrt(
            (p1.x - p2.x) ** 2 +
            (p1.y - p2.y) ** 2
        ) / shoulder_width

    features = []

    features += rel_xy(nose)
    features += rel_xy(l_sh)
    features += rel_xy(r_sh)
    features += rel_xy(l_el)
    features += rel_xy(r_el)
    features += rel_xy(l_wr)
    features += rel_xy(r_wr)

    left_upper_arm = dist(l_sh, l_el)
    left_lower_arm = dist(l_el, l_wr)
    right_upper_arm = dist(r_sh, r_el)
    right_lower_arm = dist(r_el, r_wr)
    wrist_dist = dist(l_wr, r_wr)

    features += [
        left_upper_arm,
        left_lower_arm,
        right_upper_arm,
        right_lower_arm,
        wrist_dist,
    ]

    left_angle = get_angle(
        (l_sh.x, l_sh.y),
        (l_el.x, l_el.y),
        (l_wr.x, l_wr.y),
    )
    right_angle = get_angle(
        (r_sh.x, r_sh.y),
        (r_el.x, r_el.y),
        (r_wr.x, r_wr.y),
    )

    slope = np.arctan2(
        r_sh.y - l_sh.y,
        r_sh.x - l_sh.x
    )

    features += [
        left_angle / 180.0,
        right_angle / 180.0,
        slope / np.pi,
    ]

    features += [
        nose.visibility,
        l_sh.visibility,
        r_sh.visibility,
        l_el.visibility,
        r_el.visibility,
        l_wr.visibility,
        r_wr.visibility,
    ]

    return np.array(features, dtype=np.float32)

def analyze_dbdbd_from_bytes(image_bytes: bytes) -> dict:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        print("DBDBD: 이미지 디코딩 실패")
        return {
            "success": False,
            "pose_detected": False,
            "label": "unknown"
        }

    features = extract_pose_features(image)

    if features is None:
        print("DBDBD: 상체 feature 추출 실패")
        return {
            "success": True,
            "pose_detected": False,
            "label": "none"
        }

    print("DBDBD feature shape:", features.shape)
    label, confidence = predict_pose(features)

    return {
        "success": True,
        "pose_detected": True,
        "label": label,
        "confidence": confidence
    }