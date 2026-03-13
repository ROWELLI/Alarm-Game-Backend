import cv2
import mediapipe as mp
import numpy as np

from pose_model import predict_pose

mp_pose = mp.solutions.pose

# 웹캠 연속 프레임 처리에 맞게 static_image_mode=False
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def get_angle(a, b, c):
    """
    angle ABC in degrees
    """
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


def is_upper_body_present(landmarks, vis_threshold=0.35):
    """
    '사람 상체가 화면 안에 들어와 있는가?'를 느슨하게 판정
    - 양쪽 어깨는 보여야 함
    - 팔꿈치/손목은 양쪽 전부가 아니라 하나 이상씩만 보여도 통과
    """
    l_sh = landmarks[11]
    r_sh = landmarks[12]
    l_el = landmarks[13]
    r_el = landmarks[14]
    l_wr = landmarks[15]
    r_wr = landmarks[16]

    shoulders_ok = (
        l_sh.visibility >= vis_threshold and
        r_sh.visibility >= vis_threshold
    )
    elbows_ok = (
        l_el.visibility >= vis_threshold or
        r_el.visibility >= vis_threshold
    )
    wrists_ok = (
        l_wr.visibility >= vis_threshold or
        r_wr.visibility >= vis_threshold
    )

    return shoulders_ok and elbows_ok and wrists_ok


def extract_pose_features(image_bgr):
    """
    포즈 분류용 feature 추출
    반환:
      np.array(shape=(24,), dtype=np.float32) 또는 None
    """
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks.landmark

    # 상체 존재 판정 완화
    if not is_upper_body_present(landmarks, vis_threshold=0.35):
        return None

    nose = landmarks[0]
    l_sh = landmarks[11]
    r_sh = landmarks[12]
    l_el = landmarks[13]
    r_el = landmarks[14]
    l_wr = landmarks[15]
    r_wr = landmarks[16]

    # 어깨 중심 / 어깨 폭 기반 정규화
    shoulder_cx = (l_sh.x + r_sh.x) / 2.0
    shoulder_cy = (l_sh.y + r_sh.y) / 2.0
    shoulder_width = np.sqrt((r_sh.x - l_sh.x) ** 2 + (r_sh.y - l_sh.y) ** 2)
    shoulder_width = max(shoulder_width, 1e-6)

    def rel_xy(p):
        return [
            (p.x - shoulder_cx) / shoulder_width,
            (p.y - shoulder_cy) / shoulder_width,
        ]

    def dist(p1, p2):
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) / shoulder_width

    features = []

    # 1) 주요 점 상대좌표 (14개)
    features += rel_xy(nose)
    features += rel_xy(l_sh)
    features += rel_xy(r_sh)
    features += rel_xy(l_el)
    features += rel_xy(r_el)
    features += rel_xy(l_wr)
    features += rel_xy(r_wr)

    # 2) 거리 feature (5개)
    features += [
        dist(l_sh, l_el),
        dist(l_el, l_wr),
        dist(r_sh, r_el),
        dist(r_el, r_wr),
        dist(l_wr, r_wr),
    ]

    # 3) 각도 / 기울기 feature (3개)
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
    shoulder_slope = np.arctan2(r_sh.y - l_sh.y, r_sh.x - l_sh.x)

    features += [
        left_angle / 180.0,
        right_angle / 180.0,
        shoulder_slope / np.pi,
    ]

    # 4) visibility feature (7개)
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
    """
    dbdbd 이미지 바이트를 받아 상체 존재 + 포즈 분류 결과 반환
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return {
            "success": False,
            "pose_detected": False,
            "label": "unknown",
            "confidence": 0.0,
        }

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if not result.pose_landmarks:
        return {
            "success": True,
            "pose_detected": False,
            "label": "none",
            "confidence": 0.0,
        }

    landmarks = result.pose_landmarks.landmark

    # 상체가 화면에 들어왔는지 먼저 판단
    if not is_upper_body_present(landmarks, vis_threshold=0.35):
        return {
            "success": True,
            "pose_detected": False,
            "label": "none",
            "confidence": 0.0,
        }

    # 여기까지 왔으면 최소한 상체는 있다고 판단
    features = extract_pose_features(image)

    if features is None:
        return {
            "success": True,
            "pose_detected": True,
            "label": "other",
            "confidence": 0.0,
        }

    label, confidence = predict_pose(features)

    # 낮은 confidence는 오분류 방지용으로 other 처리
    if confidence < 0.55:
        label = "other"

    return {
        "success": True,
        "pose_detected": True,
        "label": label,
        "confidence": float(confidence),
    }