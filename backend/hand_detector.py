import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def dist2d(p1, p2):
    return np.linalg.norm(np.array([p1.x - p2.x, p1.y - p2.y]))

def dist3d(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5

def angle2d(landmarks, p1, p2, p3):
    v1 = np.array([
        landmarks[p1].x - landmarks[p2].x,
        landmarks[p1].y - landmarks[p2].y
    ])
    v2 = np.array([
        landmarks[p3].x - landmarks[p2].x,
        landmarks[p3].y - landmarks[p2].y
    ])

    v1_len = np.linalg.norm(v1)
    v2_len = np.linalg.norm(v2)

    if v1_len == 0 or v2_len == 0:
        return 0.0

    dot_product = np.dot(v1, v2)
    res = np.arccos(np.clip(dot_product / (v1_len * v2_len), -1.0, 1.0))
    return np.degrees(res)

def angle3d(a, b, c):
    ab = (a.x - b.x, a.y - b.y, a.z - b.z)
    cb = (c.x - b.x, c.y - b.y, c.z - b.z)

    dot = ab[0] * cb[0] + ab[1] * cb[1] + ab[2] * cb[2]
    norm_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2)
    norm_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2 + cb[2] ** 2)

    if norm_ab == 0 or norm_cb == 0:
        return 0.0

    cos_theta = max(-1.0, min(1.0, dot / (norm_ab * norm_cb)))
    return math.degrees(math.acos(cos_theta))

def palm_center(landmarks):
    ids = [0, 5, 9, 17]
    x = sum(landmarks[i].x for i in ids) / len(ids)
    y = sum(landmarks[i].y for i in ids) / len(ids)
    z = sum(landmarks[i].z for i in ids) / len(ids)

    class P:
        pass

    p = P()
    p.x, p.y, p.z = x, y, z
    return p

def get_label_base(landmarks):
    angles = [
        angle2d(landmarks, 5, 6, 8),
        angle2d(landmarks, 9, 10, 12),
        angle2d(landmarks, 13, 14, 16),
        angle2d(landmarks, 17, 18, 20)
    ]

    thumb_angle = angle2d(landmarks, 2, 3, 4)

    base_dist = dist2d(landmarks[0], landmarks[9]) + 1e-6
    thumb_dist = dist2d(landmarks[4], landmarks[5]) / base_dist

    finger_state = []
    for a in angles:
        if a > 145:
            finger_state.append("opened")
        elif a < 125:
            finger_state.append("folded")
        else:
            finger_state.append("mid")

    thumb_opened = (thumb_angle > 155) or (thumb_dist > 0.58)

    opened_count = sum(s == "opened" for s in finger_state)
    folded_count = sum(s == "folded" for s in finger_state)

    if opened_count >= 3 and folded_count == 0:
        return "PAPER"

    if folded_count >= 3 and not thumb_opened and thumb_dist < 0.52:
        return "ROCK"

    if (
        finger_state[0] == "opened" and
        finger_state[1] == "opened" and
        finger_state[2] != "opened" and
        finger_state[3] != "opened"
    ):
        return "SCISSOR"

    if (
        thumb_opened and
        finger_state[0] == "opened" and
        finger_state[1] != "opened" and
        finger_state[2] != "opened" and
        finger_state[3] != "opened"
    ):
        return "SCISSOR"

    return "UNKNOWN"

def finger_state_robust(landmarks, mcp, pip, dip, tip):
    palm_size = dist3d(landmarks[0], landmarks[9]) + 1e-6
    ang = angle3d(landmarks[pip], landmarks[dip], landmarks[tip])
    tip_mcp = dist3d(landmarks[tip], landmarks[mcp]) / palm_size

    if ang > 160 and tip_mcp > 0.78:
        return "strong"
    elif ang > 140 and tip_mcp > 0.60:
        return "weak"
    else:
        return "folded"

def is_front_fist(landmarks, center_thr=1.0, spread_thr=0.62):
    palm_size = dist3d(landmarks[0], landmarks[9]) + 1e-6
    c = palm_center(landmarks)

    tip_ids = [8, 12, 16, 20]
    center_dists = [dist3d(landmarks[i], c) / palm_size for i in tip_ids]
    mean_center_dist = sum(center_dists) / len(center_dists)

    spread = (
        dist3d(landmarks[8], landmarks[12]) +
        dist3d(landmarks[12], landmarks[16]) +
        dist3d(landmarks[16], landmarks[20])
    ) / (3 * palm_size)

    return (mean_center_dist < center_thr) and (spread < spread_thr)

def get_label_robust(landmarks):
    palm_size = dist3d(landmarks[0], landmarks[9]) + 1e-6
    center = palm_center(landmarks)

    index = finger_state_robust(landmarks, 5, 6, 7, 8)
    middle = finger_state_robust(landmarks, 9, 10, 11, 12)
    ring = finger_state_robust(landmarks, 13, 14, 15, 16)
    pinky = finger_state_robust(landmarks, 17, 18, 19, 20)

    states = [index, middle, ring, pinky]
    strong_count = sum(s == "strong" for s in states)
    weak_or_better = sum(s in ["strong", "weak"] for s in states)
    folded_count = sum(s == "folded" for s in states)

    spread_im = dist3d(landmarks[8], landmarks[12]) / palm_size
    spread_mr = dist3d(landmarks[12], landmarks[16]) / palm_size
    spread_rp = dist3d(landmarks[16], landmarks[20]) / palm_size

    protr_index = dist3d(landmarks[8], center) / palm_size
    protr_middle = dist3d(landmarks[12], center) / palm_size
    protr_ring = dist3d(landmarks[16], center) / palm_size
    protr_pinky = dist3d(landmarks[20], center) / palm_size

    if (
        index == "strong" and
        middle == "strong" and
        ring != "strong" and
        pinky != "strong" and
        spread_im > 0.14 and
        protr_ring < protr_index * 0.92 and
        protr_pinky < protr_middle * 0.92
    ):
        return "SCISSOR"

    if (
        weak_or_better == 4 and
        folded_count == 0 and
        strong_count >= 2 and
        spread_im > 0.08 and
        spread_mr > 0.05 and
        spread_rp > 0.04
    ):
        return "PAPER"

    if folded_count >= 3:
        return "ROCK"

    if is_front_fist(landmarks):
        return "ROCK"

    return "UNKNOWN"

def get_label(landmarks):
    base_label = get_label_base(landmarks)
    if base_label == "UNKNOWN":
        return get_label_robust(landmarks)
    return base_label

def analyze_rps_from_bytes(image_bytes: bytes) -> dict:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return {
            "success": False,
            "hand_detected": False,
            "label": "unknown"
        }

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return {
            "success": True,
            "hand_detected": False,
            "label": "none"
        }

    landmarks = results.multi_hand_landmarks[0].landmark
    raw_label = get_label(landmarks)

    label_map = {
        "ROCK": "rock",
        "PAPER": "paper",
        "SCISSOR": "scissor",
        "UNKNOWN": "unknown"
    }

    return {
        "success": True,
        "hand_detected": True,
        "label": label_map.get(raw_label, "unknown")
    }