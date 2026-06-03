from ultralytics import YOLO
import cv2
import os
import math
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, Mapping, Optional, Tuple

# =========================================================
# 1. Config
# =========================================================

USE_REAR = False  # True: run rear branch; False: front-only

FRONT_MODEL_PATH = r"runs/detect/runs/Front/front_yolo26m/weights/best.pt"
REAR_MODEL_PATH = r"runs/detect/runs/Rear/rear_yolo26m/weights/best.pt"

FRONT_SOURCE = r"I:/ProgramData/课堂专注度项目/P1_YOLO_Pose_Attention/学生全景_10.30.04-p01-80.mp4"
REAR_SOURCE = r""

FRONT_CLASSES = ["discuss", "hand-raising", "read", "write", "teacher", "stand", "TurnHead"]
REAR_CLASSES = ["discuss", "hand-raising", "teacher", "stand", "screen", "blackBoard", "BowHead", "TurnHead"]

CONF_THRES = 0.2
IOU_THRES = 0.5
WINDOW_SECONDS = 5
HOLD_FRAMES = 10
SHOW_WINDOW = True
TRACKER_YAML = "bytetrack.yaml"

STABLE_MAX_GAP_FRAMES = 15
STABLE_MAX_CENTER_DIST = 80
STABLE_MIN_SIZE_RATIO = 0.5
INACTIVE_CLEANUP_FRAMES = 60

# New rule switch: use blackboard relation to refine TurnHead semantics.
ENABLE_BOARD_RULE = True
LOOK_BOARD_ANGLE_DEG = 35.0
LOOK_AWAY_ANGLE_DEG = 65.0

MAX_ATTENTION_SCORE = 100

# Deduction scheme (100-point system)
DEDUCT_DISCUSS = 2
DEDUCT_TURNHEAD = 3
DEDUCT_BOWHEAD = 4
DEDUCT_LOOK_AWAY = 5

# Duration-based deduction gate: deduct only when behavior duration in a window
# exceeds its threshold.
ENABLE_DURATION_GATED_SCORING = True
THRESHOLD_DISCUSS_SECONDS = 2.0
THRESHOLD_TURNHEAD_SECONDS = 1.5
THRESHOLD_BOWHEAD_SECONDS = 1.2
THRESHOLD_LOOK_AWAY_SECONDS = 1.0

# Bonus items: positive behaviors can recover score, final score is still capped at 100.
ENABLE_BONUS_SCORING = True
BONUS_MIN_DURATION_SECONDS = 1.0
BONUS_HAND_RAISING = 2.0
BONUS_READ = 1.0
BONUS_WRITE = 1.0
BONUS_NORMAL_TURN = 0.5

USE_REGION_MAPPING = True
MAP_X_SCALE = 1.0
MAP_Y_SCALE = 1.0
MAP_X_OFFSET = 0.0
MAP_Y_OFFSET = 0.0
MAP_MAX_NORM_DIST = 0.18
LOCAL_REAR_ONLY_FOR_AMBIGUOUS = True

ENABLE_MANUAL_SCENE_CALIBRATION = True
CALIBRATION_FRAME_INDEX = 0
CALIBRATION_WINDOW_FRONT = "front_scene_calibration"
CALIBRATION_WINDOW_REAR = "rear_scene_calibration"

EXPORT_REPRESENTATIVE_IMAGES = True
REPRESENTATIVE_DIR = "output/person_representatives"
REPRESENTATIVE_PADDING_RATIO = 0.15
EXPORT_SCORE_REPORT = True
SCORE_REPORT_DIR = "output/score_reports"
SCORE_REPORT_FORMAT = "csv"  # "csv" or "json"
PRINT_WINDOW_AVERAGES = True
PRINT_WINDOW_FUSION_DETAIL = False
MAX_PRINT_WINDOWS = 200

# Person-ID merge (merge fragmented stable_id to real person_id)
ENABLE_PERSON_ID_MERGE = True
PERSON_MERGE_MAX_CENTER_DIST = 85.0
PERSON_MERGE_OVERLAP_CENTER_DIST = 55.0
PERSON_MERGE_MIN_SIZE_RATIO = 0.45
PERSON_MERGE_MAX_OVERLAP_WINDOWS = 1


# =========================================================
# 2. TurnHead + board helpers (inlined)
# =========================================================

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Vector = Tuple[float, float]


@dataclass(frozen=True)
class TurnHeadBoardConfig:
    look_board_angle_deg: float = 35.0
    look_away_angle_deg: float = 65.0
    min_vec_norm: float = 1e-6


def bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _norm(v: Vector) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1])


def _normalize(v: Vector, eps: float) -> Optional[Vector]:
    n = _norm(v)
    if n <= eps:
        return None
    return (v[0] / n, v[1] / n)


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _angle_deg(a: Vector, b: Vector, eps: float) -> Optional[float]:
    na = _normalize(a, eps)
    nb = _normalize(b, eps)
    if na is None or nb is None:
        return None
    d = max(-1.0, min(1.0, _dot(na, nb)))
    return math.degrees(math.acos(d))


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def classify_turnhead_with_board(
    main_cls: str,
    student_bbox: BBox,
    board_bbox: Optional[BBox],
    cfg: TurnHeadBoardConfig = TurnHeadBoardConfig(),
    gaze_vector: Optional[Vector] = None,
    turn_direction: Optional[int] = None,
) -> Dict[str, object]:
    if main_cls == "BowHead":
        return {"rear_state": "bow_head", "angle_deg": None, "board_direction": None}

    if main_cls != "TurnHead":
        return {"rear_state": "not_turnhead", "angle_deg": None, "board_direction": None}

    if board_bbox is None:
        return {"rear_state": "unknown_turn", "angle_deg": None, "board_direction": None}

    student_c = bbox_center(student_bbox)
    board_c = bbox_center(board_bbox)
    board_vec = (board_c[0] - student_c[0], board_c[1] - student_c[1])
    board_direction = _sign(board_vec[0])

    if gaze_vector is not None:
        angle_deg = _angle_deg(gaze_vector, board_vec, cfg.min_vec_norm)
        if angle_deg is None:
            rear_state = "unknown_turn"
        elif angle_deg <= cfg.look_board_angle_deg:
            rear_state = "look_board"
        elif angle_deg >= cfg.look_away_angle_deg:
            rear_state = "look_away"
        else:
            rear_state = "unknown_turn"
        return {
            "rear_state": rear_state,
            "angle_deg": angle_deg,
            "board_direction": board_direction,
        }

    if turn_direction is not None and turn_direction in (-1, 1) and board_direction in (-1, 1):
        rear_state = "look_board" if turn_direction == board_direction else "look_away"
        return {"rear_state": rear_state, "angle_deg": None, "board_direction": board_direction}

    return {"rear_state": "unknown_turn", "angle_deg": None, "board_direction": board_direction}


def infer_rear_states_for_window(
    rear_window_summary: Mapping[int, Mapping[str, object]],
    board_bbox: Optional[BBox],
    pose_hints: Optional[Mapping[int, Mapping[str, object]]] = None,
    cfg: TurnHeadBoardConfig = TurnHeadBoardConfig(),
) -> Dict[int, Dict[str, object]]:
    result: Dict[int, Dict[str, object]] = {}
    pose_hints = pose_hints or {}

    for sid, info in rear_window_summary.items():
        main_cls = str(info.get("main_cls", ""))
        bbox = info.get("bbox")
        if not bbox or len(bbox) != 4:
            result[sid] = {
                "rear_state": "unknown_turn" if main_cls == "TurnHead" else "not_turnhead",
                "angle_deg": None,
                "board_direction": None,
            }
            continue

        hint = pose_hints.get(sid, {})
        gaze_vector = hint.get("gaze_vector")
        turn_direction = hint.get("turn_direction")

        result[sid] = classify_turnhead_with_board(
            main_cls=main_cls,
            student_bbox=tuple(bbox),  # type: ignore[arg-type]
            board_bbox=board_bbox,
            cfg=cfg,
            gaze_vector=gaze_vector,  # type: ignore[arg-type]
            turn_direction=turn_direction,  # type: ignore[arg-type]
        )

    return result


# =========================================================
# 3. Utils
# =========================================================

def get_video_fps(video_path, default_fps=25):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] failed to read FPS, fallback={default_fps}: {video_path}")
        return default_fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 1:
        return default_fps
    return fps


def validate_path(path, name):
    if path is None or str(path).strip() == "":
        raise ValueError(f"{name} is empty, please check config")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} does not exist: {path}")


def safe_get_class_name(class_names, cls_id):
    if 0 <= cls_id < len(class_names):
        return class_names[cls_id]
    return str(cls_id)


def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_distance(b1, b2):
    c1 = get_center(b1)
    c2 = get_center(b2)
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(1, x2 - x1) * max(1, y2 - y1)


def similar_size(b1, b2, ratio_thresh=0.5):
    a1 = bbox_area(b1)
    a2 = bbox_area(b2)
    ratio = min(a1, a2) / max(a1, a2)
    return ratio >= ratio_thresh


def mean_bbox(bboxes):
    if not bboxes:
        return None
    x1 = sum(b[0] for b in bboxes) / len(bboxes)
    y1 = sum(b[1] for b in bboxes) / len(bboxes)
    x2 = sum(b[2] for b in bboxes) / len(bboxes)
    y2 = sum(b[3] for b in bboxes) / len(bboxes)
    return (x1, y1, x2, y2)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def clamp01(v):
    return max(0.0, min(1.0, v))


def clamp_score(score):
    return float(max(0.0, min(float(MAX_ATTENTION_SCORE), float(score))))


def get_action_duration_seconds(class_counts, cls_name, fps):
    if not class_counts or not cls_name:
        return 0.0
    if fps is None or fps <= 0:
        return 0.0
    frame_count = float(class_counts.get(cls_name, 0))
    return max(0.0, frame_count / float(fps))


def duration_passes_threshold(action_key, duration_seconds):
    if not ENABLE_DURATION_GATED_SCORING:
        return True
    if action_key == "discuss":
        return duration_seconds >= THRESHOLD_DISCUSS_SECONDS
    if action_key == "TurnHead":
        return duration_seconds >= THRESHOLD_TURNHEAD_SECONDS
    if action_key in ["BowHead", "bow_head"]:
        return duration_seconds >= THRESHOLD_BOWHEAD_SECONDS
    if action_key in ["look_away"]:
        return duration_seconds >= THRESHOLD_LOOK_AWAY_SECONDS
    return True


def compute_window_bonus(front_cls, final_label, action_duration_sec):
    if not ENABLE_BONUS_SCORING:
        return 0.0, ""
    if action_duration_sec < BONUS_MIN_DURATION_SECONDS:
        return 0.0, ""

    if front_cls == "hand-raising":
        return float(BONUS_HAND_RAISING), "bonus_hand_raising"
    if front_cls == "read":
        return float(BONUS_READ), "bonus_read"
    if front_cls == "write":
        return float(BONUS_WRITE), "bonus_write"
    if final_label == "normal_turn":
        return float(BONUS_NORMAL_TURN), "bonus_normal_turn"
    return 0.0, ""


def bbox_center_norm(bbox, frame_size):
    if bbox is None or frame_size is None:
        return None
    width, height = frame_size
    if width <= 0 or height <= 0:
        return None
    cx, cy = bbox_center(bbox)
    return (cx / width, cy / height)


def map_front_to_rear_point(front_point_norm):
    if front_point_norm is None:
        return None
    x, y = front_point_norm
    return (
        clamp01(x * MAP_X_SCALE + MAP_X_OFFSET),
        clamp01(y * MAP_Y_SCALE + MAP_Y_OFFSET),
    )


def point_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def is_ambiguous_front_cls(front_cls):
    return front_cls in ["TurnHead", "discuss"]


def find_nearest_rear_sid(mapped_point_norm, rear_track_dict, rear_frame_size, max_norm_dist):
    if mapped_point_norm is None or not rear_track_dict:
        return None, None

    best_sid = None
    best_dist = None
    for sid, info in rear_track_dict.items():
        rear_bbox = info.get("bbox")
        rear_point = bbox_center_norm(rear_bbox, rear_frame_size)
        if rear_point is None:
            continue
        dist = point_distance(mapped_point_norm, rear_point)
        if dist <= max_norm_dist and (best_dist is None or dist < best_dist):
            best_sid = sid
            best_dist = dist

    return best_sid, best_dist


def read_video_frame(video_path, frame_index=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video for calibration: {video_path}")
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read calibration frame: {video_path}")
    return frame


def points_to_quad(points):
    return np.array(points, dtype=np.float32)


def draw_quad_overlay(base_frame, points, mouse_pos=None, finished=False):
    frame = base_frame.copy()
    expected_labels = [
        "1: front-left",
        "2: front-right",
        "3: rear-right",
        "4: rear-left",
    ]
    if not points:
        tips = [
            "Left click: add corner",
            "Right click: undo last corner",
            "After 4 corners: left click again to confirm",
            f"Now select: {expected_labels[0]}",
        ]
        for i, tip in enumerate(tips):
            cv2.putText(
                frame,
                tip,
                (20, 30 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        return frame

    for idx, p in enumerate(points):
        cv2.circle(frame, p, 5, (0, 255, 0), -1)
        cv2.putText(
            frame,
            expected_labels[idx],
            (p[0] + 8, p[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], (0, 255, 255), 2)

    if not finished and mouse_pos is not None and len(points) < 4:
        cv2.line(frame, points[-1], mouse_pos, (255, 200, 0), 1)

    if len(points) == 4:
        cv2.polylines(
            frame,
            [np.array(points, dtype=np.int32)],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2,
        )

    tips = [
        "Left click: add corner",
        "Right click: undo last corner",
        "After 4 corners: left click again to confirm",
        "Use same physical order in both views",
    ]
    next_idx = min(len(points), 3)
    next_tip = (
        "Now confirm selected corners"
        if len(points) == 4
        else f"Now select: {expected_labels[next_idx]}"
    )
    for i, tip in enumerate(tips):
        cv2.putText(
            frame,
            tip,
            (20, 30 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    cv2.putText(
        frame,
        "Order: front-left -> front-right -> rear-right -> rear-left",
        (20, 30 + len(tips) * 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        frame,
        next_tip,
        (20, 30 + (len(tips) + 1) * 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 0, 0),
        2,
    )
    return frame


def collect_scene_quad(frame, window_name):
    state = {
        "points": [],
        "mouse_pos": None,
        "confirmed": False,
    }

    def on_mouse(event, x, y, flags, param):
        state["mouse_pos"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(state["points"]) < 4:
                state["points"].append((x, y))
            else:
                state["confirmed"] = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            if state["points"]:
                state["points"].pop()
                state["confirmed"] = False

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        preview = draw_quad_overlay(
            base_frame=frame,
            points=state["points"],
            mouse_pos=state["mouse_pos"],
            finished=(len(state["points"]) == 4),
        )
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError(f"scene calibration cancelled: {window_name}")

        if state["confirmed"] and len(state["points"]) == 4:
            break

    cv2.destroyWindow(window_name)
    return points_to_quad(state["points"])


def calibrate_scene_quads(front_source, rear_source):
    front_frame = read_video_frame(front_source, CALIBRATION_FRAME_INDEX)
    rear_frame = read_video_frame(rear_source, CALIBRATION_FRAME_INDEX)
    front_quad = collect_scene_quad(front_frame, CALIBRATION_WINDOW_FRONT)
    rear_quad = collect_scene_quad(rear_frame, CALIBRATION_WINDOW_REAR)
    return front_quad, rear_quad


def build_homography_from_quads(front_quad, rear_quad):
    canonical = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    front_to_canonical = cv2.getPerspectiveTransform(front_quad, canonical)
    canonical_to_rear = cv2.getPerspectiveTransform(canonical, rear_quad)
    return front_to_canonical, canonical_to_rear


def map_front_bbox_to_rear_point(front_bbox, front_to_canonical, canonical_to_rear):
    if front_bbox is None:
        return None, None
    x1, y1, x2, y2 = front_bbox
    foot = np.array([[(x1 + x2) / 2.0, y2]], dtype=np.float32)
    canonical_pt = cv2.perspectiveTransform(foot.reshape(1, 1, 2), front_to_canonical)[0][0]
    rear_pt = cv2.perspectiveTransform(
        np.array([[canonical_pt]], dtype=np.float32), canonical_to_rear
    )[0][0]
    return (float(canonical_pt[0]), float(canonical_pt[1])), (float(rear_pt[0]), float(rear_pt[1]))


def find_nearest_rear_sid_by_pixel(mapped_rear_point, rear_track_dict, max_pixel_dist=160):
    if mapped_rear_point is None or not rear_track_dict:
        return None, None
    best_sid = None
    best_dist = None
    for sid, info in rear_track_dict.items():
        rear_bbox = info.get("bbox")
        if rear_bbox is None:
            continue
        rear_center = bbox_center(rear_bbox)
        dist = point_distance(mapped_rear_point, rear_center)
        if dist <= max_pixel_dist and (best_dist is None or dist < best_dist):
            best_sid = sid
            best_dist = dist
    return best_sid, best_dist


def pick_board_bbox(board_bboxes):
    """
    Pick a stable board bbox per window.
    Strategy: use average bbox of all detected blackBoard boxes in the window.
    """
    return mean_bbox(board_bboxes)


def find_matching_stable_id(
    new_bbox,
    frame_idx,
    inactive_tracks,
    max_gap_frames=15,
    max_dist=80,
    min_size_ratio=0.5,
):
    best_sid = None
    best_dist = float("inf")

    for _, info in inactive_tracks.items():
        gap = frame_idx - info["last_seen"]
        if gap > max_gap_frames:
            continue

        old_bbox = info["bbox"]
        dist = center_distance(new_bbox, old_bbox)

        if dist <= max_dist and similar_size(new_bbox, old_bbox, ratio_thresh=min_size_ratio):
            if dist < best_dist:
                best_dist = dist
                best_sid = info["stable_id"]

    return best_sid


# =========================================================
# 3.1 Stable-ID to Person-ID merge
# =========================================================

def build_stable_stats(front_summary):
    sid_to_windows = defaultdict(set)
    sid_to_centers = defaultdict(list)
    sid_to_areas = defaultdict(list)

    for win_idx, track_dict in front_summary.items():
        for sid, info in track_dict.items():
            bbox = info.get("bbox")
            if bbox is None:
                continue
            sid_to_windows[sid].add(win_idx)
            sid_to_centers[sid].append(bbox_center(bbox))
            sid_to_areas[sid].append(float(bbox_area(bbox)))

    stats = {}
    for sid in sid_to_windows.keys():
        centers = sid_to_centers[sid]
        areas = sid_to_areas[sid]
        if not centers or not areas:
            continue
        mean_center = (
            sum(c[0] for c in centers) / len(centers),
            sum(c[1] for c in centers) / len(centers),
        )
        mean_area = sum(areas) / len(areas)
        windows = sorted(sid_to_windows[sid])
        stats[sid] = {
            "windows": set(windows),
            "first_win": windows[0],
            "last_win": windows[-1],
            "mean_center": mean_center,
            "mean_area": mean_area,
        }
    return stats


def stable_should_merge(a, b):
    win_a = a["windows"]
    win_b = b["windows"]
    overlap = len(win_a & win_b)
    if overlap > PERSON_MERGE_MAX_OVERLAP_WINDOWS:
        return False

    center_dist = point_distance(a["mean_center"], b["mean_center"])
    area_ratio = min(a["mean_area"], b["mean_area"]) / max(a["mean_area"], b["mean_area"])
    if area_ratio < PERSON_MERGE_MIN_SIZE_RATIO:
        return False

    if overlap > 0:
        return center_dist <= PERSON_MERGE_OVERLAP_CENTER_DIST
    return center_dist <= PERSON_MERGE_MAX_CENTER_DIST


def build_person_id_mapping(front_summary):
    stable_stats = build_stable_stats(front_summary)
    sids = sorted(stable_stats.keys())
    if not sids:
        return {}, {}

    parent = {sid: sid for sid in sids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for i in range(len(sids)):
        sid_a = sids[i]
        for j in range(i + 1, len(sids)):
            sid_b = sids[j]
            if stable_should_merge(stable_stats[sid_a], stable_stats[sid_b]):
                union(sid_a, sid_b)

    root_to_sids = defaultdict(list)
    for sid in sids:
        root_to_sids[find(sid)].append(sid)

    person_to_sids = {}
    sid_to_person = {}
    for pid, root in enumerate(sorted(root_to_sids.keys()), start=1):
        group = sorted(root_to_sids[root])
        person_to_sids[pid] = group
        for sid in group:
            sid_to_person[sid] = pid

    return sid_to_person, person_to_sids


def remap_fused_results_to_person(fused_window_results, sid_to_person):
    fused_person_results = {}
    for win_idx, track_dict in fused_window_results.items():
        person_bucket = defaultdict(list)
        for sid, info in track_dict.items():
            pid = sid_to_person.get(sid, sid)
            person_bucket[pid].append((sid, info))

        merged_window = {}
        for pid, sid_info_list in person_bucket.items():
            sid_info_list = sorted(sid_info_list, key=lambda x: x[1].get("final_score", 0), reverse=True)
            chosen_sid, chosen_info = sid_info_list[0]
            merged_info = dict(chosen_info)
            merged_info["source_sids"] = sorted([sid for sid, _ in sid_info_list])
            merged_info["chosen_sid"] = chosen_sid
            merged_info["merged_count"] = len(sid_info_list)
            merged_window[pid] = merged_info

        fused_person_results[win_idx] = merged_window

    return fused_person_results


def build_person_front_tracks(person_to_sids, front_stable_to_tracks):
    person_to_tracks = {}
    for pid, sids in person_to_sids.items():
        tracks = set()
        for sid in sids:
            for tid in front_stable_to_tracks.get(sid, []):
                tracks.add(tid)
        person_to_tracks[pid] = sorted(tracks)
    return person_to_tracks


def merge_representative_candidates_by_person(representative_candidates, sid_to_person):
    person_candidates = {}
    for sid, info in representative_candidates.items():
        pid = sid_to_person.get(sid, sid)
        quality = float(info.get("quality", 0.0))
        prev = person_candidates.get(pid)
        if prev is None or quality > float(prev.get("quality", 0.0)):
            new_info = dict(info)
            new_info["source_sid"] = sid
            person_candidates[pid] = new_info
    return person_candidates


# =========================================================
# 4. Per-view tracking + window stats
# =========================================================

def run_view_tracking(
    model,
    source,
    class_names,
    view_name="front",
    conf=0.2,
    iou=0.5,
    window_seconds=5,
    hold_frames=10,
    show_window=True,
    tracker_yaml="bytetrack.yaml",
):
    fps = get_video_fps(source, default_fps=25)

    results = model.track(
        source=source,
        stream=True,
        persist=True,
        tracker=tracker_yaml,
        conf=conf,
        iou=iou,
        save=False,
        show=False,
        verbose=False,
    )

    frame_idx = 0
    track_memory = {}
    track_to_stable = {}
    stable_to_tracks = defaultdict(set)
    inactive_tracks = {}
    stable_id_counter = 0
    teacher_stable_ids = set()

    window_events = defaultdict(lambda: defaultdict(list))
    window_bboxes = defaultdict(lambda: defaultdict(list))
    window_board_bboxes = defaultdict(list)
    frame_size = None
    representative_candidates = {}

    for result in results:
        frame = result.orig_img.copy()
        if frame_size is None:
            frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        current_time = frame_idx / fps
        window_idx = int(current_time // window_seconds)

        current_ids = set()
        boxes = result.boxes

        to_delete = []
        for old_tid, info in inactive_tracks.items():
            if frame_idx - info["last_seen"] > INACTIVE_CLEANUP_FRAMES:
                to_delete.append(old_tid)
        for old_tid in to_delete:
            del inactive_tracks[old_tid]

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            if getattr(boxes, "id", None) is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = list(range(len(xyxy)))

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(int, xyxy[i])
                bbox = (x1, y1, x2, y2)
                cls_id = int(cls_ids[i])
                conf_val = float(confs[i])
                tid = int(track_ids[i])

                current_ids.add(tid)
                cls_name = safe_get_class_name(class_names, cls_id)

                if tid not in track_to_stable:
                    matched_sid = find_matching_stable_id(
                        new_bbox=bbox,
                        frame_idx=frame_idx,
                        inactive_tracks=inactive_tracks,
                        max_gap_frames=STABLE_MAX_GAP_FRAMES,
                        max_dist=STABLE_MAX_CENTER_DIST,
                        min_size_ratio=STABLE_MIN_SIZE_RATIO,
                    )
                    if matched_sid is not None:
                        track_to_stable[tid] = matched_sid
                    else:
                        stable_id_counter += 1
                        track_to_stable[tid] = stable_id_counter

                sid = track_to_stable[tid]
                stable_to_tracks[sid].add(tid)
                if cls_name == "teacher":
                    teacher_stable_ids.add(sid)

                track_memory[tid] = {
                    "bbox": bbox,
                    "cls": cls_name,
                    "conf": conf_val,
                    "last_seen": frame_idx,
                    "stable_id": sid,
                }

                # Teacher priority is always highest:
                # once a stable_id is identified as teacher, exclude it globally.
                if sid in teacher_stable_ids:
                    continue

                quality = conf_val * bbox_area(bbox)
                prev = representative_candidates.get(sid)
                if prev is None or quality > prev["quality"]:
                    representative_candidates[sid] = {
                        "frame_idx": frame_idx,
                        "bbox": bbox,
                        "cls": cls_name,
                        "conf": conf_val,
                        "quality": quality,
                    }

                window_events[window_idx][sid].append(cls_name)
                window_bboxes[window_idx][sid].append(bbox)

                if cls_name == "blackBoard":
                    window_board_bboxes[window_idx].append(bbox)

                label = f"{view_name} T:{tid} S:{sid} {cls_name} {conf_val:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

        for tid, info in list(track_memory.items()):
            if tid not in current_ids:
                missing = frame_idx - info["last_seen"]
                if missing <= hold_frames:
                    x1, y1, x2, y2 = info["bbox"]
                    sid = info["stable_id"]
                    label = f"{view_name} T:{tid} S:{sid} HOLD {info['cls']}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (180, 180, 180),
                        1,
                    )
                else:
                    inactive_tracks[tid] = {
                        "stable_id": info["stable_id"],
                        "bbox": info["bbox"],
                        "last_seen": info["last_seen"],
                        "last_cls": info["cls"],
                    }
                    del track_memory[tid]

        cv2.putText(
            frame,
            f"{view_name} | t={current_time:.1f}s | window={window_idx}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

        if show_window:
            cv2.imshow(f"{view_name}_tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print(f"[INFO] manual stop for {view_name}")
                break

        frame_idx += 1

    if show_window:
        cv2.destroyWindow(f"{view_name}_tracking")

    summary = defaultdict(dict)
    for win_idx, stable_dict in window_events.items():
        for sid, cls_list in stable_dict.items():
            if sid in teacher_stable_ids:
                continue
            counter = Counter(cls_list)
            main_cls = counter.most_common(1)[0][0]
            summary[win_idx][sid] = {
                "main_cls": main_cls,
                "counts": dict(counter),
                "bbox": mean_bbox(window_bboxes[win_idx].get(sid, [])),
            }

    board_bboxes = {}
    for win_idx, bboxes in window_board_bboxes.items():
        board_bboxes[win_idx] = pick_board_bbox(bboxes)

    return {
        "summary": dict(summary),
        "events": dict(window_events),
        "fps": fps,
        "frame_count": frame_idx,
        "track_to_stable": dict(track_to_stable),
        "stable_to_tracks": {sid: sorted(list(tids)) for sid, tids in stable_to_tracks.items()},
        "teacher_stable_ids": sorted(list(teacher_stable_ids)),
        "board_bboxes": board_bboxes,
        "frame_size": frame_size,
        "representative_candidates": representative_candidates,
    }


# =========================================================
# 4. Decision and fusion
# =========================================================

def front_base_decision(front_cls, front_action_duration_sec=0.0):
    full_score = MAX_ATTENTION_SCORE

    if front_cls == "hand-raising":
        return full_score, "high_focus"
    if front_cls in ["read", "write"]:
        return full_score, "focused"
    if front_cls == "discuss":
        if not duration_passes_threshold("discuss", front_action_duration_sec):
            return full_score, "normal_short_discuss"
        return full_score - DEDUCT_DISCUSS, "neutral"
    if front_cls == "TurnHead":
        if not duration_passes_threshold("TurnHead", front_action_duration_sec):
            return full_score, "normal_short_turn"
        return full_score - DEDUCT_TURNHEAD, "suspicious"
    if front_cls in ["teacher", "stand"]:
        return full_score, "ignore"
    return full_score, "normal"


def get_window_main_rear_cls(rear_track_dict):
    all_cls = []
    for _, info in rear_track_dict.items():
        cls_name = info["main_cls"]
        if cls_name not in ["teacher", "stand", "screen", "blackBoard"]:
            all_cls.append(cls_name)
    if not all_cls:
        return None
    return Counter(all_cls).most_common(1)[0][0]


def get_window_main_rear_state_with_board(rear_track_dict, rear_board_bbox):
    """
    Build a window-level rear semantic state with board relation.
    This keeps compatibility with the original fusion style (window-level signal).
    """
    cfg = TurnHeadBoardConfig(
        look_board_angle_deg=LOOK_BOARD_ANGLE_DEG,
        look_away_angle_deg=LOOK_AWAY_ANGLE_DEG,
    )
    rear_states = infer_rear_states_for_window(
        rear_window_summary=rear_track_dict,
        board_bbox=rear_board_bbox,
        pose_hints=None,
        cfg=cfg,
    )

    # Priority is conservative: look_away > bow_head > look_board > unknown_turn
    state_counter = Counter(v["rear_state"] for v in rear_states.values())
    if state_counter["look_away"] > 0:
        return "look_away"
    if state_counter["bow_head"] > 0:
        return "bow_head"
    if state_counter["look_board"] > 0:
        return "look_board"
    if state_counter["unknown_turn"] > 0:
        return "unknown_turn"
    return None


def build_rear_state_dict(rear_track_dict, rear_board_bbox):
    cfg = TurnHeadBoardConfig(
        look_board_angle_deg=LOOK_BOARD_ANGLE_DEG,
        look_away_angle_deg=LOOK_AWAY_ANGLE_DEG,
    )
    return infer_rear_states_for_window(
        rear_window_summary=rear_track_dict,
        board_bbox=rear_board_bbox,
        pose_hints=None,
        cfg=cfg,
    )


def refine_with_rear(front_cls, rear_cls, base_score, base_label, front_action_duration_sec=0.0):
    score, label = base_score, base_label

    if front_cls in ["read", "write", "hand-raising"]:
        return score, label

    if front_cls == "TurnHead":
        if not duration_passes_threshold("TurnHead", front_action_duration_sec):
            return MAX_ATTENTION_SCORE, "normal_short_turn"
        if rear_cls is None:
            return score, label
        if rear_cls == "TurnHead":
            if not duration_passes_threshold("look_away", front_action_duration_sec):
                return MAX_ATTENTION_SCORE, "normal_short_turn"
            return MAX_ATTENTION_SCORE - DEDUCT_LOOK_AWAY, "low_focus"
        if rear_cls == "BowHead":
            if not duration_passes_threshold("BowHead", front_action_duration_sec):
                return MAX_ATTENTION_SCORE, "normal_short_turn"
            return MAX_ATTENTION_SCORE - DEDUCT_BOWHEAD, "suspicious"
        return MAX_ATTENTION_SCORE, "normal_turn"

    if front_cls == "discuss":
        if not duration_passes_threshold("discuss", front_action_duration_sec):
            return MAX_ATTENTION_SCORE, "normal_short_discuss"
        if rear_cls in ["BowHead", "TurnHead"]:
            return MAX_ATTENTION_SCORE - DEDUCT_TURNHEAD, "suspicious"
        return score, label

    if rear_cls == "BowHead":
        if not duration_passes_threshold("BowHead", front_action_duration_sec):
            return score, label
        return MAX_ATTENTION_SCORE - DEDUCT_BOWHEAD, "low_focus"

    return score, label


def refine_with_rear_state(front_cls, rear_state, base_score, base_label, front_action_duration_sec=0.0):
    score, label = base_score, base_label

    if front_cls in ["read", "write", "hand-raising"]:
        return score, label

    if front_cls == "TurnHead":
        if not duration_passes_threshold("TurnHead", front_action_duration_sec):
            return MAX_ATTENTION_SCORE, "normal_short_turn"
        if rear_state is None:
            return score, label
        if rear_state == "look_board":
            return MAX_ATTENTION_SCORE, "normal_turn"
        if rear_state == "look_away":
            if not duration_passes_threshold("look_away", front_action_duration_sec):
                return MAX_ATTENTION_SCORE, "normal_short_turn"
            return MAX_ATTENTION_SCORE - DEDUCT_LOOK_AWAY, "low_focus"
        if rear_state in ["bow_head", "unknown_turn"]:
            if not duration_passes_threshold("BowHead", front_action_duration_sec):
                return MAX_ATTENTION_SCORE, "normal_short_turn"
            return MAX_ATTENTION_SCORE - DEDUCT_BOWHEAD, "suspicious"
        return score, label

    if front_cls == "discuss":
        if not duration_passes_threshold("discuss", front_action_duration_sec):
            return MAX_ATTENTION_SCORE, "normal_short_discuss"
        if rear_state in ["bow_head", "look_away"]:
            return MAX_ATTENTION_SCORE - max(DEDUCT_TURNHEAD, DEDUCT_BOWHEAD), "suspicious"
        return score, label

    if rear_state == "bow_head":
        if not duration_passes_threshold("BowHead", front_action_duration_sec):
            return score, label
        return MAX_ATTENTION_SCORE - DEDUCT_BOWHEAD, "low_focus"

    return score, label


def fuse_results(
    front_summary,
    rear_summary=None,
    use_rear=False,
    rear_board_bboxes=None,
    front_fps=25.0,
    rear_fps=25.0,
    front_frame_size=None,
    rear_frame_size=None,
    front_to_canonical=None,
    canonical_to_rear=None,
):
    fused_window_results = {}
    all_windows = sorted(set(front_summary.keys()) | set((rear_summary or {}).keys()))

    for win_idx in all_windows:
        front_track_dict = front_summary.get(win_idx, {})
        rear_track_dict = (rear_summary or {}).get(win_idx, {})
        rear_board_bbox = (rear_board_bboxes or {}).get(win_idx)

        rear_signal = None
        rear_state_dict = {}
        if use_rear:
            if ENABLE_BOARD_RULE:
                rear_signal = get_window_main_rear_state_with_board(rear_track_dict, rear_board_bbox)
                rear_state_dict = build_rear_state_dict(rear_track_dict, rear_board_bbox)
            else:
                rear_signal = get_window_main_rear_cls(rear_track_dict)

        track_results = {}
        for sid, info in front_track_dict.items():
            front_cls = info["main_cls"]
            front_counts = info.get("counts", {})
            front_action_duration_sec = get_action_duration_seconds(
                class_counts=front_counts,
                cls_name=front_cls,
                fps=front_fps,
            )
            base_score, base_label = front_base_decision(
                front_cls=front_cls,
                front_action_duration_sec=front_action_duration_sec,
            )
            local_rear_sid = None
            local_rear_signal = None
            mapped_point_norm = None

            if (
                use_rear
                and USE_REGION_MAPPING
                and ENABLE_BOARD_RULE
                and rear_track_dict
                and info.get("bbox") is not None
                and ((not LOCAL_REAR_ONLY_FOR_AMBIGUOUS) or is_ambiguous_front_cls(front_cls))
            ):
                if front_to_canonical is not None and canonical_to_rear is not None:
                    mapped_point_norm, mapped_rear_pixel = map_front_bbox_to_rear_point(
                        front_bbox=info.get("bbox"),
                        front_to_canonical=front_to_canonical,
                        canonical_to_rear=canonical_to_rear,
                    )
                    local_rear_sid, _ = find_nearest_rear_sid_by_pixel(
                        mapped_rear_point=mapped_rear_pixel,
                        rear_track_dict=rear_track_dict,
                    )
                else:
                    front_point_norm = bbox_center_norm(info.get("bbox"), front_frame_size)
                    mapped_point_norm = map_front_to_rear_point(front_point_norm)
                    local_rear_sid, _ = find_nearest_rear_sid(
                        mapped_point_norm=mapped_point_norm,
                        rear_track_dict=rear_track_dict,
                        rear_frame_size=rear_frame_size,
                        max_norm_dist=MAP_MAX_NORM_DIST,
                    )
                if local_rear_sid is not None:
                    local_info = rear_state_dict.get(local_rear_sid, {})
                    local_rear_signal = local_info.get("rear_state")

            if use_rear:
                if ENABLE_BOARD_RULE:
                    active_rear_signal = local_rear_signal if local_rear_signal is not None else rear_signal
                    final_score, final_label = refine_with_rear_state(
                        front_cls=front_cls,
                        rear_state=active_rear_signal,
                        base_score=base_score,
                        base_label=base_label,
                        front_action_duration_sec=front_action_duration_sec,
                    )
                else:
                    final_score, final_label = refine_with_rear(
                        front_cls=front_cls,
                        rear_cls=rear_signal,
                        base_score=base_score,
                        base_label=base_label,
                        front_action_duration_sec=front_action_duration_sec,
                    )
            else:
                final_score, final_label = base_score, base_label

            track_results[sid] = {
                "front_cls": front_cls,
                "front_action_duration_sec": float(front_action_duration_sec),
                "rear_cls": rear_signal if use_rear else None,
                "local_rear_sid": local_rear_sid,
                "local_rear_state": local_rear_signal,
                "mapped_point_norm": list(mapped_point_norm) if mapped_point_norm is not None else None,
                "base_score": base_score,
                "base_label": base_label,
                "final_score": clamp_score(final_score),
                "final_label": final_label,
            }

        fused_window_results[win_idx] = track_results

    return fused_window_results


def calc_window_scores(fused_window_results):
    window_scores = {}
    for win_idx, track_dict in fused_window_results.items():
        scores = [clamp_score(info["final_score"]) for _, info in track_dict.items()]
        window_scores[win_idx] = sum(scores) / len(scores) if scores else 0
    return window_scores


def calc_student_scores(fused_window_results):
    student_windows = defaultdict(dict)  # sid -> win -> info
    for win_idx in sorted(fused_window_results.keys()):
        for sid, info in fused_window_results[win_idx].items():
            student_windows[sid][win_idx] = info

    per_student = {}
    for sid in sorted(student_windows.keys()):
        running_score = float(MAX_ATTENTION_SCORE)
        total_deduction = 0.0
        total_bonus = 0.0
        window_detail = {}

        for win_idx in sorted(student_windows[sid].keys()):
            info = student_windows[sid][win_idx]
            score = clamp_score(info.get("final_score", MAX_ATTENTION_SCORE))
            label = info.get("final_label", "normal")
            front_cls = info.get("front_cls", "")
            action_duration_sec = float(info.get("front_action_duration_sec", 0.0))

            deduction = max(0.0, float(MAX_ATTENTION_SCORE) - score)
            bonus, bonus_reason = compute_window_bonus(
                front_cls=front_cls,
                final_label=label,
                action_duration_sec=action_duration_sec,
            )

            total_deduction += deduction
            total_bonus += bonus
            running_score = clamp_score(running_score - deduction + bonus)

            window_detail[win_idx] = {
                "front_cls": front_cls,
                "score": score,
                "label": label,
                "deduction": deduction,
                "bonus": bonus,
                "bonus_reason": bonus_reason,
                "action_duration_sec": action_duration_sec,
                "running_score": running_score,
            }

        per_student[sid] = {
            "start_score": float(MAX_ATTENTION_SCORE),
            "total_deduction": total_deduction,
            "total_bonus": total_bonus,
            "final_video_score": running_score,
            "window_detail": window_detail,
        }

    return {
        "per_student": per_student,  # backward compatible
        "per_person": per_student,
    }


def clip_bbox_with_padding(bbox, frame_w, frame_h, pad_ratio=0.15):
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(frame_w - 1, x2 + pad_x)
    ny2 = min(frame_h - 1, y2 + pad_y)
    return nx1, ny1, nx2, ny2


def export_representative_images(front_source, representative_candidates, output_root, id_prefix="person"):
    run_dir = ensure_dir(output_root)
    cap = cv2.VideoCapture(front_source)
    if not cap.isOpened():
        print(f"[WARN] failed to open front video for representative export: {front_source}")
        return {}

    saved = {}
    for entity_id in sorted(representative_candidates.keys()):
        info = representative_candidates[entity_id]
        frame_idx = int(info["frame_idx"])
        bbox = tuple(map(int, info["bbox"]))
        cls_name = info.get("cls", "unknown")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(
                f"[WARN] failed to read representative frame for {id_prefix}={entity_id}, "
                f"frame={frame_idx}"
            )
            continue
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = clip_bbox_with_padding(
            bbox=bbox,
            frame_w=w,
            frame_h=h,
            pad_ratio=REPRESENTATIVE_PADDING_RATIO,
        )
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"[WARN] empty crop for {id_prefix}={entity_id}")
            continue
        out_name = f"{id_prefix}_{entity_id:03d}_f{frame_idx:06d}_{cls_name}.jpg"
        out_path = run_dir / out_name
        cv2.imwrite(str(out_path), crop)
        saved[entity_id] = str(out_path)

    cap.release()
    return saved


def build_score_report(
    person_scores,
    representative_images,
    use_rear,
    front_source,
    rear_source,
):
    per_person = person_scores.get("per_person", person_scores.get("per_student", {}))
    person_items = []
    final_scores = []

    for pid in sorted(per_person.keys()):
        info = per_person[pid]
        final_score = float(info.get("final_video_score", 0.0))
        final_scores.append(final_score)
        person_items.append(
            {
                "person_id": int(pid),
                "start_score": float(info.get("start_score", MAX_ATTENTION_SCORE)),
                "total_deduction": float(info.get("total_deduction", 0.0)),
                "total_bonus": float(info.get("total_bonus", 0.0)),
                "final_video_score": final_score,
                "window_detail": info.get("window_detail", {}),
                "representative_image": representative_images.get(pid, ""),
            }
        )

    overall_final_average = (
        float(sum(final_scores) / len(final_scores)) if final_scores else 0.0
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "front + rear" if use_rear else "front-only",
        "video": {
            "front_source": front_source,
            "rear_source": rear_source if use_rear else "",
        },
        "overall": {
            "person_count": len(person_items),
            "overall_average_final_score": overall_final_average,
        },
        "per_person": person_items,
    }


def save_score_report_json(report_data, output_path):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        import json
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    return str(output_path)


def save_score_report_csv(report_data, output_path):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    overall = report_data.get("overall", {})
    rows = []
    rows.append(
        {
            "record_type": "summary",
            "person_id": "",
            "start_score": "",
            "total_deduction": "",
            "total_bonus": "",
            "final_video_score": "",
            "representative_image": "",
            "window_detail": "",
            "person_count": overall.get("person_count", 0),
            "overall_average_final_score": overall.get("overall_average_final_score", 0.0),
        }
    )

    for item in report_data.get("per_person", []):
        rows.append(
            {
                "record_type": "person",
                "person_id": item.get("person_id", ""),
                "start_score": item.get("start_score", ""),
                "total_deduction": item.get("total_deduction", ""),
                "total_bonus": item.get("total_bonus", ""),
                "final_video_score": item.get("final_video_score", ""),
                "representative_image": item.get("representative_image", ""),
                "window_detail": str(item.get("window_detail", {})),
                "person_count": "",
                "overall_average_final_score": "",
            }
        )

    fieldnames = [
        "record_type",
        "person_id",
        "start_score",
        "total_deduction",
        "total_bonus",
        "final_video_score",
        "representative_image",
        "window_detail",
        "person_count",
        "overall_average_final_score",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return str(output_path)


def print_final_report(
    person_scores,
    window_scores,
    fused_window_results,
    use_rear,
    representative_images=None,
    person_to_sids=None,
    person_to_front_tracks=None,
):
    mode_name = "front + rear" if use_rear else "front-only"
    print("\n" + "=" * 60)
    print(f"[RESULT] mode: {mode_name}")
    print("=" * 60)

    print("\n[PERSON] per-person final score (100-point deduction):")
    per_person = person_scores.get("per_person", person_scores.get("per_student", {}))
    for pid in sorted(per_person.keys()):
        person_sids = person_to_sids.get(pid, []) if person_to_sids is not None else []
        front_tracks = (
            person_to_front_tracks.get(pid, [])
            if person_to_front_tracks is not None
            else []
        )
        window_detail = per_person[pid]["window_detail"]
        score_timeline = ", ".join(
            f"w{win_idx}:{window_detail[win_idx]['score']:.1f}/{window_detail[win_idx]['label']}(deduct={window_detail[win_idx]['deduction']:.1f})"
            for win_idx in sorted(window_detail.keys())
        )
        rep_path = ""
        if representative_images is not None and pid in representative_images:
            rep_path = representative_images[pid]
        print(
            f"  person={pid:>3} "
            f"| stable_group={str(person_sids):<18} "
            f"| front_tracks={str(front_tracks):<15} "
            f"| start={per_person[pid]['start_score']:.1f} "
            f"| deduct={per_person[pid]['total_deduction']:.1f} "
            f"| final={per_person[pid]['final_video_score']:.1f} "
            f"| windows=[{score_timeline}]"
        )
        if rep_path:
            print(f"    representative_image={rep_path}")

    if PRINT_WINDOW_AVERAGES:
        print("\n[AUX] window average score (class-level statistic):")
        sorted_windows = sorted(window_scores.keys())
        if len(sorted_windows) > MAX_PRINT_WINDOWS:
            shown = sorted_windows[:MAX_PRINT_WINDOWS]
            for win_idx in shown:
                print(f"window {win_idx:03d}: score = {window_scores[win_idx]:.2f}")
            print(
                f"... ({len(sorted_windows) - MAX_PRINT_WINDOWS} more windows omitted)"
            )
        else:
            for win_idx in sorted_windows:
                print(f"window {win_idx:03d}: score = {window_scores[win_idx]:.2f}")

    overall = sum(window_scores.values()) / len(window_scores) if window_scores else 0
    print("-" * 60)
    print(f"Overall classroom average score: {overall:.2f}")
    print("=" * 60)

    if PRINT_WINDOW_FUSION_DETAIL:
        print("\n[DETAIL] per-window fusion:")
        sorted_windows = sorted(fused_window_results.keys())
        if len(sorted_windows) > MAX_PRINT_WINDOWS:
            sorted_windows = sorted_windows[:MAX_PRINT_WINDOWS]
            print(f"[INFO] only first {MAX_PRINT_WINDOWS} windows are shown.")
        for win_idx in sorted_windows:
            print(f"\nwindow {win_idx}:")
            window_dict = fused_window_results[win_idx]
            if not window_dict:
                print("  (empty)")
                continue

            for pid, info in sorted(window_dict.items(), key=lambda x: x[0]):
                front_tracks = person_to_front_tracks.get(pid, []) if person_to_front_tracks is not None else []
                source_sids = info.get("source_sids", [])

                print(
                    f"  person={pid:>3} "
                    f"| source_sids={str(source_sids):<18} "
                    f"| front_tracks={str(front_tracks):<15} "
                    f"| front={info['front_cls']:<12} "
                    f"| rear={str(info['rear_cls']):<12} "
                    f"| final={info['final_label']:<12} "
                    f"| score={info['final_score']}"
                )


# =========================================================
# 5. Main
# =========================================================

def main():
    validate_path(FRONT_MODEL_PATH, "FRONT_MODEL_PATH")
    validate_path(FRONT_SOURCE, "FRONT_SOURCE")

    if USE_REAR:
        validate_path(REAR_MODEL_PATH, "REAR_MODEL_PATH")
        validate_path(REAR_SOURCE, "REAR_SOURCE")

    print("[INFO] loading front model...")
    front_model = YOLO(FRONT_MODEL_PATH)

    rear_model = None
    if USE_REAR:
        print("[INFO] loading rear model...")
        rear_model = YOLO(REAR_MODEL_PATH)

    front_to_canonical = None
    canonical_to_rear = None
    if USE_REAR and ENABLE_MANUAL_SCENE_CALIBRATION:
        print("[INFO] start manual scene calibration...")
        front_scene_quad, rear_scene_quad = calibrate_scene_quads(FRONT_SOURCE, REAR_SOURCE)
        front_to_canonical, canonical_to_rear = build_homography_from_quads(
            front_scene_quad, rear_scene_quad
        )
        print("[INFO] manual scene calibration finished.")

    print("[INFO] running front view...")
    front_result = run_view_tracking(
        model=front_model,
        source=FRONT_SOURCE,
        class_names=FRONT_CLASSES,
        view_name="front",
        conf=CONF_THRES,
        iou=IOU_THRES,
        window_seconds=WINDOW_SECONDS,
        hold_frames=HOLD_FRAMES,
        show_window=SHOW_WINDOW,
        tracker_yaml=TRACKER_YAML,
    )

    front_summary = front_result["summary"]
    front_fps = float(front_result.get("fps", 25.0))
    front_stable_to_tracks = front_result["stable_to_tracks"]
    front_frame_size = front_result.get("frame_size")
    front_representative_candidates = front_result.get("representative_candidates", {})

    rear_summary = {}
    rear_track_to_stable = {}
    rear_stable_to_tracks = {}
    rear_board_bboxes = {}
    rear_frame_size = None
    rear_fps = 25.0

    if USE_REAR and rear_model is not None:
        print("[INFO] running rear view...")
        rear_result = run_view_tracking(
            model=rear_model,
            source=REAR_SOURCE,
            class_names=REAR_CLASSES,
            view_name="rear",
            conf=CONF_THRES,
            iou=IOU_THRES,
            window_seconds=WINDOW_SECONDS,
            hold_frames=HOLD_FRAMES,
            show_window=SHOW_WINDOW,
            tracker_yaml=TRACKER_YAML,
        )
        rear_summary = rear_result["summary"]
        rear_track_to_stable = rear_result["track_to_stable"]
        rear_stable_to_tracks = rear_result["stable_to_tracks"]
        rear_board_bboxes = rear_result.get("board_bboxes", {})
        rear_frame_size = rear_result.get("frame_size")
        rear_fps = float(rear_result.get("fps", 25.0))
    else:
        print("[INFO] rear disabled; using front-only mode.")

    fused_window_results_stable = fuse_results(
        front_summary=front_summary,
        rear_summary=rear_summary,
        use_rear=USE_REAR,
        rear_board_bboxes=rear_board_bboxes,
        front_fps=front_fps,
        rear_fps=rear_fps,
        front_frame_size=front_frame_size,
        rear_frame_size=rear_frame_size,
        front_to_canonical=front_to_canonical,
        canonical_to_rear=canonical_to_rear,
    )

    if ENABLE_PERSON_ID_MERGE:
        sid_to_person, person_to_sids = build_person_id_mapping(front_summary)
    else:
        all_sids = sorted(
            {sid for _, track_dict in fused_window_results_stable.items() for sid in track_dict.keys()}
        )
        sid_to_person = {sid: sid for sid in all_sids}
        person_to_sids = {sid: [sid] for sid in all_sids}

    fused_window_results = remap_fused_results_to_person(
        fused_window_results=fused_window_results_stable,
        sid_to_person=sid_to_person,
    )

    person_to_front_tracks = build_person_front_tracks(person_to_sids, front_stable_to_tracks)

    window_scores = calc_window_scores(fused_window_results)
    person_scores = calc_student_scores(fused_window_results)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    representative_images = {}
    if EXPORT_REPRESENTATIVE_IMAGES:
        person_representative_candidates = merge_representative_candidates_by_person(
            representative_candidates=front_representative_candidates,
            sid_to_person=sid_to_person,
        )
        scored_person_ids = set(
            person_scores.get("per_person", person_scores.get("per_student", {})).keys()
        )
        representative_candidates = {
            pid: info
            for pid, info in person_representative_candidates.items()
            if pid in scored_person_ids
        }
        representative_root = Path(REPRESENTATIVE_DIR) / run_name
        representative_images = export_representative_images(
            front_source=FRONT_SOURCE,
            representative_candidates=representative_candidates,
            output_root=representative_root,
            id_prefix="person",
        )
        print(f"[INFO] representative images exported to: {representative_root}")

    score_report = build_score_report(
        person_scores=person_scores,
        representative_images=representative_images,
        use_rear=USE_REAR,
        front_source=FRONT_SOURCE,
        rear_source=REAR_SOURCE,
    )
    score_report_path = ""
    if EXPORT_SCORE_REPORT:
        if SCORE_REPORT_FORMAT.lower() == "json":
            score_report_path = save_score_report_json(
                report_data=score_report,
                output_path=Path(SCORE_REPORT_DIR) / f"person_scores_{run_name}.json",
            )
        else:
            score_report_path = save_score_report_csv(
                report_data=score_report,
                output_path=Path(SCORE_REPORT_DIR) / f"person_scores_{run_name}.csv",
            )
        print(f"[INFO] score report saved: {score_report_path}")

    return {
        "front_summary": front_summary,
        "rear_summary": rear_summary,
        "fused_window_results_stable": fused_window_results_stable,
        "fused_window_results": fused_window_results,
        "window_scores": window_scores,
        "person_scores": person_scores,
        "student_scores": person_scores,
        "representative_images": representative_images,
        "sid_to_person": sid_to_person,
        "person_to_sids": person_to_sids,
        "person_to_front_tracks": person_to_front_tracks,
        "score_report": score_report,
        "score_report_path": score_report_path,
        "front_track_to_stable": front_result["track_to_stable"],
        "rear_track_to_stable": rear_track_to_stable,
        "front_stable_to_tracks": front_stable_to_tracks,
        "rear_stable_to_tracks": rear_stable_to_tracks,
        "rear_board_bboxes": rear_board_bboxes,
    }


# =========================================================
# 6. Run
# =========================================================

results_all = main()
