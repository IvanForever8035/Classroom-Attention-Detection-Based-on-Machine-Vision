from ultralytics import YOLO
import cv2
import os
import math
import csv
import json
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Mapping, Optional, Tuple

# =========================================================
# 1. Config
# =========================================================

USE_REAR = True  # True: run rear branch; False: front-only

FRONT_MODEL_PATH = r"runs/detect/runs/Front/front_yolo26m/weights/best.pt"
REAR_MODEL_PATH = r"runs/detect/runs/Rear/rear_yolo26m/weights/best.pt"

FRONT_SOURCE = r"I:/ProgramData/课堂专注度项目/SCB-YOLO/Multiple Cameras Fall Dataset/dataset/dataset/chute22/cam4.avi"
REAR_SOURCE = r"I:/ProgramData/课堂专注度项目/SCB-YOLO/Multiple Cameras Fall Dataset/dataset/dataset/chute22/cam5.avi"

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

EXPORT_RESULTS = True
EXPORT_DIR = "output/predict_exports"
RUN_NAME = "chute22_cam4_cam5"

# Weak cross-view region mapping:
# map front target center (normalized) to an approximate rear region and
# use the nearest rear target inside that region as local auxiliary evidence.
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

ENABLE_MAPPING_DEBUG_VIDEO = True
DEBUG_VIDEO_PATH = "output/predict_exports/mapping_debug.mp4"
DEBUG_MAX_FRAMES = None


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


def bbox_to_list(bbox):
    if bbox is None:
        return None
    return [float(v) for v in bbox]


def clamp01(v):
    return max(0.0, min(1.0, v))


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
    # Important: front_quad and rear_quad must follow the same physical order
    # chosen by the user, e.g. [front-left, front-right, rear-right, rear-left].
    # We intentionally do not reorder by image geometry, otherwise viewpoint
    # orientation information would be lost and the mapping would degenerate into
    # a same-view style warp.
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


def make_run_dir():
    base = ensure_dir(EXPORT_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = RUN_NAME.strip() if RUN_NAME.strip() else timestamp
    run_dir = base / suffix
    ensure_dir(run_dir)
    return run_dir


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_window_scores_csv(path, window_scores):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["window_idx", "score"])
        for win_idx in sorted(window_scores.keys()):
            writer.writerow([win_idx, f"{window_scores[win_idx]:.4f}"])


def save_fused_results_csv(path, fused_window_results):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "window_idx",
                "stable_id",
                "front_cls",
                "rear_cls",
                "base_score",
                "base_label",
                "final_score",
                "final_label",
            ]
        )
        for win_idx in sorted(fused_window_results.keys()):
            for sid, info in sorted(fused_window_results[win_idx].items(), key=lambda x: x[0]):
                writer.writerow(
                    [
                        win_idx,
                        sid,
                        info.get("front_cls"),
                        info.get("rear_cls"),
                        info.get("base_score"),
                        info.get("base_label"),
                        info.get("final_score"),
                        info.get("final_label"),
                    ]
                )


def save_window_scores_plot(path, window_scores):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable, skip score plot: {e}")
        return

    xs = sorted(window_scores.keys())
    ys = [window_scores[x] for x in xs]

    plt.figure(figsize=(10, 4.5))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.title("Window Attention Scores")
    plt.xlabel("Window Index")
    plt.ylabel("Average Score")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def export_results_bundle(
    run_dir,
    use_rear,
    front_source,
    rear_source,
    front_summary,
    rear_summary,
    fused_window_results,
    window_scores,
    front_track_to_stable,
    rear_track_to_stable,
    front_stable_to_tracks,
    rear_stable_to_tracks,
    rear_board_bboxes,
    front_frame_size,
    rear_frame_size,
    front_scene_quad,
    rear_scene_quad,
):
    overview = {
        "mode": "front+rear" if use_rear else "front-only",
        "front_source": front_source,
        "rear_source": rear_source if use_rear else "",
        "window_count": len(window_scores),
        "overall_score": (
            sum(window_scores.values()) / len(window_scores) if window_scores else 0
        ),
        "front_frame_size": list(front_frame_size) if front_frame_size else None,
        "rear_frame_size": list(rear_frame_size) if rear_frame_size else None,
        "use_region_mapping": USE_REGION_MAPPING,
        "manual_scene_calibration": ENABLE_MANUAL_SCENE_CALIBRATION,
        "front_scene_quad": front_scene_quad,
        "rear_scene_quad": rear_scene_quad,
    }

    normalized_front_summary = {
        str(win_idx): {
            str(sid): {
                "main_cls": info["main_cls"],
                "counts": info["counts"],
                "bbox": bbox_to_list(info.get("bbox")),
            }
            for sid, info in stable_dict.items()
        }
        for win_idx, stable_dict in front_summary.items()
    }

    normalized_rear_summary = {
        str(win_idx): {
            str(sid): {
                "main_cls": info["main_cls"],
                "counts": info["counts"],
                "bbox": bbox_to_list(info.get("bbox")),
            }
            for sid, info in stable_dict.items()
        }
        for win_idx, stable_dict in rear_summary.items()
    }

    normalized_fused = {
        str(win_idx): {str(sid): info for sid, info in stable_dict.items()}
        for win_idx, stable_dict in fused_window_results.items()
    }

    normalized_board = {
        str(win_idx): bbox_to_list(bbox) for win_idx, bbox in rear_board_bboxes.items()
    }

    save_json(run_dir / "overview.json", overview)
    save_json(run_dir / "front_summary.json", normalized_front_summary)
    save_json(run_dir / "rear_summary.json", normalized_rear_summary)
    save_json(run_dir / "fused_window_results.json", normalized_fused)
    save_json(run_dir / "window_scores.json", window_scores)
    save_json(run_dir / "front_track_to_stable.json", front_track_to_stable)
    save_json(run_dir / "rear_track_to_stable.json", rear_track_to_stable)
    save_json(run_dir / "front_stable_to_tracks.json", front_stable_to_tracks)
    save_json(run_dir / "rear_stable_to_tracks.json", rear_stable_to_tracks)
    save_json(run_dir / "rear_board_bboxes.json", normalized_board)

    save_window_scores_csv(run_dir / "window_scores.csv", window_scores)
    save_fused_results_csv(run_dir / "fused_window_results.csv", fused_window_results)
    save_window_scores_plot(run_dir / "window_scores.png", window_scores)


def open_video_capture(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    return cap


def draw_cross(frame, point, color, size=10, thickness=2):
    if point is None:
        return
    x, y = int(point[0]), int(point[1])
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)


def draw_debug_mapping_video(
    output_path,
    front_source,
    rear_source,
    fused_window_results,
    front_frame_records,
    rear_frame_records,
    front_to_canonical,
    canonical_to_rear,
    window_seconds,
    max_frames=300,
):
    if front_to_canonical is None or canonical_to_rear is None:
        print("[WARN] no homography available, skip mapping debug video")
        return

    front_cap = open_video_capture(front_source)
    rear_cap = open_video_capture(rear_source)

    front_ok, front_frame = front_cap.read()
    rear_ok, rear_frame = rear_cap.read()
    if not front_ok or not rear_ok:
        front_cap.release()
        rear_cap.release()
        print("[WARN] failed to read initial frame, skip mapping debug video")
        return

    out_dir = Path(output_path).parent
    ensure_dir(out_dir)
    h = max(front_frame.shape[0], rear_frame.shape[0])
    w = front_frame.shape[1] + rear_frame.shape[1]
    fps = min(
        max(front_cap.get(cv2.CAP_PROP_FPS), 1),
        max(rear_cap.get(cv2.CAP_PROP_FPS), 1),
    )
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    frame_idx = 0
    while front_ok and rear_ok and (max_frames is None or frame_idx < max_frames):
        left = front_frame.copy()
        right = rear_frame.copy()
        current_time = frame_idx / fps
        window_idx = int(current_time // window_seconds)
        fused = fused_window_results.get(window_idx, {})
        front_frame_dict = front_frame_records[frame_idx] if frame_idx < len(front_frame_records) else {}
        rear_frame_dict = rear_frame_records[frame_idx] if frame_idx < len(rear_frame_records) else {}
        for sid, info in sorted(fused.items(), key=lambda x: int(x[0]) if isinstance(x[0], str) else x[0]):
            sid_key = int(sid) if isinstance(sid, str) and sid.isdigit() else sid
            front_info = front_frame_dict.get(sid_key) or front_frame_dict.get(str(sid_key))

            front_bbox = front_info.get("bbox") if front_info else None
            mapped_norm, mapped_pixel = (None, None)
            local_sid = None
            local_rear_state = None
            rear_bbox = None

            if front_bbox is not None:
                mapped_norm, mapped_pixel = map_front_bbox_to_rear_point(
                    front_bbox=front_bbox,
                    front_to_canonical=front_to_canonical,
                    canonical_to_rear=canonical_to_rear,
                )
                local_sid, _ = find_nearest_rear_sid_by_pixel(
                    mapped_rear_point=mapped_pixel,
                    rear_track_dict=rear_frame_dict,
                )
                if local_sid is not None:
                    rear_info = rear_frame_dict.get(local_sid) or rear_frame_dict.get(str(local_sid))
                    if rear_info is not None:
                        rear_bbox = rear_info.get("bbox")
                        local_rear_state = rear_info.get("main_cls")

            if front_bbox is not None:
                x1, y1, x2, y2 = map(int, front_bbox)
                cv2.rectangle(left, (x1, y1), (x2, y2), (0, 255, 255), 2)

            if mapped_pixel is not None:
                draw_cross(right, mapped_pixel, (255, 0, 255), size=12, thickness=2)

            if rear_bbox is not None:
                x1, y1, x2, y2 = map(int, rear_bbox)
                cv2.rectangle(right, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if left.shape[0] < h:
            left = cv2.copyMakeBorder(left, 0, h - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if right.shape[0] < h:
            right = cv2.copyMakeBorder(right, 0, h - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        canvas = cv2.hconcat([left, right])
        writer.write(canvas)

        front_ok, front_frame = front_cap.read()
        rear_ok, rear_frame = rear_cap.read()
        frame_idx += 1

    writer.release()
    front_cap.release()
    rear_cap.release()


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
# 3. Per-view tracking + window stats
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

    window_events = defaultdict(lambda: defaultdict(list))
    window_bboxes = defaultdict(lambda: defaultdict(list))
    window_board_bboxes = defaultdict(list)
    frame_records = []
    frame_size = None

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

                track_memory[tid] = {
                    "bbox": bbox,
                    "cls": cls_name,
                    "conf": conf_val,
                    "last_seen": frame_idx,
                    "stable_id": sid,
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

        current_frame_objects = {}
        for tid, info in track_memory.items():
            sid = info["stable_id"]
            current_frame_objects[sid] = {
                "bbox": info["bbox"],
                "main_cls": info["cls"],
                "track_id": tid,
            }
        frame_records.append(current_frame_objects)

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
        "board_bboxes": board_bboxes,
        "frame_size": frame_size,
        "frame_records": frame_records,
    }


# =========================================================
# 4. Decision and fusion
# =========================================================

def front_base_decision(front_cls):
    if front_cls == "hand-raising":
        return 3, "high_focus"
    if front_cls in ["read", "write"]:
        return 2, "focused"
    if front_cls == "discuss":
        return 0, "neutral"
    if front_cls == "TurnHead":
        return -1, "suspicious"
    if front_cls in ["teacher", "stand"]:
        return 0, "ignore"
    return 0, "normal"


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


def refine_with_rear(front_cls, rear_cls, base_score, base_label):
    score, label = base_score, base_label

    if front_cls in ["read", "write", "hand-raising"]:
        return score, label

    if front_cls == "TurnHead":
        if rear_cls is None:
            return score, label
        if rear_cls == "TurnHead":
            return -2, "low_focus"
        if rear_cls == "BowHead":
            return -1, "suspicious"
        return 0, "normal_turn"

    if front_cls == "discuss":
        if rear_cls in ["BowHead", "TurnHead"]:
            return -1, "suspicious"
        return score, label

    if rear_cls == "BowHead":
        return -2, "low_focus"

    return score, label


def refine_with_rear_state(front_cls, rear_state, base_score, base_label):
    score, label = base_score, base_label

    if front_cls in ["read", "write", "hand-raising"]:
        return score, label

    if front_cls == "TurnHead":
        if rear_state is None:
            return score, label
        if rear_state == "look_board":
            return 0, "normal_turn"
        if rear_state == "look_away":
            return -2, "low_focus"
        if rear_state in ["bow_head", "unknown_turn"]:
            return -1, "suspicious"
        return score, label

    if front_cls == "discuss":
        if rear_state in ["bow_head", "look_away"]:
            return -1, "suspicious"
        return score, label

    if rear_state == "bow_head":
        return -2, "low_focus"

    return score, label


def fuse_results(
    front_summary,
    rear_summary=None,
    use_rear=False,
    rear_board_bboxes=None,
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
            base_score, base_label = front_base_decision(front_cls)
            local_rear_sid = None
            local_rear_signal = None
            mapped_point_norm = None

            if (
                use_rear
                and USE_REGION_MAPPING
                and ENABLE_BOARD_RULE
                and rear_track_dict
                and info.get("bbox") is not None
                and (
                    (not LOCAL_REAR_ONLY_FOR_AMBIGUOUS)
                    or is_ambiguous_front_cls(front_cls)
                )
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
                    )
                else:
                    final_score, final_label = refine_with_rear(
                        front_cls=front_cls,
                        rear_cls=rear_signal,
                        base_score=base_score,
                        base_label=base_label,
                    )
            else:
                final_score, final_label = base_score, base_label

            track_results[sid] = {
                "front_cls": front_cls,
                "rear_cls": rear_signal if use_rear else None,
                "local_rear_sid": local_rear_sid,
                "local_rear_state": local_rear_signal,
                "mapped_point_norm": list(mapped_point_norm) if mapped_point_norm is not None else None,
                "base_score": base_score,
                "base_label": base_label,
                "final_score": final_score,
                "final_label": final_label,
            }

        fused_window_results[win_idx] = track_results

    return fused_window_results


def calc_window_scores(fused_window_results):
    window_scores = {}
    for win_idx, track_dict in fused_window_results.items():
        scores = [info["final_score"] for _, info in track_dict.items()]
        window_scores[win_idx] = sum(scores) / len(scores) if scores else 0
    return window_scores


def print_final_report(
    window_scores,
    fused_window_results,
    use_rear,
    front_stable_to_tracks=None,
    rear_stable_to_tracks=None,
):
    mode_name = "front + rear" if use_rear else "front-only"
    print("\n" + "=" * 60)
    print(f"[RESULT] mode: {mode_name}")
    print("=" * 60)

    for win_idx in sorted(window_scores.keys()):
        print(f"window {win_idx:03d}: score = {window_scores[win_idx]:.2f}")

    overall = sum(window_scores.values()) / len(window_scores) if window_scores else 0
    print("-" * 60)
    print(f"Overall class attention score: {overall:.2f}")
    print("=" * 60)

    print("\n[DETAIL] per-window fusion:")
    for win_idx in sorted(fused_window_results.keys()):
        print(f"\nwindow {win_idx}:")
        window_dict = fused_window_results[win_idx]
        if not window_dict:
            print("  (empty)")
            continue

        for sid, info in sorted(window_dict.items(), key=lambda x: x[0]):
            front_tracks = []
            rear_tracks = []
            if front_stable_to_tracks is not None:
                front_tracks = front_stable_to_tracks.get(sid, [])
            if rear_stable_to_tracks is not None:
                rear_tracks = rear_stable_to_tracks.get(sid, [])

            print(
                f"  stable={sid:>3} "
                f"| front_tracks={str(front_tracks):<15} "
                f"| rear_tracks={str(rear_tracks):<15} "
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

    rear_summary = {}
    rear_track_to_stable = {}
    rear_stable_to_tracks = {}
    rear_board_bboxes = {}
    rear_frame_size = None
    rear_frame_records = []
    front_scene_quad = None
    rear_scene_quad = None
    front_to_canonical = None
    canonical_to_rear = None

    if USE_REAR and ENABLE_MANUAL_SCENE_CALIBRATION:
        print("[INFO] start manual scene calibration...")
        front_scene_quad_np, rear_scene_quad_np = calibrate_scene_quads(
            FRONT_SOURCE, REAR_SOURCE
        )
        front_scene_quad = front_scene_quad_np.tolist()
        rear_scene_quad = rear_scene_quad_np.tolist()
        front_to_canonical, canonical_to_rear = build_homography_from_quads(
            front_scene_quad_np, rear_scene_quad_np
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
    front_stable_to_tracks = front_result["stable_to_tracks"]
    front_frame_size = front_result.get("frame_size")
    front_frame_records = front_result.get("frame_records", [])

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
        rear_frame_records = rear_result.get("frame_records", [])
    else:
        print("[INFO] rear disabled; using front-only mode.")

    fused_window_results = fuse_results(
        front_summary=front_summary,
        rear_summary=rear_summary,
        use_rear=USE_REAR,
        rear_board_bboxes=rear_board_bboxes,
        front_frame_size=front_frame_size,
        rear_frame_size=rear_frame_size,
        front_to_canonical=front_to_canonical,
        canonical_to_rear=canonical_to_rear,
    )

    window_scores = calc_window_scores(fused_window_results)

    print_final_report(
        window_scores=window_scores,
        fused_window_results=fused_window_results,
        use_rear=USE_REAR,
        front_stable_to_tracks=front_stable_to_tracks,
        rear_stable_to_tracks=rear_stable_to_tracks,
    )

    run_dir = None
    if EXPORT_RESULTS:
        run_dir = make_run_dir()
        export_results_bundle(
            run_dir=run_dir,
            use_rear=USE_REAR,
            front_source=FRONT_SOURCE,
            rear_source=REAR_SOURCE,
            front_summary=front_summary,
            rear_summary=rear_summary,
            fused_window_results=fused_window_results,
            window_scores=window_scores,
            front_track_to_stable=front_result["track_to_stable"],
            rear_track_to_stable=rear_track_to_stable,
            front_stable_to_tracks=front_stable_to_tracks,
            rear_stable_to_tracks=rear_stable_to_tracks,
            rear_board_bboxes=rear_board_bboxes,
            front_frame_size=front_frame_size,
            rear_frame_size=rear_frame_size,
            front_scene_quad=front_scene_quad,
            rear_scene_quad=rear_scene_quad,
        )
        if ENABLE_MAPPING_DEBUG_VIDEO and USE_REAR:
            debug_video_path = run_dir / Path(DEBUG_VIDEO_PATH).name
            draw_debug_mapping_video(
                output_path=debug_video_path,
                front_source=FRONT_SOURCE,
                rear_source=REAR_SOURCE,
                fused_window_results=fused_window_results,
                front_frame_records=front_frame_records,
                rear_frame_records=rear_frame_records,
                front_to_canonical=front_to_canonical,
                canonical_to_rear=canonical_to_rear,
                window_seconds=WINDOW_SECONDS,
                max_frames=DEBUG_MAX_FRAMES,
            )
            print(f"[INFO] exported mapping debug video to: {debug_video_path}")
        print(f"[INFO] exported results to: {run_dir}")

    return {
        "front_summary": front_summary,
        "rear_summary": rear_summary,
        "fused_window_results": fused_window_results,
        "window_scores": window_scores,
        "front_track_to_stable": front_result["track_to_stable"],
        "rear_track_to_stable": rear_track_to_stable,
        "front_stable_to_tracks": front_stable_to_tracks,
        "rear_stable_to_tracks": rear_stable_to_tracks,
        "rear_board_bboxes": rear_board_bboxes,
        "front_frame_size": front_frame_size,
        "rear_frame_size": rear_frame_size,
        "front_scene_quad": front_scene_quad,
        "rear_scene_quad": rear_scene_quad,
        "export_dir": str(run_dir) if run_dir is not None else None,
    }


# =========================================================
# 6. Run
# =========================================================

results_all = main()
