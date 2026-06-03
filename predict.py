from ultralytics import YOLO
import cv2
import os
import math
from collections import defaultdict, Counter

# =========================================================
# 1. 只需要修改这里
# =========================================================

USE_REAR = False   # True: 启用 rear；False: 只运行 front

FRONT_MODEL_PATH = r"runs/detect/runs/Front/front_yolo26m/weights/best.pt"
REAR_MODEL_PATH  = r"runs/detect/runs/Rear/rear_yolo26m/weights/best.pt"

FRONT_SOURCE = r"I:/ProgramData/课堂专注度项目/P1_YOLO_Pose_Attention/学生全景_10.30.04-p01-80.mp4"
REAR_SOURCE  = r""

# front 最终类别
FRONT_CLASSES = ['discuss', 'hand-raising', 'read', 'write', 'teacher', 'stand', 'TurnHead']

# rear 最终类别
REAR_CLASSES = ['discuss', 'hand-raising', 'teacher', 'stand', 'screen', 'blackBoard', 'BowHead', 'TurnHead']

# 推理参数
CONF_THRES = 0.2
IOU_THRES = 0.5
WINDOW_SECONDS = 5
HOLD_FRAMES = 10
SHOW_WINDOW = True
TRACKER_YAML = "bytetrack.yaml"

# stable_id 融合参数
STABLE_MAX_GAP_FRAMES = 15
STABLE_MAX_CENTER_DIST = 80
STABLE_MIN_SIZE_RATIO = 0.5
INACTIVE_CLEANUP_FRAMES = 60


# =========================================================
# 2. 工具函数
# =========================================================

def get_video_fps(video_path, default_fps=25):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] 无法读取视频 FPS，使用默认 FPS={default_fps}: {video_path}")
        return default_fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 1:
        return default_fps
    return fps


def validate_path(path, name):
    if path is None or str(path).strip() == "":
        raise ValueError(f"{name} 为空，请检查路径设置。")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} 不存在: {path}")


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


def find_matching_stable_id(
    new_bbox,
    frame_idx,
    inactive_tracks,
    max_gap_frames=15,
    max_dist=80,
    min_size_ratio=0.5
):
    """
    为新的 track_id 寻找可融合的 stable_id
    """
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
# 3. 单视角跟踪 + stable_id 融合
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
    tracker_yaml="bytetrack.yaml"
):
    """
    对单个视角运行跟踪 + stable_id 融合 + 时间窗统计

    返回:
        {
            "summary": ...,
            "events": ...,
            "fps": ...,
            "frame_count": ...,
            "track_to_stable": ...,
            "stable_to_tracks": ...
        }
    """
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
        verbose=False
    )

    frame_idx = 0

    # 原始 track_id -> 最新信息
    track_memory = {}

    # 原始 track_id -> stable_id
    track_to_stable = {}

    # stable_id -> set(track_id)
    stable_to_tracks = defaultdict(set)

    # 刚消失的旧轨迹，用于后续融合
    inactive_tracks = {}

    stable_id_counter = 0

    # window_idx -> stable_id -> [cls1, cls2, ...]
    window_events = defaultdict(lambda: defaultdict(list))

    for result in results:
        frame = result.orig_img.copy()
        current_time = frame_idx / fps
        window_idx = int(current_time // window_seconds)

        current_ids = set()
        boxes = result.boxes

        # 清理太旧的 inactive 轨迹
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

                # 为新的原始 track_id 分配 stable_id
                if tid not in track_to_stable:
                    matched_sid = find_matching_stable_id(
                        new_bbox=bbox,
                        frame_idx=frame_idx,
                        inactive_tracks=inactive_tracks,
                        max_gap_frames=STABLE_MAX_GAP_FRAMES,
                        max_dist=STABLE_MAX_CENTER_DIST,
                        min_size_ratio=STABLE_MIN_SIZE_RATIO
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
                    "stable_id": sid
                }

                # 按 stable_id 统计
                window_events[window_idx][sid].append(cls_name)

                # 显示：原始 track_id + stable_id
                label = f"{view_name} T:{tid} S:{sid} {cls_name} {conf_val:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                )

        # 对本帧未出现的 track 处理
        for tid, info in list(track_memory.items()):
            if tid not in current_ids:
                missing = frame_idx - info["last_seen"]

                # 短时保留显示
                if missing <= hold_frames:
                    x1, y1, x2, y2 = info["bbox"]
                    sid = info["stable_id"]
                    label = f"{view_name} T:{tid} S:{sid} HOLD {info['cls']}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
                    cv2.putText(
                        frame, label, (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1
                    )
                else:
                    # 转入 inactive，等待后续新 track 来接 stable_id
                    inactive_tracks[tid] = {
                        "stable_id": info["stable_id"],
                        "bbox": info["bbox"],
                        "last_seen": info["last_seen"],
                        "last_cls": info["cls"]
                    }
                    del track_memory[tid]

        cv2.putText(
            frame,
            f"{view_name} | t={current_time:.1f}s | window={window_idx}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        if show_window:
            cv2.imshow(f"{view_name}_tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print(f"[INFO] 手动停止 {view_name} 处理")
                break

        frame_idx += 1

    if show_window:
        cv2.destroyWindow(f"{view_name}_tracking")

    # 汇总每个时间窗中每个 stable_id 的主行为
    summary = defaultdict(dict)
    for win_idx, stable_dict in window_events.items():
        for sid, cls_list in stable_dict.items():
            counter = Counter(cls_list)
            main_cls = counter.most_common(1)[0][0]
            summary[win_idx][sid] = {
                "main_cls": main_cls,
                "counts": dict(counter)
            }

    return {
        "summary": dict(summary),
        "events": dict(window_events),
        "fps": fps,
        "frame_count": frame_idx,
        "track_to_stable": dict(track_to_stable),
        "stable_to_tracks": {sid: sorted(list(tids)) for sid, tids in stable_to_tracks.items()}
    }


# =========================================================
# 4. 决策与融合逻辑
# =========================================================

def front_base_decision(front_cls):
    """
    front 单独运行时的基础判断
    返回:
        score, label
    """
    if front_cls == "hand-raising":
        return 3, "high_focus"
    elif front_cls in ["read", "write"]:
        return 2, "focused"
    elif front_cls == "discuss":
        return 0, "neutral"
    elif front_cls == "TurnHead":
        return -1, "suspicious"
    elif front_cls in ["teacher", "stand"]:
        return 0, "ignore"
    else:
        return 0, "normal"


def get_window_main_rear_cls(rear_track_dict):
    """
    从 rear 某个时间窗中提取一个主要辅助类别
    忽略 teacher / stand / screen / blackBoard
    """
    all_cls = []
    for _, info in rear_track_dict.items():
        cls_name = info["main_cls"]
        if cls_name not in ["teacher", "stand", "screen", "blackBoard"]:
            all_cls.append(cls_name)

    if not all_cls:
        return None

    counter = Counter(all_cls)
    return counter.most_common(1)[0][0]


def refine_with_rear(front_cls, rear_cls, base_score, base_label):
    """
    rear 只做增强修正，不替代 front
    """
    score, label = base_score, base_label

    if front_cls in ["read", "write", "hand-raising"]:
        return score, label

    if front_cls == "TurnHead":
        if rear_cls is None:
            return score, label
        elif rear_cls == "TurnHead":
            return -2, "low_focus"
        elif rear_cls == "BowHead":
            return -1, "suspicious"
        else:
            return 0, "normal_turn"

    if front_cls == "discuss":
        if rear_cls in ["BowHead", "TurnHead"]:
            return -1, "suspicious"
        return score, label

    if rear_cls == "BowHead":
        return -2, "low_focus"

    return score, label


def fuse_results(front_summary, rear_summary=None, use_rear=False):
    """
    自动根据 use_rear 决定运行 front-only 还是 front+rear
    注意：这里的键已经是 stable_id
    """
    fused_window_results = {}
    all_windows = sorted(set(front_summary.keys()) | set((rear_summary or {}).keys()))

    for win_idx in all_windows:
        front_track_dict = front_summary.get(win_idx, {})
        rear_track_dict = (rear_summary or {}).get(win_idx, {})

        rear_main_cls = None
        if use_rear:
            rear_main_cls = get_window_main_rear_cls(rear_track_dict)

        track_results = {}

        for sid, info in front_track_dict.items():
            front_cls = info["main_cls"]
            base_score, base_label = front_base_decision(front_cls)

            if use_rear:
                final_score, final_label = refine_with_rear(
                    front_cls=front_cls,
                    rear_cls=rear_main_cls,
                    base_score=base_score,
                    base_label=base_label
                )
            else:
                final_score, final_label = base_score, base_label

            track_results[sid] = {
                "front_cls": front_cls,
                "rear_cls": rear_main_cls if use_rear else None,
                "base_score": base_score,
                "base_label": base_label,
                "final_score": final_score,
                "final_label": final_label
            }

        fused_window_results[win_idx] = track_results

    return fused_window_results


def calc_window_scores(fused_window_results):
    window_scores = {}
    for win_idx, track_dict in fused_window_results.items():
        scores = []
        for _, info in track_dict.items():
            scores.append(info["final_score"])
        window_scores[win_idx] = sum(scores) / len(scores) if scores else 0
    return window_scores


def print_final_report(
    window_scores,
    fused_window_results,
    use_rear,
    front_stable_to_tracks=None,
    rear_stable_to_tracks=None
):
    mode_name = "front + rear" if use_rear else "front-only"
    print("\n" + "=" * 60)
    print(f"[RESULT] 运行模式: {mode_name}")
    print("=" * 60)

    for win_idx in sorted(window_scores.keys()):
        print(f"window {win_idx:03d}: score = {window_scores[win_idx]:.2f}")

    overall = sum(window_scores.values()) / len(window_scores) if window_scores else 0
    print("-" * 60)
    print(f"Overall class attention score: {overall:.2f}")
    print("=" * 60)

    print("\n[DETAIL] 每个 window 的 stable_id 融合结果:")
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
                f"| rear={str(info['rear_cls']):<10} "
                f"| final={info['final_label']:<12} "
                f"| score={info['final_score']}"
            )


# =========================================================
# 5. 主流程：全自动运行
# =========================================================

def main():
    validate_path(FRONT_MODEL_PATH, "FRONT_MODEL_PATH")
    validate_path(FRONT_SOURCE, "FRONT_SOURCE")

    if USE_REAR:
        validate_path(REAR_MODEL_PATH, "REAR_MODEL_PATH")
        validate_path(REAR_SOURCE, "REAR_SOURCE")

    print("[INFO] 正在加载 front 模型...")
    front_model = YOLO(FRONT_MODEL_PATH)

    rear_model = None
    if USE_REAR:
        print("[INFO] 正在加载 rear 模型...")
        rear_model = YOLO(REAR_MODEL_PATH)

    print("[INFO] 开始运行 front 视角...")
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
        tracker_yaml=TRACKER_YAML
    )

    front_summary = front_result["summary"]
    front_stable_to_tracks = front_result["stable_to_tracks"]

    rear_summary = {}
    rear_track_to_stable = {}
    rear_stable_to_tracks = {}

    if USE_REAR and rear_model is not None:
        print("[INFO] 开始运行 rear 视角...")
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
            tracker_yaml=TRACKER_YAML
        )
        rear_summary = rear_result["summary"]
        rear_track_to_stable = rear_result["track_to_stable"]
        rear_stable_to_tracks = rear_result["stable_to_tracks"]
    else:
        print("[INFO] rear 未启用，自动使用 front-only 模式。")

    fused_window_results = fuse_results(
        front_summary=front_summary,
        rear_summary=rear_summary,
        use_rear=USE_REAR
    )

    window_scores = calc_window_scores(fused_window_results)

    print_final_report(
        window_scores=window_scores,
        fused_window_results=fused_window_results,
        use_rear=USE_REAR,
        front_stable_to_tracks=front_stable_to_tracks,
        rear_stable_to_tracks=rear_stable_to_tracks
    )

    return {
        "front_summary": front_summary,
        "rear_summary": rear_summary,
        "fused_window_results": fused_window_results,
        "window_scores": window_scores,
        "front_track_to_stable": front_result["track_to_stable"],
        "rear_track_to_stable": rear_track_to_stable,
        "front_stable_to_tracks": front_stable_to_tracks,
        "rear_stable_to_tracks": rear_stable_to_tracks
    }


# =========================================================
# 6. 执行
# =========================================================

results_all = main()