"""Auditoria objetiva do detector de componentes e de suas anotações.

O módulo não altera imagens, labels nem pesos. Ele mede os problemas que antes
eram tratados por tentativa e erro: caixas grandes, caixas sobrepostas,
componentes pequenos, desequilíbrio de classes, robustez a cores e falsos
positivos. A execução usa o conjunto real de validação, nunca o treino.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import json
import math
import threading

import cv2
import numpy as np

from backend.real_dataset import (
    REAL_IMAGES_DIR,
    active_dataset_dirs,
    list_real_images,
    load_classes,
)
from backend.recognition import MODEL_PATH


BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_PATH = BASE_DIR / "models" / "component_diagnostic_report.json"
STATUS_PATH = BASE_DIR / "models" / "component_diagnostic_status.json"
_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp.replace(path)


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _intersection(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a, b):
    inter = _intersection(a, b)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


def audit_real_annotations():
    """Inspeciona todas as caixas reais sem depender do YOLO."""
    classes = {int(item["id"]): item["key"] for item in load_classes()}
    records = list_real_images()
    counts = Counter()
    area_ratios = defaultdict(list)
    width_pixels = defaultdict(list)
    height_pixels = defaultdict(list)
    issues = []
    image_sizes = []
    duplicate_pairs = 0
    overlapping_different_classes = 0

    for record in records:
        width = int(record["width"])
        height = int(record["height"])
        image_sizes.append((width, height))
        boxes = list(record.get("annotations") or [])
        image = None
        stored_filename = record.get("stored_filename")
        if stored_filename:
            image_path = REAL_IMAGES_DIR / stored_filename
            if image_path.exists():
                image = cv2.imdecode(
                    np.fromfile(str(image_path), dtype=np.uint8),
                    cv2.IMREAD_GRAYSCALE,
                )
        normalized = []
        for index, box in enumerate(boxes):
            class_id = int(box["class_id"])
            class_name = classes.get(class_id, f"class_{class_id}")
            x1, y1 = float(box["x1"]), float(box["y1"])
            x2, y2 = float(box["x2"]), float(box["y2"])
            bw, bh = x2 - x1, y2 - y1
            ratio = (bw * bh) / max(1.0, width * height)
            counts[class_name] += 1
            area_ratios[class_name].append(ratio)
            width_pixels[class_name].append(bw)
            height_pixels[class_name].append(bh)
            normalized.append((x1, y1, x2, y2, class_id, index))
            reasons = []
            if bw < 8 or bh < 8:
                reasons.append("box_too_small_in_pixels")
            if ratio < 0.00012:
                reasons.append("component_too_small_on_full_page")
            if ratio > 0.12:
                reasons.append("box_covers_more_than_12_percent_of_image")
            if bw / max(1.0, width) > 0.55 or bh / max(1.0, height) > 0.55:
                reasons.append("box_dimension_exceeds_55_percent_of_image")
            suggested_bbox = None
            ink_occupancy = None
            if image is not None:
                ix1 = max(0, min(width - 1, int(math.floor(x1))))
                iy1 = max(0, min(height - 1, int(math.floor(y1))))
                ix2 = max(ix1 + 1, min(width, int(math.ceil(x2))))
                iy2 = max(iy1 + 1, min(height, int(math.ceil(y2))))
                crop = image[iy1:iy2, ix1:ix2]
                if crop.size:
                    border = np.concatenate(
                        (crop[0], crop[-1], crop[:, 0], crop[:, -1])
                    )
                    background = float(np.median(border))
                    ink = crop.astype(np.float32) < background - 18.0
                    ys, xs = np.where(ink)
                    ink_occupancy = float(np.count_nonzero(ink)) / crop.size
                    if len(xs) >= 12:
                        tx1, tx2 = np.percentile(xs, (1.0, 99.0))
                        ty1, ty2 = np.percentile(ys, (1.0, 99.0))
                        content_w = max(1.0, tx2 - tx1)
                        content_h = max(1.0, ty2 - ty1)
                        if content_w < 0.62 * crop.shape[1] and content_h < 0.62 * crop.shape[0]:
                            reasons.append("box_has_excessive_empty_margin")
                            margin = max(3, int(round(0.06 * max(content_w, content_h))))
                            suggested_bbox = [
                                round(max(0.0, ix1 + tx1 - margin), 2),
                                round(max(0.0, iy1 + ty1 - margin), 2),
                                round(min(float(width), ix1 + tx2 + margin), 2),
                                round(min(float(height), iy1 + ty2 + margin), 2),
                            ]
            if reasons:
                issues.append({
                    "image_id": record["image_id"],
                    "box_index": index,
                    "class": class_name,
                    "bbox": [x1, y1, x2, y2],
                    "area_ratio": round(ratio, 6),
                    "ink_occupancy": (
                        None if ink_occupancy is None else round(ink_occupancy, 6)
                    ),
                    "suggested_tighter_bbox": suggested_bbox,
                    "reasons": reasons,
                })

        for first in range(len(normalized)):
            a = normalized[first]
            for second in range(first + 1, len(normalized)):
                b = normalized[second]
                overlap = _iou(a[:4], b[:4])
                if overlap >= 0.80:
                    duplicate_pairs += 1
                    issues.append({
                        "image_id": record["image_id"],
                        "box_indices": [a[5], b[5]],
                        "classes": [classes.get(a[4]), classes.get(b[4])],
                        "iou": round(overlap, 4),
                        "reasons": ["possible_duplicate_annotation"],
                    })
                elif overlap >= 0.35 and a[4] != b[4]:
                    overlapping_different_classes += 1
                    issues.append({
                        "image_id": record["image_id"],
                        "box_indices": [a[5], b[5]],
                        "classes": [classes.get(a[4]), classes.get(b[4])],
                        "iou": round(overlap, 4),
                        "reasons": ["different_classes_compete_in_annotation"],
                    })

    positive_counts = [value for value in counts.values() if value > 0]
    imbalance = (
        max(positive_counts) / max(1, min(positive_counts))
        if positive_counts else None
    )

    def stats(values):
        if not values:
            return None
        values = np.asarray(values, dtype=np.float64)
        return {
            "min": round(float(np.min(values)), 6),
            "median": round(float(np.median(values)), 6),
            "p90": round(float(np.percentile(values, 90)), 6),
            "max": round(float(np.max(values)), 6),
        }

    by_class = {}
    for class_name in classes.values():
        by_class[class_name] = {
            "instances": int(counts[class_name]),
            "area_ratio": stats(area_ratios[class_name]),
            "width_pixels": stats(width_pixels[class_name]),
            "height_pixels": stats(height_pixels[class_name]),
        }
    return {
        "image_count": len(records),
        "annotated_image_count": sum(
            1 for record in records if record.get("annotations")
        ),
        "reviewed_negative_image_count": sum(
            1 for record in records
            if record.get("annotations_reviewed") and not record.get("annotations")
        ),
        "instance_count": int(sum(counts.values())),
        "instances_by_class": dict(sorted(counts.items())),
        "class_imbalance_max_over_min": (
            None if imbalance is None else round(float(imbalance), 3)
        ),
        "possible_duplicate_annotation_pairs": duplicate_pairs,
        "overlapping_different_class_pairs": overlapping_different_classes,
        "issue_count": len(issues),
        "issues": issues[:1000],
        "issues_truncated": len(issues) > 1000,
        "box_statistics_by_class": by_class,
        "image_width": stats([item[0] for item in image_sizes]),
        "image_height": stats([item[1] for item in image_sizes]),
    }


def _load_yolo_labels(label_path: Path, width: int, height: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:])
        boxes.append({
            "class_id": class_id,
            "bbox": (
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            ),
        })
    return boxes


def _colorize_ink(image, bgr):
    """Mantém papel/iluminação e troca somente a tinta escura de cor."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    paper = float(np.median(border))
    darkness = np.clip((paper - gray.astype(np.float32)) / max(45.0, paper), 0, 1)
    strength = np.power(darkness, 0.85)[..., None]
    color = np.asarray(bgr, dtype=np.float32).reshape(1, 1, 3)
    base = image.astype(np.float32)
    out = base * (1.0 - strength) + color * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def _prediction_boxes(result):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    return [
        {
            "class_id": int(class_id),
            "confidence": float(confidence),
            "bbox": tuple(map(float, coords)),
        }
        for coords, class_id, confidence in zip(xyxy, classes, confidences)
    ]


def _score_predictions(predictions, ground_truth, iou_threshold=0.50):
    matched_gt = set()
    tp = 0
    fp = 0
    by_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for prediction in sorted(
        predictions, key=lambda item: item["confidence"], reverse=True
    ):
        best_index = None
        best_iou = 0.0
        for index, target in enumerate(ground_truth):
            if index in matched_gt or target["class_id"] != prediction["class_id"]:
                continue
            overlap = _iou(prediction["bbox"], target["bbox"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index is not None and best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_index)
            by_class[int(prediction["class_id"])]["tp"] += 1
        else:
            fp += 1
            by_class[int(prediction["class_id"])]["fp"] += 1
    fn = len(ground_truth) - len(matched_gt)
    for index, target in enumerate(ground_truth):
        if index not in matched_gt:
            by_class[int(target["class_id"])]["fn"] += 1
    return tp, fp, fn, dict(by_class)


def evaluate_pen_colors(max_images=60, confidence=0.25, imgsz=640):
    """Mede TP/FP/FN em preto, azul, vermelho e verde no holdout real."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modelo ativo não encontrado.")
    try:
        from ultralytics import YOLO
    except Exception as error:
        raise RuntimeError("A biblioteca ultralytics não está instalada.") from error
    _, real_eval_dir = active_dataset_dirs()
    image_dir = real_eval_dir / "images" / "val"
    label_dir = real_eval_dir / "labels" / "val"
    candidates = [
        path for path in sorted(image_dir.glob("*"))
        if path.is_file()
        and "_mono_" not in path.stem
        and "_focus_" not in path.stem
        and "_background_" not in path.stem
    ][:max(1, min(250, int(max_images)))]
    if not candidates:
        raise ValueError("Não há imagens reais originais no split de validação.")
    colors = {
        "black": (15, 15, 15),
        "blue": (190, 70, 25),
        "red": (35, 35, 190),
        "green": (45, 145, 45),
    }
    totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in colors}
    model = YOLO(str(MODEL_PATH))
    model_names = getattr(model, "names", {}) or {}
    if isinstance(model_names, (list, tuple)):
        model_names = {index: name for index, name in enumerate(model_names)}
    per_color_class = {name: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}) for name in colors}
    for image_path in candidates:
        image = cv2.imdecode(
            np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            continue
        h, w = image.shape[:2]
        targets = _load_yolo_labels(label_dir / f"{image_path.stem}.txt", w, h)
        names = list(colors)
        variants = [_colorize_ink(image, colors[name]) for name in names]
        results = model.predict(
            source=variants,
            conf=float(confidence),
            iou=0.55,
            imgsz=int(imgsz),
            verbose=False,
        )
        for name, result in zip(names, results):
            tp, fp, fn, by_class = _score_predictions(
                _prediction_boxes(result), targets
            )
            totals[name]["tp"] += tp
            totals[name]["fp"] += fp
            totals[name]["fn"] += fn
            for class_id, values in by_class.items():
                class_name = str(model_names.get(int(class_id), class_id))
                for key in ("tp", "fp", "fn"):
                    per_color_class[name][class_name][key] += int(values[key])

    for metrics in totals.values():
        tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        metrics.update({
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(
                2 * precision * recall / max(1e-9, precision + recall), 6
            ),
        })
    for color_name, class_totals in per_color_class.items():
        rendered = {}
        for class_name, values in sorted(class_totals.items()):
            tp, fp, fn = values["tp"], values["fp"], values["fn"]
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            rendered[class_name] = {
                **values,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(
                    2 * precision * recall / max(1e-9, precision + recall), 6
                ),
            }
        totals[color_name]["by_class"] = rendered
    f1_values = [item["f1"] for item in totals.values()]
    return {
        "image_count": len(candidates),
        "confidence": float(confidence),
        "imgsz": int(imgsz),
        "metrics_by_pen_color": totals,
        "f1_spread_max_minus_min": round(max(f1_values) - min(f1_values), 6),
    }


def _compare_reports(previous, current):
    if not previous:
        return None
    comparison = {
        "annotation_issue_delta": (
            int(current["dataset_audit"]["issue_count"])
            - int(previous.get("dataset_audit", {}).get("issue_count", 0))
        ),
        "pen_color_f1_delta": {},
    }
    old_colors = previous.get("pen_color_evaluation", {}).get(
        "metrics_by_pen_color", {}
    )
    new_colors = current.get("pen_color_evaluation", {}).get(
        "metrics_by_pen_color", {}
    )
    for color, metrics in new_colors.items():
        if color in old_colors:
            comparison["pen_color_f1_delta"][color] = round(
                float(metrics.get("f1", 0.0))
                - float(old_colors[color].get("f1", 0.0)),
                6,
            )
    return comparison


def _diagnostic_worker(config):
    status = {
        "state": "running",
        "started_at": _now(),
        "message": "Auditando labels, caixas, modelo e cores de caneta...",
        **config,
    }
    _write_json(STATUS_PATH, status)
    try:
        previous_report = _read_json(REPORT_PATH)
        report = {
            "created_at": _now(),
            "dataset_audit": audit_real_annotations(),
            "pen_color_evaluation": evaluate_pen_colors(**config),
            "interpretation": {
                "annotation_issues_must_be_fixed_before_retraining": True,
                "acceptable_color_f1_spread": 0.08,
                "duplicate_annotation_iou_threshold": 0.80,
                "different_class_overlap_warning_iou": 0.35,
            },
        }
        report["before_after_comparison"] = _compare_reports(
            previous_report, report
        )
        _write_json(REPORT_PATH, report)
        status.update({
            "state": "completed",
            "completed_at": _now(),
            "message": "Auditoria completa concluída.",
            "report": report,
        })
    except Exception as error:
        status.update({
            "state": "failed",
            "completed_at": _now(),
            "message": str(error),
        })
    finally:
        _write_json(STATUS_PATH, status)


def start_component_diagnostic(max_images=60, confidence=0.25, imgsz=640):
    global _THREAD
    config = {
        "max_images": max(1, min(250, int(max_images))),
        "confidence": max(0.05, min(0.95, float(confidence))),
        "imgsz": max(320, min(1280, int(imgsz))),
    }
    with _LOCK:
        if _THREAD and _THREAD.is_alive():
            raise RuntimeError("Já existe uma auditoria em andamento.")
        _THREAD = threading.Thread(
            target=_diagnostic_worker,
            args=(config,),
            daemon=True,
            name="component-diagnostic",
        )
        _THREAD.start()
    return {"state": "running", **config}


def component_diagnostic_status():
    status = _read_json(
        STATUS_PATH,
        {"state": "idle", "message": "Nenhuma auditoria foi executada."},
    )
    with _LOCK:
        alive = bool(_THREAD and _THREAD.is_alive())
    if status.get("state") == "running" and not alive:
        status = dict(status)
        status["state"] = "interrupted"
        status["message"] = "A auditoria foi interrompida pelo reinício do servidor."
    if REPORT_PATH.exists() and "report" not in status:
        status["last_report"] = _read_json(REPORT_PATH)
    return status
