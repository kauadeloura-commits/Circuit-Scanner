from __future__ import annotations

"""Treino YOLO-Pose dos terminais corrigidos de múltiplos componentes.

O modelo usa três slots de keypoint para manter uma cabeça única:
  * transistor: B, C, E
  * diodo/LED: A, K, ausente
  * resistor/capacitor: 1, 2, ausente
  * VCC: VCC, ausente, ausente

As divisões são feitas por project_id, portanto componentes do mesmo circuito
jamais aparecem simultaneamente no treino e na validação/teste.
"""

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import random
import shutil
from typing import Any

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "terminal_training_all_components" / "raw"
DATASET_DIR = BASE_DIR / "terminal_training_all_components" / "pose_dataset_7_classes"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "all_component_terminal_keypoints.pt"
SUMMARY_PATH = MODELS_DIR / "all_component_terminal_keypoints_summary.json"
BJT_MODEL_PATH = MODELS_DIR / "terminal_keypoints.pt"
BJT_SUMMARY_PATH = MODELS_DIR / "terminal_keypoints_summary.json"
BJT_DATASET_DIR = BASE_DIR / "terminal_training_all_components" / "pose_dataset_bjt_corrected"
_MODEL_CACHE: dict[str, Any] = {"signature": None, "model": None, "error": None}

CLASS_NAMES = (
    "vcc",
    "transistor_npn",
    "transistor_pnp",
    "diode",
    "led",
    "resistor",
    "capacitor",
)
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
KEYPOINT_SCHEMAS = {
    "vcc": ("VCC", None, None),
    "transistor_npn": ("B", "C", "E"),
    "transistor_pnp": ("B", "C", "E"),
    "diode": ("A", "K", None),
    "led": ("A", "K", None),
    "resistor": ("1", "2", None),
    "capacitor": ("1", "2", None),
}


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _split_for_project(project_id: str) -> str:
    # 70/15/15, estável entre execuções. O sal 0 foi auditado: todas as sete
    # classes possuem amostras nos três subconjuntos.
    bucket = int(hashlib.sha1(f"0|{project_id}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else ("val" if bucket < 85 else "test")


def _load_samples() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    problems: list[str] = []
    for class_name in CLASS_NAMES:
        for metadata_path in sorted((RAW_DIR / class_name).glob("*.json")):
            try:
                data = json.loads(metadata_path.read_text(encoding="utf-8"))
                image_path = metadata_path.with_name(str(data.get("crop_image") or f"{metadata_path.stem}.png"))
                if data.get("type") != class_name or not image_path.exists():
                    raise ValueError("tipo divergente ou recorte ausente")
                by_pin = {str(item.get("pin")): item for item in data.get("keypoints", [])}
                required = [pin for pin in KEYPOINT_SCHEMAS[class_name] if pin]
                if any(pin not in by_pin for pin in required):
                    raise ValueError("keypoint obrigatório ausente")
                samples.append({
                    "sample_id": str(data.get("sample_id") or metadata_path.stem),
                    "project_id": str(data.get("project_id") or "unknown"),
                    "class_name": class_name,
                    "image_path": image_path,
                    "metadata_path": metadata_path,
                    "metadata": data,
                })
            except Exception as exc:
                problems.append(f"{metadata_path}: {exc}")
    if problems:
        raise ValueError("Amostras inválidas:\n" + "\n".join(problems[:25]))
    return samples


def _audit_npn_emitter_rule(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Confere a regra nos ground truths sem reinterpretar símbolos rotacionados."""
    checked = 0
    by_side = Counter()
    violations: list[str] = []
    for sample in samples:
        if sample["class_name"] != "transistor_npn":
            continue
        data = sample["metadata"]
        points = {str(item["pin"]): item["global"] for item in data["keypoints"]}
        if not all(pin in points for pin in ("B", "C", "E")):
            continue
        bbox = data["bbox"]
        bw = max(1.0, float(bbox["x2"]) - float(bbox["x1"]))
        bh = max(1.0, float(bbox["y2"]) - float(bbox["y1"]))
        b, c, e = points["B"], points["C"], points["E"]
        dcx, dex = float(c["x"]) - float(b["x"]), float(e["x"]) - float(b["x"])
        clear_horizontal = (
            dcx * dex > 0.0
            and min(abs(dcx), abs(dex)) > 0.08 * bw
            and abs(float(c["y"]) - float(e["y"])) > 0.15 * bh
        )
        if not clear_horizontal:
            continue
        checked += 1
        side = "right" if (dcx + dex) > 0.0 else "left"
        by_side[side] += 1
        if float(e["y"]) <= float(c["y"]):
            violations.append(sample["sample_id"])
    return {
        "clear_horizontal_npn_checked": checked,
        "by_fork_side": dict(by_side),
        "rule": "E must be lower-right for right-facing NPN and lower-left for mirrored NPN",
        "violation_count": len(violations),
        "violation_sample_ids": violations,
        "healthy": not violations,
    }


def _yolo_label(sample: dict[str, Any]) -> str:
    data = sample["metadata"]
    crop = data["crop_bbox"]
    cw = max(1.0, float(crop["x2"]) - float(crop["x1"]))
    ch = max(1.0, float(crop["y2"]) - float(crop["y1"]))
    bbox = data["bbox"]
    bx1 = _clip((float(bbox["x1"]) - float(crop["x1"])) / cw)
    by1 = _clip((float(bbox["y1"]) - float(crop["y1"])) / ch)
    bx2 = _clip((float(bbox["x2"]) - float(crop["x1"])) / cw)
    by2 = _clip((float(bbox["y2"]) - float(crop["y1"])) / ch)
    by_pin = {str(item["pin"]): item for item in data["keypoints"]}

    points: list[tuple[float, float, int]] = []
    visible_xy: list[tuple[float, float]] = []
    for pin in KEYPOINT_SCHEMAS[sample["class_name"]]:
        if pin is None:
            points.append((0.0, 0.0, 0))
            continue
        point = by_pin[pin]["crop_normalized"]
        x, y = _clip(point["x"]), _clip(point["y"])
        points.append((x, y, 2))
        visible_xy.append((x, y))

    # A caixa inclui o corpo e todos os terminais. Isso evita keypoints fora da
    # caixa e torna o aprendizado mais estável nos símbolos com fios longos.
    xs = [bx1, bx2] + [p[0] for p in visible_xy]
    ys = [by1, by2] + [p[1] for p in visible_xy]
    x1, x2 = max(0.0, min(xs) - 0.015), min(1.0, max(xs) + 0.015)
    y1, y2 = max(0.0, min(ys) - 0.015), min(1.0, max(ys) + 0.015)
    fields = [
        str(CLASS_IDS[sample["class_name"]]),
        f"{(x1+x2)/2:.7f}", f"{(y1+y2)/2:.7f}",
        f"{max(0.01, x2-x1):.7f}", f"{max(0.01, y2-y1):.7f}",
    ]
    for x, y, visibility in points:
        fields.extend((f"{x:.7f}", f"{y:.7f}", str(visibility)))
    return " ".join(fields) + "\n"


def _photometric_variant(image: np.ndarray, seed: int) -> np.ndarray:
    """Aumentação que não move os pontos: luz, contraste, ruído e desfoque."""
    rng = np.random.default_rng(seed)
    alpha = float(rng.uniform(0.78, 1.24))
    beta = float(rng.uniform(-24.0, 24.0))
    out = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if rng.random() < 0.55:
        sigma = float(rng.uniform(1.0, 5.0))
        noise = rng.normal(0.0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.30:
        out = cv2.GaussianBlur(out, (3, 3), float(rng.uniform(0.25, 0.8)))
    return out


def build_dataset(*, clean: bool = True, balance_target: int = 60, resistor_cap: int = 120) -> dict:
    samples = _load_samples()
    npn_rule_audit = _audit_npn_emitter_rule(samples)
    if not npn_rule_audit["healthy"]:
        raise ValueError(
            "Há amostras NPN horizontais com C/E contrários à regra do emissor: "
            + ", ".join(npn_rule_audit["violation_sample_ids"][:20])
        )
    if clean and DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for split in ("train", "val", "test"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    original_counts = {split: Counter() for split in ("train", "val", "test")}
    project_splits: dict[str, str] = {}
    manifest: list[dict[str, Any]] = []
    for sample in samples:
        split = _split_for_project(sample["project_id"])
        project_splits[sample["project_id"]] = split
        grouped[f"{split}:{sample['class_name']}"] .append(sample)
        original_counts[split][sample["class_name"]] += 1

    rng = random.Random(20260924)
    written_counts = {split: Counter() for split in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            class_samples = list(grouped[f"{split}:{class_name}"])
            if split == "train" and class_name == "resistor" and len(class_samples) > resistor_cap:
                rng.shuffle(class_samples)
                class_samples = class_samples[:resistor_cap]
            copies = list(class_samples)
            if split == "train" and copies and len(copies) < balance_target:
                index = 0
                while len(copies) < balance_target:
                    copies.append(class_samples[index % len(class_samples)])
                    index += 1

            occurrence = Counter()
            for sample in copies:
                occurrence[sample["sample_id"]] += 1
                copy_index = occurrence[sample["sample_id"]] - 1
                suffix = "" if copy_index == 0 else f"_aug{copy_index:02d}"
                output_stem = f"{sample['sample_id']}{suffix}"
                image_out = DATASET_DIR / "images" / split / f"{output_stem}.png"
                label_out = DATASET_DIR / "labels" / split / f"{output_stem}.txt"
                image = cv2.imread(str(sample["image_path"]))
                if image is None:
                    raise ValueError(f"Não foi possível abrir {sample['image_path']}")
                augmented = copy_index > 0
                if augmented:
                    seed = int(hashlib.sha1(output_stem.encode()).hexdigest()[:8], 16)
                    image = _photometric_variant(image, seed)
                if not cv2.imwrite(str(image_out), image):
                    raise OSError(f"Não foi possível gravar {image_out}")
                label_out.write_text(_yolo_label(sample), encoding="utf-8")
                written_counts[split][class_name] += 1
                manifest.append({
                    "output": output_stem,
                    "sample_id": sample["sample_id"],
                    "project_id": sample["project_id"],
                    "class_name": class_name,
                    "split": split,
                    "augmented": augmented,
                })

    yaml_text = "\n".join([
        f"path: {DATASET_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "kpt_shape: [3, 3]",
        "flip_idx: [0, 1, 2]",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
        "",
    ])
    (DATASET_DIR / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (DATASET_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selected_classes": list(CLASS_NAMES),
        "keypoint_schemas": {key: list(value) for key, value in KEYPOINT_SCHEMAS.items()},
        "raw_sample_count": len(samples),
        "project_count": len(project_splits),
        "split_is_project_grouped": True,
        "original_counts": {split: dict(counts) for split, counts in original_counts.items()},
        "written_counts": {split: dict(counts) for split, counts in written_counts.items()},
        "balance_target": balance_target,
        "resistor_train_cap": resistor_cap,
        "npn_emitter_rule_audit": npn_rule_audit,
    }
    (DATASET_DIR / "dataset_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_bjt_dataset_from_corrections() -> dict[str, Any]:
    """Deriva o dataset BJT dedicado dos mesmos pontos corrigidos da interface."""
    if not (DATASET_DIR / "manifest.json").exists():
        build_dataset()
    if BJT_DATASET_DIR.exists():
        shutil.rmtree(BJT_DATASET_DIR)
    counts = {split: Counter() for split in ("train", "val", "test")}
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
    remap = {CLASS_IDS["transistor_npn"]: 0, CLASS_IDS["transistor_pnp"]: 1}
    for item in manifest:
        class_name = item["class_name"]
        if class_name not in {"transistor_npn", "transistor_pnp"}:
            continue
        split, stem = item["split"], item["output"]
        image_source = DATASET_DIR / "images" / split / f"{stem}.png"
        label_source = DATASET_DIR / "labels" / split / f"{stem}.txt"
        image_target = BJT_DATASET_DIR / "images" / split / image_source.name
        label_target = BJT_DATASET_DIR / "labels" / split / label_source.name
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_source, image_target)
        fields = label_source.read_text(encoding="utf-8").strip().split()
        fields[0] = str(remap[int(fields[0])])
        label_target.write_text(" ".join(fields) + "\n", encoding="utf-8")
        counts[split][class_name] += 1
    (BJT_DATASET_DIR / "data.yaml").write_text("\n".join([
        f"path: {BJT_DATASET_DIR.as_posix()}",
        "train: images/train", "val: images/val", "test: images/test",
        "kpt_shape: [3, 3]", "flip_idx: [0, 1, 2]", "names:",
        "  0: transistor_npn", "  1: transistor_pnp", "",
    ]), encoding="utf-8")
    report = {
        "source": "terminal_training_all_components/raw",
        "uses_user_corrected_points": True,
        "keypoint_order": ["B", "C", "E"],
        "counts": {split: dict(value) for split, value in counts.items()},
        "npn_emitter_rule_audit": json.loads(
            (DATASET_DIR / "dataset_summary.json").read_text(encoding="utf-8")
        )["npn_emitter_rule_audit"],
    }
    (BJT_DATASET_DIR / "dataset_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _metrics_dict(metrics: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    results = getattr(metrics, "results_dict", {}) or {}
    for key, value in results.items():
        try:
            output[str(key)] = float(value)
        except (TypeError, ValueError):
            output[str(key)] = str(value)
    return output


def _per_class_metrics(
    metrics: Any, class_names: tuple[str, ...] = CLASS_NAMES
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    box = getattr(metrics, "box", None)
    pose = getattr(metrics, "pose", None)
    for index, class_name in enumerate(class_names):
        item: dict[str, float] = {}
        for prefix, metric in (("box", box), ("pose", pose)):
            if metric is None:
                continue
            for field, attr in (("precision", "p"), ("recall", "r"), ("map50", "ap50"), ("map50_95", "maps")):
                values = getattr(metric, attr, None)
                try:
                    item[f"{prefix}_{field}"] = float(values[index])
                except (TypeError, ValueError, IndexError):
                    pass
        output[class_name] = item
    return output


def model_runtime_status(*, load_model: bool = False) -> dict[str, Any]:
    exists = MODEL_PATH.exists()
    info: dict[str, Any] = {
        "model_path": str(MODEL_PATH),
        "model_exists": exists,
        "selected_classes": list(CLASS_NAMES),
        "model_loadable": False,
    }
    if not exists:
        return info
    signature = (MODEL_PATH.stat().st_mtime_ns, MODEL_PATH.stat().st_size)
    if load_model and _MODEL_CACHE["signature"] != signature:
        try:
            from ultralytics import YOLO
            _MODEL_CACHE.update(signature=signature, model=YOLO(str(MODEL_PATH)), error=None)
        except Exception as exc:
            _MODEL_CACHE.update(signature=signature, model=None, error=f"{type(exc).__name__}: {exc}")
    info["model_loadable"] = bool(_MODEL_CACHE.get("model")) if load_model else True
    info["model_load_error"] = _MODEL_CACHE.get("error")
    return info


def _runtime_crop(image: np.ndarray, bbox: dict, pad_ratio: float) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(bbox[key]) for key in ("x1", "y1", "x2", "y2"))
    pad = max(8.0, max(x2 - x1, y2 - y1) * pad_ratio)
    cx1, cy1 = max(0, int(np.floor(x1 - pad))), max(0, int(np.floor(y1 - pad)))
    cx2, cy2 = min(width, int(np.ceil(x2 + pad)) + 1), min(height, int(np.ceil(y2 + pad)) + 1)
    return image[cy1:cy2, cx1:cx2], (cx1, cy1)


def predict_component_terminals(
    *, image: np.ndarray, bbox: dict, component_type: str,
    min_box_confidence: float = 0.28, min_keypoint_confidence: float = 0.30,
) -> dict[str, Any] | None:
    """Prediz pinos no quadro global; só aceita a classe esperada e pontos confiáveis."""
    if component_type not in CLASS_IDS or image is None or not MODEL_PATH.exists():
        return None
    status = model_runtime_status(load_model=True)
    model = _MODEL_CACHE.get("model")
    if not status.get("model_loadable") or model is None:
        return None
    expected_class = CLASS_IDS[component_type]
    expected_pins = [pin for pin in KEYPOINT_SCHEMAS[component_type] if pin]
    candidates: list[dict[str, Any]] = []
    # 0,24 reproduz a maioria dos recortes salvos. 0,40 recupera terminais
    # manuais mais afastados sem penalizar toda inferência com contexto extra.
    for pad_ratio in (0.24, 0.40):
        crop, (ox, oy) = _runtime_crop(image, bbox, pad_ratio)
        if crop.size == 0:
            continue
        result = model.predict(crop, imgsz=416, conf=0.12, iou=0.55, verbose=False)[0]
        if result.boxes is None or result.keypoints is None:
            continue
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        box_confidences = result.boxes.conf.detach().cpu().numpy()
        keypoint_xy = result.keypoints.xy.detach().cpu().numpy()
        keypoint_conf = result.keypoints.conf
        keypoint_conf = keypoint_conf.detach().cpu().numpy() if keypoint_conf is not None else None
        for index, predicted_class in enumerate(classes):
            if predicted_class != expected_class or index >= len(keypoint_xy):
                continue
            point_items: dict[str, dict[str, float]] = {}
            confidences: dict[str, float] = {}
            for point_index, pin in enumerate(expected_pins):
                x, y = keypoint_xy[index][point_index]
                conf = float(keypoint_conf[index][point_index]) if keypoint_conf is not None else float(box_confidences[index])
                point_items[pin] = {"x": round(float(x) + ox, 3), "y": round(float(y) + oy, 3)}
                confidences[pin] = conf
            mean_keypoint_conf = float(np.mean(list(confidences.values())))
            candidates.append({
                "points": point_items,
                "box_confidence": float(box_confidences[index]),
                "keypoint_confidence": mean_keypoint_conf,
                "keypoint_confidences": confidences,
                "pad_ratio": pad_ratio,
                "score": float(box_confidences[index]) * mean_keypoint_conf,
            })
        if candidates:
            break
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item["score"])
    accepted = bool(
        best["box_confidence"] >= min_box_confidence
        and best["keypoint_confidence"] >= min_keypoint_confidence
        and all(value >= 0.18 for value in best["keypoint_confidences"].values())
    )
    best.update({
        "accepted": accepted,
        "component_type": component_type,
        "source": "trained_all_component_terminal_model",
        "review_required": not accepted or best["keypoint_confidence"] < 0.55,
    })
    return best


def train_model(*, epochs: int = 80, imgsz: int = 512, batch: int = 8, patience: int = 20) -> dict:
    from ultralytics import YOLO

    dataset_summary = build_dataset()
    initial_model = BASE_DIR / "models" / "terminal_keypoints.pt"
    if not initial_model.exists():
        initial_model = Path("yolo11n-pose.pt")
    run_root = DATASET_DIR / "runs"
    model = YOLO(str(initial_model))
    result = model.train(
        data=str(DATASET_DIR / "data.yaml"),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        patience=int(patience),
        device="cpu",
        workers=0,
        project=str(run_root),
        name="terminals_7_classes",
        exist_ok=True,
        pretrained=True,
        seed=20260924,
        deterministic=True,
        cache=False,
        plots=True,
        verbose=True,
        degrees=8.0,
        translate=0.06,
        scale=0.20,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.15,
        close_mosaic=10,
    )
    save_dir = Path(result.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Treino terminou sem best.pt em {save_dir}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, MODEL_PATH)
    trained = YOLO(str(MODEL_PATH))
    val_metrics = trained.val(data=str(DATASET_DIR / "data.yaml"), split="val", device="cpu", workers=0, plots=True)
    test_metrics = trained.val(data=str(DATASET_DIR / "data.yaml"), split="test", device="cpu", workers=0, plots=True)
    summary = {
        "schema_version": "all-component-terminal-pose-1.0",
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_path": str(MODEL_PATH),
        "initial_model": str(initial_model),
        "classes": list(CLASS_NAMES),
        "keypoint_schemas": {key: list(value) for key, value in KEYPOINT_SCHEMAS.items()},
        "dataset": dataset_summary,
        "training": {"epochs_requested": epochs, "imgsz": imgsz, "batch": batch, "patience": patience, "save_dir": str(save_dir)},
        "validation_metrics": _metrics_dict(val_metrics),
        "validation_per_class": _per_class_metrics(val_metrics),
        "test_metrics": _metrics_dict(test_metrics),
        "test_per_class": _per_class_metrics(test_metrics),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def train_bjt_model_from_corrections(
    *, epochs: int = 40, imgsz: int = 416, batch: int = 16, patience: int = 12
) -> dict[str, Any]:
    from ultralytics import YOLO

    build_dataset()
    dataset_summary = build_bjt_dataset_from_corrections()
    if not BJT_MODEL_PATH.exists():
        raise FileNotFoundError("models/terminal_keypoints.pt não encontrado para ajuste fino.")
    backup = MODELS_DIR / "terminal_keypoints_before_npn_rule_v5.7.1.pt"
    if not backup.exists():
        shutil.copy2(BJT_MODEL_PATH, backup)
    model = YOLO(str(BJT_MODEL_PATH))
    result = model.train(
        data=str(BJT_DATASET_DIR / "data.yaml"), epochs=epochs, imgsz=imgsz,
        batch=batch, patience=patience, device="cpu", workers=0,
        project=str(BJT_DATASET_DIR / "runs"), name="bjt_corrected_v571",
        exist_ok=True, pretrained=True, seed=20260924, deterministic=True,
        cache=False, plots=True, verbose=True, degrees=8.0, translate=0.06,
        scale=0.20, fliplr=0.0, flipud=0.0, mosaic=0.10, close_mosaic=8,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Treino BJT terminou sem best.pt em {result.save_dir}")
    shutil.copy2(best, BJT_MODEL_PATH)
    trained = YOLO(str(BJT_MODEL_PATH))
    val = trained.val(data=str(BJT_DATASET_DIR / "data.yaml"), split="val", device="cpu", workers=0, plots=False)
    test = trained.val(data=str(BJT_DATASET_DIR / "data.yaml"), split="test", device="cpu", workers=0, plots=False)
    summary = {
        "schema_version": "bjt-corrected-terminal-pose-1.0",
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_path": str(BJT_MODEL_PATH),
        "backup_path": str(backup),
        "classes": ["transistor_npn", "transistor_pnp"],
        "keypoint_order": ["B", "C", "E"],
        "dataset": dataset_summary,
        "training": {"epochs_requested": epochs, "imgsz": imgsz, "batch": batch, "patience": patience, "save_dir": str(result.save_dir)},
        "validation_metrics": _metrics_dict(val),
        "validation_per_class": _per_class_metrics(val, ("transistor_npn", "transistor_pnp")),
        "test_metrics": _metrics_dict(test),
        "test_per_class": _per_class_metrics(test, ("transistor_npn", "transistor_pnp")),
    }
    BJT_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "train", "train-bjt"), nargs="?", default="build")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    if args.action == "build":
        result = build_dataset()
    elif args.action == "train-bjt":
        result = train_bjt_model_from_corrections(
            epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience
        )
    else:
        result = train_model(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
