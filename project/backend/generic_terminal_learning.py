from __future__ import annotations

from datetime import datetime
from hashlib import sha1
from pathlib import Path
import json
import re

import cv2


BASE_DIR = Path(__file__).resolve().parent.parent
GENERIC_TERMINAL_DATA_DIR = BASE_DIR / "terminal_training_all_components"
GENERIC_RAW_DIR = GENERIC_TERMINAL_DATA_DIR / "raw"
GENERIC_REGISTRY_PATH = GENERIC_TERMINAL_DATA_DIR / "classes.json"


def _ensure_dirs() -> None:
    GENERIC_RAW_DIR.mkdir(parents=True, exist_ok=True)


def _safe_name(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "unknown")).strip("._")
    return clean or "unknown"


def _read_registry() -> dict:
    _ensure_dirs()
    if not GENERIC_REGISTRY_PATH.exists():
        return {"schema_version": "all-component-terminals-1.0", "classes": {}}
    try:
        data = json.loads(GENERIC_REGISTRY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("classes"), dict):
            return data
    except Exception:
        pass
    return {"schema_version": "all-component-terminals-1.0", "classes": {}}


def _update_registry(component_type: str, pin_schema: list[dict]) -> None:
    data = _read_registry()
    classes = data.setdefault("classes", {})
    item = classes.setdefault(component_type, {"sample_count": 0, "pin_schema_variants": []})
    signature = [{"pin": str(p["pin"]), "role": str(p.get("role") or p["pin"])} for p in pin_schema]
    if signature not in item["pin_schema_variants"]:
        item["pin_schema_variants"].append(signature)
    class_dir = GENERIC_RAW_DIR / _safe_name(component_type)
    item["sample_count"] = len(list(class_dir.glob("*.json"))) if class_dir.exists() else 0
    item["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    data["updated_at"] = item["updated_at"]
    GENERIC_REGISTRY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_generic_terminal_training_sample(
    *,
    image_path: Path,
    project_id: str,
    component: dict,
    points: dict,
    pin_schema: list[dict],
) -> dict:
    """Salva ground truth de terminais para qualquer classe de componente.

    O ID e estavel por projeto/componente/bbox. Corrigir novamente o mesmo
    componente substitui a amostra anterior em vez de duplicar um erro no
    futuro conjunto de treino.
    """
    _ensure_dirs()
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel abrir {image_path}.")
    h, w = image.shape[:2]
    component_type = str(component.get("type") or "unknown")
    component_id = str(component.get("component_id") or component.get("detection_id") or "COMP")
    bbox = component.get("bbox") or {}
    try:
        bx1, by1, bx2, by2 = map(float, (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
    except Exception as exc:
        raise ValueError(f"{component_id}: bbox invalida para amostra de terminais.") from exc
    if bx2 <= bx1 or by2 <= by1:
        raise ValueError(f"{component_id}: bbox vazia para amostra de terminais.")

    clean_points = {}
    for item in pin_schema:
        pin = str(item.get("pin") or "").strip()
        if not pin or pin not in points:
            raise ValueError(f"{component_id}: ponto ausente para o pino {pin or '?'}.")
        try:
            x, y = float(points[pin]["x"]), float(points[pin]["y"])
        except Exception as exc:
            raise ValueError(f"{component_id}.{pin}: coordenada invalida.") from exc
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(f"{component_id}.{pin}: ponto fora da imagem.")
        clean_points[pin] = {"x": round(x, 3), "y": round(y, 3)}

    pad = max(6, int(round(max(bx2 - bx1, by2 - by1) * 0.18)))
    point_xs = [float(p["x"]) for p in clean_points.values()]
    point_ys = [float(p["y"]) for p in clean_points.values()]
    x1 = max(0, min(int(bx1) - pad, int(min(point_xs)) - 8))
    y1 = max(0, min(int(by1) - pad, int(min(point_ys)) - 8))
    x2 = min(w, max(int(round(bx2)) + pad + 1, int(max(point_xs)) + 9))
    y2 = min(h, max(int(round(by2)) + pad + 1, int(max(point_ys)) + 9))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"{component_id}: recorte vazio para amostra de terminais.")

    stable_key = json.dumps(
        {"project_id": project_id, "component_id": component_id, "type": component_type, "bbox": bbox},
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = sha1(stable_key.encode("utf-8")).hexdigest()[:14]
    sample_id = f"{_safe_name(component_type)}_{digest}"
    class_dir = GENERIC_RAW_DIR / _safe_name(component_type)
    class_dir.mkdir(parents=True, exist_ok=True)
    image_out = class_dir / f"{sample_id}.png"
    metadata_out = class_dir / f"{sample_id}.json"
    if not cv2.imwrite(str(image_out), crop):
        raise OSError(f"Nao foi possivel salvar {image_out.name}.")

    cw, ch = max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))
    bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
    annotations = []
    for order, schema_item in enumerate(pin_schema, start=1):
        pin = str(schema_item["pin"])
        p = clean_points[pin]
        annotations.append({
            "pin": pin,
            "role": str(schema_item.get("role") or pin),
            "order": order,
            "global": p,
            "crop_normalized": {
                "x": round((p["x"] - x1) / cw, 7),
                "y": round((p["y"] - y1) / ch, 7),
            },
            "bbox_normalized": {
                "x": round((p["x"] - bx1) / bw, 7),
                "y": round((p["y"] - by1) / bh, 7),
            },
            "visibility": 2,
        })

    payload = {
        "schema_version": "all-component-terminals-1.0",
        "sample_id": sample_id,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_id": project_id,
        "component_id": component_id,
        "type": component_type,
        "source_image": image_path.name,
        "crop_image": image_out.name,
        "image_size": {"width": w, "height": h},
        "bbox": {"x1": bx1, "y1": by1, "x2": bx2, "y2": by2},
        "crop_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "pin_schema": [{"pin": str(p["pin"]), "role": str(p.get("role") or p["pin"])} for p in pin_schema],
        "keypoints": annotations,
        "ground_truth": True,
        "source": "user_terminal_correction",
    }
    metadata_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _update_registry(component_type, payload["pin_schema"])
    return {
        "sample_id": sample_id,
        "type": component_type,
        "image": str(image_out),
        "metadata": str(metadata_out),
        "pin_count": len(pin_schema),
    }


def generic_terminal_learning_status() -> dict:
    data = _read_registry()
    by_class = {}
    total = 0
    schema_conflicts = []
    for component_type, item in sorted(data.get("classes", {}).items()):
        class_dir = GENERIC_RAW_DIR / _safe_name(component_type)
        count = len(list(class_dir.glob("*.json"))) if class_dir.exists() else 0
        variants = item.get("pin_schema_variants") or []
        by_class[component_type] = {"sample_count": count, "pin_schema_variants": variants}
        total += count
        if len(variants) > 1:
            schema_conflicts.append(component_type)
    return {
        "schema_version": "all-component-terminals-1.0",
        "dataset_dir": str(GENERIC_TERMINAL_DATA_DIR),
        "sample_count": total,
        "class_count": len(by_class),
        "samples_by_class": by_class,
        "classes_with_multiple_pin_schemas": schema_conflicts,
        "ready_for_future_training": total > 0 and not schema_conflicts,
    }


_ensure_dirs()
