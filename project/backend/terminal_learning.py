from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from typing import Any

import cv2
import numpy as np
from backend.resistor_geometry import cached_vision,session_result

@cached_vision
def _training_gray(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image.copy()
    return cv2.normalize(cv2.GaussianBlur(gray,(3,3),0),None,0,255,cv2.NORM_MINMAX)

@cached_vision
def _training_features(gray,method):
    detector=(cv2.SIFT_create(nfeatures=450,contrastThreshold=0.02,edgeThreshold=12)
              if method=='SIFT' else cv2.ORB_create(nfeatures=700,fastThreshold=8))
    points,descriptors=detector.detectAndCompute(gray,None)
    return np.asarray([point.pt for point in points],np.float32),descriptors

BASE_DIR = Path(__file__).resolve().parent.parent
TERMINAL_DATA_DIR = BASE_DIR / "terminal_keypoints"
RAW_IMAGES_DIR = TERMINAL_DATA_DIR / "raw" / "images"
RAW_LABELS_DIR = TERMINAL_DATA_DIR / "raw" / "labels"
RAW_META_DIR = TERMINAL_DATA_DIR / "raw" / "meta"
POSE_DATASET_DIR = TERMINAL_DATA_DIR / "pose_dataset"
MODELS_DIR = BASE_DIR / "models"
PROJECTS_DIR = BASE_DIR / "projects"
TERMINAL_MODEL_PATH = MODELS_DIR / "terminal_keypoints.pt"
TRAINING_SUMMARY_PATH = MODELS_DIR / "terminal_keypoints_summary.json"

TRANSISTOR_CLASSES = {
    "transistor_npn": 0,
    "transistor_pnp": 1,
}
TRANSISTOR_CLASS_NAMES = {v: k for k, v in TRANSISTOR_CLASSES.items()}
KEYPOINT_ORDER = ("B", "C", "E")

for directory in (RAW_IMAGES_DIR, RAW_LABELS_DIR, RAW_META_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

_MODEL_CACHE: dict[str, Any] = {"signature": None, "model": None, "load_error": None, "loaded_at": None}
_VISUAL_PRIOR_CACHE: dict[str, Any] = {"signature": None, "entries": []}


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _crop_bounds(image: np.ndarray, bbox: dict, pad_ratio: float = 0.28) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    x1, y1 = float(bbox["x1"]), float(bbox["y1"])
    x2, y2 = float(bbox["x2"]), float(bbox["y2"])
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad = max(10.0, min(90.0, max(bw, bh) * pad_ratio))
    cx1 = max(0, int(np.floor(x1 - pad)))
    cy1 = max(0, int(np.floor(y1 - pad)))
    cx2 = min(w, int(np.ceil(x2 + pad)) + 1)
    cy2 = min(h, int(np.ceil(y2 + pad)) + 1)
    return cx1, cy1, cx2, cy2


def _training_crop_bounds(
    image: np.ndarray,
    bbox: dict,
    corrected_points: dict[str, tuple[float, float]],
) -> tuple[int, int, int, int]:
    """Recorte que contém bbox + TODOS os keypoints manuais com folga.

    A versão antiga podia cortar C/E quando o terminal ficava fora da bbox do
    detector e depois normalizar o label para 0/1. Isso contaminava o treino.
    """
    h, w = image.shape[:2]
    xs = [float(bbox["x1"]), float(bbox["x2"])] + [float(p[0]) for p in corrected_points.values()]
    ys = [float(bbox["y1"]), float(bbox["y2"])] + [float(p[1]) for p in corrected_points.values()]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    span = max(1.0, x2-x1, y2-y1)
    pad = max(12.0, min(120.0, span * 0.22))
    return (
        max(0, int(np.floor(x1-pad))),
        max(0, int(np.floor(y1-pad))),
        min(w, int(np.ceil(x2+pad))+1),
        min(h, int(np.ceil(y2+pad))+1),
    )


def _audit_raw_samples(samples: list[dict]) -> dict:
    edge_hits = 0
    invalid_labels = 0
    checked = 0
    per_pin_edge = {pin: 0 for pin in KEYPOINT_ORDER}
    problematic_samples: list[str] = []
    role_topology_suspects: list[str] = []
    collapsed_role_samples: list[str] = []
    for sample in samples:
        label_path = RAW_LABELS_DIR / str(sample.get("label_file") or "")
        try:
            fields = label_path.read_text(encoding="utf-8").strip().split()
            if len(fields) < 14:
                raise ValueError("label curto")
            sample_problem = False
            for i, pin in enumerate(KEYPOINT_ORDER):
                x = float(fields[5 + i*3])
                y = float(fields[6 + i*3])
                vis = float(fields[7 + i*3])
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and vis >= 0.0):
                    raise ValueError("keypoint fora do formato")
                if x <= 0.015 or x >= 0.985 or y <= 0.015 or y >= 0.985:
                    edge_hits += 1
                    per_pin_edge[pin] += 1
                    sample_problem = True
            # Auditoria semântica adicional. B/C/E precisam ser distintos.
            # Quando só um keypoint fica muito mais perto do centro da bbox,
            # ele normalmente é a base; outro rótulo nessa posição é listado
            # como suspeito para revisão antes do próximo treino.
            coords = {
                pin: np.array([
                    float(fields[5 + i*3]), float(fields[6 + i*3])
                ], dtype=np.float64)
                for i, pin in enumerate(KEYPOINT_ORDER)
            }
            pair_distances = [
                float(np.linalg.norm(coords[a] - coords[b]))
                for a, b in (("B", "C"), ("B", "E"), ("C", "E"))
            ]
            sid = str(sample.get("sample_id") or label_path.stem)
            if min(pair_distances) < 0.035:
                collapsed_role_samples.append(sid)
                sample_problem = True
            box_center = np.array([float(fields[1]), float(fields[2])], dtype=np.float64)
            radial = {pin: float(np.linalg.norm(coords[pin] - box_center)) for pin in KEYPOINT_ORDER}
            closest_pin = min(radial, key=radial.get)
            ordered = sorted(radial.values())
            if closest_pin != "B" and ordered[1] - ordered[0] >= 0.10:
                role_topology_suspects.append(sid)
            checked += 1
            if sample_problem:
                problematic_samples.append(str(sample.get("sample_id") or label_path.stem))
        except Exception:
            invalid_labels += 1
            problematic_samples.append(str(sample.get("sample_id") or label_path.stem))
    return {
        "samples_checked": checked,
        "invalid_label_count": invalid_labels,
        "edge_clipped_keypoint_count": edge_hits,
        "edge_clipped_by_pin": per_pin_edge,
        "problematic_sample_count": len(set(problematic_samples)),
        "problematic_sample_ids_preview": list(dict.fromkeys(problematic_samples))[:20],
        "collapsed_role_sample_count": len(set(collapsed_role_samples)),
        "collapsed_role_sample_ids_preview": list(dict.fromkeys(collapsed_role_samples))[:20],
        "role_topology_suspect_count": len(set(role_topology_suspects)),
        "role_topology_suspect_ids_preview": list(dict.fromkeys(role_topology_suspects))[:20],
        "healthy": invalid_labels == 0 and edge_hits == 0 and not collapsed_role_samples,
        "note": "Keypoints cortados/colapsados bloqueiam um dataset saudável; suspeitas semânticas B/C/E são listadas para revisão manual.",
    }


def _validate_points(points: dict, image_width: int, image_height: int) -> dict[str, tuple[float, float]]:
    corrected: dict[str, tuple[float, float]] = {}
    missing = [pin for pin in KEYPOINT_ORDER if pin not in points]
    if missing:
        raise ValueError(f"Faltam pontos do transistor: {', '.join(missing)}.")
    for pin in KEYPOINT_ORDER:
        item = points[pin]
        if not isinstance(item, dict) or "x" not in item or "y" not in item:
            raise ValueError(f"Ponto {pin} deve conter x e y.")
        x, y = float(item["x"]), float(item["y"])
        if not (0 <= x < image_width and 0 <= y < image_height):
            raise ValueError(f"Ponto {pin} está fora da imagem.")
        corrected[pin] = (x, y)
    return corrected


def save_transistor_training_sample(
    *,
    image_path: Path,
    project_id: str,
    component: dict,
    points: dict,
) -> dict:
    component_type = str(component.get("type") or "")
    if component_type not in TRANSISTOR_CLASSES:
        raise ValueError("O treino de keypoints atual aceita somente transistor_npn e transistor_pnp.")
    bbox = component.get("bbox")
    if not isinstance(bbox, dict):
        raise ValueError("Componente sem bbox válida.")
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    h, w = image.shape[:2]
    corrected = _validate_points(points, w, h)
    cx1, cy1, cx2, cy2 = _training_crop_bounds(image, bbox, corrected)
    crop = image[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    if cw < 16 or ch < 16:
        raise ValueError("Recorte do transistor ficou pequeno demais para virar amostra.")

    component_id = _safe_id(component.get("component_id") or component.get("detection_id") or "COMP")
    # Uma amostra por transistor/projeto. Se o usuário corrigir de novo,
    # substituímos o ground truth anterior em vez de duplicar a mesma imagem.
    digest = hashlib.sha1(f"{project_id}|{component_id}".encode()).hexdigest()[:8]
    sample_id = f"{_safe_id(project_id)}_{component_id}_{digest}"
    image_file = RAW_IMAGES_DIR / f"{sample_id}.png"
    label_file = RAW_LABELS_DIR / f"{sample_id}.txt"
    meta_file = RAW_META_DIR / f"{sample_id}.json"
    cv2.imwrite(str(image_file), crop)

    bx1 = _clip((float(bbox["x1"]) - cx1) / cw, 0.0, 1.0)
    by1 = _clip((float(bbox["y1"]) - cy1) / ch, 0.0, 1.0)
    bx2 = _clip((float(bbox["x2"]) - cx1) / cw, 0.0, 1.0)
    by2 = _clip((float(bbox["y2"]) - cy1) / ch, 0.0, 1.0)
    center_x = (bx1 + bx2) / 2.0
    center_y = (by1 + by2) / 2.0
    box_w = max(0.01, bx2 - bx1)
    box_h = max(0.01, by2 - by1)
    fields = [str(TRANSISTOR_CLASSES[component_type]), f"{center_x:.6f}", f"{center_y:.6f}", f"{box_w:.6f}", f"{box_h:.6f}"]
    for pin in KEYPOINT_ORDER:
        px, py = corrected[pin]
        nx = _clip((px - cx1) / cw, 0.0, 1.0)
        ny = _clip((py - cy1) / ch, 0.0, 1.0)
        fields.extend((f"{nx:.6f}", f"{ny:.6f}", "2"))
    label_file.write_text(" ".join(fields) + "\n", encoding="utf-8")

    meta = {
        "schema_version": "terminal-keypoint-sample-1.1",
        "crop_strategy": "bbox_plus_manual_keypoints",
        "sample_id": sample_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_id": project_id,
        "component_id": component_id,
        "type": component_type,
        "keypoint_order": list(KEYPOINT_ORDER),
        "source_image": image_path.name,
        "crop": {"x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2, "width": cw, "height": ch},
        "bbox": bbox,
        "points": {pin: {"x": corrected[pin][0], "y": corrected[pin][1]} for pin in KEYPOINT_ORDER},
        "image_file": image_file.name,
        "label_file": label_file.name,
    }
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _load_raw_samples() -> list[dict]:
    samples: list[dict] = []
    for path in sorted(RAW_META_DIR.glob("*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            image_path = RAW_IMAGES_DIR / meta["image_file"]
            label_path = RAW_LABELS_DIR / meta["label_file"]
            if image_path.exists() and label_path.exists() and meta.get("type") in TRANSISTOR_CLASSES:
                meta["_meta_path"] = str(path)
                samples.append(meta)
        except Exception:
            continue
    return samples





def _visual_prior_canvas(image: np.ndarray, size: int = 96) -> tuple[np.ndarray, float, int, int]:
    """Normaliza um recorte preservando aspecto e retorna feature + mapeamento.

    A feature combina bordas e tinta escura. Ela não tenta reconhecer a mesma
    foto exatamente: serve como prior visual de vizinhos rotulados da mesma
    classe quando o YOLO-Pose rejeita/colapsa B/C/E.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    h, w = gray.shape[:2]
    scale = min((size - 8) / max(1, w), (size - 8) / max(1, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    background = int(np.median(gray))
    canvas = np.full((size, size), background, np.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    edges = cv2.Canny(canvas, 40, 120).astype(np.float32) / 255.0
    darkness = (255.0 - canvas.astype(np.float32)) / 255.0
    edge_small = cv2.resize(edges, (48, 48), interpolation=cv2.INTER_AREA)
    dark_small = cv2.resize(darkness, (48, 48), interpolation=cv2.INTER_AREA)
    feature = np.concatenate((edge_small.reshape(-1), 0.5 * dark_small.reshape(-1))).astype(np.float32)
    feature = (feature - float(feature.mean())) / (float(feature.std()) + 1e-6)
    return feature, float(scale), int(ox), int(oy)


def _visual_prior_signature(samples: list[dict]) -> tuple[int, int]:
    latest = 0
    for sample in samples:
        for folder, key in ((RAW_IMAGES_DIR, "image_file"), (RAW_LABELS_DIR, "label_file")):
            path = folder / str(sample.get(key) or "")
            try:
                latest = max(latest, int(path.stat().st_mtime_ns))
            except Exception:
                pass
    return len(samples), latest


def _load_visual_prior_entries() -> list[dict]:
    samples = _load_raw_samples()
    signature = _visual_prior_signature(samples)
    if _VISUAL_PRIOR_CACHE.get("signature") == signature:
        return list(_VISUAL_PRIOR_CACHE.get("entries") or [])
    entries: list[dict] = []
    for sample in samples:
        raw_path = RAW_IMAGES_DIR / str(sample.get("image_file") or "")
        label_path = RAW_LABELS_DIR / str(sample.get("label_file") or "")
        raw = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE) if raw_path.exists() else None
        if raw is None or not label_path.exists():
            continue
        try:
            fields = [float(v) for v in label_path.read_text(encoding="utf-8").strip().split()]
            if len(fields) < 14:
                continue
            feature, scale, ox, oy = _visual_prior_canvas(raw)
            h, w = raw.shape[:2]
            points = []
            for i, pin in enumerate(KEYPOINT_ORDER):
                x = float(fields[5 + i * 3]) * w
                y = float(fields[6 + i * 3]) * h
                points.append((x * scale + ox, y * scale + oy))
            sb = sample.get("bbox") or {}
            bw = max(1.0, float(sb.get("width") or (float(sb.get("x2", 1)) - float(sb.get("x1", 0)))))
            bh = max(1.0, float(sb.get("height") or (float(sb.get("y2", 1)) - float(sb.get("y1", 0)))))
            entries.append({
                "sample_id": sample.get("sample_id"),
                "project_id": sample.get("project_id"),
                "type": sample.get("type"),
                "feature": feature,
                "canvas_points": np.asarray(points, dtype=np.float32),
                "bbox_aspect": float(bw / bh),
            })
        except Exception:
            continue
    _VISUAL_PRIOR_CACHE["signature"] = signature
    _VISUAL_PRIOR_CACHE["entries"] = entries
    return list(entries)


def predict_transistor_visual_prior(
    *,
    image: np.ndarray | None,
    bbox: dict,
    component_type: str,
    accept_score: float = 0.45,
) -> dict | None:
    """Fallback supervisionado não-paramétrico para B/C/E.

    Compara o recorte atual com as amostras MANUALMENTE corrigidas da mesma
    classe. Os três vizinhos mais parecidos votam nos keypoints em um canvas
    normalizado. Em validação leave-one-project-out sobre o dataset recebido,
    score >= 0.45 colocou todos os 3 pontos a <15 px normalizados em ~92% dos
    casos; por isso o prior só é publicado em similaridade alta.
    """
    if image is None or component_type not in TRANSISTOR_CLASSES or not isinstance(bbox, dict):
        return None
    entries = [e for e in _load_visual_prior_entries() if e.get("type") == component_type]
    if len(entries) < 3:
        return None
    try:
        cx1, cy1, cx2, cy2 = _crop_bounds(image, bbox, pad_ratio=0.28)
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        q_feature, q_scale, q_ox, q_oy = _visual_prior_canvas(crop)
        bw = max(1.0, float(bbox["x2"]) - float(bbox["x1"]))
        bh = max(1.0, float(bbox["y2"]) - float(bbox["y1"]))
        q_aspect = bw / bh
    except Exception:
        return None

    ranked = []
    for entry in entries:
        feature = entry["feature"]
        if feature.shape != q_feature.shape:
            continue
        similarity = float(np.mean(q_feature * feature))
        aspect_penalty = abs(float(np.log(max(1e-6, q_aspect / max(1e-6, float(entry["bbox_aspect"]))))))
        score = similarity - 0.10 * aspect_penalty
        ranked.append((score, similarity, entry))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    top = ranked[:3]
    best_score = float(top[0][0])
    if best_score < float(accept_score):
        return None

    scores = np.asarray([item[0] for item in top], dtype=np.float64)
    weights = np.exp((scores - scores.max()) * 8.0)
    weights /= max(1e-9, float(weights.sum()))
    canvas_points = sum(float(weights[i]) * top[i][2]["canvas_points"] for i in range(len(top)))
    mapped: dict[str, dict] = {}
    for i, pin in enumerate(KEYPOINT_ORDER):
        x_local = (float(canvas_points[i][0]) - q_ox) / max(1e-6, q_scale)
        y_local = (float(canvas_points[i][1]) - q_oy) / max(1e-6, q_scale)
        mapped[pin] = {
            "x": float(cx1 + x_local),
            "y": float(cy1 + y_local),
            "confidence": float(max(0.0, min(1.0, (best_score - 0.35) / 0.30))),
        }

    # Sanidade mínima: não publica três pontos colapsados nem absurdamente longe.
    coords = np.asarray([[mapped[p]["x"], mapped[p]["y"]] for p in KEYPOINT_ORDER], dtype=np.float64)
    diag = max(1.0, float(np.hypot(bw, bh)))
    min_sep = max(4.0, diag * 0.05)
    distances = [float(np.linalg.norm(coords[a] - coords[b])) for a, b in ((0, 1), (0, 2), (1, 2))]
    if min(distances) < min_sep:
        return None
    xlo, xhi = float(bbox["x1"]) - bw * 1.0, float(bbox["x2"]) + bw * 1.0
    ylo, yhi = float(bbox["y1"]) - bh * 1.0, float(bbox["y2"]) + bh * 1.0
    if any(not (xlo <= x <= xhi and ylo <= y <= yhi) for x, y in coords):
        return None

    confidence = float(max(0.0, min(1.0, (best_score - 0.35) / 0.30)))
    return {
        "accepted": True,
        "review_required": bool(best_score < 0.55),
        "source": "training_visual_prior",
        "points": mapped,
        "box_confidence": confidence,
        "keypoint_confidence": confidence,
        "keypoint_confidences": {pin: confidence for pin in KEYPOINT_ORDER},
        "visual_prior_score": round(best_score, 5),
        "visual_similarity": round(float(top[0][1]), 5),
        "neighbors": [
            {
                "sample_id": item[2].get("sample_id"),
                "project_id": item[2].get("project_id"),
                "score": round(float(item[0]), 5),
                "similarity": round(float(item[1]), 5),
                "weight": round(float(weights[i]), 5),
            }
            for i, item in enumerate(top)
        ],
        "validation_note": "Prior de alta similaridade; score>=0.45. Use revisão se score<0.55.",
    }


def _bbox_iou_local(a: dict | None, b: dict | None) -> float:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = map(float, (a["x1"], a["y1"], a["x2"], a["y2"]))
        bx1, by1, bx2, by2 = map(float, (b["x1"], b["y1"], b["x2"], b["y2"]))
    except Exception:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw * ih
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def _normalized_image_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similaridade robusta para reconhecer a MESMA foto reaberta/recomprimida.

    Não é usada para generalização; somente para recuperar ground truth já
    ensinado. Por isso o threshold de uso é deliberadamente alto.
    """
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a.copy()
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b.copy()
    ar_a = ga.shape[1] / max(1.0, float(ga.shape[0]))
    ar_b = gb.shape[1] / max(1.0, float(gb.shape[0]))
    if abs(ar_a - ar_b) / max(ar_a, ar_b, 1e-6) > 0.035:
        return 0.0
    size = (96, 96)
    ga = cv2.resize(ga, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    gb = cv2.resize(gb, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    ga = cv2.GaussianBlur(ga, (3, 3), 0)
    gb = cv2.GaussianBlur(gb, (3, 3), 0)
    ga = (ga - float(ga.mean())) / (float(ga.std()) + 1e-6)
    gb = (gb - float(gb.mean())) / (float(gb.std()) + 1e-6)
    corr = float(np.mean(ga * gb))
    return max(0.0, min(1.0, (corr + 1.0) / 2.0))



def _bbox_context_crop(image: np.ndarray, bbox: dict, pad_ratio: float = 0.62) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
    """Recorta contexto local em torno da bbox para memória visual.

    Usa proporção fixa em relação ao componente, portanto funciona mesmo se a
    imagem completa tiver sido reenviada, redimensionada ou recomprimida.
    """
    if image is None or image.size == 0:
        return None, (0, 0, 0, 0)
    h, w = image.shape[:2]
    try:
        x1, y1 = float(bbox["x1"]), float(bbox["y1"])
        x2, y2 = float(bbox["x2"]), float(bbox["y2"])
    except Exception:
        return None, (0, 0, 0, 0)
    bw, bh = max(1.0, x2-x1), max(1.0, y2-y1)
    pad = max(12.0, min(140.0, max(bw, bh) * pad_ratio))
    cx1 = max(0, int(np.floor(x1-pad)))
    cy1 = max(0, int(np.floor(y1-pad)))
    cx2 = min(w, int(np.ceil(x2+pad))+1)
    cy2 = min(h, int(np.ceil(y2+pad))+1)
    if cx2-cx1 < 12 or cy2-cy1 < 12:
        return None, (cx1, cy1, cx2, cy2)
    return image[cy1:cy2, cx1:cx2].copy(), (cx1, cy1, cx2, cy2)


def _normalized_context_image(image: np.ndarray | None, size: int = 144) -> np.ndarray | None:
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape[:2]
    scale = min((size-12) / max(1, w), (size-12) / max(1, h))
    nw, nh = max(1, int(round(w*scale))), max(1, int(round(h*scale)))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((size, size), int(np.median(gray)), np.uint8)
    ox, oy = (size-nw)//2, (size-nh)//2
    canvas[oy:oy+nh, ox:ox+nw] = resized
    return canvas


def _context_visual_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Similaridade local robusta a JPEG, brilho leve e pequenas mudanças de bbox."""
    na = _normalized_context_image(a)
    nb = _normalized_context_image(b)
    if na is None or nb is None:
        return 0.0
    # Correlação de intensidade normalizada.
    fa = na.astype(np.float32); fb = nb.astype(np.float32)
    fa = (fa-fa.mean()) / (fa.std()+1e-6)
    fb = (fb-fb.mean()) / (fb.std()+1e-6)
    corr = float(np.mean(fa*fb))
    corr01 = max(0.0, min(1.0, (corr+1.0)/2.0))
    # Estrutura de bordas; dilatação torna tolerante a 1-2 px de deslocamento.
    ea = cv2.Canny(na, 45, 135)
    eb = cv2.Canny(nb, 45, 135)
    k = np.ones((3,3), np.uint8)
    da = cv2.dilate(ea, k)
    db = cv2.dilate(eb, k)
    a_on = ea > 0; b_on = eb > 0
    if np.any(a_on) and np.any(b_on):
        recall_ab = float(np.mean(db[a_on] > 0))
        recall_ba = float(np.mean(da[b_on] > 0))
        edge = 0.5*(recall_ab+recall_ba)
    else:
        edge = 0.0
    return float(max(0.0, min(1.0, corr01*0.68 + edge*0.32)))


def _sample_bbox_in_raw_crop(sample: dict) -> dict | None:
    crop = sample.get("crop") or {}
    bbox = sample.get("bbox") or {}
    try:
        ox, oy = float(crop["x1"]), float(crop["y1"])
        return {
            "x1": float(bbox["x1"])-ox,
            "y1": float(bbox["y1"])-oy,
            "x2": float(bbox["x2"])-ox,
            "y2": float(bbox["y2"])-oy,
        }
    except Exception:
        return None


def _map_sample_points_by_bbox(sample: dict, current_bbox: dict) -> dict[str, dict] | None:
    """Mapeia GT manual para a bbox atual por translação/escala.

    Para a mesma foto em resolução diferente, isso preserva a posição relativa
    dos keypoints mesmo que a detecção atual varie alguns pixels.
    """
    sb = sample.get("bbox") or {}
    pts = sample.get("points") or {}
    if not all(pin in pts for pin in KEYPOINT_ORDER):
        return None
    try:
        sw = max(1e-6, float(sb["x2"])-float(sb["x1"]))
        sh = max(1e-6, float(sb["y2"])-float(sb["y1"]))
        cw = max(1e-6, float(current_bbox["x2"])-float(current_bbox["x1"]))
        ch = max(1e-6, float(current_bbox["y2"])-float(current_bbox["y1"]))
        scx = (float(sb["x1"])+float(sb["x2"]))/2.0
        scy = (float(sb["y1"])+float(sb["y2"]))/2.0
        ccx = (float(current_bbox["x1"])+float(current_bbox["x2"]))/2.0
        ccy = (float(current_bbox["y1"])+float(current_bbox["y2"]))/2.0
        sx, sy = cw/sw, ch/sh
        return {
            pin: {
                "x": ccx + (float(pts[pin]["x"])-scx)*sx,
                "y": ccy + (float(pts[pin]["y"])-scy)*sy,
                "confidence": 1.0,
            }
            for pin in KEYPOINT_ORDER
        }
    except Exception:
        return None



def _feature_match_training_sample(
    image: np.ndarray,
    bbox: dict,
    sample: dict,
    raw: np.ndarray | None,
) -> dict | None:
    """Reconhece uma amostra já ensinada por features locais + RANSAC.

    Diferente de template matching, tolera mudança de escala, pequena rotação,
    recompressão e diferenças de bbox. O transform é estimado do crop de treino
    para uma região ampla ao redor da bbox atual e depois usado para transportar
    B/C/E manualmente marcados.
    """
    if raw is None or raw.size == 0 or image is None:
        return None
    crop_meta = sample.get("crop") or {}
    points = sample.get("points") or {}
    sample_bbox = sample.get("bbox") or {}
    if not all(pin in points for pin in KEYPOINT_ORDER):
        return None
    try:
        crop_x1 = float(crop_meta["x1"]); crop_y1 = float(crop_meta["y1"])
        sbx1 = float(sample_bbox["x1"]); sby1 = float(sample_bbox["y1"])
        sbx2 = float(sample_bbox["x2"]); sby2 = float(sample_bbox["y2"])
    except Exception:
        return None

    h, w = image.shape[:2]
    cbw = max(2.0, float(bbox["x2"]) - float(bbox["x1"]))
    cbh = max(2.0, float(bbox["y2"]) - float(bbox["y1"]))
    cx = (float(bbox["x1"]) + float(bbox["x2"])) / 2.0
    cy = (float(bbox["y1"]) + float(bbox["y2"])) / 2.0
    pad_x = max(70.0, cbw * 2.0)
    pad_y = max(70.0, cbh * 2.0)
    rx1 = max(0, int(np.floor(cx - cbw/2.0 - pad_x)))
    ry1 = max(0, int(np.floor(cy - cbh/2.0 - pad_y)))
    rx2 = min(w, int(np.ceil(cx + cbw/2.0 + pad_x)))
    ry2 = min(h, int(np.ceil(cy + cbh/2.0 + pad_y)))
    search = image[ry1:ry2, rx1:rx2]
    if search.size == 0:
        return None

    def gray(im: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im.copy()
        g = cv2.GaussianBlur(g, (3, 3), 0)
        return cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)

    raw_g, search_g = _training_gray(raw), _training_gray(search)
    detector_name = "SIFT"
    try:
        kp1, des1 = _training_features(raw_g,'SIFT')
        kp2, des2 = _training_features(search_g,'SIFT')
        norm = cv2.NORM_L2
    except Exception:
        detector_name = "ORB"
        kp1, des1 = _training_features(raw_g,'ORB')
        kp2, des2 = _training_features(search_g,'ORB')
        norm = cv2.NORM_HAMMING
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None
    try:
        matcher = cv2.BFMatcher(norm)
        pairs = matcher.knnMatch(des1, des2, k=2)
    except Exception:
        return None
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < 0.76 * max(1e-6, n.distance):
            good.append(m)
    if len(good) < 4:
        return None
    src = np.float32([kp1[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0,
        maxIters=2500, confidence=0.995, refineIters=10,
    )
    if matrix is None or inlier_mask is None:
        return None
    inliers = int(np.count_nonzero(inlier_mask))
    inlier_ratio = inliers / max(1, len(good))
    if inliers < 4 or inlier_ratio < 0.34:
        return None
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    scale = float(np.hypot(a, b))
    if not (0.28 <= scale <= 3.6):
        return None

    def transform_local(x: float, y: float) -> tuple[float, float]:
        out = matrix @ np.array([x, y, 1.0], dtype=np.float64)
        return float(out[0] + rx1), float(out[1] + ry1)

    mapped = {}
    for pin in KEYPOINT_ORDER:
        lx = float(points[pin]["x"]) - crop_x1
        ly = float(points[pin]["y"]) - crop_y1
        gx, gy = transform_local(lx, ly)
        mapped[pin] = {"x": gx, "y": gy, "confidence": 1.0}

    # Transforma a bbox antiga (coordenadas da foto antiga -> crop local -> atual).
    corners = [
        (sbx1-crop_x1, sby1-crop_y1), (sbx2-crop_x1, sby1-crop_y1),
        (sbx2-crop_x1, sby2-crop_y1), (sbx1-crop_x1, sby2-crop_y1),
    ]
    tc = [transform_local(x, y) for x, y in corners]
    transformed_bbox = {
        "x1": min(x for x, _ in tc), "y1": min(y for _, y in tc),
        "x2": max(x for x, _ in tc), "y2": max(y for _, y in tc),
    }
    iou = _bbox_iou_local(bbox, transformed_bbox)
    ccx = (float(bbox["x1"])+float(bbox["x2"]))/2.0
    ccy = (float(bbox["y1"])+float(bbox["y2"]))/2.0
    tccx = (transformed_bbox["x1"]+transformed_bbox["x2"])/2.0
    tccy = (transformed_bbox["y1"]+transformed_bbox["y2"])/2.0
    center_dist = float(np.hypot(ccx-tccx, ccy-tccy))
    diag = max(1.0, float(np.hypot(cbw, cbh)))
    center_score = max(0.0, 1.0-center_dist/(diag*1.15))
    feature_density = min(1.0, inliers / 14.0)
    score = 0.42*inlier_ratio + 0.22*feature_density + 0.24*iou + 0.12*center_score
    return {
        "score": float(max(0.0, min(1.0, score))),
        "points": mapped,
        "inliers": inliers,
        "good_matches": len(good),
        "inlier_ratio": float(inlier_ratio),
        "scale": scale,
        "bbox_iou": float(iou),
        "center_score": float(center_score),
        "detector": detector_name,
        "transformed_bbox": transformed_bbox,
    }


def _template_match_training_sample(
    image: np.ndarray,
    bbox: dict,
    sample: dict,
    raw: np.ndarray | None,
) -> dict | None:
    """Procura o crop de treino diretamente na imagem atual em múltiplas escalas.

    Esta memória é propositalmente independente da bbox atual: a bbox serve
    apenas para limitar a região de busca. Quando a mesma foto/circuito é
    reimportada e o detector muda a caixa alguns pixels, o crop ensinado ainda
    pode ser localizado pelo conteúdo visual e os B/C/E antigos são
    transportados pela mesma transformação de escala/translação.
    """
    if raw is None or raw.size == 0:
        return None
    crop_meta = sample.get("crop") or {}
    sample_bbox = sample.get("bbox") or {}
    points = sample.get("points") or {}
    if not all(pin in points for pin in KEYPOINT_ORDER):
        return None
    try:
        sbw = max(2.0, float(sample_bbox["x2"]) - float(sample_bbox["x1"]))
        sbh = max(2.0, float(sample_bbox["y2"]) - float(sample_bbox["y1"]))
        cbw = max(2.0, float(bbox["x2"]) - float(bbox["x1"]))
        cbh = max(2.0, float(bbox["y2"]) - float(bbox["y1"]))
        crop_x1 = float(crop_meta["x1"]); crop_y1 = float(crop_meta["y1"])
    except Exception:
        return None

    # Escala esperada pela bbox, mas testamos uma faixa larga porque a bbox
    # pode ter sido muito diferente entre duas execuções do detector.
    sx0 = cbw / sbw
    sy0 = cbh / sbh
    scale0 = float(np.sqrt(max(1e-6, sx0 * sy0)))
    multipliers = (0.62, 0.72, 0.82, 0.92, 1.0, 1.08, 1.18, 1.30, 1.45)
    scales = []
    for m in multipliers:
        sc = max(0.30, min(3.20, scale0 * m))
        if all(abs(sc - old) > 0.025 for old in scales):
            scales.append(sc)

    h, w = image.shape[:2]
    cx = (float(bbox["x1"]) + float(bbox["x2"])) / 2.0
    cy = (float(bbox["y1"]) + float(bbox["y2"])) / 2.0
    # Região de busca ampla o bastante para bbox deslocada, mas local para não
    # confundir dois transistores idênticos em posições distantes do circuito.
    pad_x = max(55.0, cbw * 1.55)
    pad_y = max(55.0, cbh * 1.55)
    rx1 = max(0, int(np.floor(cx - cbw / 2.0 - pad_x)))
    ry1 = max(0, int(np.floor(cy - cbh / 2.0 - pad_y)))
    rx2 = min(w, int(np.ceil(cx + cbw / 2.0 + pad_x)))
    ry2 = min(h, int(np.ceil(cy + cbh / 2.0 + pad_y)))
    search = image[ry1:ry2, rx1:rx2]
    if search.size == 0:
        return None

    def prep_gray(img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        g = cv2.GaussianBlur(g, (3, 3), 0)
        return cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)

    search_g = _training_gray(search)
    search_e = cv2.Canny(search_g, 45, 135)
    best = None
    for sc in scales:
        tw = int(round(raw.shape[1] * sc)); th = int(round(raw.shape[0] * sc))
        if tw < 12 or th < 12 or tw > search.shape[1] or th > search.shape[0]:
            continue
        interp = cv2.INTER_AREA if sc < 1.0 else cv2.INTER_LINEAR
        tpl = cv2.resize(raw, (tw, th), interpolation=interp)
        tpl_g = prep_gray(tpl)
        tpl_e = cv2.Canny(tpl_g, 45, 135)
        # Templates quase vazios não são úteis.
        if int(np.count_nonzero(tpl_e)) < max(12, int(tw * th * 0.002)):
            continue
        edge_map = cv2.matchTemplate(search_e, tpl_e, cv2.TM_CCOEFF_NORMED)
        _mn, edge_max, _ml, edge_loc = cv2.minMaxLoc(edge_map)
        gray_map = cv2.matchTemplate(search_g, tpl_g, cv2.TM_CCOEFF_NORMED)
        _mn2, gray_max, _ml2, gray_loc = cv2.minMaxLoc(gray_map)
        # Usa a localização da evidência mais forte, mas combina as duas
        # métricas avaliadas nessa mesma posição para reduzir falsos positivos.
        loc = edge_loc if edge_max >= gray_max else gray_loc
        ex, ey = int(loc[0]), int(loc[1])
        edge_here = float(edge_map[ey, ex]) if ey < edge_map.shape[0] and ex < edge_map.shape[1] else float(edge_max)
        gray_here = float(gray_map[ey, ex]) if ey < gray_map.shape[0] and ex < gray_map.shape[1] else float(gray_max)
        score = max(0.0, min(1.0, 0.72 * max(0.0, edge_here) + 0.28 * max(0.0, gray_here)))
        item = {"score": score, "scale": sc, "loc": loc, "edge": edge_here, "gray": gray_here, "tw": tw, "th": th}
        if best is None or item["score"] > best["score"]:
            best = item

    if best is None:
        return None
    mx = float(rx1 + best["loc"][0]); my = float(ry1 + best["loc"][1])
    sc = float(best["scale"])
    mapped = {}
    for pin in KEYPOINT_ORDER:
        lx = float(points[pin]["x"]) - crop_x1
        ly = float(points[pin]["y"]) - crop_y1
        mapped[pin] = {"x": mx + lx * sc, "y": my + ly * sc, "confidence": 1.0}
    return {
        "score": float(best["score"]),
        "edge_score": float(best["edge"]),
        "gray_score": float(best["gray"]),
        "scale": sc,
        "matched_rect": {"x1": mx, "y1": my, "x2": mx + best["tw"], "y2": my + best["th"]},
        "points": mapped,
    }


def lookup_known_training_ground_truth(
    *,
    image: np.ndarray | None,
    bbox: dict,
    component_type: str,
) -> dict | None:
    if image is None or component_type not in TRANSISTOR_CLASSES:return None
    key=('training-ground-truth',str(RAW_META_DIR),str(PROJECTS_DIR),image.shape,
         hashlib.sha256(memoryview(image) if image.flags.c_contiguous else image.tobytes()).digest(),
         tuple(sorted(bbox.items())),component_type)
    return session_result(key,lambda:_lookup_known_training_ground_truth(image=image,bbox=bbox,component_type=component_type))

def _lookup_known_training_ground_truth(*,image,bbox,component_type):
    """Recupera B/C/E de amostras já ensinadas usando memória visual local.

    Ordem de evidência:
      1) mesma imagem-fonte (quando disponível), mas sem exigir 98.5%;
      2) contexto visual local do transistor comparado ao crop salvo no treino;
      3) compatibilidade de aspecto da bbox.

    Isso permite reconhecer a mesma foto reimportada, redimensionada ou JPEG,
    inclusive em outro project_id.
    """
    if image is None or component_type not in TRANSISTOR_CLASSES:
        return None
    current_h, current_w = image.shape[:2]
    current_context, _current_ctx_bounds = _bbox_context_crop(image, bbox, pad_ratio=0.15)
    current_context_wide, _ = _bbox_context_crop(image, bbox, pad_ratio=0.45)
    if current_context is None:
        return None

    samples = [s for s in _load_raw_samples() if str(s.get("type") or "") == component_type]
    if not samples:
        return None

    source_cache: dict[str, tuple[np.ndarray | None, int, int, float]] = {}
    candidates: list[dict] = []
    cbw = max(1.0, float(bbox["x2"])-float(bbox["x1"]))
    cbh = max(1.0, float(bbox["y2"])-float(bbox["y1"]))
    current_ar = cbw/cbh

    for sample in samples:
        pts = sample.get("points") or {}
        sb = sample.get("bbox") or {}
        if not isinstance(sb, dict) or not all(pin in pts for pin in KEYPOINT_ORDER):
            continue

        # 1) Similaridade da imagem completa quando o source antigo ainda existe.
        image_similarity = 0.0
        old_w = old_h = 0
        project_id = str(sample.get("project_id") or "")
        source_name = str(sample.get("source_image") or "")
        if project_id and source_name:
            source_path = PROJECTS_DIR / project_id / source_name
            key = str(source_path)
            if key not in source_cache:
                old = cv2.imread(str(source_path)) if source_path.exists() else None
                if old is None:
                    source_cache[key] = (None, 0, 0, 0.0)
                else:
                    oh, ow = old.shape[:2]
                    # Only dimensions/similarity are used below. Retaining all
                    # full photos for hundreds of examples caused RAM pressure.
                    source_cache[key] = (None, ow, oh, _normalized_image_similarity(image, old))
            _old, old_w, old_h, image_similarity = source_cache[key]

        # 2) Similaridade LOCAL com o crop salvo no dataset.
        raw_path = RAW_IMAGES_DIR / str(sample.get("image_file") or "")
        raw = cv2.imread(str(raw_path)) if raw_path.exists() else None
        template_match = feature_match = None
        sample_local_bbox = _sample_bbox_in_raw_crop(sample)
        sample_context = None
        sample_context_wide = None
        if raw is not None and sample_local_bbox is not None:
            sample_context, _ = _bbox_context_crop(raw, sample_local_bbox, pad_ratio=0.15)
            sample_context_wide, _ = _bbox_context_crop(raw, sample_local_bbox, pad_ratio=0.45)
        local_similarity = _context_visual_similarity(current_context, sample_context)
        wide_similarity = _context_visual_similarity(current_context_wide, sample_context_wide)

        try:
            sbw = max(1.0, float(sb["x2"])-float(sb["x1"]))
            sbh = max(1.0, float(sb["y2"])-float(sb["y1"]))
            sample_ar = sbw/sbh
            aspect_score = max(0.0, 1.0 - abs(np.log(max(1e-6, current_ar/sample_ar))) / 0.55)
        except Exception:
            aspect_score = 0.0

        # Mesmo source em outra resolução: bbox escalada ajuda a desambiguar.
        iou = 0.0
        center_score = 0.0
        if image_similarity >= 0.88 and old_w > 0 and old_h > 0:
            sx_img = current_w / max(1.0, float(old_w))
            sy_img = current_h / max(1.0, float(old_h))
            scaled_bbox = {
                "x1": float(sb["x1"])*sx_img, "y1": float(sb["y1"])*sy_img,
                "x2": float(sb["x2"])*sx_img, "y2": float(sb["y2"])*sy_img,
            }
            iou = _bbox_iou_local(bbox, scaled_bbox)
            cbx = (float(bbox["x1"])+float(bbox["x2"]))/2.0
            cby = (float(bbox["y1"])+float(bbox["y2"]))/2.0
            sbx = (scaled_bbox["x1"]+scaled_bbox["x2"])/2.0
            sby = (scaled_bbox["y1"]+scaled_bbox["y2"])/2.0
            dist = float(np.hypot(cbx-sbx, cby-sby))
            diag = max(1.0, float(np.hypot(cbw, cbh)))
            center_score = max(0.0, 1.0-dist/(diag*1.6))

        template_similarity = float((template_match or {}).get("score") or 0.0)
        feature_similarity = float((feature_match or {}).get("score") or 0.0)
        # Features locais recebem peso alto: são invariantes a pequena rotação,
        # escala e mudança de bbox, justamente onde template matching falhava.
        score = local_similarity*1.55 + wide_similarity*0.50 + image_similarity*0.60 + aspect_score*0.16 + iou*0.35 + center_score*0.16 + template_similarity*2.50 + feature_similarity*3.40
        candidates.append({
            "score": float(score), "sample": sample,
            "local_similarity": float(local_similarity),
            "wide_similarity": float(wide_similarity),
            "image_similarity": float(image_similarity),
            "aspect_score": float(aspect_score), "bbox_iou": float(iou),
            "old_w": int(old_w), "old_h": int(old_h),
            "template_match": template_match,
            "template_similarity": template_similarity,
            "feature_match": feature_match,
            "feature_similarity": feature_similarity,
        })

    if not candidates:
        return None
    # Retrieve similar local contexts first. Nine large template correlations
    # plus SIFT for every historical correction made this grow without bound.
    # Keep all strong same-source matches as well as 24 closest contexts; each
    # selected example must still pass the existing visual acceptance checks.
    candidates.sort(key=lambda c: c['score'],reverse=True)
    shortlist=[c for i,c in enumerate(candidates) if i<24 or
               c['image_similarity']>=.93 and c['bbox_iou']>=.08]
    from backend.processing_progress import report
    for index,candidate in enumerate(shortlist):
        report('Conferindo os terminais com os exemplos corrigidos…',index,len(shortlist))
        sample=candidate['sample'];path=RAW_IMAGES_DIR/str(sample.get('image_file') or '')
        raw=cv2.imread(str(path)) if path.is_file() else None
        template=_template_match_training_sample(image,bbox,sample,raw)
        features=_feature_match_training_sample(image,bbox,sample,raw)
        ts=float((template or {}).get('score') or 0.);fs=float((features or {}).get('score') or 0.)
        candidate.update(template_match=template,feature_match=features,template_similarity=ts,feature_similarity=fs,
                         score=candidate['score']+ts*2.50+fs*3.40)
    candidates=shortlist
    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]
    second_score = candidates[1]["score"] if len(candidates) > 1 else -999.0
    local_sim = best["local_similarity"]
    wide_sim = best.get("wide_similarity", 0.0)
    full_sim = best["image_similarity"]
    # Aceita se o contexto local é forte. Se estiver só moderado, exige também
    # evidência da imagem completa/bbox para evitar confundir dois transistores parecidos.
    template_sim = float(best.get("template_similarity") or 0.0)
    feature_sim = float(best.get("feature_similarity") or 0.0)
    strong_local = local_sim >= 0.84
    very_strong_local = local_sim >= 0.93
    context_supported = wide_sim >= 0.54
    source_supported = full_sim >= 0.88 and (best["bbox_iou"] >= 0.08 or best["aspect_score"] >= 0.62)
    template_strong = template_sim >= 0.62
    template_very_strong = template_sim >= 0.74
    feature_match = best.get("feature_match") or {}
    feature_strong = feature_sim >= 0.48 and int(feature_match.get("inliers") or 0) >= 5 and float(feature_match.get("bbox_iou") or 0.0) >= 0.10
    feature_very_strong = feature_sim >= 0.60 and int(feature_match.get("inliers") or 0) >= 7
    margin = best["score"] - second_score
    margin_ok = margin >= 0.05 or very_strong_local or source_supported or template_very_strong or feature_very_strong
    if not ((feature_strong and margin_ok) or (template_strong and margin_ok) or (strong_local and (context_supported or margin_ok)) or source_supported):
        return None

    sample = best["sample"]
    # Quando a foto completa foi reconhecida, preserva exatamente a geometria
    # manual usando somente a escala global da imagem. Isso evita que pequenas
    # diferenças da bbox do detector desloquem B/C/E conhecidos.
    if full_sim >= 0.93 and best.get("old_w", 0) > 0 and best.get("old_h", 0) > 0:
        sx_img = current_w / max(1.0, float(best["old_w"]))
        sy_img = current_h / max(1.0, float(best["old_h"]))
        points = {
            pin: {
                "x": float(sample["points"][pin]["x"]) * sx_img,
                "y": float(sample["points"][pin]["y"]) * sy_img,
                "confidence": 1.0,
            }
            for pin in KEYPOINT_ORDER
        }
        point_mapping = "full_image_scale"
    elif feature_strong and best.get("feature_match") is not None:
        points = best["feature_match"]["points"]
        point_mapping = "feature_affine_transform"
    elif template_sim >= 0.62 and best.get("template_match") is not None:
        points = best["template_match"]["points"]
        point_mapping = "multiscale_template_transform"
    else:
        points = _map_sample_points_by_bbox(sample, bbox)
        point_mapping = "bbox_relative_scale"
    if points is None:
        return None
    return {
        "accepted": True,
        "review_required": False,
        "source": "known_training_ground_truth",
        "match_method": "local_visual_memory",
        "points": points,
        "box_confidence": 1.0,
        "keypoint_confidence": 1.0,
        "keypoint_confidences": {pin: 1.0 for pin in KEYPOINT_ORDER},
        "sample_id": sample.get("sample_id"),
        "source_project_id": sample.get("project_id"),
        "image_similarity": round(full_sim, 5),
        "local_similarity": round(local_sim, 5),
        "wide_similarity": round(wide_sim, 5),
        "aspect_score": round(best["aspect_score"], 5),
        "bbox_iou": round(best["bbox_iou"], 5),
        "candidate_margin": round(float(best["score"]-second_score), 5) if len(candidates)>1 else None,
        "candidate_count": len(candidates),
        "template_similarity": round(template_sim, 5),
        "template_scale": round(float((best.get("template_match") or {}).get("scale") or 0.0), 5),
        "template_edge_score": round(float((best.get("template_match") or {}).get("edge_score") or 0.0), 5),
        "template_gray_score": round(float((best.get("template_match") or {}).get("gray_score") or 0.0), 5),
        "feature_similarity": round(feature_sim, 5),
        "feature_inliers": int((best.get("feature_match") or {}).get("inliers") or 0),
        "feature_good_matches": int((best.get("feature_match") or {}).get("good_matches") or 0),
        "feature_inlier_ratio": round(float((best.get("feature_match") or {}).get("inlier_ratio") or 0.0), 5),
        "feature_bbox_iou": round(float((best.get("feature_match") or {}).get("bbox_iou") or 0.0), 5),
        "feature_detector": (best.get("feature_match") or {}).get("detector"),
        "point_mapping": point_mapping,
    }


def terminal_model_runtime_status(*, load_model: bool = True) -> dict:
    """Diagnóstico real do arquivo de keypoints e do carregamento em runtime.

    `model_exists` sozinho não é suficiente: um .pt pode existir e falhar ao
    abrir por versão do Ultralytics/PyTorch, arquivo corrompido ou tarefa
    incompatível. Este diagnóstico deixa isso explícito no Swagger.
    """
    info: dict[str, Any] = {
        "base_dir": str(BASE_DIR),
        "models_dir": str(MODELS_DIR),
        "terminal_data_dir": str(TERMINAL_DATA_DIR),
        "model_path": str(TERMINAL_MODEL_PATH),
        "model_exists": TERMINAL_MODEL_PATH.exists(),
        "model_loadable": False,
        "model_task": None,
        "model_names": None,
        "model_file_size_bytes": None,
        "model_modified_at": None,
        "model_load_error": None,
    }
    if not TERMINAL_MODEL_PATH.exists():
        info["model_load_error"] = "terminal_keypoints.pt não encontrado no diretório models do projeto."
        return info
    try:
        stat = TERMINAL_MODEL_PATH.stat()
        info["model_file_size_bytes"] = int(stat.st_size)
        info["model_modified_at"] = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")
    except Exception as exc:
        info["model_load_error"] = f"Não foi possível ler metadados do modelo: {exc}"
        return info
    if not load_model:
        return info
    model = _load_keypoint_model()
    info["model_loadable"] = model is not None
    info["model_load_error"] = _MODEL_CACHE.get("load_error")
    if model is not None:
        try:
            info["model_task"] = getattr(model, "task", None)
        except Exception:
            pass
        try:
            names = getattr(model, "names", None)
            if isinstance(names, dict):
                info["model_names"] = {str(k): str(v) for k, v in names.items()}
            elif names is not None:
                info["model_names"] = names
        except Exception:
            pass
    return info


def terminal_learning_status() -> dict:
    samples = _load_raw_samples()
    counts = {name: 0 for name in TRANSISTOR_CLASSES}
    for sample in samples:
        counts[sample["type"]] += 1
    summary = None
    if TRAINING_SUMMARY_PATH.exists():
        try:
            summary = json.loads(TRAINING_SUMMARY_PATH.read_text(encoding="utf-8"))
        except Exception:
            summary = None
    runtime = terminal_model_runtime_status(load_model=True)
    dataset_quality = _audit_raw_samples(samples)
    trained_samples = int((summary or {}).get("samples") or 0)
    return {
        "sample_count": len(samples),
        "samples_by_class": counts,
        "keypoint_order": list(KEYPOINT_ORDER),
        "model_exists": runtime["model_exists"],
        "model_loadable": runtime["model_loadable"],
        "model_path": runtime["model_path"],
        "models_dir": runtime["models_dir"],
        "terminal_data_dir": runtime["terminal_data_dir"],
        "model_task": runtime["model_task"],
        "model_names": runtime["model_names"],
        "model_file_size_bytes": runtime["model_file_size_bytes"],
        "model_modified_at": runtime["model_modified_at"],
        "model_load_error": runtime["model_load_error"],
        "recommended_minimum": {"transistor_npn": 25, "transistor_pnp": 25},
        "recommended_good": {"transistor_npn": 60, "transistor_pnp": 60},
        "runtime_policy": {
            "trained_model_has_priority": True,
            "heuristic_used_when_model_loadable": False,
            "hard_min_box_confidence": 0.12,
            "hard_min_mean_keypoint_confidence": 0.20,
            "hard_min_single_keypoint_confidence": 0.05,
            "review_below_box_confidence": 0.40,
            "review_below_mean_keypoint_confidence": 0.40,
            "visual_prior_enabled": True,
            "visual_prior_accept_score": 0.45,
            "visual_prior_review_below_score": 0.55,
        },
        "training_summary": summary,
        "new_samples_since_training": max(0, len(samples) - trained_samples),
        "dataset_quality": dataset_quality,
    }

def repair_training_samples_from_metadata() -> dict:
    """Reconstrói crops/labels antigos usando o ground truth guardado no meta.

    Não exige nova marcação do usuário. A operação sobrescreve a amostra do
    mesmo project_id/component_id com a estratégia de crop v1.1.
    """
    samples = _load_raw_samples()
    repaired = 0
    missing_source: list[str] = []
    failed: list[dict] = []
    projects_dir = BASE_DIR / "projects"
    for sample in samples:
        project_id = str(sample.get("project_id") or "")
        component_id = str(sample.get("component_id") or "")
        source_name = str(sample.get("source_image") or "")
        if not project_id or not component_id or not source_name:
            failed.append({"sample_id": sample.get("sample_id"), "error": "meta_incompleto"})
            continue
        image_path = projects_dir / project_id / source_name
        if not image_path.exists():
            # Fallback comum: projeto pode ter sido salvo como original.png/jpg
            candidates = list((projects_dir / project_id).glob("original.*")) if (projects_dir / project_id).exists() else []
            if candidates:
                image_path = candidates[0]
            else:
                missing_source.append(str(sample.get("sample_id") or component_id))
                continue
        component = {
            "component_id": component_id,
            "type": sample.get("type"),
            "bbox": sample.get("bbox"),
        }
        points = sample.get("points") or {}
        try:
            save_transistor_training_sample(
                image_path=image_path,
                project_id=project_id,
                component=component,
                points=points,
            )
            repaired += 1
        except Exception as exc:
            failed.append({
                "sample_id": sample.get("sample_id"),
                "error": f"{type(exc).__name__}: {exc}",
            })
    status = terminal_learning_status()
    return {
        "total_samples": len(samples),
        "repaired": repaired,
        "missing_source_count": len(missing_source),
        "missing_source_preview": missing_source[:20],
        "failed_count": len(failed),
        "failed_preview": failed[:20],
        "dataset_quality_after_repair": status.get("dataset_quality"),
        "note": "O modelo .pt existente não é alterado. Se labels foram reparados, retreine depois.",
    }


def build_pose_dataset(val_fraction: float = 0.2) -> dict:
    samples = _load_raw_samples()
    if len(samples) < 4:
        raise ValueError("São necessárias pelo menos 4 correções salvas para montar o dataset.")
    if POSE_DATASET_DIR.exists():
        shutil.rmtree(POSE_DATASET_DIR)
    for split in ("train", "val"):
        (POSE_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (POSE_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Evita vazamento: transistores vindos do MESMO circuito/projeto ficam no
    # mesmo split. A versão antiga dividia por amostra e podia colocar dois
    # transistores da mesma foto em train e val, inflando artificialmente a
    # validação.
    groups: dict[str, list[dict]] = {}
    for sample in samples:
        project_id = str(sample.get("project_id") or sample.get("sample_id") or "unknown")
        groups.setdefault(project_id, []).append(sample)
    project_ids = sorted(
        groups,
        key=lambda pid: hashlib.sha1(pid.encode("utf-8")).hexdigest(),
    )
    target_val = max(1, int(round(len(samples) * float(max(0.05, min(0.45, val_fraction))))))
    val_projects: set[str] = set()
    val_count = 0
    # Prefere grupos menores primeiro para aproximar melhor o alvo, mas usa
    # hash como desempate determinístico.
    candidates = sorted(project_ids, key=lambda pid: (len(groups[pid]), hashlib.sha1(pid.encode()).hexdigest()))
    for pid in candidates:
        if len(project_ids) - len(val_projects) <= 1:
            break
        size = len(groups[pid])
        if val_count < target_val or not val_projects:
            val_projects.add(pid)
            val_count += size
        if val_count >= target_val:
            break
    if not val_projects and len(project_ids) > 1:
        val_projects.add(project_ids[0])
    if len(val_projects) == len(project_ids) and len(project_ids) > 1:
        val_projects.remove(project_ids[-1])

    split_counts = {"train": 0, "val": 0}
    split_projects = {"train": set(), "val": set()}
    class_counts = {
        "train": {name: 0 for name in TRANSISTOR_CLASSES},
        "val": {name: 0 for name in TRANSISTOR_CLASSES},
    }
    for sample in samples:
        project_id = str(sample.get("project_id") or sample.get("sample_id") or "unknown")
        split = "val" if project_id in val_projects else "train"
        src_img = RAW_IMAGES_DIR / sample["image_file"]
        src_lbl = RAW_LABELS_DIR / sample["label_file"]
        shutil.copy2(src_img, POSE_DATASET_DIR / "images" / split / src_img.name)
        shutil.copy2(src_lbl, POSE_DATASET_DIR / "labels" / split / src_lbl.name)
        split_counts[split] += 1
        split_projects[split].add(project_id)
        if sample.get("type") in TRANSISTOR_CLASSES:
            class_counts[split][sample["type"]] += 1

    if split_counts["train"] == 0 or split_counts["val"] == 0:
        raise ValueError("Não foi possível criar train/val sem misturar o mesmo projeto. Adicione exemplos de mais projetos.")

    yaml_path = POSE_DATASET_DIR / "data.yaml"
    yaml_text = "\n".join([
        f"path: {POSE_DATASET_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        "kpt_shape: [3, 3]",
        "flip_idx: [0, 1, 2]",
        "names:",
        "  0: transistor_npn",
        "  1: transistor_pnp",
        "",
    ])
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return {
        "dataset_yaml": str(yaml_path),
        **split_counts,
        "total": len(samples),
        "split_strategy": "grouped_by_project_id",
        "train_project_count": len(split_projects["train"]),
        "val_project_count": len(split_projects["val"]),
        "train_class_counts": class_counts["train"],
        "val_class_counts": class_counts["val"],
    }

def train_terminal_keypoint_model(
    *,
    epochs: int = 80,
    imgsz: int = 320,
    batch: int = 8,
    base_model: str = "yolo11n-pose.pt",
    device: str | None = None,
) -> dict:
    status = terminal_learning_status()
    if status["sample_count"] < 10:
        raise ValueError("Salve pelo menos 10 transistores corrigidos antes do primeiro treino. Para boa precisão, use 25+ de cada classe.")
    dataset = build_pose_dataset()
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Ultralytics não está instalado. Instale com: pip install ultralytics") from exc

    model = YOLO(base_model)
    kwargs = {
        "data": dataset["dataset_yaml"],
        "epochs": int(max(1, min(500, epochs))),
        "imgsz": int(max(160, min(1024, imgsz))),
        "batch": int(max(1, min(64, batch))),
        "project": str(TERMINAL_DATA_DIR / "runs"),
        "name": "terminal_keypoints",
        "exist_ok": True,
        "patience": 80,
        "degrees": 180.0,
        "fliplr": 0.5,
        "flipud": 0.5,
        # Dá mais peso à identidade/posição de B/C/E. Augmentations muito
        # agressivas podem apagar a seta pequena e prejudicar a distinção C/E.
        "pose": 20.0,
        "kobj": 2.0,
        "translate": 0.05,
        "scale": 0.28,
        "shear": 2.0,
        "perspective": 0.0,
        "mosaic": 0.35,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 15,
    }
    if device:
        kwargs["device"] = device
    results = model.train(**kwargs)
    save_dir = Path(getattr(results, "save_dir", TERMINAL_DATA_DIR / "runs" / "terminal_keypoints"))
    training_metrics = {}
    try:
        results_dict = getattr(results, "results_dict", None)
        if isinstance(results_dict, dict):
            training_metrics = {str(k): float(v) if isinstance(v, (int, float, np.floating)) else str(v) for k, v in results_dict.items()}
    except Exception:
        training_metrics = {}
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Treino terminou, mas best.pt não foi encontrado em {best}")
    shutil.copy2(best, TERMINAL_MODEL_PATH)
    summary = {
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "samples": status["sample_count"],
        "samples_by_class": status["samples_by_class"],
        "epochs": kwargs["epochs"],
        "imgsz": kwargs["imgsz"],
        "batch": kwargs["batch"],
        "base_model": base_model,
        "dataset": dataset,
        "model_path": str(TERMINAL_MODEL_PATH),
        "training_metrics": training_metrics,
        "dataset_quality_before_training": status.get("dataset_quality"),
    }
    TRAINING_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _MODEL_CACHE["signature"] = None
    _MODEL_CACHE["model"] = None
    return summary


def _load_keypoint_model():
    if not TERMINAL_MODEL_PATH.exists():
        _MODEL_CACHE["load_error"] = "arquivo não encontrado"
        _MODEL_CACHE["model"] = None
        return None
    try:
        stat = TERMINAL_MODEL_PATH.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
    except Exception as exc:
        _MODEL_CACHE["load_error"] = f"erro ao ler arquivo: {exc}"
        _MODEL_CACHE["model"] = None
        return None
    if _MODEL_CACHE["signature"] == signature and _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"]
    try:
        from ultralytics import YOLO
        model = YOLO(str(TERMINAL_MODEL_PATH))
        task = getattr(model, "task", None)
        if task not in (None, "pose"):
            raise RuntimeError(f"modelo carregado com task={task!r}; esperado 'pose'")
    except Exception as exc:
        _MODEL_CACHE["signature"] = signature
        _MODEL_CACHE["model"] = None
        _MODEL_CACHE["load_error"] = f"{type(exc).__name__}: {exc}"
        _MODEL_CACHE["loaded_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        return None
    _MODEL_CACHE["signature"] = signature
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["load_error"] = None
    _MODEL_CACHE["loaded_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    return model


def _keypoint_geometry_sanity(points: dict[str, dict], bbox: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    try:
        coords = {pin: (float(points[pin]["x"]), float(points[pin]["y"])) for pin in KEYPOINT_ORDER}
    except Exception:
        return False, ["keypoints_incompletos"]
    bw = max(1.0, float(bbox["x2"]) - float(bbox["x1"]))
    bh = max(1.0, float(bbox["y2"]) - float(bbox["y1"]))
    diag = float(np.hypot(bw, bh))
    min_sep = max(3.0, diag * 0.045)
    for i, a in enumerate(KEYPOINT_ORDER):
        for b in KEYPOINT_ORDER[i + 1:]:
            d = float(np.hypot(coords[a][0] - coords[b][0], coords[a][1] - coords[b][1]))
            if d < min_sep:
                reasons.append(f"{a}_{b}_colapsados")
    # Keypoints podem ficar um pouco fora da bbox porque o terminal elétrico
    # costuma estar na ponta do fio. Só rejeitamos previsões muito distantes.
    padx, pady = bw * 0.85, bh * 0.85
    xlo, xhi = float(bbox["x1"]) - padx, float(bbox["x2"]) + padx
    ylo, yhi = float(bbox["y1"]) - pady, float(bbox["y2"]) + pady
    for pin, (x, y) in coords.items():
        if not (xlo <= x <= xhi and ylo <= y <= yhi):
            reasons.append(f"{pin}_muito_longe_da_bbox")
    return not reasons, reasons



def _pose_inference_view(crop: np.ndarray, mode: str) -> np.ndarray:
    if mode == "original":
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _normalized_center_distance(a: dict, b: dict) -> float:
    acx = (float(a["x1"]) + float(a["x2"])) / 2.0
    acy = (float(a["y1"]) + float(a["y2"])) / 2.0
    bcx = (float(b["x1"]) + float(b["x2"])) / 2.0
    bcy = (float(b["y1"]) + float(b["y2"])) / 2.0
    diag = max(1.0, float(np.hypot(float(b["x2"])-float(b["x1"]), float(b["y2"])-float(b["y1"]))))
    return float(np.hypot(acx-bcx, acy-bcy) / diag)


def _run_pose_multicrop_candidates(model, image: np.ndarray, bbox: dict, expected_class: int) -> list[dict]:
    """Executa o pose model em múltiplos recortes/contrastes.

    Em fotos reais a bbox do detector pode conter muito contexto ou cortar parte
    da seta/forquilha. Uma única janela fazia B/C/E colapsarem apesar de o modelo
    ter confiança alta. Esta rotina testa vários contextos e escolhe somente uma
    pose geometricamente válida associada à bbox original.
    """
    variants = [
        (0.10, "original", 384),
        (0.20, "original", 384),
        (0.28, "original", 384),
        (0.42, "original", 416),
        (0.60, "original", 416),
        (0.20, "clahe", 384),
        (0.42, "clahe", 416),
    ]
    candidates: list[dict] = []
    for pad_ratio, view_mode, imgsz in variants:
        cx1, cy1, cx2, cy2 = _crop_bounds(image, bbox, pad_ratio=pad_ratio)
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        view = _pose_inference_view(crop, view_mode)
        try:
            results = model.predict(view, imgsz=imgsz, conf=0.015, verbose=False, max_det=10)
        except Exception as exc:
            candidates.append({
                "accepted": False,
                "variant": {"pad_ratio": pad_ratio, "view": view_mode, "imgsz": imgsz},
                "error": f"{type(exc).__name__}:{exc}",
                "score": -999.0,
            })
            continue
        if not results:
            continue
        result = results[0]
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            continue
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        box_conf = result.boxes.conf.detach().cpu().numpy()
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        kp_xy_all = result.keypoints.xy.detach().cpu().numpy()
        try:
            kp_conf_all = result.keypoints.conf.detach().cpu().numpy()
        except Exception:
            kp_conf_all = np.ones((len(classes), len(KEYPOINT_ORDER)), dtype=np.float32)

        for i in range(len(classes)):
            if i >= len(kp_xy_all) or kp_xy_all[i].shape[0] < len(KEYPOINT_ORDER):
                continue
            pred_bbox = {
                "x1": float(xyxy[i][0] + cx1), "y1": float(xyxy[i][1] + cy1),
                "x2": float(xyxy[i][2] + cx1), "y2": float(xyxy[i][3] + cy1),
            }
            iou = _bbox_iou_local(pred_bbox, bbox)
            center_dist = _normalized_center_distance(pred_bbox, bbox)
            points: dict[str, dict] = {}
            confs: list[float] = []
            for k, pin in enumerate(KEYPOINT_ORDER):
                conf = float(kp_conf_all[i][k]) if k < len(kp_conf_all[i]) else 1.0
                confs.append(conf)
                points[pin] = {
                    "x": float(kp_xy_all[i][k][0] + cx1),
                    "y": float(kp_xy_all[i][k][1] + cy1),
                    "confidence": conf,
                }
            mean_conf = float(np.mean(confs)) if confs else 0.0
            min_conf = float(np.min(confs)) if confs else 0.0
            geometry_ok, geometry_reasons = _keypoint_geometry_sanity(points, bbox)
            cls = int(classes[i])
            cls_match = cls == int(expected_class)
            bconf = float(box_conf[i])
            # Associação à bbox principal evita escolher outro transistor que
            # apareceu dentro de um crop mais largo.
            associated = (iou >= 0.05 or center_dist <= 0.70)
            hard_ok = (
                bconf >= 0.08
                and mean_conf >= 0.18
                and min_conf >= 0.04
                and geometry_ok
                and associated
                and cls_match
            )
            score = (
                bconf * 1.00
                + mean_conf * 1.70
                + min_conf * 0.65
                + iou * 1.55
                + max(0.0, 1.0-center_dist) * 0.85
                + (0.25 if cls_match else 0.0)
                + (1.30 if geometry_ok else -2.00)
            )
            candidates.append({
                "accepted": bool(hard_ok),
                "score": float(score),
                "variant": {"pad_ratio": pad_ratio, "view": view_mode, "imgsz": imgsz,
                            "crop": {"x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2}},
                "predicted_class": cls,
                "pose_class_matches_detector": cls_match,
                "box_confidence": bconf,
                "keypoint_confidence": mean_conf,
                "min_keypoint_confidence": min_conf,
                "keypoint_confidences": {pin: float(points[pin]["confidence"]) for pin in KEYPOINT_ORDER},
                "points": points,
                "predicted_bbox": pred_bbox,
                "bbox_iou": float(iou),
                "center_distance": float(center_dist),
                "geometry_sane": bool(geometry_ok),
                "geometry_reasons": list(geometry_reasons),
            })
    return candidates


def diagnose_transistor_keypoints(
    *,
    image: np.ndarray | None,
    bbox: dict,
    component_type: str,
) -> dict:
    """Diagnóstico robusto usando ensemble multi-crop do modelo treinado."""
    diagnostic: dict[str, Any] = {
        "component_type": component_type,
        "model_path": str(TERMINAL_MODEL_PATH),
        "model_exists": TERMINAL_MODEL_PATH.exists(),
        "model_loadable": False,
        "accepted": False,
        "review_required": True,
        "rejection_reasons": [],
        "source": "trained_terminal_keypoint_model",
        "box_confidence": 0.0,
        "keypoint_confidence": 0.0,
        "keypoint_confidences": {},
        "points": {},
        "inference_strategy": "multicrop_pose_ensemble_v1",
    }
    if image is None:
        diagnostic["rejection_reasons"].append("imagem_ausente")
        return diagnostic
    if component_type not in TRANSISTOR_CLASSES:
        diagnostic["rejection_reasons"].append("classe_nao_suportada")
        return diagnostic
    model = _load_keypoint_model()
    diagnostic["model_loadable"] = model is not None
    if model is None:
        diagnostic["rejection_reasons"].append(_MODEL_CACHE.get("load_error") or "modelo_nao_carregado")
        return diagnostic

    expected_class = TRANSISTOR_CLASSES[component_type]
    candidates = _run_pose_multicrop_candidates(model, image, bbox, expected_class)
    real_candidates = [c for c in candidates if "points" in c]
    accepted = [c for c in real_candidates if c.get("accepted")]
    selected = max(accepted, key=lambda c: c.get("score", -999.0)) if accepted else (
        max(real_candidates, key=lambda c: c.get("score", -999.0)) if real_candidates else None
    )

    # Resumo compacto das tentativas para o diagnostics sem explodir o JSON.
    diagnostic["candidate_count"] = len(real_candidates)
    diagnostic["accepted_candidate_count"] = len(accepted)
    diagnostic["candidate_summary"] = [
        {
            "accepted": bool(c.get("accepted")),
            "score": round(float(c.get("score", 0.0)), 4),
            "pad_ratio": c.get("variant", {}).get("pad_ratio"),
            "view": c.get("variant", {}).get("view"),
            "imgsz": c.get("variant", {}).get("imgsz"),
            "box_confidence": round(float(c.get("box_confidence", 0.0)), 4),
            "keypoint_confidence": round(float(c.get("keypoint_confidence", 0.0)), 4),
            "min_keypoint_confidence": round(float(c.get("min_keypoint_confidence", 0.0)), 4),
            "bbox_iou": round(float(c.get("bbox_iou", 0.0)), 4),
            "center_distance": round(float(c.get("center_distance", 0.0)), 4),
            "geometry_sane": bool(c.get("geometry_sane")),
            "geometry_reasons": c.get("geometry_reasons", []),
            "pose_class_matches_detector": bool(c.get("pose_class_matches_detector", False)),
        }
        for c in sorted(real_candidates, key=lambda x: x.get("score", -999.0), reverse=True)[:8]
    ]

    if selected is None:
        diagnostic["rejection_reasons"].append("sem_box_ou_keypoints_em_multicrop")
        return diagnostic

    diagnostic["selected_variant"] = selected.get("variant")
    diagnostic["selected_score"] = float(selected.get("score", 0.0))
    diagnostic["box_confidence"] = float(selected.get("box_confidence", 0.0))
    diagnostic["keypoint_confidence"] = float(selected.get("keypoint_confidence", 0.0))
    diagnostic["keypoint_confidences"] = selected.get("keypoint_confidences", {})
    diagnostic["min_keypoint_confidence"] = float(selected.get("min_keypoint_confidence", 0.0))
    diagnostic["points"] = selected.get("points", {})
    diagnostic["predicted_bbox"] = selected.get("predicted_bbox")
    diagnostic["bbox_iou"] = float(selected.get("bbox_iou", 0.0))
    diagnostic["center_distance"] = float(selected.get("center_distance", 0.0))
    diagnostic["detector_authoritative_class"] = int(expected_class)
    diagnostic["pose_predicted_class"] = int(selected.get("predicted_class", expected_class))
    diagnostic["pose_class_matches_detector"] = bool(selected.get("pose_class_matches_detector", True))
    diagnostic["geometry_sane"] = bool(selected.get("geometry_sane", False))
    diagnostic["geometry_reasons"] = list(selected.get("geometry_reasons", []))

    hard_reasons: list[str] = []
    if not selected.get("accepted"):
        if float(selected.get("box_confidence", 0.0)) < 0.08:
            hard_reasons.append("box_confidence_muito_baixa")
        if float(selected.get("keypoint_confidence", 0.0)) < 0.18:
            hard_reasons.append("media_keypoints_muito_baixa")
        if float(selected.get("min_keypoint_confidence", 0.0)) < 0.04:
            hard_reasons.append("keypoint_individual_muito_baixo")
        hard_reasons.extend(selected.get("geometry_reasons", []))
        if float(selected.get("bbox_iou", 0.0)) < 0.05 and float(selected.get("center_distance", 99.0)) > 0.70:
            hard_reasons.append("pose_nao_associada_a_bbox_do_detector")
        if not selected.get("pose_class_matches_detector", False):
            hard_reasons.append("classe_pose_diverge_do_detector_principal")
        if not hard_reasons:
            hard_reasons.append("nenhuma_variante_multicrop_aprovada")
    diagnostic["rejection_reasons"] = list(dict.fromkeys(hard_reasons))
    diagnostic["accepted"] = bool(selected.get("accepted"))

    bconf = diagnostic["box_confidence"]
    mean_conf = diagnostic["keypoint_confidence"]
    min_kp = diagnostic["min_keypoint_confidence"]
    diagnostic["review_required"] = (
        not diagnostic["accepted"]
        or bconf < 0.35
        or mean_conf < 0.35
        or min_kp < 0.12
        or not diagnostic.get("pose_class_matches_detector", True)
    )
    diagnostic["review_reasons"] = []
    if not diagnostic.get("pose_class_matches_detector", True):
        diagnostic["review_reasons"].append("pose_class_diverge_do_detector_principal")
    if bconf < 0.35:
        diagnostic["review_reasons"].append("box_confidence_moderada")
    if mean_conf < 0.35:
        diagnostic["review_reasons"].append("keypoints_confidence_moderada")
    if min_kp < 0.12:
        diagnostic["review_reasons"].append("algum_keypoint_baixa_confianca")
    diagnostic["acceptance_policy"] = {
        "hard_min_box_confidence": 0.08,
        "hard_min_mean_keypoint_confidence": 0.18,
        "hard_min_single_keypoint_confidence": 0.04,
        "association_min_iou_or_max_center_distance": "IoU>=0.05 OR center_distance<=0.70",
        "geometry_must_be_sane": True,
        "pose_class_must_match_detector": True,
        "review_box_confidence": 0.35,
        "review_mean_keypoint_confidence": 0.35,
        "review_single_keypoint_confidence": 0.12,
    }
    return diagnostic

def predict_transistor_keypoints(
    *,
    image: np.ndarray | None,
    bbox: dict,
    component_type: str,
    min_confidence: float | None = None,
) -> dict | None:
    """Compatibilidade com a Etapa 7: retorna apenas previsões aceitas.

    `min_confidence` é mantido por compatibilidade, mas a política atual usa
    thresholds separados para caixa/keypoints e uma checagem geométrica.
    """
    diagnostic = diagnose_transistor_keypoints(image=image, bbox=bbox, component_type=component_type)
    if not diagnostic.get("accepted"):
        return None
    return {
        "points": diagnostic["points"],
        "box_confidence": diagnostic["box_confidence"],
        "keypoint_confidence": diagnostic["keypoint_confidence"],
        "keypoint_confidences": diagnostic["keypoint_confidences"],
        "review_required": diagnostic["review_required"],
        "source": "trained_terminal_keypoint_model",
        "diagnostic": diagnostic,
    }

def _cli():
    parser = argparse.ArgumentParser(description="Treino dos pontos B/C/E dos transistores")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status")
    sub.add_parser("repair")
    train = sub.add_parser("train")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--imgsz", type=int, default=320)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--model", default="yolo11n-pose.pt")
    train.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(train_terminal_keypoint_model(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, base_model=args.model, device=args.device), ensure_ascii=False, indent=2))
    elif args.command == "repair":
        print(json.dumps(repair_training_samples_from_metadata(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(terminal_learning_status(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
