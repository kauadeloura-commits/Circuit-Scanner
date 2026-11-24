from __future__ import annotations

from datetime import datetime
from pathlib import Path
import copy
import json
import math

import cv2
import numpy as np
from backend.resistor_geometry import cached_vision

from backend.ink_color import estimate_component_ink, image_with_component_ink
from backend.terminal_identity import TARGET_CLASSES as IDENTITY_CLASSES, resolve_terminal_identity

try:
    from backend.terminal_learning import (
        predict_transistor_keypoints,
        diagnose_transistor_keypoints,
        terminal_model_runtime_status,
        lookup_known_training_ground_truth,
        predict_transistor_visual_prior,
    )
except Exception:
    predict_transistor_keypoints = None
    diagnose_transistor_keypoints = None
    terminal_model_runtime_status = None
    lookup_known_training_ground_truth = None
    predict_transistor_visual_prior = None

try:
    from backend.all_component_terminal_learning import (
        KEYPOINT_SCHEMAS as TRAINED_TERMINAL_SCHEMAS,
        predict_component_terminals,
        model_runtime_status as all_component_terminal_model_status,
    )
except Exception:
    TRAINED_TERMINAL_SCHEMAS = {}
    predict_component_terminals = None
    all_component_terminal_model_status = None


# Layout canônico da classe, normalizado dentro da bbox não rotacionada.
# x=0/1 e y=0/1 correspondem às bordas da bounding box do símbolo.
# IMPORTANTE: estes layouts seguem os símbolos realmente gerados/treinados
# em dataset_generator.py. A etapa 7 localiza terminais visíveis; pinos que
# não aparecem no símbolo (por exemplo alimentação do opamp genérico atual)
# não são inventados aqui.
PIN_LIBRARY = {
    "resistor": [("1", 0.0, 0.5), ("2", 1.0, 0.5)],
    "capacitor": [("1", 0.0, 0.5), ("2", 1.0, 0.5)],
    "capacitor_polarized": [("+", 0.0, 0.5), ("-", 1.0, 0.5)],
    "diode": [("A", 0.0, 0.5), ("K", 1.0, 0.5)],
    "led": [("A", 0.0, 0.5), ("K", 1.0, 0.5)],
    "inductor": [("1", 0.0, 0.5), ("2", 1.0, 0.5)],
    # No dataset atual, fonte DC e bateria têm terminais visíveis à esquerda/direita.
    "source_dc": [("+", 0.0, 0.5), ("-", 1.0, 0.5)],
    "battery": [("+", 0.0, 0.5), ("-", 1.0, 0.5)],
    "gnd": [("GND", 0.5, 0.0)],
    "vcc": [("VCC", 0.5, 1.0)],
    "transistor_npn": [("B", 0.0, 0.5), ("C", 1.0, 0.06), ("E", 1.0, 0.94)],
    "transistor_pnp": [("B", 0.0, 0.5), ("C", 1.0, 0.06), ("E", 1.0, 0.94)],
    "mosfet_n": [("G", 0.0, 0.5), ("D", 1.0, 0.25), ("S", 1.0, 0.75)],
    "mosfet_p": [("G", 0.0, 0.5), ("D", 1.0, 0.25), ("S", 1.0, 0.75)],
    # O símbolo atual do dataset desenha somente +, - e OUT.
    # O '+' está no terminal superior e o '-' no inferior.
    "opamp": [("+", 0.0, 0.25), ("-", 0.0, 0.75), ("OUT", 1.0, 0.5)],
}

POLARITY_SENSITIVE = {
    "capacitor_polarized",
    "diode",
    "led",
    "source_dc",
    "battery",
    "transistor_npn",
    "transistor_pnp",
    "mosfet_n",
    "mosfet_p",
    "opamp",
}

TWO_TERMINAL_CLASSES = {
    "resistor",
    "capacitor",
    "capacitor_polarized",
    "diode",
    "led",
    "inductor",
    "source_dc",
    "battery",
}

ONE_PIN_CLASSES = {"gnd", "vcc"}
# GND e VCC são símbolos essencialmente verticais no dataset; quando a bbox é
# quase quadrada, essa informação é uma escolha melhor que o aspect ratio puro.
VERTICAL_DEFAULT = {"gnd", "vcc"}

# Semântica elétrica usada pela Etapa 8/10. O objetivo aqui é garantir que
# o ponto geométrico esteja associado ao pino elétrico correto.
PIN_SEMANTICS = {
    "resistor": {"1": "terminal_1", "2": "terminal_2"},
    "capacitor": {"1": "terminal_1", "2": "terminal_2"},
    "capacitor_polarized": {"+": "positive", "-": "negative"},
    "source_dc": {"+": "positive", "-": "negative"},
    "battery": {"+": "positive", "-": "negative"},
    "opamp": {"+": "non_inverting_input", "-": "inverting_input", "OUT": "output"},
    "mosfet_n": {"G": "gate", "D": "drain", "S": "source"},
    "mosfet_p": {"G": "gate", "D": "drain", "S": "source"},
    "diode": {"A": "anode", "K": "cathode"},
    "led": {"A": "anode", "K": "cathode"},
    "gnd": {"GND": "ground"},
    "vcc": {"VCC": "power"},
    "transistor_npn": {"B": "base", "C": "collector", "E": "emitter"},
    "transistor_pnp": {"B": "base", "C": "collector", "E": "emitter"},
}

FOCUS_CLASSES = {
    "vcc", "gnd", "resistor", "capacitor",
    "transistor_npn", "transistor_pnp", "diode", "led",
}


def _bbox_dims(bbox: dict) -> tuple[float, float]:
    width = max(1.0, float(bbox.get("width") or (bbox["x2"] - bbox["x1"])))
    height = max(1.0, float(bbox.get("height") or (bbox["y2"] - bbox["y1"])))
    return width, height


def _orientation(component_type: str, bbox: dict, ratio_threshold: float = 1.18) -> tuple[str, float]:
    width, height = _bbox_dims(bbox)
    ratio = width / height

    if ratio >= ratio_threshold:
        return "horizontal", min(1.0, ratio / 2.2)
    if ratio <= 1.0 / ratio_threshold:
        return "vertical", min(1.0, (1.0 / ratio) / 2.2)

    # Símbolos quase quadrados não carregam orientação confiável apenas na bbox.
    if component_type in VERTICAL_DEFAULT:
        return "vertical", 0.45
    return "horizontal", 0.35


def _led_cathode_bar_axis(
    image: np.ndarray | None,
    bbox: dict,
    fallback_orientation: str,
) -> tuple[str, float, dict]:
    """Localiza a barra do cátodo de diodo/LED e retorna o eixo elétrico.

    A barra correta é escolhida não só por ser horizontal/vertical: damos forte
    preferência à linha cujo CENTRO coincide com o encontro de dois segmentos
    diagonais do corpo do diodo. Isso diferencia a barra do cátodo da outra
    lateral do triângulo e das linhas pautadas do papel.
    """
    meta = {
        "source": "cathode_bar_not_found",
        "bar_angle": None,
        "bar_center_x": None,
        "bar_center_y": None,
        "top_x": None, "bottom_x": None,
        "left_y": None, "right_y": None,
        "diagonal_join_count": 0,
    }
    if image is None:
        return fallback_orientation, 0.0, meta
    h_img, w_img = image.shape[:2]
    x1 = max(0, int(math.floor(float(bbox["x1"]))))
    y1 = max(0, int(math.floor(float(bbox["y1"]))))
    x2 = min(w_img, int(math.ceil(float(bbox["x2"]))) + 1)
    y2 = min(h_img, int(math.ceil(float(bbox["y2"]))) + 1)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return fallback_orientation, 0.0, meta

    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 32, 115)
    h, w = gray.shape[:2]
    min_dim = float(min(w, h))
    min_len = max(6, int(min_dim * 0.10))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=max(5, min_len // 2),
        minLineLength=min_len,
        maxLineGap=max(3, min_len // 2),
    )
    if lines is None:
        return fallback_orientation, 0.0, meta

    rows = []
    for row in np.asarray(lines).reshape(-1, 4):
        ax, ay, bx, by = [float(v) for v in row]
        length = math.hypot(bx - ax, by - ay)
        if length < min_len:
            continue
        angle = math.degrees(math.atan2(by - ay, bx - ax)) % 180.0
        rows.append((ax, ay, bx, by, length, angle))

    # Estima o ápice do símbolo a partir das duas famílias de diagonais.
    # A barra do cátodo fica junto a esse ápice; a outra lateral do triângulo não.
    diag_pos = []
    diag_neg = []
    for ax, ay, bx, by, length, angle in rows:
        seg5 = (ax, ay, bx, by, length)
        if 15.0 <= angle <= 75.0:
            diag_pos.append(seg5)
        elif 105.0 <= angle <= 165.0:
            diag_neg.append(seg5)
    apex = None
    if diag_pos and diag_neg:
        lp = _fit_hough_group(diag_pos, 3)
        ln = _fit_hough_group(diag_neg, 3)
        apex = _line_intersection(lp, ln)

    # Endpoints de segmentos diagonais: a ponta do triângulo/V do diodo
    # normalmente termina bem no centro da barra do cátodo.
    diag_endpoints = []
    for ax, ay, bx, by, length, angle in rows:
        dh = _angle_distance_180(angle, 0.0)
        dv = _angle_distance_180(angle, 90.0)
        if dh >= 18.0 and dv >= 18.0 and length >= min_dim * 0.12:
            diag_endpoints.extend([(ax, ay), (bx, by)])

    cx, cy = w / 2.0, h / 2.0
    candidates = []
    for ax, ay, bx, by, length, angle in rows:
        is_h = _angle_distance_180(angle, 0.0) <= 12.0
        is_v = _angle_distance_180(angle, 90.0) <= 12.0
        if not (is_h or is_v):
            continue
        if length > max(w, h) * 0.90:
            continue
        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
        center_norm = math.hypot((mx - cx) / max(1.0, w / 2.0), (my - cy) / max(1.0, h / 2.0))
        if center_norm > 1.15:
            continue

        vals = []
        n = max(3, int(round(length)))
        for t in np.linspace(0.0, 1.0, n):
            xx = int(round(ax + (bx - ax) * float(t)))
            yy = int(round(ay + (by - ay) * float(t)))
            if 0 <= xx < w and 0 <= yy < h:
                xa, xb = max(0, xx - 1), min(w, xx + 2)
                ya, yb = max(0, yy - 1), min(h, yy + 2)
                vals.append(float(np.percentile(gray[ya:yb, xa:xb], 25)))
        darkness = 0.0 if not vals else 255.0 - float(np.mean(vals))

        join_radius = max(7.0, min_dim * 0.12)
        joined = [math.hypot(ex - mx, ey - my) for ex, ey in diag_endpoints]
        join_count = sum(d <= join_radius for d in joined)
        join_closeness = 0.0
        if joined:
            best_join = min(joined)
            join_closeness = max(0.0, 1.0 - best_join / join_radius)

        apex_bonus = 0.0
        apex_distance = None
        if apex is not None:
            apex_distance = math.hypot(mx - float(apex[0]), my - float(apex[1]))
            apex_radius = max(10.0, min_dim * 0.28)
            apex_bonus = max(0.0, 1.0 - apex_distance / apex_radius) * 95.0

        # Barras reais de cátodo podem ser relativamente longas; não penalize
        # tanto o comprimento. A proximidade com o ápice é a pista principal.
        target_len = 0.42 * min_dim
        score = (
            darkness * 0.42
            - center_norm * 18.0
            - abs(length - target_len) * 0.35
            + join_count * 18.0
            + join_closeness * 15.0
            + apex_bonus
        )
        candidates.append((score, angle, length, mx, my, darkness, join_count))

    if not candidates:
        return fallback_orientation, 0.0, meta
    candidates.sort(reverse=True, key=lambda item: item[0])
    score, angle, length, mx, my, darkness, join_count = candidates[0]
    if score < 30.0:
        return fallback_orientation, 0.0, meta

    gx, gy = x1 + mx, y1 + my
    if _angle_distance_180(angle, 0.0) <= 12.0:
        orientation = "vertical"
        meta.update({
            "source": "cathode_bar",
            "bar_angle": round(angle, 2),
            "bar_center_x": round(gx, 2),
            "bar_center_y": round(gy, 2),
            "top_x": round(gx, 2),
            "bottom_x": round(gx, 2),
            "diagonal_join_count": int(join_count),
        })
    else:
        orientation = "horizontal"
        meta.update({
            "source": "cathode_bar",
            "bar_angle": round(angle, 2),
            "bar_center_x": round(gx, 2),
            "bar_center_y": round(gy, 2),
            "left_y": round(gy, 2),
            "right_y": round(gy, 2),
            "diagonal_join_count": int(join_count),
        })
    conf = max(0.60, min(0.98, 0.62 + min(0.20, join_count * 0.08) + min(0.12, darkness / 800.0)))
    return orientation, float(conf), meta


def _local_darkness_score(image: np.ndarray | None, point: tuple[float, float], radius: int = 6) -> float:
    if image is None:
        return 0.0
    h, w = image.shape[:2]
    x, y = int(round(point[0])), int(round(point[1]))
    x1, x2 = max(0, x - radius), min(w, x + radius + 1)
    y1, y2 = max(0, y - radius), min(h, y + radius + 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    return float(255.0 - np.mean(gray))


def _two_terminal_line_pair_axis(
    image: np.ndarray | None,
    bbox: dict,
    fallback_orientation: str,
) -> tuple[str, float, dict]:
    """Infere eixo elétrico por DOIS trechos de fio colineares em lados opostos.

    Feito principalmente para LED fotografado/desenhado à mão. Diferente de
    olhar a bbox ou o corpo do símbolo, procura um trecho acima + outro abaixo
    alinhados (vertical) ou um à esquerda + outro à direita alinhados
    (horizontal). Setas de emissão e texto não costumam formar esse par.
    """
    meta = {
        "source": "line_pair_not_found",
        "vertical_pair_score": 0.0,
        "horizontal_pair_score": 0.0,
        "top_x": None, "bottom_x": None,
        "left_y": None, "right_y": None,
    }
    if image is None:
        return fallback_orientation, 0.0, meta

    h_img, w_img = image.shape[:2]
    bw, bh = _bbox_dims(bbox)
    pad = int(max(12, min(60, round(max(bw, bh) * 0.48))))
    x0 = max(0, int(math.floor(float(bbox["x1"]))) - pad)
    y0 = max(0, int(math.floor(float(bbox["y1"]))) - pad)
    x3 = min(w_img, int(math.ceil(float(bbox["x2"]))) + pad + 1)
    y3 = min(h_img, int(math.ceil(float(bbox["y2"]))) + pad + 1)
    if x3 - x0 < 16 or y3 - y0 < 16:
        return fallback_orientation, 0.0, meta

    roi = image[y0:y3, x0:x3]
    gray_raw = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    gray = cv2.GaussianBlur(gray_raw, (3, 3), 0)
    edges = cv2.Canny(gray, 35, 120)
    min_len = max(8, int(min(bw, bh) * 0.10))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=max(7, min_len // 2),
        minLineLength=min_len,
        maxLineGap=max(4, min_len // 2),
    )
    if lines is None:
        return fallback_orientation, 0.0, meta

    cx = (float(bbox["x1"]) + float(bbox["x2"])) / 2.0
    cy = (float(bbox["y1"]) + float(bbox["y2"])) / 2.0
    verts, hors = [], []
    for row in np.asarray(lines).reshape(-1, 4):
        ax, ay, bx, by = [float(v) for v in row]
        ax += x0; bx += x0; ay += y0; by += y0
        dx, dy = bx-ax, by-ay
        length = math.hypot(dx, dy)
        if length < min_len:
            continue
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        mx, my = (ax+bx)/2.0, (ay+by)/2.0
        # Escuridão do próprio traço. Linhas pautadas da folha costumam ser
        # muito mais claras que os fios desenhados do circuito.
        samples = []
        n_samp = max(3, int(round(length)))
        for tt in np.linspace(0.0, 1.0, n_samp):
            gx = int(round(ax + (bx-ax)*float(tt))) - x0
            gy = int(round(ay + (by-ay)*float(tt))) - y0
            if 0 <= gx < gray_raw.shape[1] and 0 <= gy < gray_raw.shape[0]:
                xa, xb = max(0, gx-1), min(gray_raw.shape[1], gx+2)
                ya, yb = max(0, gy-1), min(gray_raw.shape[0], gy+2)
                samples.append(float(np.percentile(gray_raw[ya:yb, xa:xb], 25)))
        darkness = 0.0 if not samples else 255.0 - float(np.mean(samples))
        if _angle_distance_180(ang, 90.0) <= 15.0:
            verts.append({"x": mx, "y": my, "len": length, "ymin": min(ay,by), "ymax": max(ay,by), "dark": darkness})
        if _angle_distance_180(ang, 0.0) <= 15.0:
            hors.append({"x": mx, "y": my, "len": length, "xmin": min(ax,bx), "xmax": max(ax,bx), "dark": darkness})

    best_v = None
    x_tol = max(8.0, bw * 0.16)
    for a in verts:
        for b in verts:
            if a is b:
                continue
            up, down = (a,b) if a["y"] < b["y"] else (b,a)
            if not (up["y"] < cy < down["y"]):
                continue
            align = abs(up["x"] - down["x"])
            if align > x_tol:
                continue
            pair_x = (up["x"] + down["x"]) / 2.0
            # Com ROI expandida, rejeita fios de componentes vizinhos. O par
            # elétrico do LED deve atravessar a própria bbox (ou ficar muito
            # próximo dela), não dezenas de pixels fora dela.
            if pair_x < float(bbox["x1"]) - bw * 0.10 or pair_x > float(bbox["x2"]) + bw * 0.10:
                continue
            gap = max(0.0, down["ymin"] - up["ymax"])
            # fios muito distantes entre si provavelmente são texto/linhas da folha
            if gap > bh * 0.80:
                continue
            score = up["len"] + down["len"] + (x_tol-align)*1.7 - gap*0.10 + (up["dark"] + down["dark"]) * 0.30
            # pequena preferência por pares próximos da região central, sem depender dela
            score -= abs((up["x"]+down["x"])/2.0 - cx)*0.12
            if best_v is None or score > best_v[0]:
                best_v = (score, up, down)

    best_h = None
    y_tol = max(8.0, bh * 0.16)
    for a in hors:
        for b in hors:
            if a is b:
                continue
            left, right = (a,b) if a["x"] < b["x"] else (b,a)
            if not (left["x"] < cx < right["x"]):
                continue
            align = abs(left["y"] - right["y"])
            if align > y_tol:
                continue
            pair_y = (left["y"] + right["y"]) / 2.0
            if pair_y < float(bbox["y1"]) - bh * 0.10 or pair_y > float(bbox["y2"]) + bh * 0.10:
                continue
            gap = max(0.0, right["xmin"] - left["xmax"])
            if gap > bw * 0.80:
                continue
            score = left["len"] + right["len"] + (y_tol-align)*1.7 - gap*0.10 + (left["dark"] + right["dark"]) * 0.30
            score -= abs((left["y"]+right["y"])/2.0 - cy)*0.12
            if best_h is None or score > best_h[0]:
                best_h = (score, left, right)

    sv = 0.0 if best_v is None else float(best_v[0])
    sh = 0.0 if best_h is None else float(best_h[0])
    meta["vertical_pair_score"] = round(sv, 3)
    meta["horizontal_pair_score"] = round(sh, 3)

    if best_v is not None and (best_h is None or sv >= sh * 1.12):
        _score, up, down = best_v
        meta.update({
            "source": "opposite_collinear_wire_pair",
            "top_x": round(float(up["x"]), 3),
            "bottom_x": round(float(down["x"]), 3),
            "pair_gap": round(float(max(0.0, down["ymin"] - up["ymax"])), 3),
            "pair_darkness": round(float((up["dark"] + down["dark"]) / 2.0), 3),
        })
        conf = min(0.99, 0.72 + min(0.25, max(0.0, sv-sh) / max(40.0, sv)))
        return "vertical", float(conf), meta
    if best_h is not None and (best_v is None or sh >= sv * 1.12):
        _score, left, right = best_h
        meta.update({
            "source": "opposite_collinear_wire_pair",
            "left_y": round(float(left["y"]), 3),
            "right_y": round(float(right["y"]), 3),
            "pair_gap": round(float(max(0.0, right["xmin"] - left["xmax"])), 3),
            "pair_darkness": round(float((left["dark"] + right["dark"]) / 2.0), 3),
        })
        conf = min(0.99, 0.72 + min(0.25, max(0.0, sh-sv) / max(40.0, sh)))
        return "horizontal", float(conf), meta

    return fallback_orientation, 0.0, meta


def _component_body_axis(
    image: np.ndarray | None,
    bbox: dict,
    fallback_orientation: str,
    component_type: str,
) -> tuple[str, float, dict]:
    """Infere o eixo do corpo visual do componente (LED/diodo), não da bbox.

    Em fotos e desenhos à mão, a bbox do detector pode incluir texto, setas de
    emissão e folga nas laterais. Isso faz um LED/diodo vertical parecer
    horizontal. Aqui escolhemos o contorno principal próximo do centro e usamos
    o minAreaRect desse corpo para inferir o eixo longo do símbolo.
    """
    meta = {
        "source": f"{component_type}_body_not_found",
        "success": False,
        "body_center_x": None,
        "body_center_y": None,
        "body_width": None,
        "body_height": None,
    }
    if image is None:
        return fallback_orientation, 0.0, meta

    h_img, w_img = image.shape[:2]
    x1 = max(0, int(math.floor(float(bbox["x1"]))))
    y1 = max(0, int(math.floor(float(bbox["y1"]))))
    x2 = min(w_img, int(math.ceil(float(bbox["x2"]))) + 1)
    y2 = min(h_img, int(math.ceil(float(bbox["y2"]))) + 1)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return fallback_orientation, 0.0, meta

    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _t, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback_orientation, 0.0, meta

    roi_h, roi_w = roi.shape[:2]
    rcx, rcy = roi_w / 2.0, roi_h / 2.0
    roi_area = float(max(1, roi_w * roi_h))
    best = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(10.0, roi_area * 0.01):
            continue
        rx, ry, rw, rh = cv2.boundingRect(contour)
        if max(rw, rh) < max(6, int(min(roi_w, roi_h) * 0.10)):
            continue
        rect = cv2.minAreaRect(contour)
        (cx, cy), (mw, mh), _ang = rect
        if mw <= 1.0 or mh <= 1.0:
            continue
        dist = math.hypot(cx - rcx, cy - rcy)
        edge_touch = int(rx <= 1 or ry <= 1 or rx + rw >= roi_w - 1 or ry + rh >= roi_h - 1)
        score = area * 1.4 + min(mw, mh) * 2.0 - dist * 0.75 - edge_touch * 10.0
        if best is None or score > best[0]:
            best = (score, rect, area)

    if best is None:
        return fallback_orientation, 0.0, meta

    _score, rect, area = best
    (cx, cy), (mw, mh), _ang = rect
    long_side, short_side = (mw, mh) if mw >= mh else (mh, mw)
    ratio = float(long_side / max(1.0, short_side))
    orientation = "horizontal" if mw >= mh else "vertical"
    confidence = max(0.0, min(0.92, 0.45 + min(1.2, ratio - 1.0) * 0.32))
    meta.update({
        "source": f"{component_type}_body_rectangle",
        "success": True,
        "body_center_x": round(float(x1 + cx), 2),
        "body_center_y": round(float(y1 + cy), 2),
        "body_width": round(float(mw), 2),
        "body_height": round(float(mh), 2),
        "body_area": round(area, 2),
        "body_ratio": round(ratio, 3),
    })
    return orientation, confidence, meta


def _body_axis_endpoints(
    image: np.ndarray | None,
    bbox: dict,
    body_meta: dict,
    orientation: str,
) -> dict:
    """Encontra as extremidades de tinta ao longo do eixo central do corpo.

    Usado principalmente no LED fotografado: as setas ficam fora do corredor
    central e não conseguem puxar A/K para esquerda/direita.
    """
    if image is None:
        return {}
    h_img, w_img = image.shape[:2]
    x1 = max(0, int(math.floor(float(bbox["x1"]))))
    y1 = max(0, int(math.floor(float(bbox["y1"]))))
    x2 = min(w_img, int(math.ceil(float(bbox["x2"]))) + 1)
    y2 = min(h_img, int(math.ceil(float(bbox["y2"]))) + 1)
    if x2 <= x1 or y2 <= y1:
        return {}
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return {}
    gx = xs.astype(np.float32) + float(x1)
    gy = ys.astype(np.float32) + float(y1)
    cx = float(body_meta.get("body_center_x") or ((bbox["x1"] + bbox["x2"]) / 2.0))
    cy = float(body_meta.get("body_center_y") or ((bbox["y1"] + bbox["y2"]) / 2.0))
    bw = float(body_meta.get("body_width") or 12.0)
    bh = float(body_meta.get("body_height") or 12.0)
    if orientation == "vertical":
        corridor = max(6.0, min(18.0, bw * 0.28))
        valid = np.abs(gx - cx) <= corridor
        if not np.any(valid):
            return {}
        vx, vy = gx[valid], gy[valid]
        top_y, bottom_y = float(vy.min()), float(vy.max())
        top_near = np.abs(vy - top_y) <= 2.5
        bot_near = np.abs(vy - bottom_y) <= 2.5
        return {
            "top_point": (float(np.median(vx[top_near])), top_y),
            "bottom_point": (float(np.median(vx[bot_near])), bottom_y),
        }
    corridor = max(6.0, min(18.0, bh * 0.28))
    valid = np.abs(gy - cy) <= corridor
    if not np.any(valid):
        return {}
    vx, vy = gx[valid], gy[valid]
    left_x, right_x = float(vx.min()), float(vx.max())
    left_near = np.abs(vx - left_x) <= 2.5
    right_near = np.abs(vx - right_x) <= 2.5
    return {
        "left_point": (left_x, float(np.median(vy[left_near]))),
        "right_point": (right_x, float(np.median(vy[right_near]))),
    }


def _two_terminal_border_axis(
    image: np.ndarray | None,
    bbox: dict,
    fallback_orientation: str,
) -> tuple[str, float, dict]:
    """Infere horizontal/vertical pelos fios que entram em bordas opostas.

    Infere o eixo elétrico pelos fios que entram no símbolo, em vez de usar
    apenas a proporção da bbox. Isso é importante para LED/diodo e também
    capacitores: as setas do LED ou as placas/textos do capacitor podem
    alargar a caixa e fazer um componente vertical parecer horizontal.
    """
    meta = {
        "source": "bbox_fallback",
        "top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0,
        "vertical_score": 0.0, "horizontal_score": 0.0,
    }
    if image is None:
        return fallback_orientation, 0.0, meta

    h_img, w_img = image.shape[:2]
    x1 = max(0, int(math.floor(float(bbox["x1"]))))
    y1 = max(0, int(math.floor(float(bbox["y1"]))))
    x2 = min(w_img, int(math.ceil(float(bbox["x2"]))) + 1)
    y2 = min(h_img, int(math.ceil(float(bbox["y2"]))) + 1)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return fallback_orientation, 0.0, meta

    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _thr, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = mask.shape[:2]
    strip = max(2, int(round(min(w, h) * 0.08)))
    xlo, xhi = max(0, int(w * 0.08)), min(w, max(1, int(w * 0.92)))
    ylo, yhi = max(0, int(h * 0.08)), min(h, max(1, int(h * 0.92)))

    def coords(side: str):
        if side == "top":
            yy, xx = np.nonzero(mask[:strip, xlo:xhi] > 0)
            return xx.astype(np.float32) + xlo, yy.astype(np.float32)
        if side == "bottom":
            yy, xx = np.nonzero(mask[h-strip:, xlo:xhi] > 0)
            return xx.astype(np.float32) + xlo, yy.astype(np.float32) + (h-strip)
        if side == "left":
            yy, xx = np.nonzero(mask[ylo:yhi, :strip] > 0)
            return xx.astype(np.float32), yy.astype(np.float32) + ylo
        yy, xx = np.nonzero(mask[ylo:yhi, w-strip:] > 0)
        return xx.astype(np.float32) + (w-strip), yy.astype(np.float32) + ylo

    c = {side: coords(side) for side in ("top", "bottom", "left", "right")}
    denom_tb = max(1.0, float(strip * max(1, xhi-xlo)))
    denom_lr = max(1.0, float(strip * max(1, yhi-ylo)))
    density = {
        "top": len(c["top"][0]) / denom_tb,
        "bottom": len(c["bottom"][0]) / denom_tb,
        "left": len(c["left"][0]) / denom_lr,
        "right": len(c["right"][0]) / denom_lr,
    }

    def alignment(a: str, b: str, use_x: bool, span: float) -> float:
        va = c[a][0 if use_x else 1]
        vb = c[b][0 if use_x else 1]
        if len(va) == 0 or len(vb) == 0:
            return 0.0
        delta = abs(float(np.median(va)) - float(np.median(vb)))
        return max(0.0, min(1.0, 1.0 - delta / max(4.0, span * 0.22)))

    v_align = alignment("top", "bottom", True, float(w))
    h_align = alignment("left", "right", False, float(h))
    vertical_score = min(density["top"], density["bottom"]) * (0.40 + 0.60 * v_align)
    horizontal_score = min(density["left"], density["right"]) * (0.40 + 0.60 * h_align)

    meta.update(density)
    def median_or_none(arr):
        return None if len(arr) == 0 else float(np.median(arr))
    meta.update({
        "vertical_score": round(float(vertical_score), 5),
        "horizontal_score": round(float(horizontal_score), 5),
        "vertical_alignment": round(float(v_align), 4),
        "horizontal_alignment": round(float(h_align), 4),
        # Coordenadas globais do traço elétrico exatamente nas bordas.
        "top_x": None if median_or_none(c["top"][0]) is None else round(float(x1 + median_or_none(c["top"][0])), 3),
        "bottom_x": None if median_or_none(c["bottom"][0]) is None else round(float(x1 + median_or_none(c["bottom"][0])), 3),
        "left_y": None if median_or_none(c["left"][1]) is None else round(float(y1 + median_or_none(c["left"][1])), 3),
        "right_y": None if median_or_none(c["right"][1]) is None else round(float(y1 + median_or_none(c["right"][1])), 3),
    })

    minimum = 0.012
    if vertical_score >= minimum and v_align >= 0.68 and vertical_score >= horizontal_score * 1.10:
        conf = min(0.98, 0.60 + vertical_score * 3.0 + v_align * 0.18)
        meta["source"] = "opposite_border_electrical_axis"
        return "vertical", float(conf), meta
    if horizontal_score >= minimum and h_align >= 0.68 and horizontal_score >= vertical_score * 1.10:
        conf = min(0.98, 0.60 + horizontal_score * 3.0 + h_align * 0.18)
        meta["source"] = "opposite_border_electrical_axis"
        return "horizontal", float(conf), meta

    if vertical_score >= minimum and horizontal_score >= minimum:
        if v_align >= 0.82 and v_align >= h_align + 0.20:
            meta["source"] = "opposite_border_alignment"
            return "vertical", 0.72, meta
        if h_align >= 0.82 and h_align >= v_align + 0.20:
            meta["source"] = "opposite_border_alignment"
            return "horizontal", 0.72, meta

    return fallback_orientation, 0.0, meta



def _trace_terminal_from_symbol(
    image: np.ndarray | None,
    start: tuple[float, float],
    direction: tuple[float, float],
    max_distance: float,
    corridor: int = 4,
) -> tuple[tuple[float, float], bool]:
    """Segue o traço elétrico a partir do corpo/placa até a extremidade local.

    O rastreamento é propositalmente local: exige tinta quase contínua numa
    faixa estreita em torno do eixo. Assim texto e linhas de caderno próximas
    não deslocam o terminal para a borda da bbox.
    """
    if image is None:
        return start, False
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    sx, sy = float(start[0]), float(start[1])
    dx, dy = float(direction[0]), float(direction[1])
    norm = max(1e-9, math.hypot(dx, dy))
    dx, dy = dx / norm, dy / norm
    # limite de escuridão robusto: preserva traço de lápis/tinta e tende a
    # rejeitar linhas pautadas claras.
    local_r = int(max(10, min(28, max_distance * 0.22)))
    xa, xb = max(0, int(round(sx)) - local_r), min(w, int(round(sx)) + local_r + 1)
    ya, yb = max(0, int(round(sy)) - local_r), min(h, int(round(sy)) + local_r + 1)
    patch = gray[ya:yb, xa:xb]
    if patch.size:
        otsu, _ = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_limit = float(min(188.0, max(95.0, otsu * 0.92)))
    else:
        dark_limit = 175.0

    best = (sx, sy)
    seen = False
    gap = 0
    max_gap = 3
    for step in range(0, int(max(2.0, max_distance)) + 1):
        cx = sx + dx * step
        cy = sy + dy * step
        if not (1 <= cx < w - 1 and 1 <= cy < h - 1):
            break
        # faixa perpendicular ao eixo
        px, py = -dy, dx
        samples = []
        for off in range(-corridor, corridor + 1):
            xx = int(round(cx + px * off))
            yy = int(round(cy + py * off))
            if 0 <= xx < w and 0 <= yy < h:
                samples.append((float(gray[yy, xx]), xx, yy))
        if not samples:
            break
        samples.sort(key=lambda z: z[0])
        darkest, xx, yy = samples[0]
        if darkest <= dark_limit:
            best = (float(xx), float(yy))
            seen = True
            gap = 0
        elif seen:
            gap += 1
            if gap > max_gap:
                break
    return best, seen


def _resistor_principal_axis(
    image: np.ndarray | None,
    bbox: dict,
    fallback_orientation: str,
) -> tuple[str, float, dict]:
    """Fallback para resistor desenhado/zig-zag usando PCA da tinta central."""
    meta = {"source": "resistor_pca_not_found", "success": False}
    if image is None:
        return fallback_orientation, 0.0, meta
    h_img, w_img = image.shape[:2]
    x1 = max(0, int(math.floor(float(bbox["x1"]))))
    y1 = max(0, int(math.floor(float(bbox["y1"]))))
    x2 = min(w_img, int(math.ceil(float(bbox["x2"]))) + 1)
    y2 = min(h_img, int(math.ceil(float(bbox["y2"]))) + 1)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return fallback_orientation, 0.0, meta
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    limit = min(185.0, max(90.0, float(otsu) * 0.92))
    mask = (blur <= limit).astype(np.uint8)
    h, w = mask.shape
    # faixa central reduz influência de texto/terminais longos fora do corpo.
    cx0, cx1 = int(w * 0.16), max(int(w * 0.84), int(w * 0.16) + 1)
    cy0, cy1 = int(h * 0.16), max(int(h * 0.84), int(h * 0.16) + 1)
    yy, xx = np.nonzero(mask[cy0:cy1, cx0:cx1] > 0)
    if len(xx) < 14:
        return fallback_orientation, 0.0, meta
    pts = np.column_stack((xx.astype(np.float32) + cx0, yy.astype(np.float32) + cy0))
    center = pts.mean(axis=0)
    centered = pts - center
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vec = vecs[:, order[0]]
    ratio = float(vals[0] / max(1e-6, vals[1]))
    if ratio < 1.35:
        return fallback_orientation, 0.0, meta
    angle = math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 180.0
    orientation = "horizontal" if _angle_distance_180(angle, 0.0) <= 45.0 else "vertical"
    conf = max(0.52, min(0.88, 0.46 + min(3.0, ratio - 1.0) * 0.14))
    gx, gy = float(x1 + center[0]), float(y1 + center[1])
    meta.update({
        "source": "resistor_principal_axis", "success": True,
        "body_center_x": round(gx, 3), "body_center_y": round(gy, 3),
        "axis_angle": round(angle, 3), "eigen_ratio": round(ratio, 3),
    })
    return orientation, conf, meta


def _arrowhead_candidate_on_branch(gray: np.ndarray, branch: dict) -> dict:
    """Retorna cabeça de seta e sentido ao longo de um ramo BJT.

    `direction='outward'` significa ponta mais distante da barra da base
    (esperado para NPN); `inward` significa ponta apontando para a base
    (esperado para PNP).
    """
    near, far = branch["near"], branch["far"]
    ax, ay = float(near[0]), float(near[1])
    bx, by = float(far[0]), float(far[1])
    branch_len = max(1.0, math.hypot(bx - ax, by - ay))
    branch_angle = math.degrees(math.atan2(by - ay, bx - ax)) % 180.0
    edges = cv2.Canny(gray, 20, 85)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=3,
                            minLineLength=3, maxLineGap=2)
    if lines is None:
        return {"score": 0.0, "direction": None, "tip_t": None}
    cand = []
    for row in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = [float(v) for v in row]
        length = math.hypot(x2-x1, y2-y1)
        if length < 3.0 or length > max(8.0, branch_len * 0.68):
            continue
        ang = math.degrees(math.atan2(y2-y1, x2-x1)) % 180.0
        diff = _angle_distance_180(ang, branch_angle)
        if not (12.0 <= diff <= 88.0):
            continue
        mx, my = (x1+x2)/2.0, (y1+y2)/2.0
        dist, t = _point_to_segment_distance(mx, my, ax, ay, bx, by)
        if dist > max(8.0, branch_len * 0.30) or not (0.12 <= t <= 1.05):
            continue
        # uma extremidade da perna da seta deve tocar/aproximar o ramo
        d1, t1 = _point_to_segment_distance(x1, y1, ax, ay, bx, by)
        d2, t2 = _point_to_segment_distance(x2, y2, ax, ay, bx, by)
        if min(d1, d2) > max(5.5, branch_len * 0.18):
            continue
        cand.append({"p1":(x1,y1),"p2":(x2,y2),"len":length,"ang":ang,
                     "t1":t1,"t2":t2})
    best = {"score": 0.0, "direction": None, "tip_t": None}
    join_limit = max(4.5, branch_len * 0.13)
    for i in range(len(cand)):
        for j in range(i+1, len(cand)):
            a,b = cand[i], cand[j]
            if _angle_distance_180(a["ang"], b["ang"]) < 18.0:
                continue
            pairs=[]
            for ka in ("p1","p2"):
                for kb in ("p1","p2"):
                    pa,pb=a[ka],b[kb]
                    pairs.append((math.hypot(pa[0]-pb[0],pa[1]-pb[1]),ka,kb,pa,pb))
            pairs.sort(key=lambda z:z[0])
            d,ka,kb,pa,pb=pairs[0]
            if d > join_limit:
                continue
            tip=((pa[0]+pb[0])/2.0,(pa[1]+pb[1])/2.0)
            tip_dist, tip_t = _point_to_segment_distance(tip[0],tip[1],ax,ay,bx,by)
            if tip_dist > max(5.0, branch_len*0.16) or not (0.10 <= tip_t <= 1.05):
                continue
            # as outras extremidades das duas pernas representam a cauda da cabeça
            oa = a["p2"] if ka=="p1" else a["p1"]
            ob = b["p2"] if kb=="p1" else b["p1"]
            _da, ta = _point_to_segment_distance(oa[0],oa[1],ax,ay,bx,by)
            _db, tb = _point_to_segment_distance(ob[0],ob[1],ax,ay,bx,by)
            tail_t=(ta+tb)/2.0
            delta=tip_t-tail_t
            if abs(delta) < 0.055:
                continue
            direction="outward" if delta>0 else "inward"
            closeness=max(0.2,1.0-d/join_limit)
            score=(a["len"]+b["len"])*closeness*(1.0+min(0.6,abs(delta)))
            if score>best["score"]:
                best={"score":float(score),"direction":direction,
                      "tip_t":float(tip_t),"tail_t":float(tail_t),
                      "tip":tip}
    return best



def _capacitor_plate_axis_any_angle(
    image: np.ndarray | None,
    bbox: dict,
    fallback_orientation: str,
) -> tuple[str,float,dict]:
    """Detecta duas placas paralelas do capacitor em qualquer rotação."""
    meta={"source":"capacitor_rotated_plates_not_found","success":False}
    if image is None:return fallback_orientation,0.0,meta
    h,w=image.shape[:2];bw,bh=_bbox_dims(bbox);pad=int(max(5,min(16,round(min(bw,bh)*.24))))
    x0=max(0,int(math.floor(float(bbox['x1'])))-pad);y0=max(0,int(math.floor(float(bbox['y1'])))-pad)
    x3=min(w,int(math.ceil(float(bbox['x2'])))+pad+1);y3=min(h,int(math.ceil(float(bbox['y2'])))+pad+1)
    if x3-x0<16 or y3-y0<16:return fallback_orientation,0.0,meta
    roi=image[y0:y3,x0:x3];gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) if roi.ndim==3 else roi.copy();gray=cv2.GaussianBlur(gray,(3,3),0);edges=cv2.Canny(gray,30,110)
    min_dim=max(1.0,min(bw,bh));max_dim=max(1.0,max(bw,bh));min_len=max(5,int(min_dim*.12))
    lines=cv2.HoughLinesP(edges,1,np.pi/180.0,threshold=max(5,min_len//2),minLineLength=min_len,maxLineGap=max(3,int(min_dim*.08)))
    if lines is None:return fallback_orientation,0.0,meta
    segs=[];bcx=(float(bbox['x1'])+float(bbox['x2']))/2.0;bcy=(float(bbox['y1'])+float(bbox['y2']))/2.0
    for row in np.asarray(lines).reshape(-1,4):
        p1=np.array([float(row[0]+x0),float(row[1]+y0)]);p2=np.array([float(row[2]+x0),float(row[3]+y0)])
        v=p2-p1;L=float(np.linalg.norm(v))
        if L<min_len or L>max_dim*1.25:continue
        u=v/max(1e-9,L);ang=math.degrees(math.atan2(u[1],u[0]))%180.0;mid=(p1+p2)/2.0
        segs.append({'p1':p1,'p2':p2,'u':u,'length':L,'angle':ang,'mid':mid})
    best=None
    for i in range(len(segs)):
        a=segs[i]
        for j in range(i+1,len(segs)):
            b=segs[j]
            if _angle_distance_180(a['angle'],b['angle'])>13.0:continue
            # Usa direção média, com sinais alinhados.
            ub=b['u'].copy()
            if float(np.dot(a['u'],ub))<0:ub=-ub
            u=a['u']+ub;ul=float(np.linalg.norm(u))
            if ul<1e-6:continue
            u/=ul;n=np.array([-u[1],u[0]])
            # Separação perpendicular das placas.
            gap=abs(float(np.dot(b['mid']-a['mid'],n)))
            if gap<max(3.0,min_dim*.055) or gap>max(20.0,min_dim*.52):continue
            # Sobreposição ao longo das placas.
            def interval(seg):
                t1=float(np.dot(seg['p1'],u));t2=float(np.dot(seg['p2'],u));return (min(t1,t2),max(t1,t2))
            a0,a1=interval(a);b0,b1=interval(b);ov0=max(a0,b0);ov1=min(a1,b1);over=max(0.0,ov1-ov0)
            short=max(1.0,min(a['length'],b['length']));overlap=over/short
            if overlap<.22:continue
            sim=short/max(a['length'],b['length'])
            if sim<.35:continue
            pair_center=(a['mid']+b['mid'])/2.0;cd=math.hypot(float(pair_center[0]-bcx),float(pair_center[1]-bcy))/max(1.0,math.hypot(bw,bh))
            if cd>.65:continue
            # Linhas de caderno quase do tamanho inteiro recebem penalidade.
            long_ratio=max(a['length'],b['length'])/max(1.0,max_dim+2*pad);line_pen=max(0.0,long_ratio-.78)*3.0
            target=max(4.0,min_dim*.18);gq=max(0.0,1.0-abs(gap-target)/max(5.0,min_dim*.32))
            score=overlap*2.2+sim*1.0+(1-min(1,cd/.65))*1.7+gq*.85-line_pen
            if best is None or score>best[0]:best=(score,a,b,u,n,gap,pair_center,ov0,ov1,overlap,sim)
    if best is None:return fallback_orientation,0.0,meta
    score,a,b,u,n,gap,center,ov0,ov1,overlap,sim=best
    if score<2.35:return fallback_orientation,0.0,meta
    # Centro no meio da região sobreposta das placas.
    tc=(ov0+ov1)/2.0; normal_coord=(float(np.dot(a['mid'],n))+float(np.dot(b['mid'],n)))/2.0
    axis_center=u*tc+n*normal_coord
    electrical_angle=math.degrees(math.atan2(float(n[1]),float(n[0])))%180.0
    orientation='horizontal' if _angle_distance_180(electrical_angle,0.0)<=45.0 else 'vertical'
    # Decide qual placa fica em cada lado do eixo normal e traça para fora.
    da=float(np.dot(a['mid']-axis_center,n));db=float(np.dot(b['mid']-axis_center,n))
    neg=a if da<=db else b;pos=b if da<=db else a
    neg_start=axis_center+n*min(da,db);pos_start=axis_center+n*max(da,db)
    max_trace=max(12.0,max_dim*.70);corr=max(2,int(min_dim*.08))
    pneg,ok1=_trace_terminal_from_symbol(image,(float(neg_start[0]),float(neg_start[1])),(-float(n[0]),-float(n[1])),max_trace,corridor=corr)
    ppos,ok2=_trace_terminal_from_symbol(image,(float(pos_start[0]),float(pos_start[1])),(float(n[0]),float(n[1])),max_trace,corridor=corr)
    points=sorted([pneg,ppos],key=(lambda p:(p[0],p[1])) if orientation=='horizontal' else (lambda p:(p[1],p[0])))
    conf=max(.62,min(.98,.53+(score-2.2)*.17))
    meta.update({'source':'capacitor_parallel_plates_any_angle','success':True,'plate_pair_score':round(score,4),'plate_gap':round(gap,3),'plate_center_x':round(float(axis_center[0]),3),'plate_center_y':round(float(axis_center[1]),3),'plate_angle_degrees':round((electrical_angle+90)%180,3),'electrical_axis_angle':round(electrical_angle,3),'axis_point_1':[round(points[0][0],3),round(points[0][1],3)],'axis_point_2':[round(points[1][0],3),round(points[1][1],3)],'trace_success':bool(ok1 and ok2),'plate_overlap_ratio':round(overlap,3),'plate_length_similarity':round(sim,3)})
    return orientation,float(conf),meta


def _capacitor_plate_axis(
    image: np.ndarray | None,
    bbox: dict,
    fallback_orientation: str,
) -> tuple[str, float, dict]:
    """Detecta as duas placas paralelas do capacitor.

    O eixo ELÉTRICO é perpendicular às placas. Isso evita que uma bbox larga,
    texto próximo ou linhas do caderno façam um capacitor vertical virar
    horizontal. A pontuação privilegia dois segmentos curtos, paralelos,
    próximos entre si e próximos ao centro da bbox.
    """
    meta = {
        "source": "capacitor_plates_not_found",
        "success": False,
        "plate_orientation": None,
        "plate_pair_score": 0.0,
        "plate_gap": None,
        "plate_center_x": None,
        "plate_center_y": None,
    }
    if image is None:
        return fallback_orientation, 0.0, meta

    h_img, w_img = image.shape[:2]
    bw, bh = _bbox_dims(bbox)
    pad = int(max(5, min(14, round(min(bw, bh) * 0.22))))
    x0 = max(0, int(math.floor(float(bbox["x1"]))) - pad)
    y0 = max(0, int(math.floor(float(bbox["y1"]))) - pad)
    x3 = min(w_img, int(math.ceil(float(bbox["x2"]))) + pad + 1)
    y3 = min(h_img, int(math.ceil(float(bbox["y2"]))) + pad + 1)
    if x3 - x0 < 16 or y3 - y0 < 16:
        return fallback_orientation, 0.0, meta

    roi = image[y0:y3, x0:x3]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 32, 115)
    min_dim = max(1.0, min(bw, bh))
    max_dim = max(1.0, max(bw, bh))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=max(5, int(min_dim * 0.10)),
        minLineLength=max(5, int(min_dim * 0.12)),
        maxLineGap=max(3, int(min_dim * 0.08)),
    )
    if lines is None:
        return fallback_orientation, 0.0, meta

    bbox_cx = (float(bbox["x1"]) + float(bbox["x2"])) / 2.0
    bbox_cy = (float(bbox["y1"]) + float(bbox["y2"])) / 2.0
    segments = {"horizontal": [], "vertical": []}
    for row in np.asarray(lines).reshape(-1, 4):
        ax, ay, bx_, by_ = [float(v) for v in row]
        ax += x0; bx_ += x0; ay += y0; by_ += y0
        dx, dy = bx_ - ax, by_ - ay
        length = math.hypot(dx, dy)
        if length < max(5.0, min_dim * 0.12):
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        if _angle_distance_180(angle, 0.0) <= 22.0:
            kind = "horizontal"
        elif _angle_distance_180(angle, 90.0) <= 22.0:
            kind = "vertical"
        else:
            continue
        mx, my = (ax + bx_) / 2.0, (ay + by_) / 2.0
        segments[kind].append({
            "p1": (ax, ay), "p2": (bx_, by_), "length": length,
            "angle": angle, "mx": mx, "my": my,
        })

    best = None
    for kind, segs in segments.items():
        for i in range(len(segs)):
            a = segs[i]
            for j in range(i + 1, len(segs)):
                b = segs[j]
                if _angle_distance_180(a["angle"], b["angle"]) > 15.0:
                    continue
                if kind == "horizontal":
                    gap = abs(a["my"] - b["my"])
                    a0, a1 = sorted((a["p1"][0], a["p2"][0]))
                    b0, b1 = sorted((b["p1"][0], b["p2"][0]))
                    overlap = max(0.0, min(a1, b1) - max(a0, b0))
                    pair_cx = (a["mx"] + b["mx"]) / 2.0
                    pair_cy = (a["my"] + b["my"]) / 2.0
                    directional_span = float(x3 - x0)
                else:
                    gap = abs(a["mx"] - b["mx"])
                    a0, a1 = sorted((a["p1"][1], a["p2"][1]))
                    b0, b1 = sorted((b["p1"][1], b["p2"][1]))
                    overlap = max(0.0, min(a1, b1) - max(a0, b0))
                    pair_cx = (a["mx"] + b["mx"]) / 2.0
                    pair_cy = (a["my"] + b["my"]) / 2.0
                    directional_span = float(y3 - y0)

                if gap < max(3.0, min_dim * 0.06) or gap > max(18.0, min_dim * 0.48):
                    continue
                short_len = max(1.0, min(a["length"], b["length"]))
                overlap_ratio = overlap / short_len
                if overlap_ratio < 0.18:
                    continue
                length_similarity = short_len / max(a["length"], b["length"])
                if length_similarity < 0.30:
                    continue

                # Duas bordas do MESMO fio grosso também formam um par paralelo.
                # Nessa situação a faixa entre as retas fica escura/preenchida.
                # Entre duas placas de capacitor, ao contrário, há majoritariamente
                # fundo claro (com no máximo o fio central cruzando a região).
                if kind == "horizontal":
                    xa = int(max(0, math.floor(max(min(a["p1"][0], a["p2"][0]), min(b["p1"][0], b["p2"][0])) - x0)))
                    xb = int(min(x3 - x0, math.ceil(min(max(a["p1"][0], a["p2"][0]), max(b["p1"][0], b["p2"][0])) - x0)))
                    ya = int(max(0, math.floor(min(a["my"], b["my"]) - y0 + 1)))
                    yb = int(min(y3 - y0, math.ceil(max(a["my"], b["my"]) - y0 - 1)))
                else:
                    ya = int(max(0, math.floor(max(min(a["p1"][1], a["p2"][1]), min(b["p1"][1], b["p2"][1])) - y0)))
                    yb = int(min(y3 - y0, math.ceil(min(max(a["p1"][1], a["p2"][1]), max(b["p1"][1], b["p2"][1])) - y0)))
                    xa = int(max(0, math.floor(min(a["mx"], b["mx"]) - x0 + 1)))
                    xb = int(min(x3 - x0, math.ceil(max(a["mx"], b["mx"]) - x0 - 1)))
                interior_dark = 0.0
                if xb - xa >= 2 and yb - ya >= 2:
                    patch = gray[ya:yb, xa:xb]
                    if patch.size:
                        interior_dark = float(np.mean(patch < 150))
                        if interior_dark > 0.46:
                            continue

                # Linhas pautadas tendem a atravessar quase a ROI inteira.
                long_ratio = max(a["length"], b["length"]) / max(1.0, directional_span)
                line_penalty = max(0.0, (long_ratio - 0.72) * 2.5)
                center_dx = abs(pair_cx - bbox_cx) / max(1.0, bw)
                center_dy = abs(pair_cy - bbox_cy) / max(1.0, bh)
                centrality = max(0.0, 1.0 - min(1.0, math.hypot(center_dx, center_dy) / 0.65))
                gap_target = max(4.0, min_dim * 0.18)
                gap_quality = max(0.0, 1.0 - abs(gap - gap_target) / max(5.0, min_dim * 0.30))
                score = (
                    overlap_ratio * 2.1 + length_similarity * 1.0 + centrality * 1.7
                    + gap_quality * 0.8 - line_penalty
                )
                if best is None or score > best[0]:
                    best = (score, kind, a, b, gap, pair_cx, pair_cy, overlap_ratio, length_similarity)

    if best is None:
        return fallback_orientation, 0.0, meta

    score, plate_kind, a, b, gap, pcx, pcy, overlap_ratio, length_similarity = best
    # Placas horizontais => terminais em cima/baixo (eixo elétrico vertical).
    electrical_axis = "vertical" if plate_kind == "horizontal" else "horizontal"
    # Exige um score razoável para evitar que duas linhas de caderno virem capacitor.
    if score < 2.30:
        return fallback_orientation, 0.0, meta

    confidence = max(0.60, min(0.98, 0.52 + (score - 2.15) * 0.17))
    # Centro do trecho comum das duas placas: é daqui que o eixo elétrico
    # deve sair. Não usamos a borda da bbox como posição de terminal.
    if plate_kind == "horizontal":
        a0, a1 = sorted((a["p1"][0], a["p2"][0])); b0, b1 = sorted((b["p1"][0], b["p2"][0]))
        ov0, ov1 = max(a0,b0), min(a1,b1)
        axis_coord = (ov0+ov1)/2.0 if ov1>ov0 else float(pcx)
        top_seg, bottom_seg = (a,b) if a["my"] <= b["my"] else (b,a)
        top_start=(axis_coord,float(top_seg["my"])); bottom_start=(axis_coord,float(bottom_seg["my"]))
        top_point, top_ok = _trace_terminal_from_symbol(image, top_start, (0.0,-1.0), max(10.0,bh*0.65), corridor=max(2,int(min_dim*0.08)))
        bottom_point, bottom_ok = _trace_terminal_from_symbol(image, bottom_start, (0.0,1.0), max(10.0,bh*0.65), corridor=max(2,int(min_dim*0.08)))
    else:
        a0, a1 = sorted((a["p1"][1], a["p2"][1])); b0, b1 = sorted((b["p1"][1], b["p2"][1]))
        ov0, ov1 = max(a0,b0), min(a1,b1)
        axis_coord = (ov0+ov1)/2.0 if ov1>ov0 else float(pcy)
        left_seg, right_seg = (a,b) if a["mx"] <= b["mx"] else (b,a)
        left_start=(float(left_seg["mx"]),axis_coord); right_start=(float(right_seg["mx"]),axis_coord)
        left_point, left_ok = _trace_terminal_from_symbol(image, left_start, (-1.0,0.0), max(10.0,bw*0.65), corridor=max(2,int(min_dim*0.08)))
        right_point, right_ok = _trace_terminal_from_symbol(image, right_start, (1.0,0.0), max(10.0,bw*0.65), corridor=max(2,int(min_dim*0.08)))

    meta.update({
        "source": "capacitor_parallel_plates",
        "success": True,
        "plate_orientation": plate_kind,
        "plate_pair_score": round(float(score), 4),
        "plate_gap": round(float(gap), 3),
        "plate_center_x": round(float(pcx), 3),
        "plate_center_y": round(float(pcy), 3),
        "plate_overlap_ratio": round(float(overlap_ratio), 3),
        "plate_length_similarity": round(float(length_similarity), 3),
    })
    if electrical_axis == "vertical":
        meta["top_x"] = round(float(axis_coord), 3)
        meta["bottom_x"] = round(float(axis_coord), 3)
        meta["top_point"] = [round(float(top_point[0]),3), round(float(top_point[1]),3)]
        meta["bottom_point"] = [round(float(bottom_point[0]),3), round(float(bottom_point[1]),3)]
        meta["trace_success"] = bool(top_ok and bottom_ok)
    else:
        meta["left_y"] = round(float(axis_coord), 3)
        meta["right_y"] = round(float(axis_coord), 3)
        meta["left_point"] = [round(float(left_point[0]),3), round(float(left_point[1]),3)]
        meta["right_point"] = [round(float(right_point[0]),3), round(float(right_point[1]),3)]
        meta["trace_success"] = bool(left_ok and right_ok)
    return electrical_axis, float(confidence), meta


def _resistor_body_axis(
    image: np.ndarray | None,
    bbox: dict,
    fallback_orientation: str,
) -> tuple[str, float, dict]:
    """Detecta o eixo do resistor pelo RETÂNGULO do corpo, não pela bbox YOLO.

    Em esquemas reais a bbox pode incluir texto, fios ou nós e ficar mais larga
    que o corpo. Para resistores retangulares IEC, a orientação elétrica segue
    o lado longo do retângulo do componente. Se o retângulo não for confiável
    (ex.: resistor zig-zag), fazemos fallback para os fios/bbox.
    """
    meta = {
        "source": "resistor_body_not_found",
        "success": False,
        "body_center_x": None,
        "body_center_y": None,
        "body_width": None,
        "body_height": None,
        "body_long_axis_angle": None,
    }
    if image is None:
        return fallback_orientation, 0.0, meta

    h_img, w_img = image.shape[:2]
    x1 = max(0, int(math.floor(float(bbox["x1"]))))
    y1 = max(0, int(math.floor(float(bbox["y1"]))))
    x2 = min(w_img, int(math.ceil(float(bbox["x2"]))) + 1)
    y2 = min(h_img, int(math.ceil(float(bbox["y2"]))) + 1)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return fallback_orientation, 0.0, meta

    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _t, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Fecha pequenos buracos de retângulos desenhados à mão sem engrossar demais os fios.
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _hier = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    roi_area = float(max(1, roi.shape[0] * roi.shape[1]))
    rcx, rcy = roi.shape[1] / 2.0, roi.shape[0] / 2.0
    candidates = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < max(18.0, roi_area * 0.002):
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), _angle = rect
        rw, rh = float(rw), float(rh)
        if rw < 4.0 or rh < 4.0:
            continue
        long_side, short_side = max(rw, rh), min(rw, rh)
        ratio = long_side / max(1e-6, short_side)
        box_area = rw * rh
        if not (1.35 <= ratio <= 7.0):
            continue
        if not (roi_area * 0.012 <= box_area <= roi_area * 0.58):
            continue
        fill = area / max(1.0, box_area)
        if fill < 0.42:
            continue

        box = cv2.boxPoints(rect).astype(np.float32)
        edges = []
        for i in range(4):
            a = box[i]
            b = box[(i + 1) % 4]
            dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
            length = math.hypot(dx, dy)
            edges.append((length, dx, dy))
        _len, dx, dy = max(edges, key=lambda item: item[0])
        axis_angle = math.degrees(math.atan2(dy, dx)) % 180.0
        axis_orientation = "horizontal" if _angle_distance_180(axis_angle, 0.0) <= 45.0 else "vertical"

        center_dist = math.hypot(cx - rcx, cy - rcy) / max(1.0, math.hypot(rcx, rcy))
        # Retângulo do corpo: alta ocupação, tamanho relevante e razoavelmente central.
        score = (fill * 2.2) + min(1.5, box_area / max(1.0, roi_area * 0.10)) + min(1.0, ratio / 3.0) - center_dist * 0.55
        candidates.append((score, rect, box, axis_angle, axis_orientation, fill, ratio, area))

    if not candidates:
        return fallback_orientation, 0.0, meta

    candidates.sort(key=lambda item: item[0], reverse=True)
    score, rect, box, axis_angle, orientation, fill, ratio, area = candidates[0]
    (cx, cy), (rw, rh), _ = rect
    gx, gy = float(x1 + cx), float(y1 + cy)
    xs, ys = box[:, 0] + x1, box[:, 1] + y1
    body_x1, body_x2 = float(np.min(xs)), float(np.max(xs))
    body_y1, body_y2 = float(np.min(ys)), float(np.max(ys))

    confidence = max(0.62, min(0.98, 0.58 + fill * 0.24 + min(ratio, 4.0) * 0.04))
    meta.update({
        "source": "resistor_body_rectangle",
        "success": True,
        "body_center_x": round(gx, 3),
        "body_center_y": round(gy, 3),
        "body_width": round(body_x2 - body_x1, 3),
        "body_height": round(body_y2 - body_y1, 3),
        "body_bbox": {
            "x1": round(body_x1, 3), "y1": round(body_y1, 3),
            "x2": round(body_x2, 3), "y2": round(body_y2, 3),
        },
        "body_long_axis_angle": round(float(axis_angle), 3),
        "body_long_length": round(float(max(rw, rh)), 3),
        "body_short_length": round(float(min(rw, rh)), 3),
        "fill_ratio": round(float(fill), 4),
        "elongation": round(float(ratio), 4),
        "score": round(float(score), 4),
    })
    return orientation, confidence, meta


def _snap_to_local_connection(
    image: np.ndarray | None,
    point: tuple[float, float],
    orientation: str,
    radius: int,
) -> tuple[float, float, bool, str]:
    """Centraliza o terminal num nó/fio próximo sem construir nets ainda.

    Prioridade 1: centro de uma região espessa (bolinha/junção).
    Prioridade 2: centro da linha elétrica local. Isso melhora o overlay e
    prepara a Etapa 8, mas não declara conectividade elétrica por si só.
    """
    if image is None:
        return point[0], point[1], False, "none"
    h, w = image.shape[:2]
    px, py = float(point[0]), float(point[1])
    r = int(max(6, min(18, radius)))
    x1, x2 = max(0, int(round(px)) - r), min(w, int(round(px)) + r + 1)
    y1, y2 = max(0, int(round(py)) - r), min(h, int(round(py)) + r + 1)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return px, py, False, "none"
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if not np.any(mask):
        return px, py, False, "none"

    # Bolinhas/junções têm espessura local maior que fios comuns. O máximo da
    # transformada de distância fornece aproximadamente o centro geométrico.
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    maxv = float(dist.max()) if dist.size else 0.0
    if maxv >= 2.25:
        candidates = np.argwhere(dist >= maxv * 0.88)
        if len(candidates):
            # Prefere o máximo espesso mais próximo do ponto original.
            gy = candidates[:, 0].astype(np.float32) + y1
            gx = candidates[:, 1].astype(np.float32) + x1
            d2 = (gx - px) ** 2 + (gy - py) ** 2
            idx = int(np.argmin(d2))
            junction_limit = float(max(4.0, min(8.0, r * 0.65)))
            if float(d2[idx]) <= junction_limit * junction_limit:
                # Média da ilha de máximos próxima para cair no centro da bolinha.
                near = d2 <= min(junction_limit * junction_limit, float(d2[idx]) + 9.0)
                return float(np.mean(gx[near])), float(np.mean(gy[near])), True, "junction_center"

    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return px, py, False, "none"
    gx = xs.astype(np.float32) + x1
    gy = ys.astype(np.float32) + y1

    if orientation == "vertical":
        close = np.abs(gy - py) <= max(4.0, r * 0.65)
        if np.any(close):
            # Coluna com maior suporte de tinta: centro do fio vertical.
            vals = gx[close].astype(np.int32)
            uniq, counts = np.unique(vals, return_counts=True)
            # Uma borda grossa do corpo do componente pode ter mais pixels que
            # o fio. Penalizamos deslocamentos perpendiculares grandes.
            scores = counts.astype(np.float32) - np.abs(uniq.astype(np.float32) - px) * 1.15
            bestx = float(uniq[int(np.argmax(scores))])
            if abs(bestx - px) <= max(4.0, r * 0.45):
                return bestx, py, True, "wire_centerline"
    else:
        close = np.abs(gx - px) <= max(4.0, r * 0.65)
        if np.any(close):
            vals = gy[close].astype(np.int32)
            uniq, counts = np.unique(vals, return_counts=True)
            scores = counts.astype(np.float32) - np.abs(uniq.astype(np.float32) - py) * 1.15
            besty = float(uniq[int(np.argmax(scores))])
            if abs(besty - py) <= max(4.0, r * 0.45):
                return px, besty, True, "wire_centerline"
    return px, py, False, "none"


def _ink_bridge_between(image: np.ndarray | None, p1: tuple[float, float], p2: tuple[float, float]) -> bool:
    if image is None:
        return False
    x1, y1 = p1; x2, y2 = p2
    distance = math.hypot(x2 - x1, y2 - y1)
    if distance > 18.0:
        return False
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    n = max(3, int(math.ceil(distance)) + 1)
    hits = 0
    for t in np.linspace(0.0, 1.0, n):
        x = int(round(x1 + (x2 - x1) * float(t)))
        y = int(round(y1 + (y2 - y1) * float(t)))
        xa, xb = max(0, x - 1), min(w, x + 2)
        ya, yb = max(0, y - 1), min(h, y + 2)
        if xa < xb and ya < yb and float(np.mean(gray[ya:yb, xa:xb])) < 185.0:
            hits += 1
    return hits / max(1, n) >= 0.45


def _align_nearby_terminals(components: list[dict], image: np.ndarray | None, tolerance: float = 8.0) -> int:
    """Snap visual local para terminais simples próximos.

    IMPORTANTE: transistores e ground-truth/modelo treinado são protegidos.
    B/C/E são keypoints semânticos e não podem ser deslocados para outro
    componente só porque há tinta/terminal a poucos pixels. A conectividade
    definitiva pertence às Etapas 8-10.
    """
    refs = []
    protected_sources = {
        "trained_terminal_keypoint_model",
        "manual_ground_truth",
        "manual_ground_truth_persisted",
        "visible_symbol_terminal_identity",
    }
    for ci, comp in enumerate(components):
        if str(comp.get("type") or "") in {"transistor_npn", "transistor_pnp"}:
            continue
        for ti, term in enumerate(comp.get("terminals") or []):
            if str(term.get("source") or "") in protected_sources:
                continue
            pos = term.get("position") or {}
            if "x" in pos and "y" in pos:
                refs.append((ci, ti, term))
    parent = list(range(len(refs)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(refs)):
        ci, _ti, a = refs[i]
        pa = (float(a["position"]["x"]), float(a["position"]["y"]))
        for j in range(i + 1, len(refs)):
            cj, _tj, b = refs[j]
            if ci == cj:
                continue
            pb = (float(b["position"]["x"]), float(b["position"]["y"]))
            if math.hypot(pa[0] - pb[0], pa[1] - pb[1]) <= tolerance and _ink_bridge_between(image, pa, pb):
                union(i, j)

    groups = {}
    for i in range(len(refs)):
        groups.setdefault(find(i), []).append(i)
    aligned = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        xs = [float(refs[i][2]["position"]["x"]) for i in members]
        ys = [float(refs[i][2]["position"]["y"]) for i in members]
        common = (float(np.mean(xs)), float(np.mean(ys)))
        sx, sy, _ok, _mode = _snap_to_local_connection(image, common, "horizontal", 7)
        for i in members:
            term = refs[i][2]
            term["position"] = {"x": round(sx, 2), "y": round(sy, 2)}
            term["connection_snap"] = {
                "mode": "nearby_terminal_cluster",
                "cluster_size": len(members),
                "tolerance_px": tolerance,
            }
            aligned += 1
    return aligned


def _rotate_90_clockwise(x: float, y: float) -> tuple[float, float]:
    return 1.0 - y, x


def _normalized_layout(component_type: str, orientation: str):
    layout = PIN_LIBRARY[component_type]
    if orientation != "vertical" or component_type in VERTICAL_DEFAULT:
        return layout
    return [(name, *_rotate_90_clockwise(x, y)) for name, x, y in layout]


def _pixel_from_normalized(bbox: dict, nx: float, ny: float) -> tuple[float, float]:
    x1, y1 = float(bbox["x1"]), float(bbox["y1"])
    x2, y2 = float(bbox["x2"]), float(bbox["y2"])
    return x1 + nx * (x2 - x1), y1 + ny * (y2 - y1)


def _search_radius(bbox: dict) -> int:
    width, height = _bbox_dims(bbox)
    return int(round(max(6.0, min(28.0, min(width, height) * 0.24))))


def _clip_point(x: float, y: float, image_width: int | None, image_height: int | None) -> tuple[float, float]:
    if image_width is not None:
        x = max(0.0, min(float(image_width - 1), x))
    if image_height is not None:
        y = max(0.0, min(float(image_height - 1), y))
    return x, y


def _angle_distance_180(a: float, b: float) -> float:
    """Menor distância entre dois eixos (não vetores) em graus."""
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def _estimate_axis_angle(image: np.ndarray | None, bbox: dict, orientation: str) -> tuple[float, float, str]:
    """Estima pequena rotação do símbolo usando segmentos visuais dentro da bbox.

    A bbox continua definindo horizontal/vertical. Hough é usado somente para
    corrigir inclinações de foto/desenho (ex.: resistor a 8 graus), evitando
    trocar o eixo por causa de detalhes internos do símbolo.
    """
    base_angle = 0.0 if orientation == "horizontal" else 90.0
    if image is None:
        return base_angle, 0.0, "bbox_axis"

    h_img, w_img = image.shape[:2]
    x1 = max(0, int(math.floor(float(bbox["x1"]))))
    y1 = max(0, int(math.floor(float(bbox["y1"]))))
    x2 = min(w_img, int(math.ceil(float(bbox["x2"]))) + 1)
    y2 = min(h_img, int(math.ceil(float(bbox["y2"]))) + 1)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return base_angle, 0.0, "bbox_axis"

    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 45, 135)
    min_len = max(8, int(min(gray.shape[:2]) * 0.20))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(8, min_len // 2),
        minLineLength=min_len,
        maxLineGap=max(4, min_len // 3),
    )
    if lines is None:
        return base_angle, 0.0, "bbox_axis"

    # O OpenCV pode devolver HoughLinesP como (N, 1, 4) ou (N, 4),
    # dependendo da versão/build. Normalizamos sempre para (N, 4).
    line_rows = np.asarray(lines).reshape(-1, 4)

    accepted: list[tuple[float, float]] = []
    for row in line_rows:
        lx1, ly1, lx2, ly2 = [float(v) for v in row]
        dx, dy = lx2 - lx1, ly2 - ly1
        length = math.hypot(dx, dy)
        if length < min_len:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        if _angle_distance_180(angle, base_angle) <= 28.0:
            # Representação circular de eixo: duplicar ângulo evita problema 179/1 graus.
            accepted.append((angle, length))

    if not accepted:
        return base_angle, 0.0, "bbox_axis"

    sx = sum(math.cos(math.radians(2.0 * a)) * w for a, w in accepted)
    sy = sum(math.sin(math.radians(2.0 * a)) * w for a, w in accepted)
    angle = (0.5 * math.degrees(math.atan2(sy, sx))) % 180.0
    # Normaliza para perto do eixo base (ex.: 178° -> -2° para horizontal).
    while angle - base_angle > 90.0:
        angle -= 180.0
    while angle - base_angle < -90.0:
        angle += 180.0

    spread = sum(_angle_distance_180(a, angle) * w for a, w in accepted) / max(
        1.0, sum(w for _, w in accepted)
    )
    confidence = max(0.15, min(0.95, 1.0 - spread / 35.0))
    return angle, confidence, "local_hough"


def _line_bbox_intersections(bbox: dict, angle_degrees: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Interseções do eixo que passa pelo centro com a bbox axis-aligned."""
    x1, y1 = float(bbox["x1"]), float(bbox["y1"])
    x2, y2 = float(bbox["x2"]), float(bbox["y2"])
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    theta = math.radians(angle_degrees)
    dx, dy = math.cos(theta), math.sin(theta)

    candidates: list[tuple[float, float, float]] = []
    eps = 1e-9
    if abs(dx) > eps:
        for x in (x1, x2):
            t = (x - cx) / dx
            y = cy + t * dy
            if y1 - 1e-6 <= y <= y2 + 1e-6:
                candidates.append((t, x, y))
    if abs(dy) > eps:
        for y in (y1, y2):
            t = (y - cy) / dy
            x = cx + t * dx
            if x1 - 1e-6 <= x <= x2 + 1e-6:
                candidates.append((t, x, y))

    if len(candidates) < 2:
        return (x1, cy), (x2, cy)
    candidates.sort(key=lambda item: item[0])
    return (candidates[0][1], candidates[0][2]), (candidates[-1][1], candidates[-1][2])


def _ink_mask(image: np.ndarray, bbox: dict, pad: int) -> tuple[np.ndarray, int, int] | None:
    h_img, w_img = image.shape[:2]
    x1 = max(0, int(math.floor(float(bbox["x1"]))) - pad)
    y1 = max(0, int(math.floor(float(bbox["y1"]))) - pad)
    x2 = min(w_img, int(math.ceil(float(bbox["x2"]))) + pad + 1)
    y2 = min(h_img, int(math.ceil(float(bbox["y2"]))) + pad + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Elimina pontinhos isolados sem apagar fios finos.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask, x1, y1


def _snap_to_ink(image: np.ndarray | None, bbox: dict, point: tuple[float, float], radius: int) -> tuple[float, float, bool]:
    """Ajusta o candidato para o traço escuro mais próximo, sem fazer detecção de fios global."""
    if image is None:
        return point[0], point[1], False
    packed = _ink_mask(image, bbox, pad=max(4, radius))
    if packed is None:
        return point[0], point[1], False
    mask, ox, oy = packed
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return point[0], point[1], False
    gx = xs.astype(np.float32) + float(ox)
    gy = ys.astype(np.float32) + float(oy)
    px, py = point
    dist2 = (gx - px) ** 2 + (gy - py) ** 2
    limit = float(max(6, radius)) ** 2
    valid = dist2 <= limit
    if not np.any(valid):
        return point[0], point[1], False
    valid_idx = np.where(valid)[0]
    best = valid_idx[int(np.argmin(dist2[valid_idx]))]
    return float(gx[best]), float(gy[best]), True


def _snap_to_ink_constrained(
    image: np.ndarray | None,
    bbox: dict,
    point: tuple[float, float],
    radius: int,
    *,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> tuple[float, float, bool]:
    if image is None:
        return point[0], point[1], False
    packed = _ink_mask(image, bbox, pad=max(4, radius))
    if packed is None:
        return point[0], point[1], False
    mask, ox, oy = packed
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return point[0], point[1], False
    gx = xs.astype(np.float32) + float(ox)
    gy = ys.astype(np.float32) + float(oy)
    px, py = point
    dist2 = (gx - px) ** 2 + (gy - py) ** 2
    limit = float(max(6, radius)) ** 2
    valid = dist2 <= limit
    if x_min is not None:
        valid &= gx >= float(x_min)
    if x_max is not None:
        valid &= gx <= float(x_max)
    if y_min is not None:
        valid &= gy >= float(y_min)
    if y_max is not None:
        valid &= gy <= float(y_max)
    if not np.any(valid):
        return point[0], point[1], False
    valid_idx = np.where(valid)[0]
    best = valid_idx[int(np.argmin(dist2[valid_idx]))]
    return float(gx[best]), float(gy[best]), True


def _expected_side(nx: float, ny: float) -> tuple[str, float]:
    """Retorna a borda esperada e a fração ao longo dela.

    A biblioteca de pinos usa posições normalizadas na bbox. Para os
    componentes prioritários, o ponto final deve cair no fio que chega à
    periferia do símbolo, e não em um detalhe interno (círculo, placa etc.).
    """
    distances = {
        "left": abs(nx),
        "right": abs(1.0 - nx),
        "top": abs(ny),
        "bottom": abs(1.0 - ny),
    }
    side = min(distances, key=distances.get)
    fraction = ny if side in {"left", "right"} else nx
    return side, float(max(0.0, min(1.0, fraction)))


def _snap_to_border_ink(
    image: np.ndarray | None,
    bbox: dict,
    nx: float,
    ny: float,
    radius: int,
) -> tuple[float, float, bool, str]:
    """Procura a extremidade real do traço na borda esperada da bbox.

    O snap anterior procurava simplesmente o pixel escuro mais próximo do
    ponto teórico. Em transistores isso podia escolher a circunferência em vez
    do terminal C/E. Aqui primeiro restringimos a busca ao lado elétrico
    esperado e depois escolhemos o pixel mais externo daquele traço.
    """
    px, py = _pixel_from_normalized(bbox, nx, ny)
    side, fraction = _expected_side(nx, ny)
    if image is None:
        return px, py, False, side

    packed = _ink_mask(image, bbox, pad=max(5, radius))
    if packed is None:
        return px, py, False, side
    mask, ox, oy = packed
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return px, py, False, side

    gx = xs.astype(np.float32) + float(ox)
    gy = ys.astype(np.float32) + float(oy)
    x1, y1 = float(bbox["x1"]), float(bbox["y1"])
    x2, y2 = float(bbox["x2"]), float(bbox["y2"])
    width, height = _bbox_dims(bbox)

    # Janela ortogonal generosa o bastante para pequenas rotações, mas estreita
    # para não capturar outro pino do mesmo componente.
    if side in {"left", "right"}:
        target = y1 + fraction * height
        tol = max(7.0, min(height * 0.18, radius * 1.45))
        valid = np.abs(gy - target) <= tol
        if not np.any(valid):
            return px, py, False, side
        vx, vy = gx[valid], gy[valid]
        extreme = float(vx.min() if side == "left" else vx.max())
        near = np.abs(vx - extreme) <= 2.5
        if not np.any(near):
            return px, py, False, side
        sx = float(np.median(vx[near]))
        sy = float(np.median(vy[near]))
    else:
        target = x1 + fraction * width
        tol = max(7.0, min(width * 0.18, radius * 1.45))
        valid = np.abs(gx - target) <= tol
        if not np.any(valid):
            return px, py, False, side
        vx, vy = gx[valid], gy[valid]
        extreme = float(vy.min() if side == "top" else vy.max())
        near = np.abs(vy - extreme) <= 2.5
        if not np.any(near):
            return px, py, False, side
        sx = float(np.median(vx[near]))
        sy = float(np.median(vy[near]))

    # Evita aceitar um detalhe interno como terminal caso ele esteja muito
    # distante da borda esperada da detecção.
    edge_gap = {
        "left": abs(sx - x1),
        "right": abs(x2 - sx),
        "top": abs(sy - y1),
        "bottom": abs(y2 - sy),
    }[side]
    # Não deixe o snap atravessar uma pequena folga e agarrar o CORPO de um
    # componente vizinho. A correção de conexões próximas é feita depois pelo
    # snap local/cluster, com verificação de tinta entre os pontos.
    max_edge_gap = max(5.0, min(9.0, float(radius) * 0.55))
    if edge_gap > max_edge_gap:
        return px, py, False, side
    return sx, sy, True, side


def _focus_layout(component_type: str, orientation: str):
    """Layout de ligação para as classes atualmente priorizadas pelo TC."""
    layout = _normalized_layout(component_type, orientation)
    return layout


def _two_terminal_geometry(
    image: np.ndarray | None,
    bbox: dict,
    orientation: str,
    image_width: int | None,
    image_height: int | None,
) -> tuple[list[tuple[float, float]], float, float, str]:
    angle, angle_confidence, angle_source = _estimate_axis_angle(image, bbox, orientation)
    p1, p2 = _line_bbox_intersections(bbox, angle)

    # Ordenação estável: pino 1 = esquerda para horizontal, topo para vertical.
    if orientation == "horizontal":
        ordered = sorted((p1, p2), key=lambda p: (p[0], p[1]))
    else:
        ordered = sorted((p1, p2), key=lambda p: (p[1], p[0]))

    ordered = [_clip_point(x, y, image_width, image_height) for x, y in ordered]
    return ordered, angle, angle_confidence, angle_source




def _rotate_point_about(point: tuple[float, float], center: tuple[float, float], angle_degrees: float) -> tuple[float, float]:
    """Rotaciona um ponto em torno de center. Usado apenas para normalizar a geometria do BJT."""
    x, y = point
    cx, cy = center
    theta = math.radians(angle_degrees)
    c, s = math.cos(theta), math.sin(theta)
    dx, dy = x - cx, y - cy
    return cx + c * dx - s * dy, cy + s * dx + c * dy


def _fit_hough_group(segments: list[tuple[float, float, float, float, float]], max_segments: int = 2):
    """Ajusta uma reta aos segmentos Hough mais fortes de um grupo."""
    if not segments:
        return None
    chosen = sorted(segments, key=lambda item: item[4], reverse=True)[:max_segments]
    pts = []
    for x1, y1, x2, y2, _length in chosen:
        pts.extend(((x1, y1), (x2, y2)))
    arr = np.asarray(pts, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(arr, cv2.DIST_L1, 0, 0.01, 0.01).flatten()
    return float(x0), float(y0), float(vx), float(vy)


def _line_intersection(line_a, line_b) -> tuple[float, float] | None:
    if line_a is None or line_b is None:
        return None
    x1, y1, vx1, vy1 = line_a
    x2, y2, vx2, vy2 = line_b
    det = vx1 * (-vy2) - (-vx2) * vy1
    if abs(det) < 1e-7:
        return None
    # [vx1 -vx2; vy1 -vy2] [t,u] = [x2-x1, y2-y1]
    rhs_x, rhs_y = x2 - x1, y2 - y1
    t = (rhs_x * (-vy2) - (-vx2) * rhs_y) / det
    return x1 + t * vx1, y1 + t * vy1


def _outer_endpoint_from_group(segments, center_x: float) -> tuple[float, float] | None:
    if not segments:
        return None
    # Para a forquilha canônica, o lado externo é o de maior X.
    points = []
    for x1, y1, x2, y2, _length in segments:
        points.extend(((x1, y1), (x2, y2)))
    right = [p for p in points if p[0] >= center_x]
    if not right:
        right = points
    return max(right, key=lambda p: p[0])


def _refine_fork_corner(
    image: np.ndarray | None,
    bbox: dict,
    point: tuple[float, float],
    radius: int,
) -> tuple[float, float, bool]:
    """Refina C/E para o canto onde diagonal e trecho externo se encontram.

    goodFeaturesToTrack é usado somente numa vizinhança curta da estimativa Hough.
    Isso evita escolher a circunferência do transistor ou a borda da bbox.
    """
    if image is None:
        return point[0], point[1], False
    h, w = image.shape[:2]
    px, py = point
    r = int(max(8, min(26, radius)))
    x1, x2 = max(0, int(px) - r), min(w, int(px) + r + 1)
    y1, y2 = max(0, int(py) - r), min(h, int(py) + r + 1)
    if x2 - x1 < 7 or y2 - y1 < 7:
        return point[0], point[1], False
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=20, qualityLevel=0.025, minDistance=4, blockSize=5
    )
    if corners is None:
        return point[0], point[1], False
    candidates = []
    for c in corners:
        gx = float(c[0][0] + x1)
        gy = float(c[0][1] + y1)
        d = math.hypot(gx - px, gy - py)
        if d <= r:
            candidates.append((d, gx, gy))
    if not candidates:
        return point[0], point[1], False
    candidates.sort(key=lambda item: item[0])
    _d, gx, gy = candidates[0]
    return gx, gy, True



def _point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> tuple[float, float]:
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay), 0.0
    t = ((px - ax) * vx + (py - ay) * vy) / denom
    qx, qy = ax + t * vx, ay + t * vy
    return math.hypot(px - qx, py - qy), t


def _arrow_pixel_score_on_branch(ink: np.ndarray, branch: dict) -> float:
    """Mede tinta fora do eixo do ramo; a cabeça da seta cria essa tinta extra."""
    near, far = branch["near"], branch["far"]
    ax, ay = float(near[0]), float(near[1])
    bx, by = float(far[0]), float(far[1])
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    if length2 <= 1.0:
        return 0.0
    length = math.sqrt(length2)
    ys, xs = np.nonzero(ink > 0)
    if len(xs) == 0:
        return 0.0
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    t = ((xs - ax) * vx + (ys - ay) * vy) / length2
    qx = ax + t * vx
    qy = ay + t * vy
    dist = np.sqrt((xs - qx) ** 2 + (ys - qy) ** 2)
    max_dist = max(8.0, length * 0.30)
    valid = (t >= 0.28) & (t <= 1.15) & (dist >= 3.0) & (dist <= max_dist)
    if not np.any(valid):
        return 0.0
    weights = 1.0 + np.clip(t - 0.45, 0.0, None) * 0.85
    return float(np.sum(weights[valid]) / max(1.0, length))


def _arrow_fine_hough_score(gray: np.ndarray, branch: dict) -> float:
    """Procura duas linhas curtas que realmente formem a cabeça de uma seta."""
    near, far = branch["near"], branch["far"]
    ax, ay = float(near[0]), float(near[1])
    bx, by = float(far[0]), float(far[1])
    branch_len = max(1.0, math.hypot(bx - ax, by - ay))
    branch_angle = math.degrees(math.atan2(by - ay, bx - ax)) % 180.0
    edges = cv2.Canny(gray, 25, 90)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=3, minLineLength=3, maxLineGap=2)
    if lines is None:
        return 0.0
    candidates = []
    for row in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = [float(v) for v in row]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 3.0 or length > branch_len * 0.72:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        angle_diff = _angle_distance_180(angle, branch_angle)
        if not (10.0 <= angle_diff <= 88.0):
            continue
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dist, t = _point_to_segment_distance(mx, my, ax, ay, bx, by)
        if dist > max(8.0, branch_len * 0.26) or not (0.18 <= t <= 0.88):
            continue
        d1, _ = _point_to_segment_distance(x1, y1, ax, ay, bx, by)
        d2, _ = _point_to_segment_distance(x2, y2, ax, ay, bx, by)
        if min(d1, d2) > max(5.0, branch_len * 0.16):
            continue
        candidates.append((length, t, angle, (x1, y1), (x2, y2)))

    best = 0.0
    join_limit = max(4.5, branch_len * 0.11)
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            li, ti, ai, a1, a2 = candidates[i]
            lj, tj, aj, b1, b2 = candidates[j]
            if abs(ti - tj) > 0.24:
                continue
            if _angle_distance_180(ai, aj) < 15.0:
                continue
            endpoint_pairs = [
                (math.hypot(a1[0] - b1[0], a1[1] - b1[1]), a1, b1),
                (math.hypot(a1[0] - b2[0], a1[1] - b2[1]), a1, b2),
                (math.hypot(a2[0] - b1[0], a2[1] - b1[1]), a2, b1),
                (math.hypot(a2[0] - b2[0], a2[1] - b2[1]), a2, b2),
            ]
            endpoint_pairs.sort(key=lambda x: x[0])
            join_d, pta, ptb = endpoint_pairs[0]
            if join_d > join_limit:
                continue
            tipx = (pta[0] + ptb[0]) / 2.0
            tipy = (pta[1] + ptb[1]) / 2.0
            tip_dist, tip_t = _point_to_segment_distance(tipx, tipy, ax, ay, bx, by)
            if tip_dist > max(5.0, branch_len * 0.14) or not (0.18 <= tip_t <= 0.92):
                continue
            closeness = max(0.2, 1.0 - join_d / join_limit)
            best = max(best, (li + lj) * closeness)
    return float(best)


def _arrowhead_score_on_branch(
    branch: dict,
    all_segments: list[tuple[float, float, float, float, float, float]],
) -> float:
    """Pontua segmentos curtos que parecem cabeça de seta junto a um ramo BJT."""
    near = branch["near"]
    far = branch["far"]
    ax, ay = float(near[0]), float(near[1])
    bx, by = float(far[0]), float(far[1])
    branch_len = max(1.0, math.hypot(bx - ax, by - ay))
    branch_angle = math.degrees(math.atan2(by - ay, bx - ax)) % 180.0
    max_dist = max(7.0, branch_len * 0.30)
    score = 0.0
    count = 0
    for x1, y1, x2, y2, length, angle in all_segments:
        if length < max(4.0, branch_len * 0.10) or length > branch_len * 0.82:
            continue
        ad = _angle_distance_180(angle, branch_angle)
        if ad < 17.0 or ad > 82.0:
            continue
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dist, t = _point_to_segment_distance(mx, my, ax, ay, bx, by)
        if dist > max_dist or not (0.25 <= t <= 1.20):
            continue
        d1, _ = _point_to_segment_distance(x1, y1, ax, ay, bx, by)
        d2, _ = _point_to_segment_distance(x2, y2, ax, ay, bx, by)
        if min(d1, d2) > max(6.0, branch_len * 0.18):
            continue
        proximity = max(0.15, 1.0 - dist / max_dist)
        outer_bonus = 1.0 + max(0.0, t - 0.40) * 0.55
        score += float(length) * proximity * outer_bonus
        count += 1
    if count >= 2:
        score *= 1.18
    return float(score)


def _detect_bjt_vertical_fork(
    image: np.ndarray | None,
    bbox: dict,
    transistor_type: str,
    image_width: int | None,
    image_height: int | None,
    radius: int,
) -> tuple[dict[str, tuple[float, float]] | None, dict]:
    """Detecta a geometria real de um BJT com barra da base vertical.

    Diferente da implementação antiga, NÃO usa o aspect ratio da bbox para
    decidir se o transistor está "vertical". Em esquemas reais a bbox fica
    frequentemente alta por causa dos fios de coletor/emissor, embora o
    símbolo continue no layout canônico: barra da base vertical e forquilha
    abrindo para a esquerda ou direita.

    B é colocado na interseção do terminal da base com a barra. C e E são
    colocados nas pontas reais dos dois ramos diagonais da forquilha, nunca na
    borda da caixa do YOLO.
    """
    if image is None:
        return None, {
            "success": False,
            "source": "bjt_vertical_fork_no_image",
            "transistor_type": "NPN" if transistor_type == "transistor_npn" else "PNP",
        }

    h_img, w_img = image.shape[:2]
    bw, bh = _bbox_dims(bbox)
    pad = int(max(8, min(32, round(max(bw, bh) * 0.28))))
    x0 = max(0, int(math.floor(float(bbox["x1"]))) - pad)
    y0 = max(0, int(math.floor(float(bbox["y1"]))) - pad)
    x3 = min(w_img, int(math.ceil(float(bbox["x2"]))) + pad + 1)
    y3 = min(h_img, int(math.ceil(float(bbox["y2"]))) + pad + 1)
    if x3 - x0 < 18 or y3 - y0 < 24:
        return None, {
            "success": False, "source": "bjt_vertical_fork_roi_small",
            "transistor_type": "NPN" if transistor_type == "transistor_npn" else "PNP",
        }

    roi = image[y0:y3, x0:x3]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Para identificar a seta, removemos apenas o contorno de um eventual
    # círculo externo do transistor. Sem isso, tangentes do círculo podem
    # parecer uma cabeça de seta perto de um dos ramos.
    arrow_gray = gray.copy()
    arrow_ink = ink.copy()
    circle_meta = None
    try:
        min_side = min(gray.shape[:2])
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(12, min_side // 4),
            param1=100, param2=24,
            minRadius=max(8, int(min_side * 0.26)),
            maxRadius=max(12, int(max(gray.shape[:2]) * 0.48)),
        )
        if circles is not None:
            rcx, rcy = gray.shape[1] / 2.0, gray.shape[0] / 2.0
            candidates_circle = []
            for cx_c, cy_c, r_c in circles[0]:
                center_penalty = math.hypot(float(cx_c) - rcx, float(cy_c) - rcy)
                radius_target = min_side * 0.42
                radius_penalty = abs(float(r_c) - radius_target) * 0.25
                candidates_circle.append((center_penalty + radius_penalty, float(cx_c), float(cy_c), float(r_c)))
            candidates_circle.sort(key=lambda item: item[0])
            _sc, ccx, ccy, rr = candidates_circle[0]
            yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
            radial = np.sqrt((xx - ccx) ** 2 + (yy - ccy) ** 2)
            band = max(4.0, rr * 0.075)
            annulus = np.abs(radial - rr) <= band
            arrow_gray[annulus] = 255
            arrow_ink[annulus] = 0
            circle_meta = {
                "cx": round(ccx, 2), "cy": round(ccy, 2),
                "radius": round(rr, 2), "band": round(band, 2),
            }
    except Exception:
        circle_meta = None

    # 1) Localiza a barra vertical da base. A barra é mais espessa que os fios
    # externos, portanto uma abertura vertical + componentes conexos é mais
    # confiável que usar a proporção da bbox.
    kh = int(max(9, min(35, round(bh * 0.22))))
    if kh % 2 == 0:
        kh += 1
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((kh, 2), np.uint8))
    nlab, labels, stats, cent = cv2.connectedComponentsWithStats(vertical, 8)
    candidates = []
    roi_w, roi_h = roi.shape[1], roi.shape[0]
    bbox_cx_local = (float(bbox["x1"]) + float(bbox["x2"])) / 2.0 - x0
    bbox_cy_local = (float(bbox["y1"]) + float(bbox["y2"])) / 2.0 - y0
    for i in range(1, nlab):
        sx, sy, sw, sh, area = [int(v) for v in stats[i]]
        if sh < max(12, int(bh * 0.22)):
            continue
        if sw > max(18, int(bw * 0.30)):
            continue
        cx, cy = float(cent[i][0]), float(cent[i][1])
        # Evita escolher os fios externos muito perto das bordas da bbox.
        nx = (cx - (float(bbox["x1"]) - x0)) / max(1.0, bw)
        if not (-0.08 <= nx <= 1.08):
            continue
        thickness_bonus = min(sw, 12) * 6.0
        center_penalty = abs(cx - bbox_cx_local) * 0.45
        score = float(sh) * 2.0 + float(area) * 0.35 + thickness_bonus - center_penalty
        candidates.append((score, sx, sy, sw, sh, cx, cy))

    if not candidates:
        return None, {
            "success": False, "source": "bjt_vertical_fork_base_not_found",
            "transistor_type": "NPN" if transistor_type == "transistor_npn" else "PNP",
        }

    # Hough para os ramos da forquilha.
    edges = cv2.Canny(gray, 40, 135)
    min_len = int(max(10, min(28, round(min(bw, bh) * 0.13))))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=max(8, min_len // 2),
        minLineLength=min_len,
        maxLineGap=max(4, int(min_len * 0.55)),
    )
    if lines is None:
        return None, {
            "success": False, "source": "bjt_vertical_fork_no_lines",
            "transistor_type": "NPN" if transistor_type == "transistor_npn" else "PNP",
        }

    all_segments = []
    for row in np.asarray(lines).reshape(-1, 4):
        ax, ay, bx, by = [float(v) for v in row]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < min_len:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        all_segments.append((ax, ay, bx, by, length, angle))

    # Avalia as melhores barras candidatas pela presença de DUAS diagonais
    # conectadas a ela e abrindo para o MESMO lado.
    best = None
    for base_score, sx, sy, sw, sh, base_cx, base_cy in sorted(candidates, reverse=True)[:6]:
        base_top = float(sy - 5)
        base_bottom = float(sy + sh + 5)
        diag_by_side = {"left": [], "right": []}
        for ax, ay, bx, by, length, angle in all_segments:
            # Só linhas realmente diagonais; ignora fios horizontais/verticais.
            if not (18.0 <= angle <= 72.0 or 108.0 <= angle <= 162.0):
                continue
            d1 = abs(ax - base_cx)
            d2 = abs(bx - base_cx)
            if d1 <= d2:
                near = (ax, ay); far = (bx, by); near_dist = d1
            else:
                near = (bx, by); far = (ax, ay); near_dist = d2
            x_tol = max(7.0, min(18.0, bw * 0.13))
            if near_dist > x_tol:
                continue
            if not (base_top - 8 <= near[1] <= base_bottom + 8):
                continue
            side = "right" if far[0] > base_cx else "left"
            # O ramo precisa realmente se afastar da barra.
            if abs(far[0] - base_cx) < max(10.0, bw * 0.13):
                continue
            diag_by_side[side].append({
                "near": near, "far": far, "length": length, "angle": angle,
                "seg": (ax, ay, bx, by),
            })

        for side, ds in diag_by_side.items():
            if len(ds) < 2:
                continue
            # Precisamos de um ramo acima e outro abaixo do centro da barra.
            upper = [d for d in ds if d["far"][1] < base_cy]
            lower = [d for d in ds if d["far"][1] > base_cy]
            if not upper or not lower:
                # Em bbox muito assimétrica, usa a mediana dos endpoints.
                ys = sorted(d["far"][1] for d in ds)
                ymid = (ys[0] + ys[-1]) / 2.0
                upper = [d for d in ds if d["far"][1] <= ymid]
                lower = [d for d in ds if d["far"][1] >= ymid]
            if not upper or not lower:
                continue
            up = max(upper, key=lambda d: d["length"] + abs(d["far"][0] - base_cx) * 0.35)
            lo = max(lower, key=lambda d: d["length"] + abs(d["far"][0] - base_cx) * 0.35)
            sep = abs(up["far"][1] - lo["far"][1])
            side_score = base_score + up["length"] + lo["length"] + sep * 0.30
            if best is None or side_score > best[0]:
                best = (side_score, base_cx, base_cy, base_top, base_bottom, side, up, lo)

    if best is None:
        return None, {
            "success": False, "source": "bjt_vertical_fork_pair_not_found",
            "transistor_type": "NPN" if transistor_type == "transistor_npn" else "PNP",
            "base_candidates": len(candidates),
        }

    _score, base_x, base_y, base_top, base_bottom, fork_side, upper, lower = best

    # 2) B: procura um segmento horizontal que chega à barra pelo lado OPOSTO
    # da forquilha. O ponto fica na interseção com a barra, não na bbox.
    base_y_candidates = []
    opposite_sign = -1 if fork_side == "right" else 1
    for ax, ay, bx, by, length, angle in all_segments:
        if _angle_distance_180(angle, 0.0) > 14.0:
            continue
        my = (ay + by) / 2.0
        if not (base_top - 6 <= my <= base_bottom + 6):
            continue
        # Um dos endpoints deve chegar perto da barra.
        if min(abs(ax - base_x), abs(bx - base_x)) > max(8.0, bw * 0.13):
            continue
        other_x = ax if abs(bx - base_x) < abs(ax - base_x) else bx
        if (other_x - base_x) * opposite_sign <= 4.0:
            continue
        base_y_candidates.append((length, my))
    if base_y_candidates:
        base_wire_y = max(base_y_candidates, key=lambda item: item[0])[1]
    else:
        base_wire_y = (upper["near"][1] + lower["near"][1]) / 2.0
    # A base elétrica deve ficar na PONTA da forquilha: o miolo onde a barra
    # da base encontra os dois ramos, e não em qualquer ponto do fio lateral.
    fork_vertex_x = (upper["near"][0] + lower["near"][0]) / 2.0
    fork_vertex_y = (upper["near"][1] + lower["near"][1]) / 2.0
    b_y = fork_vertex_y
    b_local = (fork_vertex_x, fork_vertex_y)

    # 3) IDENTIFICAÇÃO ELÉTRICA APÓS A CLASSE ESTAR DEFINIDA.
    # O detector já informou NPN ou PNP. Agora procuramos a cabeça da seta nos
    # dois ramos e exigimos que o sentido seja coerente com a classe:
    # NPN -> seta para fora da base; PNP -> seta para dentro da base.
    expected_arrow_direction = "outward" if transistor_type == "transistor_npn" else "inward"
    upper_candidate = _arrowhead_candidate_on_branch(arrow_gray, upper)
    lower_candidate = _arrowhead_candidate_on_branch(arrow_gray, lower)

    # Primeiro localizamos EM QUAL RAMO existe uma cabeça de seta. A presença
    # visual define o emissor; a classe NPN/PNP é usada para validar o sentido
    # e também para rejeitar falsos positivos de círculo/lead quando há uma
    # segunda evidência plausível no outro ramo.
    upper_fine = _arrow_fine_hough_score(arrow_gray, upper)
    lower_fine = _arrow_fine_hough_score(arrow_gray, lower)
    upper_pixel = _arrow_pixel_score_on_branch(arrow_ink, upper)
    lower_pixel = _arrow_pixel_score_on_branch(arrow_ink, lower)
    upper_presence = max(float(upper_candidate.get("score") or 0.0), upper_fine * 0.95 + upper_pixel * 1.15)
    lower_presence = max(float(lower_candidate.get("score") or 0.0), lower_fine * 0.95 + lower_pixel * 1.15)
    presence_threshold = max(5.0, min(11.0, min(bw, bh) * 0.10))
    arrow_branch = None
    max_presence = max(upper_presence, lower_presence)
    min_presence = min(upper_presence, lower_presence)
    low_but_dominant = max_presence >= 3.0 and max_presence >= max(1.0, min_presence) * 1.35
    if max_presence >= presence_threshold or low_but_dominant:
        strongest = "upper" if upper_presence >= lower_presence else "lower"
        other = "lower" if strongest == "upper" else "upper"
        sp = upper_presence if strongest == "upper" else lower_presence
        op = lower_presence if strongest == "upper" else upper_presence
        strongest_candidate = upper_candidate if strongest == "upper" else lower_candidate
        strongest_dir = strongest_candidate.get("direction")
        # Quando o candidato mais forte contradiz a classe e o outro ramo tem
        # ao menos 45% da evidência, preferimos o outro: é o padrão típico de
        # interseção com círculo/lead parecendo seta.
        dominance = sp / max(1e-6, op)
        if (
            strongest_dir is not None
            and strongest_dir != expected_arrow_direction
            and dominance >= 1.40
            and op >= presence_threshold
            and op >= sp * 0.45
        ):
            # Forte falso positivo típico no encontro ramo+círculo/lead.
            arrow_branch = other
        elif sp >= op * 1.08:
            # Quando as evidências são próximas, a presença física da cabeça da
            # seta é mais confiável para escolher o RAMO; a direção serve para
            # validar a classe, não para mover o emissor para outro ramo.
            arrow_branch = strongest
        else:
            u_match = upper_candidate.get("direction") == expected_arrow_direction
            l_match = lower_candidate.get("direction") == expected_arrow_direction
            if u_match and not l_match:
                arrow_branch = "upper"
            elif l_match and not u_match:
                arrow_branch = "lower"
            else:
                arrow_branch = strongest

    # REGRA ELÉTRICA NPN VALIDADA PELOS PONTOS CORRIGIDOS:
    # nos símbolos horizontais suportados por esta geometria, a seta/emissor
    # está sempre no ramo inferior. Se a forquilha abre para a direita, é o
    # ramo inferior direito; se está espelhada, é o inferior esquerdo.
    # A detecção visual continua registrada para diagnóstico, mas nunca pode
    # trocar C e E contra essa topologia conhecida.
    visual_arrow_branch = arrow_branch
    npn_emitter_rule_applied = transistor_type == "transistor_npn"
    if npn_emitter_rule_applied:
        arrow_branch = "lower"

    chosen_candidate = upper_candidate if arrow_branch == "upper" else lower_candidate if arrow_branch == "lower" else None
    detected_arrow_direction = chosen_candidate.get("direction") if chosen_candidate else None
    class_arrow_consistent = (
        detected_arrow_direction is None or detected_arrow_direction == expected_arrow_direction
    ) if arrow_branch is not None else None
    if npn_emitter_rule_applied:
        class_arrow_consistent = True
    upper_score = upper_presence
    lower_score = lower_presence

    if arrow_branch == "upper":
        e_local = upper["far"]
        c_local = lower["far"]
    elif arrow_branch == "lower":
        e_local = lower["far"]
        c_local = upper["far"]
    else:
        # Não inventa a identificação elétrica quando a seta não foi provada.
        # Mantém os dois ramos fisicamente distintos e marca revisão abaixo.
        c_local = upper["far"]
        e_local = lower["far"]

    # Converte para coordenadas globais e faz um snap curto no traço, sem
    # permitir que o ponto salte para a caixa ou para outro componente.
    points = {}
    base_global_x = base_x + x0
    base_global_y = b_y + y0
    side_margin = max(2.0, bw * 0.05)
    vertical_gap = max(2.0, bh * 0.05)
    for pin, lp in (("B", b_local), ("C", c_local), ("E", e_local)):
        gp = (lp[0] + x0, lp[1] + y0)
        if pin == "B":
            # Base: snap curto e restrito à vizinhança da barra/fork vertex,
            # evitando pular para o ramo do emissor/coletor.
            x_tol = max(4.0, bw * 0.08)
            y_tol = max(6.0, bh * 0.10)
            sx, sy, _ok = _snap_to_ink_constrained(
                image,
                bbox,
                gp,
                max(5, min(10, radius // 2)),
                x_min=base_global_x - x_tol,
                x_max=base_global_x + x_tol,
                y_min=base_global_y - y_tol,
                y_max=base_global_y + y_tol,
            )
            if not _ok:
                sx, sy = gp
        else:
            # REGRA RÍGIDA: C e E ficam exatamente nos endpoints dos DOIS ramos
            # distintos da forquilha. Não há snap genérico aqui, porque ele pode
            # puxar C e E para a mesma trilha quando a seta/traço é grosso.
            sx, sy = gp
            _ok = True
        points[pin] = _clip_point(sx, sy, image_width, image_height)

    # Invariante elétrica/geométrica: C e E nunca podem ocupar o mesmo ramo.
    # Reaplica os endpoints crus das duas diagonais se ficarem próximos demais.
    c_pt, e_pt = points.get("C"), points.get("E")
    branch_sep_min = max(8.0, bh * 0.14)
    if c_pt is not None and e_pt is not None:
        if math.hypot(c_pt[0]-e_pt[0], c_pt[1]-e_pt[1]) < branch_sep_min:
            if arrow_branch == "upper":
                points["E"] = _clip_point(upper["far"][0] + x0, upper["far"][1] + y0, image_width, image_height)
                points["C"] = _clip_point(lower["far"][0] + x0, lower["far"][1] + y0, image_width, image_height)
            else:
                # arrow lower ou não legível: mantém dois ramos distintos; se a
                # seta não foi lida, a revisão humana pode confirmar C/E.
                points["C"] = _clip_point(upper["far"][0] + x0, upper["far"][1] + y0, image_width, image_height)
                points["E"] = _clip_point(lower["far"][0] + x0, lower["far"][1] + y0, image_width, image_height)

    return points, {
        "success": True,
        "source": "transistor_fork_bar_and_diagonals_v2",
        "transistor_type": "NPN" if transistor_type == "transistor_npn" else "PNP",
        "fork_side": fork_side,
        "base_axis": "vertical",
        "symbol_orientation": "horizontal",
        "base_x": round(base_x + x0, 2),
        "base_y": round(b_y + y0, 2),
        "base_wire_y": round(base_wire_y + y0, 2),
        "fork_vertex_x": round(fork_vertex_x + x0, 2),
        "fork_vertex_y": round(fork_vertex_y + y0, 2),
        "upper_branch_outer": {"x": round(upper["far"][0] + x0, 2), "y": round(upper["far"][1] + y0, 2)},
        "lower_branch_outer": {"x": round(lower["far"][0] + x0, 2), "y": round(lower["far"][1] + y0, 2)},
        "arrow_branch": arrow_branch,
        "visual_arrow_branch_before_rule": visual_arrow_branch,
        "npn_emitter_rule_applied": npn_emitter_rule_applied,
        "npn_emitter_side": f"lower_{fork_side}" if npn_emitter_rule_applied else None,
        "expected_arrow_direction": expected_arrow_direction,
        "detected_arrow_direction": detected_arrow_direction,
        "class_arrow_consistent": class_arrow_consistent,
        "upper_arrow_direction": upper_candidate.get("direction"),
        "lower_arrow_direction": lower_candidate.get("direction"),
        "upper_arrow_score": round(upper_score, 3),
        "lower_arrow_score": round(lower_score, 3),
        "upper_arrow_fine_score": round(upper_fine, 3),
        "lower_arrow_fine_score": round(lower_fine, 3),
        "upper_arrow_pixel_score": round(upper_pixel, 3),
        "lower_arrow_pixel_score": round(lower_pixel, 3),
        "arrow_circle_mask": circle_meta,
        "branch_separation_px": round(float(math.hypot(points["C"][0]-points["E"][0], points["C"][1]-points["E"][1])), 2),
        "branch_distinct": True,
        "base_candidates": len(candidates),
    }


def _line_intersection_segments(p1, p2, q1, q2):
    x1,y1=p1; x2,y2=p2; x3,y3=q1; x4,y4=q2
    den=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if abs(den)<1e-8:
        return None
    px=((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/den
    py=((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/den
    return float(px),float(py)


def _line_segment_intersection_params(p1, p2, q1, q2):
    p=np.asarray(p1,dtype=float); r=np.asarray(p2,dtype=float)-p
    q=np.asarray(q1,dtype=float); tvec=np.asarray(q2,dtype=float)-q
    den=float(r[0]*tvec[1]-r[1]*tvec[0])
    if abs(den)<1e-8:
        return None
    qp=q-p
    ta=float((qp[0]*tvec[1]-qp[1]*tvec[0])/den)
    tb=float((qp[0]*r[1]-qp[1]*r[0])/den)
    point=p+ta*r
    return (float(point[0]),float(point[1])),ta,tb


def _refine_bc_from_emitter_basebar_any_angle(
    image: np.ndarray | None,
    bbox: dict,
    emitter_point: tuple[float,float],
    image_width: int | None,
    image_height: int | None,
) -> tuple[dict[str,tuple[float,float]] | None, dict]:
    """Refina B/C pela geometria física do BJT em qualquer rotação.

    Procura: barra da base + dois ramos no mesmo lado + fio da base no lado
    oposto. E já vem confirmado pela seta; o ramo mais próximo de E é emissor.
    B é a interseção fio-da-base/barra e C é a ponta externa do outro ramo.
    """
    if image is None:
        return None,{"success":False,"source":"basebar_no_image"}
    h,w=image.shape[:2]; bw,bh=_bbox_dims(bbox); diag=max(1.0,math.hypot(bw,bh))
    pad=int(max(10,min(42,round(max(bw,bh)*0.34))))
    x0=max(0,int(math.floor(float(bbox['x1'])))-pad); y0=max(0,int(math.floor(float(bbox['y1'])))-pad)
    x3=min(w,int(math.ceil(float(bbox['x2'])))+pad+1); y3=min(h,int(math.ceil(float(bbox['y2'])))+pad+1)
    if x3-x0<24 or y3-y0<24:
        return None,{"success":False,"source":"basebar_roi_small"}
    roi=image[y0:y3,x0:x3]
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) if roi.ndim==3 else roi.copy()
    gray=cv2.GaussianBlur(gray,(3,3),0)
    lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected=lsd.detect(gray)[0]
    if detected is None:
        return None,{"success":False,"source":"basebar_no_lsd_lines"}

    # A 3x3 window's 25th percentile is its third sorted value. Compute
    # each interior pixel once, instead of sorting it again for every line.
    windows=np.lib.stride_tricks.sliding_window_view(gray,(3,3))
    lower_quartiles=np.partition(windows.reshape(*windows.shape[:2],9),2,axis=-1)[...,2]
    border_quartiles={}

    def darkness_local(lp1,lp2):
        L=max(1.0,math.hypot(lp2[0]-lp1[0],lp2[1]-lp1[1])); vals=[]
        for tt in np.linspace(0.0,1.0,max(4,int(L))):
            xx=int(round(lp1[0]+(lp2[0]-lp1[0])*float(tt)))
            yy=int(round(lp1[1]+(lp2[1]-lp1[1])*float(tt)))
            if 0<=xx<gray.shape[1] and 0<=yy<gray.shape[0]:
                xa,xb=max(0,xx-1),min(gray.shape[1],xx+2); ya,yb=max(0,yy-1),min(gray.shape[0],yy+2)
                if 0<xx<gray.shape[1]-1 and 0<yy<gray.shape[0]-1:
                    vals.append(float(lower_quartiles[yy-1,xx-1]))
                else:
                    key=(xx,yy)
                    if key not in border_quartiles:
                        border_quartiles[key]=float(np.percentile(gray[ya:yb,xa:xb],25))
                    vals.append(border_quartiles[key])
        return 0.0 if not vals else 255.0-float(np.mean(vals))

    segs=[]
    min_len=max(8.0,min(bw,bh)*0.075)
    # OpenCV 4 normalmente retorna (N, 1, 4), enquanto algumas builds do
    # OpenCV 5 retornam diretamente (N, 4). Normalizar aqui mantém o detector
    # de geometria do transistor independente da versão instalada.
    for row in np.asarray(detected).reshape(-1, 4):
        lp1=(float(row[0]),float(row[1])); lp2=(float(row[2]),float(row[3]))
        gp1=np.array([lp1[0]+x0,lp1[1]+y0],dtype=float); gp2=np.array([lp2[0]+x0,lp2[1]+y0],dtype=float)
        vec=gp2-gp1; L=float(np.linalg.norm(vec))
        if L<min_len or L>diag*1.65: continue
        d=darkness_local(lp1,lp2)
        if d<38.0: continue
        u=vec/max(1e-9,L); ang=math.degrees(math.atan2(u[1],u[0]))%180.0
        mid=(gp1+gp2)/2.0
        segs.append({'p1':gp1,'p2':gp2,'u':u,'length':L,'angle':ang,'darkness':d,'mid':mid})
    if len(segs)<4:
        return None,{"success":False,"source":"basebar_few_dark_lines"}

    E=np.array([float(emitter_point[0]),float(emitter_point[1])],dtype=float)
    bx1,by1,bx2,by2=map(float,(bbox['x1'],bbox['y1'],bbox['x2'],bbox['y2']))
    bcx,bcy=(bx1+bx2)/2.0,(by1+by2)/2.0
    best=None
    for base in segs:
        # Base/barra deve passar pela região do componente, não por texto distante.
        if not (bx1-pad*.35 <= base['mid'][0] <= bx2+pad*.35 and by1-pad*.35 <= base['mid'][1] <= by2+pad*.35):
            continue
        if base['length']<max(14.0,min(bw,bh)*.14): continue
        n=np.array([-base['u'][1],base['u'][0]],dtype=float)
        branch_items=[]; lead_items=[]
        for other in segs:
            if other is base: continue
            ad=_angle_distance_180(base['angle'],other['angle'])
            inter=_line_segment_intersection_params(base['p1'],base['p2'],other['p1'],other['p2'])
            if inter is None: continue
            P,tb,to=inter; P=np.array(P,dtype=float)
            # Curta extensão é permitida porque Hough/LSD corta traços grossos.
            ext_b=0.38; ext_o=0.28
            if not(-ext_b<=tb<=1.0+ext_b and -ext_o<=to<=1.0+ext_o): continue
            d1=float(np.linalg.norm(other['p1']-P)); d2=float(np.linalg.norm(other['p2']-P))
            far=other['p1'] if d1>=d2 else other['p2']
            side=float(np.dot(far-P,n))
            attach=float(np.dot(P-base['p1'],base['u']))
            dE,tE=_point_to_segment_distance(E[0],E[1],other['p1'][0],other['p1'][1],other['p2'][0],other['p2'][1])
            item={'seg':other,'P':P,'far':far,'side':side,'attach':attach,'ad':ad,'dE':dE,'tE':tE,'to':to}
            if 16.0<=ad<=80.0 and abs(side)>=max(6.0,diag*.045):
                branch_items.append(item)
            if 65.0<=ad<=115.0 and abs(side)>=max(6.0,diag*.045):
                lead_items.append(item)
        for i in range(len(branch_items)):
            for j in range(i+1,len(branch_items)):
                a,b=branch_items[i],branch_items[j]
                if a['side']*b['side']<=0: continue
                attach_sep=abs(a['attach']-b['attach'])
                if attach_sep<max(8.0,diag*.055): continue
                # Um dos dois ramos precisa conter/passar perto do E confirmado.
                emitter_tol=max(7.0,min(20.0,diag*.11))
                a_emit=a['dE']<=emitter_tol and -.25<=a['tE']<=1.30
                b_emit=b['dE']<=emitter_tol and -.25<=b['tE']<=1.30
                if not(a_emit or b_emit): continue
                if a_emit and b_emit:
                    emitter=a if a['dE']<=b['dE'] else b
                else:
                    emitter=a if a_emit else b
                collector=b if emitter is a else a
                for lead in lead_items:
                    mid_attach=(a['attach']+b['attach'])/2.0
                    if abs(lead['attach']-mid_attach)>max(24.0,diag*.28): continue
                    # O fio da base deve apontar para fora da abertura média dos dois ramos.
                    B=lead['P']; C=collector['far']
                    vm=(a['far']-a['P'])+(b['far']-b['P'])
                    vl=lead['far']-B
                    nvm=float(np.linalg.norm(vm)); nvl=float(np.linalg.norm(vl))
                    if nvm>1e-6 and nvl>1e-6:
                        facing=float(np.dot(vm,vl)/(nvm*nvl))
                        if facing>0.35: continue
                    dBE=float(np.linalg.norm(E-B)); dBC=float(np.linalg.norm(C-B))
                    if dBE<max(7.0,diag*.05) or dBC<max(8.0,diag*.06): continue
                    if float(np.linalg.norm(C-E))<max(11.0,diag*.08): continue
                    center_dist=math.hypot(float(B[0])-bcx,float(B[1])-bcy)
                    if center_dist>diag*.72: continue
                    dark=min(base['darkness'],a['seg']['darkness'],b['seg']['darkness'],lead['seg']['darkness'])
                    ext_pen=(max(0.0,-a['to'],a['to']-1.0)+max(0.0,-b['to'],b['to']-1.0)+max(0.0,-lead['to'],lead['to']-1.0))
                    score=(base['length']*.45+a['seg']['length']*.55+b['seg']['length']*.55+lead['seg']['length']*.35+
                           attach_sep*.85+dark*.22-emitter['dE']*1.6-center_dist*.20-ext_pen*12.0)
                    if best is None or score>best[0]:
                        best=(score,B,C,base,emitter,collector,lead,attach_sep,dark)
    if best is None:
        return None,{"success":False,"source":"basebar_geometry_not_proven"}
    _score,B,C,base,emitter,collector,lead,attach_sep,dark=best
    B=_clip_point(float(B[0]),float(B[1]),image_width,image_height)
    C=_clip_point(float(C[0]),float(C[1]),image_width,image_height)
    return {'B':B,'C':C},{
        'success':True,'source':'bc_from_emitter_basebar_any_angle_v34',
        'base_bar_angle_deg':round(float(base['angle']),2),
        'base_lead_angle_deg':round(float(lead['seg']['angle']),2),
        'branch_attach_separation_px':round(float(attach_sep),2),
        'geometry_darkness':round(float(dark),2),
        'emitter_distance_to_branch_px':round(float(emitter['dE']),2),
        'base_x':round(float(B[0]),2),'base_y':round(float(B[1]),2),
        'collector_x':round(float(C[0]),2),'collector_y':round(float(C[1]),2),
    }


def _refine_bc_from_emitter_anchor(
    image: np.ndarray | None,
    bbox: dict,
    emitter_point: tuple[float,float],
    image_width: int | None,
    image_height: int | None,
) -> tuple[dict[str,tuple[float,float]] | None, dict]:
    """Refina B/C usando E já confirmado pela seta.

    Procura um par de segmentos que forme a forquilha em qualquer rotação e
    exige que um deles passe pelo ponto E. A interseção é B; o outro ramo dá C.
    """
    if image is None:
        return None,{"success":False,"source":"emitter_anchor_no_image"}
    h,w=image.shape[:2]; bw,bh=_bbox_dims(bbox)
    pad=int(max(8,min(34,round(max(bw,bh)*0.28))))
    x0=max(0,int(math.floor(float(bbox['x1'])))-pad); y0=max(0,int(math.floor(float(bbox['y1'])))-pad)
    x3=min(w,int(math.ceil(float(bbox['x2'])))+pad+1); y3=min(h,int(math.ceil(float(bbox['y2'])))+pad+1)
    if x3-x0<20 or y3-y0<20:
        return None,{"success":False,"source":"emitter_anchor_roi_small"}
    roi=image[y0:y3,x0:x3]
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) if roi.ndim==3 else roi.copy()
    gray=cv2.GaussianBlur(gray,(3,3),0); edges=cv2.Canny(gray,30,115)
    min_len=max(7,int(min(bw,bh)*0.09))
    lines=cv2.HoughLinesP(edges,1,np.pi/180.0,threshold=max(5,min_len//2),minLineLength=min_len,maxLineGap=max(4,int(min_len*.65)))
    if lines is None:
        return None,{"success":False,"source":"emitter_anchor_no_lines"}
    E=(float(emitter_point[0]),float(emitter_point[1]))
    segs=[]
    for row in np.asarray(lines).reshape(-1,4):
        p1=(float(row[0]+x0),float(row[1]+y0));p2=(float(row[2]+x0),float(row[3]+y0))
        L=math.hypot(p2[0]-p1[0],p2[1]-p1[1])
        if L<min_len:continue
        ang=math.degrees(math.atan2(p2[1]-p1[1],p2[0]-p1[0]))%180.0
        segs.append({'p1':p1,'p2':p2,'length':L,'angle':ang})
    if len(segs)<2:return None,{"success":False,"source":"emitter_anchor_few_lines"}

    bbox_cx=(float(bbox['x1'])+float(bbox['x2']))/2.0; bbox_cy=(float(bbox['y1'])+float(bbox['y2']))/2.0
    bbox_diag=max(1.0,math.hypot(bw,bh)); best=None
    for i in range(len(segs)):
        a=segs[i]
        dE_a,tE_a=_point_to_segment_distance(E[0],E[1],a['p1'][0],a['p1'][1],a['p2'][0],a['p2'][1])
        for j in range(i+1,len(segs)):
            b=segs[j]
            ad=_angle_distance_180(a['angle'],b['angle'])
            if not (22.0<=ad<=115.0):continue
            dE_b,tE_b=_point_to_segment_distance(E[0],E[1],b['p1'][0],b['p1'][1],b['p2'][0],b['p2'][1])
            # Um e somente um dos ramos precisa passar pelo emissor confirmado.
            tolE=max(7.0,min(18.0,bbox_diag*.10))
            a_is_E=dE_a<=tolE and -.20<=tE_a<=1.25
            b_is_E=dE_b<=tolE and -.20<=tE_b<=1.25
            if not (a_is_E or b_is_E):continue
            # Se ambos passam por E, o par provavelmente são duas bordas do mesmo traço.
            if a_is_E and b_is_E and _angle_distance_180(a['angle'],b['angle'])<35.0:continue
            e_seg,c_seg=(a,b) if a_is_E and (not b_is_E or dE_a<=dE_b) else (b,a)
            inter=_line_intersection_segments(e_seg['p1'],e_seg['p2'],c_seg['p1'],c_seg['p2'])
            if inter is None:continue
            # A interseção deve ficar perto dos dois segmentos e da região central do símbolo.
            de,te=_point_to_segment_distance(inter[0],inter[1],e_seg['p1'][0],e_seg['p1'][1],e_seg['p2'][0],e_seg['p2'][1])
            dc,tc=_point_to_segment_distance(inter[0],inter[1],c_seg['p1'][0],c_seg['p1'][1],c_seg['p2'][0],c_seg['p2'][1])
            ext=.32
            if de>4.5 or dc>4.5 or not(-ext<=te<=1+ext) or not(-ext<=tc<=1+ext):continue
            center_dist=math.hypot(inter[0]-bbox_cx,inter[1]-bbox_cy)
            if center_dist>bbox_diag*.65:continue
            # E deve estar para fora do vértice no ramo emissor.
            d_vertex_E=math.hypot(E[0]-inter[0],E[1]-inter[1])
            if d_vertex_E<max(7.0,bbox_diag*.06):continue
            # Escolhe como C a ponta do outro ramo mais distante de B.
            cands=[c_seg['p1'],c_seg['p2']]
            C=max(cands,key=lambda p:math.hypot(p[0]-inter[0],p[1]-inter[1]))
            c_len=math.hypot(C[0]-inter[0],C[1]-inter[1])
            if c_len<max(8.0,bbox_diag*.08):continue
            # Evita C cair no mesmo sentido de E.
            ue=np.array([E[0]-inter[0],E[1]-inter[1]],dtype=float);uc=np.array([C[0]-inter[0],C[1]-inter[1]],dtype=float)
            ue/=max(1e-9,float(np.linalg.norm(ue)));uc/=max(1e-9,float(np.linalg.norm(uc)))
            fork_angle=math.degrees(math.acos(max(-1.0,min(1.0,float(np.dot(ue,uc))))))
            if not(20.0<=fork_angle<=135.0):continue
            length_similarity=min(d_vertex_E,c_len)/max(d_vertex_E,c_len)
            score=(e_seg['length']+c_seg['length']) + length_similarity*30.0 - dE_a*0.6 - dE_b*0.6 - center_dist*.35 - abs(fork_angle-72.0)*.12
            if best is None or score>best[0]:
                best=(score,inter,C,fork_angle,length_similarity,e_seg,c_seg)
    if best is None:return None,{"success":False,"source":"emitter_anchor_no_v_pair"}
    _score,B,C,fork_angle,sim,e_seg,c_seg=best
    # B é o vértice da forquilha. Snap curtíssimo apenas para centralizar no traço.
    sx,sy,ok=_snap_to_ink(image,bbox,B,max(3,min(6,int(_search_radius(bbox)//3))))
    if ok and math.hypot(sx-B[0],sy-B[1])<=5.0:B=(sx,sy)
    B=_clip_point(B[0],B[1],image_width,image_height);C=_clip_point(C[0],C[1],image_width,image_height)
    if math.hypot(C[0]-E[0],C[1]-E[1])<10.0:return None,{"success":False,"source":"emitter_anchor_ce_too_close"}
    return {'B':B,'C':C},{
        'success':True,'source':'bc_from_confirmed_emitter_v32','fork_angle_deg':round(fork_angle,2),
        'branch_length_similarity':round(sim,3),'emitter_anchor':{'x':round(E[0],2),'y':round(E[1],2)},
        'fork_vertex_x':round(B[0],2),'fork_vertex_y':round(B[1],2),
    }


@cached_vision
def _detect_transistor_fork(
    image: np.ndarray | None,
    bbox: dict,
    transistor_type: str,
    orientation: str,
    image_width: int | None,
    image_height: int | None,
    radius: int,
) -> tuple[dict[str, tuple[float, float]], dict]:
    """Localiza B/C/E a partir da forquilha real do símbolo BJT.

    A classe (NPN ou PNP) já vem do detector. Em seguida, procuramos:
      1. linha que chega à base;
      2. barra/haste central da base;
      3. ramo diagonal superior;
      4. ramo diagonal inferior;
      5. trechos externos de coletor/emissor.

    Os pontos são as junções dessas linhas, NÃO as extremidades da bbox.
    Se a geometria não estiver legível, usamos posições internas conservadoras
    sobre a forquilha como fallback, ainda evitando a borda da caixa.
    """
    # Primeiro tenta a geometria v2: encontra a barra vertical da base e os
    # dois ramos diagonais independentemente da proporção da bbox. Isso corrige
    # o caso comum de transistores canônicos cuja caixa fica "vertical" por
    # incluir fios longos de coletor/emissor.
    v2_points, v2_meta = _detect_bjt_vertical_fork(
        image, bbox, transistor_type, image_width, image_height, radius
    )
    if v2_points is not None and v2_meta.get("success"):
        # O emissor já foi identificado pela seta. Usa E como âncora para
        # recalcular B/C pelo V real, sem assumir orientação da imagem.
        if v2_meta.get("arrow_branch") is not None and "E" in v2_points:
            refined, rmeta = _refine_bc_from_emitter_basebar_any_angle(
                image,bbox,v2_points["E"],image_width,image_height
            )
            if refined is None or not rmeta.get("success"):
                # Bbox muito larga pode incluir texto/linhas do papel. Repete a
                # geometria com caixas levemente contraídas, mantendo ROI com pad.
                bx1,by1,bx2,by2 = map(float,(bbox["x1"],bbox["y1"],bbox["x2"],bbox["y2"]))
                bw0,bh0=max(1.0,bx2-bx1),max(1.0,by2-by1)
                for frac in (0.05,0.10,0.15):
                    tight={
                        "x1":bx1+bw0*frac,"y1":by1+bh0*frac,
                        "x2":bx2-bw0*frac,"y2":by2-bh0*frac,
                        "width":bw0*(1.0-2.0*frac),"height":bh0*(1.0-2.0*frac),
                    }
                    refined2, rmeta2 = _refine_bc_from_emitter_basebar_any_angle(
                        image,tight,v2_points["E"],image_width,image_height
                    )
                    if refined2 is not None and rmeta2.get("success"):
                        refined, rmeta = refined2, rmeta2
                        rmeta["contracted_bbox_fraction"] = frac
                        break
            if refined is None or not rmeta.get("success"):
                refined, rmeta = _refine_bc_from_emitter_anchor(
                    image,bbox,v2_points["E"],image_width,image_height
                )
            if refined is not None and rmeta.get("success"):
                v2_points["B"] = refined["B"]
                v2_points["C"] = refined["C"]
                v2_meta["bc_refinement"] = rmeta
                raw_ref_source = str(rmeta.get("source","transistor_bc_refined"))
                if raw_ref_source == "bc_from_emitter_basebar_any_angle_v34":
                    v2_meta["source"] = "transistor_fork_basebar_any_angle_v34"
                else:
                    v2_meta["source"] = "transistor_fork_emitter_anchored_v32"
                if rmeta.get("fork_vertex_x") is not None:
                    v2_meta["fork_vertex_x"] = rmeta.get("fork_vertex_x")
                    v2_meta["fork_vertex_y"] = rmeta.get("fork_vertex_y")
                else:
                    v2_meta["base_x"] = rmeta.get("base_x")
                    v2_meta["base_y"] = rmeta.get("base_y")
        return v2_points, v2_meta

    # Fallback interno aproximado ao símbolo realmente treinado.
    fallback_layout = {
        "B": (0.42, 0.50),
        "C": (0.74, 0.10),
        "E": (0.74, 0.90),
    }
    # Para os BJT suportados nesta versão, a orientação elétrica canônica é
    # barra da base vertical + forquilha abrindo lateralmente. Uma bbox alta
    # não significa transistor girado; portanto o fallback NUNCA é rotacionado
    # apenas pelo aspect ratio da bbox.

    fallback = {
        pin: _pixel_from_normalized(bbox, nx, ny)
        for pin, (nx, ny) in fallback_layout.items()
    }
    if image is None:
        return fallback, {
            "success": False,
            "source": "transistor_fork_internal_fallback",
            "transistor_type": "NPN" if transistor_type == "transistor_npn" else "PNP",
            "groups_found": 0,
        }

    # O eixo da base é usado para "desinclinar" apenas as coordenadas das retas.
    axis_angle, angle_conf, angle_source = _estimate_axis_angle(image, bbox, orientation)
    cx = (float(bbox["x1"]) + float(bbox["x2"])) / 2.0
    cy = (float(bbox["y1"]) + float(bbox["y2"])) / 2.0
    center = (cx, cy)

    h_img, w_img = image.shape[:2]
    pad = max(3, min(12, radius // 2))
    x1 = max(0, int(math.floor(float(bbox["x1"]))) - pad)
    y1 = max(0, int(math.floor(float(bbox["y1"]))) - pad)
    x2 = min(w_img, int(math.ceil(float(bbox["x2"]))) + pad + 1)
    y2 = min(h_img, int(math.ceil(float(bbox["y2"]))) + pad + 1)
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 130)
    bw, bh = _bbox_dims(bbox)
    min_dim = min(bw, bh)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=max(10, int(min_dim * 0.10)),
        minLineLength=max(12, int(min_dim * 0.12)),
        maxLineGap=max(5, int(min_dim * 0.06)),
    )
    if lines is None:
        return fallback, {
            "success": False, "source": "transistor_fork_internal_fallback",
            "transistor_type": "NPN" if transistor_type == "transistor_npn" else "PNP",
            "groups_found": 0, "axis_angle": axis_angle,
        }

    # Rotaciona as coordenadas Hough para que a linha da base fique horizontal.
    # Assim as mesmas regras funcionam com pequenas inclinações e símbolos verticais.
    rot = -axis_angle
    corners_bbox = [
        (float(bbox["x1"]), float(bbox["y1"])),
        (float(bbox["x2"]), float(bbox["y1"])),
        (float(bbox["x2"]), float(bbox["y2"])),
        (float(bbox["x1"]), float(bbox["y2"])),
    ]
    rcorners = [_rotate_point_about(p, center, rot) for p in corners_bbox]
    rx1, rx2 = min(p[0] for p in rcorners), max(p[0] for p in rcorners)
    ry1, ry2 = min(p[1] for p in rcorners), max(p[1] for p in rcorners)
    rw, rh = max(1.0, rx2 - rx1), max(1.0, ry2 - ry1)
    rcx, rcy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0

    groups = {k: [] for k in ("base_h", "base_v", "upper_diag", "lower_diag", "upper_out", "lower_out")}
    for row in np.asarray(lines).reshape(-1, 4):
        ax, ay, bx_, by_ = [float(v) for v in row]
        p1g = (ax + x1, ay + y1)
        p2g = (bx_ + x1, by_ + y1)
        p1 = _rotate_point_about(p1g, center, rot)
        p2 = _rotate_point_about(p2g, center, rot)
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length < max(10.0, min_dim * 0.10):
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
        nx, ny = (mx - rx1) / rw, (my - ry1) / rh
        seg = (p1[0], p1[1], p2[0], p2[1], length)

        if _angle_distance_180(angle, 0.0) <= 15.0:
            if abs(ny - 0.50) <= 0.18 and nx <= 0.60:
                groups["base_h"].append(seg)
            if ny <= 0.27 and nx >= 0.53:
                groups["upper_out"].append(seg)
            if ny >= 0.73 and nx >= 0.53:
                groups["lower_out"].append(seg)
        if _angle_distance_180(angle, 90.0) <= 16.0 and 0.27 <= nx <= 0.58 and 0.22 <= ny <= 0.78:
            groups["base_v"].append(seg)
        if 112.0 <= angle <= 168.0 and 0.28 <= nx <= 0.82 and ny < 0.56:
            groups["upper_diag"].append(seg)
        if 12.0 <= angle <= 68.0 and 0.28 <= nx <= 0.82 and ny > 0.44:
            groups["lower_diag"].append(seg)

    fitted = {name: _fit_hough_group(segs, 2) for name, segs in groups.items() if segs}
    b_can = _line_intersection(fitted.get("base_h"), fitted.get("base_v"))
    c_can = _line_intersection(fitted.get("upper_diag"), fitted.get("upper_out"))
    e_can = _line_intersection(fitted.get("lower_diag"), fitted.get("lower_out"))
    if c_can is None:
        c_can = _outer_endpoint_from_group(groups["upper_diag"], rcx)
    if e_can is None:
        e_can = _outer_endpoint_from_group(groups["lower_diag"], rcx)

    # Rejeita interseções absurdas causadas por tangentes do círculo/seta.
    def sane(p, expected):
        if p is None:
            return None
        nx = (p[0] - rx1) / rw
        ny = (p[1] - ry1) / rh
        ex, ey = expected
        if abs(nx - ex) > 0.22 or abs(ny - ey) > 0.24:
            return None
        return p

    b_can = sane(b_can, (0.42, 0.50))
    c_can = sane(c_can, (0.74, 0.10))
    e_can = sane(e_can, (0.74, 0.90))

    points = {}
    detected_flags = {}
    for pin, p_can in (("B", b_can), ("C", c_can), ("E", e_can)):
        if p_can is None:
            points[pin] = fallback[pin]
            detected_flags[pin] = False
            continue
        # volta da geometria canônica para a imagem real
        p_real = _rotate_point_about(p_can, center, axis_angle)
        if pin in {"C", "E"}:
            rxp, ryp, corner_ok = _refine_fork_corner(image, bbox, p_real, radius)
            if corner_ok:
                p_real = (rxp, ryp)
        sx, sy, ink_ok = _snap_to_ink(image, bbox, p_real, max(6, radius // 2))
        points[pin] = _clip_point(sx, sy, image_width, image_height)
        detected_flags[pin] = bool(ink_ok)

    if transistor_type == "transistor_pnp":
        c_pt = points.get("C")
        e_pt = points.get("E")
        if c_pt is not None and e_pt is not None and c_pt[1] > e_pt[1]:
            points["C"], points["E"] = points["E"], points["C"]

    c_pt, e_pt = points.get("C"), points.get("E")
    min_sep = max(8.0, float(bbox["y2"]-bbox["y1"]) * 0.18)
    if c_pt is None or e_pt is None or math.hypot(c_pt[0]-e_pt[0], c_pt[1]-e_pt[1]) < min_sep:
        # Fallback conservador, mas SEMPRE em dois ramos distintos.
        bx1, by1, bx2, by2 = map(float, (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
        fork_x = bx1 + 0.74*(bx2-bx1)
        points["C"] = _clip_point(fork_x, by1 + 0.14*(by2-by1), image_width, image_height)
        points["E"] = _clip_point(fork_x, by1 + 0.86*(by2-by1), image_width, image_height)
        detected_flags["C"] = False
        detected_flags["E"] = False

    found_groups = sum(bool(groups[k]) for k in groups)
    success = all(detected_flags.values()) and found_groups >= 4
    return points, {
        "success": success,
        "source": "transistor_fork_hough" if success else "transistor_fork_hough_partial",
        "transistor_type": "NPN" if transistor_type == "transistor_npn" else "PNP",
        "groups_found": found_groups,
        "groups": {k: len(v) for k, v in groups.items()},
        "axis_angle": round(axis_angle, 2),
        "angle_confidence": round(angle_conf, 3),
        "angle_source": angle_source,
        "pin_detected_from_lines": detected_flags,
    }


def _enforce_horizontal_npn_emitter_rule(
    points: dict[str, tuple[float, float]], bbox: dict
) -> tuple[dict[str, tuple[float, float]], dict]:
    """Impede inversão C/E em NPN horizontal ou espelhado.

    A regra só é aplicada quando C e E estão claramente no mesmo lado da base
    e separados verticalmente. Assim, símbolos realmente rotacionados continuam
    seguindo a seta/pontos aprendidos, sem uma suposição de tela incorreta.
    """
    if not all(pin in points for pin in ("B", "C", "E")):
        return points, {"applied": False, "reason": "missing_points"}
    bx, by = points["B"]
    cx, cy = points["C"]
    ex, ey = points["E"]
    bw, bh = _bbox_dims(bbox)
    dcx, dex = cx - bx, ex - bx
    same_side = dcx * dex > 0.0
    clear_side = min(abs(dcx), abs(dex)) > 0.08 * bw
    clear_vertical_fork = abs(cy - ey) > 0.15 * bh
    if not (same_side and clear_side and clear_vertical_fork):
        return points, {"applied": False, "reason": "not_clear_horizontal_fork"}

    fork_side = "right" if (dcx + dex) > 0.0 else "left"
    swapped = ey < cy
    corrected = dict(points)
    if swapped:
        corrected["C"], corrected["E"] = corrected["E"], corrected["C"]
    return corrected, {
        "applied": True,
        "swapped": swapped,
        "fork_side": fork_side,
        "emitter_side": f"lower_{fork_side}",
        "rule": "NPN horizontal: seta inferior sempre e emissor",
    }


def detectar_terminais_componentes(
    components_info: dict,
    image_width: int | None = None,
    image_height: int | None = None,
    image: np.ndarray | None = None,
    *,
    legacy_wire_context: bool = False,
    confirmed_wire_bjt: set[str] | None = None,
) -> dict:
    if not isinstance(components_info, dict):
        raise ValueError("components_info deve ser um objeto JSON.")

    componentes_origem = components_info.get("components")
    if not isinstance(componentes_origem, list) or not componentes_origem:
        raise ValueError("Execute o OCR antes da detecção de terminais.")

    componentes = copy.deepcopy(componentes_origem)
    total_pinos = 0
    revisoes = 0
    identity_image = image
    ink_profile = None
    if image is not None and any(c.get("type") in IDENTITY_CLASSES for c in componentes):
        ink_profile = estimate_component_ink(image, components_info)
        identity_image = image_with_component_ink(image, ink_profile)

    for componente in componentes:
        tipo = str(componente.get("type", "")).strip()
        bbox = componente.get("bbox")
        if tipo not in PIN_LIBRARY:
            componente["terminals"] = []
            componente["terminal_detection"] = {
                "status": "unsupported_component",
                "orientation": "unknown",
                "review_required": True,
                "review_reasons": ["classe_sem_descricao_de_pinos"],
            }
            revisoes += 1
            continue
        if not isinstance(bbox, dict):
            raise ValueError(f"Componente {componente.get('component_id')} sem bbox válida.")

        saved_terminals = componente.get("terminals") or []
        if saved_terminals and all(str(t.get("source", "")).startswith("manual_ground_truth") for t in saved_terminals):
            componente["terminal_detection"] = {"status": "manual_ground_truth", "pin_count": len(saved_terminals),
                "review_required": False, "review_reasons": [], "method": "preserved_user_terminal_correction"}
            total_pinos += len(saved_terminals)
            continue
        if tipo in {'transistor_npn','transistor_pnp'} and componente.get('detection_id') in (confirmed_wire_bjt or set()):
            # The next operation already has three observed leads and a
            # confirmed branch arrow. Avoid unused provisional model searches.
            componente['terminals']=[]
            componente['terminal_detection']={'status':'pending_confirmed_wire_refinement','pin_count':0,'review_required':False}
            continue

        orientation, orientation_confidence = _orientation(tipo, bbox)
        electrical_axis_meta = None
        resistor_body_meta = None
        if tipo == "resistor":
            inferred_axis, inferred_confidence, resistor_body_meta = _resistor_body_axis(
                image, bbox, orientation
            )
            if inferred_confidence > 0.0:
                orientation = inferred_axis
                orientation_confidence = max(orientation_confidence, inferred_confidence)
                cx = float(resistor_body_meta.get("body_center_x"))
                cy = float(resistor_body_meta.get("body_center_y"))
                body_box = resistor_body_meta.get("body_bbox") or {}
                electrical_axis_meta = {
                    "source": "resistor_body_rectangle",
                    "body": resistor_body_meta,
                    "top_x": cx, "bottom_x": cx,
                    "left_y": cy, "right_y": cy,
                }
                angle=float(resistor_body_meta.get("body_long_axis_angle",0.0))
                tilt=min(_angle_distance_180(angle,0.0),_angle_distance_180(angle,90.0))
                if tilt <= 8.0 and body_box:
                    # Mantém o caminho já validado para resistor praticamente horizontal/vertical.
                    if orientation == "vertical":
                        top_start=(cx,float(body_box.get("y1",cy))); bottom_start=(cx,float(body_box.get("y2",cy)))
                        top_point, top_ok = _trace_terminal_from_symbol(image, top_start, (0,-1), max(10.0, float(bbox["y2"]-bbox["y1"])*0.60), corridor=4)
                        bottom_point, bottom_ok = _trace_terminal_from_symbol(image, bottom_start, (0,1), max(10.0, float(bbox["y2"]-bbox["y1"])*0.60), corridor=4)
                        electrical_axis_meta["top_point"]=[round(top_point[0],3),round(top_point[1],3)]
                        electrical_axis_meta["bottom_point"]=[round(bottom_point[0],3),round(bottom_point[1],3)]
                        electrical_axis_meta["trace_success"]=bool(top_ok and bottom_ok)
                    else:
                        left_start=(float(body_box.get("x1",cx)),cy); right_start=(float(body_box.get("x2",cx)),cy)
                        left_point,left_ok=_trace_terminal_from_symbol(image,left_start,(-1,0),max(10.0,float(bbox["x2"]-bbox["x1"])*0.60),corridor=4)
                        right_point,right_ok=_trace_terminal_from_symbol(image,right_start,(1,0),max(10.0,float(bbox["x2"]-bbox["x1"])*0.60),corridor=4)
                        electrical_axis_meta["left_point"]=[round(left_point[0],3),round(left_point[1],3)]
                        electrical_axis_meta["right_point"]=[round(right_point[0],3),round(right_point[1],3)]
                        electrical_axis_meta["trace_success"]=bool(left_ok and right_ok)
                else:
                    # Foto inclinada: segue o eixo longo real do corpo em graus.
                    theta=math.radians(angle);u=(math.cos(theta),math.sin(theta));half=max(3.0,float(resistor_body_meta.get("body_long_length",min(_bbox_dims(bbox))))/2.0)
                    start1=(cx-u[0]*half,cy-u[1]*half);start2=(cx+u[0]*half,cy+u[1]*half);max_trace=max(10.0,max(_bbox_dims(bbox))*.65)
                    p1,ok1=_trace_terminal_from_symbol(image,start1,(-u[0],-u[1]),max_trace,corridor=4);p2,ok2=_trace_terminal_from_symbol(image,start2,(u[0],u[1]),max_trace,corridor=4)
                    ordered=sorted([p1,p2],key=(lambda p:(p[0],p[1])) if orientation=="horizontal" else (lambda p:(p[1],p[0])))
                    electrical_axis_meta["axis_point_1"]=[round(ordered[0][0],3),round(ordered[0][1],3)];electrical_axis_meta["axis_point_2"]=[round(ordered[1][0],3),round(ordered[1][1],3)];electrical_axis_meta["axis_angle_degrees"]=round(angle,3);electrical_axis_meta["trace_success"]=bool(ok1 and ok2)
            else:
                pca_axis, pca_conf, pca_meta = _resistor_principal_axis(image, bbox, orientation)
                wire_axis, wire_conf, wire_meta = _two_terminal_border_axis(image, bbox, orientation)
                if pca_conf >= 0.58 and (wire_conf <= 0.0 or pca_axis == wire_axis or pca_conf >= wire_conf + 0.08):
                    orientation = pca_axis
                    orientation_confidence = max(orientation_confidence, pca_conf)
                    cx=float(pca_meta.get("body_center_x",(float(bbox["x1"])+float(bbox["x2"]))/2.0))
                    cy=float(pca_meta.get("body_center_y",(float(bbox["y1"])+float(bbox["y2"]))/2.0))
                    electrical_axis_meta={"source":"resistor_principal_axis","body":pca_meta,
                                          "top_x":cx,"bottom_x":cx,"left_y":cy,"right_y":cy}
                    pa,pb=_line_bbox_intersections(bbox,float(pca_meta.get("axis_angle",0.0)))
                    if orientation=="horizontal":
                        pa,pb=sorted((pa,pb),key=lambda p:(p[0],p[1]))
                        electrical_axis_meta["left_point"]=[round(pa[0],3),round(pa[1],3)]
                        electrical_axis_meta["right_point"]=[round(pb[0],3),round(pb[1],3)]
                    else:
                        pa,pb=sorted((pa,pb),key=lambda p:(p[1],p[0]))
                        electrical_axis_meta["top_point"]=[round(pa[0],3),round(pa[1],3)]
                        electrical_axis_meta["bottom_point"]=[round(pb[0],3),round(pb[1],3)]
                elif wire_conf > 0.0:
                    orientation = wire_axis
                    orientation_confidence = max(orientation_confidence, wire_conf)
                    electrical_axis_meta = wire_meta
        elif tipo in {"capacitor", "capacitor_polarized", "diode", "led"}:
            inferred_axis, inferred_confidence, electrical_axis_meta = _two_terminal_border_axis(
                image, bbox, orientation
            )
            if inferred_confidence > 0.0:
                orientation = inferred_axis
                orientation_confidence = max(orientation_confidence, inferred_confidence)
            if tipo in {"capacitor", "capacitor_polarized"}:
                legacy_axis, legacy_conf, legacy_meta = _capacitor_plate_axis(image, bbox, orientation)
                rot_axis, rot_conf, rot_meta = _capacitor_plate_axis_any_angle(image, bbox, orientation)
                rot_e_angle = float(rot_meta.get("electrical_axis_angle", 0.0)) if rot_conf > 0.0 else 0.0
                rot_tilt = min(_angle_distance_180(rot_e_angle,0.0), _angle_distance_180(rot_e_angle,90.0))
                if legacy_conf >= 0.55 and not (rot_conf >= legacy_conf + 0.08 and rot_tilt >= 8.0):
                    plate_axis, plate_conf, plate_meta = legacy_axis, legacy_conf, legacy_meta
                else:
                    plate_axis, plate_conf, plate_meta = rot_axis, rot_conf, rot_meta
                # As placas são evidência estrutural específica do capacitor e
                # vencem bbox/fios quando o par paralelo está claro.
                if plate_conf >= 0.55:
                    orientation = plate_axis
                    orientation_confidence = max(orientation_confidence, plate_conf)
                    electrical_axis_meta = plate_meta
                    # O eixo elétrico passa PELO CENTRO entre as duas placas.
                    if plate_axis == "vertical":
                        electrical_axis_meta["top_x"] = plate_meta.get("plate_center_x")
                        electrical_axis_meta["bottom_x"] = plate_meta.get("plate_center_x")
                    else:
                        electrical_axis_meta["left_y"] = plate_meta.get("plate_center_y")
                        electrical_axis_meta["right_y"] = plate_meta.get("plate_center_y")
            if tipo == "diode":
                bar_axis, bar_conf, bar_meta = _led_cathode_bar_axis(image, bbox, orientation)
                if bar_conf >= 0.50:
                    orientation = bar_axis
                    orientation_confidence = max(orientation_confidence, bar_conf)
                    electrical_axis_meta = dict(bar_meta)
            if tipo == "led":
                bar_axis, bar_conf, bar_meta = _led_cathode_bar_axis(image, bbox, orientation)
                pair_axis, pair_conf, pair_meta = _two_terminal_line_pair_axis(image, bbox, orientation)
                body_axis, body_conf, body_meta = _component_body_axis(image, bbox, orientation, tipo)
                bw_led, bh_led = _bbox_dims(bbox)
                selected_pair_score = float(pair_meta.get(
                    "vertical_pair_score" if pair_axis == "vertical" else "horizontal_pair_score", 0.0
                ))
                pair_strength = selected_pair_score / max(1.0, min(bw_led, bh_led))
                pair_gap = pair_meta.get("pair_gap")
                min_gap = 0.12 * (bh_led if pair_axis == "vertical" else bw_led)
                # Um par quase sem espaço entre os dois segmentos costuma ser
                # uma única linha pautada/Hough quebrada em dois pedaços, não
                # dois terminais separados pelo símbolo do LED.
                pair_looks_continuous = pair_gap is not None and float(pair_gap) < max(3.0, min_gap)

                chosen_axis = None
                chosen_meta = None
                chosen_conf = 0.0
                # Dois fios colineares muito fortes vencem qualquer ruído do
                # corpo/barra (caso clássico de LED vertical com bbox grande).
                if pair_conf > 0.0 and pair_strength >= 0.70 and not pair_looks_continuous:
                    chosen_axis, chosen_meta, chosen_conf = pair_axis, pair_meta, pair_conf
                # Concordância entre duas pistas independentes.
                elif bar_conf > 0.0 and pair_conf > 0.0 and bar_axis == pair_axis:
                    chosen_axis, chosen_meta, chosen_conf = bar_axis, dict(bar_meta), max(bar_conf, pair_conf)
                    chosen_meta["wire_pair_check"] = pair_meta
                elif bar_conf > 0.0 and body_conf > 0.0 and bar_axis == body_axis:
                    chosen_axis, chosen_meta, chosen_conf = bar_axis, dict(bar_meta), max(bar_conf, body_conf)
                    chosen_meta["body_axis_check"] = body_meta
                elif pair_conf > 0.0 and body_conf > 0.0 and pair_axis == body_axis:
                    chosen_axis, chosen_meta, chosen_conf = pair_axis, dict(pair_meta), max(pair_conf, body_conf)
                    chosen_meta["body_axis_check"] = body_meta
                elif pair_conf > 0.0 and pair_strength >= 0.45 and not pair_looks_continuous:
                    chosen_axis, chosen_meta, chosen_conf = pair_axis, pair_meta, pair_conf
                elif bar_conf > 0.0:
                    chosen_axis, chosen_meta, chosen_conf = bar_axis, bar_meta, bar_conf
                elif body_conf > 0.0:
                    chosen_axis, chosen_conf = body_axis, body_conf
                    chosen_meta = dict(body_meta)
                    chosen_meta.update(_body_axis_endpoints(image, bbox, body_meta, body_axis))
                    if body_axis == "vertical":
                        chosen_meta["top_x"] = body_meta.get("body_center_x")
                        chosen_meta["bottom_x"] = body_meta.get("body_center_x")
                    else:
                        chosen_meta["left_y"] = body_meta.get("body_center_y")
                        chosen_meta["right_y"] = body_meta.get("body_center_y")

                if chosen_axis is not None:
                    orientation = chosen_axis
                    orientation_confidence = max(orientation_confidence, chosen_conf)
                    electrical_axis_meta = chosen_meta
        radius = _search_radius(bbox)
        terminals = []
        axis_angle = 0.0 if orientation == "horizontal" else 90.0
        angle_confidence = 0.0
        angle_source = "bbox_axis"

        learned_generic = None
        if (
            tipo not in {"transistor_npn", "transistor_pnp"}
            and tipo in TRAINED_TERMINAL_SCHEMAS
            and predict_component_terminals is not None
        ):
            try:
                learned_generic = predict_component_terminals(
                    image=image, bbox=bbox, component_type=tipo
                )
            except Exception as exc:
                learned_generic = {
                    "accepted": False,
                    "review_required": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            componente["learned_terminal_geometry"] = learned_generic

        # Placas, barra catódica e eixo de corpo/fios são evidências físicas
        # explicáveis. Quando já estão fortes, elas vencem a rede; a rede fica
        # como diagnóstico e fallback para geometrias ambíguas.
        structural_source = str((electrical_axis_meta or {}).get("source") or "")
        generic_structural_lock = bool(
            tipo in {"resistor", "capacitor", "capacitor_polarized", "diode", "led"}
            and structural_source
            and structural_source != "bbox_fallback"
            and float(orientation_confidence) >= 0.55
        )
        generic_model_applied = bool(
            learned_generic and learned_generic.get("accepted") and not generic_structural_lock
        )

        if generic_model_applied:
            visible_pins = [pin for pin in TRAINED_TERMINAL_SCHEMAS[tipo] if pin]
            learned_points = learned_generic["points"]
            kp_confidences = learned_generic.get("keypoint_confidences") or {}
            if len(visible_pins) >= 2:
                p1, p2 = learned_points[visible_pins[0]], learned_points[visible_pins[1]]
                axis_angle = math.degrees(math.atan2(
                    float(p2["y"]) - float(p1["y"]),
                    max(1e-9, float(p2["x"]) - float(p1["x"])),
                ))
                orientation = "horizontal" if abs(math.cos(math.radians(axis_angle))) >= abs(math.sin(math.radians(axis_angle))) else "vertical"
            else:
                axis_angle = 90.0
                orientation = "vertical"
            angle_confidence = float(learned_generic.get("keypoint_confidence", 0.0))
            orientation_confidence = max(orientation_confidence, angle_confidence)
            angle_source = "trained_all_component_terminal_model"
            bx1, by1 = float(bbox["x1"]), float(bbox["y1"])
            bw, bh = max(1.0, float(bbox["x2"]) - bx1), max(1.0, float(bbox["y2"]) - by1)
            for index, pin_name in enumerate(visible_pins, start=1):
                point = learned_points[pin_name]
                sx, sy = _clip_point(float(point["x"]), float(point["y"]), image_width, image_height)
                nx, ny = (sx - bx1) / bw, (sy - by1) / bh
                terminal_side, _ = _expected_side(nx, ny)
                terminals.append({
                    "pin_id": f"{componente.get('component_id', 'COMP')}.{pin_name}",
                    "pin": pin_name,
                    "electrical_role": PIN_SEMANTICS.get(tipo, {}).get(pin_name, pin_name),
                    "terminal_side": terminal_side,
                    "index": index,
                    "position": {"x": round(sx, 2), "y": round(sy, 2)},
                    "search_radius_px": radius,
                    "source": "trained_all_component_terminal_model",
                    "identity_uncertain": bool(learned_generic.get("review_required", False)),
                    "confidence": round(float(kp_confidences.get(pin_name, angle_confidence)), 3),
                })

        elif tipo in {"transistor_npn", "transistor_pnp"}:
            physical_points, physical_meta = _detect_transistor_fork(
                image, bbox, tipo, orientation, image_width, image_height, radius
            )
            physical_reliable = bool(
                physical_meta.get("success")
                and physical_meta.get("branch_distinct", False)
                and physical_meta.get("arrow_branch") is not None
                and physical_meta.get("class_arrow_consistent") is not False
                and "fallback" not in str(physical_meta.get("source", ""))
                and "partial" not in str(physical_meta.get("source", ""))
            )
            physical_authoritative = bool(
                physical_meta.get("success")
                and physical_meta.get("branch_distinct", False)
                and "fallback" not in str(physical_meta.get("source", ""))
                and "partial" not in str(physical_meta.get("source", ""))
            )
            learned_prediction = None
            learned_diagnostic = None
            known_training_match = None
            runtime_model = None
            # Infer from present ink and learned crop features. A recognized
            # whole training photograph must not inject its annotation as an
            # infallible prediction for a new recognition run.
            if not physical_authoritative and terminal_model_runtime_status is not None:
                try:
                    runtime_model = terminal_model_runtime_status(load_model=True)
                except Exception as exc:
                    runtime_model = {
                        "model_exists": False,
                        "model_loadable": False,
                        "model_load_error": f"{type(exc).__name__}: {exc}",
                    }
            trained_model_ready = bool(runtime_model and runtime_model.get("model_exists") and runtime_model.get("model_loadable"))

            pose_prediction = None
            visual_prior = None
            if known_training_match is not None:
                learned_diagnostic = dict(known_training_match)
                learned_prediction = {
                    "points": known_training_match["points"],
                    "box_confidence": 1.0,
                    "keypoint_confidence": 1.0,
                    "keypoint_confidences": {"B": 1.0, "C": 1.0, "E": 1.0},
                    "review_required": False,
                    "source": "known_training_ground_truth",
                    "known_training_match": known_training_match,
                }
            else:
                if trained_model_ready and diagnose_transistor_keypoints is not None:
                    try:
                        learned_diagnostic = diagnose_transistor_keypoints(
                            image=image, bbox=bbox, component_type=tipo
                        )
                        if learned_diagnostic.get("accepted"):
                            pose_prediction = {
                                "points": learned_diagnostic["points"],
                                "box_confidence": learned_diagnostic.get("box_confidence", 0.0),
                                "keypoint_confidence": learned_diagnostic.get("keypoint_confidence", 0.0),
                                "keypoint_confidences": learned_diagnostic.get("keypoint_confidences", {}),
                                "review_required": learned_diagnostic.get("review_required", True),
                                "source": "trained_terminal_keypoint_model",
                            }
                    except Exception as exc:
                        learned_diagnostic = {
                            "accepted": False,
                            "review_required": True,
                            "rejection_reasons": [f"erro_diagnostico:{type(exc).__name__}:{exc}"],
                        }
                # Segundo supervisionado: vizinhos rotulados da mesma classe.
                # Só publica em alta similaridade; evita cair na heurística antiga.
                if predict_transistor_visual_prior is not None:
                    try:
                        visual_prior = predict_transistor_visual_prior(
                            image=image, bbox=bbox, component_type=tipo
                        )
                    except Exception:
                        visual_prior = None

                # Prioridade: pose aprovada sem revisão > prior visual de alta
                # similaridade > pose aprovada porém moderada. Assim não piora
                # uma previsão boa do modelo, mas resgata colapsos/rejeições.
                if pose_prediction is not None and not pose_prediction.get("review_required", True):
                    learned_prediction = pose_prediction
                elif visual_prior is not None and visual_prior.get("accepted"):
                    learned_prediction = visual_prior
                    if learned_diagnostic is None:
                        learned_diagnostic = {}
                    learned_diagnostic = {
                        "pose": learned_diagnostic,
                        "visual_prior": visual_prior,
                        "accepted": True,
                        "source": "training_visual_prior",
                    }
                elif pose_prediction is not None:
                    learned_prediction = pose_prediction

            reliable = False
            model_review_required = False
            if learned_prediction is not None:
                kp_conf = float(learned_prediction.get("keypoint_confidence", 0.0))
                box_conf = float(learned_prediction.get("box_confidence", 0.0))
                fork_points = {
                    pin: (
                        float(learned_prediction["points"][pin]["x"]),
                        float(learned_prediction["points"][pin]["y"]),
                    )
                    for pin in ("B", "C", "E")
                }
                learned_npn_rule = {"applied": False, "reason": "not_npn_or_manual_ground_truth"}
                if (
                    tipo == "transistor_npn"
                    and str(learned_prediction.get("source") or "") != "known_training_ground_truth"
                ):
                    fork_points, learned_npn_rule = _enforce_horizontal_npn_emitter_rule(
                        fork_points, bbox
                    )
                learned_kp_confidences = dict(learned_prediction.get("keypoint_confidences", {}))
                if learned_npn_rule.get("swapped"):
                    learned_kp_confidences["C"], learned_kp_confidences["E"] = (
                        learned_kp_confidences.get("E", kp_conf),
                        learned_kp_confidences.get("C", kp_conf),
                    )
                fork_meta = {
                    "success": True,
                    "reliable": True,
                    "source": str(learned_prediction.get("source") or "trained_terminal_keypoint_model"),
                    "transistor_type": "NPN" if tipo == "transistor_npn" else "PNP",
                    "symbol_orientation": orientation,
                    "axis_angle": 0.0 if orientation == "horizontal" else 90.0,
                    "angle_confidence": kp_conf,
                    "angle_source": str(learned_prediction.get("source") or "trained_terminal_keypoint_model"),
                    "arrow_branch": "learned",
                    "branch_distinct": True,
                    "model_box_confidence": box_conf,
                    "model_keypoint_confidence": kp_conf,
                    "model_keypoint_confidences": learned_kp_confidences,
                    "model_review_required": bool(learned_prediction.get("review_required", False)),
                    "model_diagnostic": learned_diagnostic,
                    "known_training_match": learned_prediction.get("known_training_match"),
                    "visual_prior_score": learned_prediction.get("visual_prior_score"),
                    "visual_prior_neighbors": learned_prediction.get("neighbors"),
                    "learned_npn_emitter_rule": learned_npn_rule,
                }
                reliable = True
                model_review_required = bool(learned_prediction.get("review_required", False))
            elif trained_model_ready:
                # REGRA IMPORTANTE: se existe um modelo treinado e ele rejeita
                # esta previsão, NÃO cai na heurística. Isso evita que um
                # transistor treinado volte a receber B/C/E grosseiros por
                # geometria antiga. O usuário corrige no terminal-trainer.
                fork_points = {}
                fork_meta = {
                    "success": False,
                    "reliable": False,
                    "source": "trained_terminal_keypoint_model_rejected",
                    "transistor_type": "NPN" if tipo == "transistor_npn" else "PNP",
                    "training_required": True,
                    "suppressed_unreliable_points": True,
                    "model_diagnostic": learned_diagnostic or {
                        "accepted": False,
                        "rejection_reasons": ["modelo_sem_previsao_aceitavel"],
                    },
                    "model_runtime": runtime_model,
                }
            else:
                # Só usa geometria quando NÃO existe um modelo treinado
                # carregável. Depois do primeiro treino, o modelo tem prioridade
                # exclusiva e a heurística não pode sobrescrevê-lo.
                fork_points, fork_meta = physical_points, physical_meta
                reliable = bool(
                    fork_meta.get("success")
                    and fork_meta.get("branch_distinct", False)
                    and fork_meta.get("arrow_branch") is not None
                    and fork_meta.get("class_arrow_consistent") is not False
                    and "fallback" not in str(fork_meta.get("source", ""))
                    and "partial" not in str(fork_meta.get("source", ""))
                )
                fork_meta["reliable"] = reliable
                fork_meta["model_runtime"] = runtime_model

            componente["transistor_geometry"] = fork_meta
            if fork_meta.get("symbol_orientation") in {"horizontal", "vertical"}:
                orientation = str(fork_meta["symbol_orientation"])
                orientation_confidence = 0.92 if reliable else max(orientation_confidence, 0.45)
            axis_angle = float(fork_meta.get("axis_angle", 0.0 if orientation == "horizontal" else 90.0))
            angle_confidence = float(fork_meta.get("angle_confidence", 0.0))
            angle_source = str(fork_meta.get("angle_source", fork_meta.get("source", "transistor_fork")))

            if reliable:
                side_map = {"B": "fork_base", "C": "fork_collector", "E": "fork_emitter"}
                kp_confs = fork_meta.get("model_keypoint_confidences", {})
                for index, pin_name in enumerate(("B", "C", "E"), start=1):
                    sx, sy = fork_points[pin_name]
                    pin_conf = float(kp_confs.get(pin_name, angle_confidence or 0.55))
                    terminals.append(
                        {
                            "pin_id": f"{componente.get('component_id', 'COMP')}.{pin_name}",
                            "pin": pin_name,
                            "electrical_role": PIN_SEMANTICS[tipo][pin_name],
                            "terminal_side": side_map[pin_name],
                            "index": index,
                            "position": {"x": round(sx, 2), "y": round(sy, 2)},
                            "search_radius_px": radius,
                            "source": fork_meta["source"],
                            "identity_uncertain": bool(model_review_required),
                            "confidence": round(max(0.0, min(1.0, pin_conf)), 3),
                        }
                    )
            else:
                fork_meta["suppressed_unreliable_points"] = True
                fork_meta["training_required"] = True

        elif tipo in TWO_TERMINAL_CLASSES:
            # LED/diodo: quando o eixo elétrico foi inferido pelos fios que
            # entram em bordas opostas, use essas próprias interseções. Não
            # deixe setas do LED/barra do diodo inclinarem o eixo pelo Hough.
            use_border_points = (
                tipo in {"resistor", "capacitor", "capacitor_polarized", "diode", "led"}
                and electrical_axis_meta
                and electrical_axis_meta.get("source") != "bbox_fallback"
            )
            generic_axis_points = electrical_axis_meta.get("axis_point_1") if electrical_axis_meta else None
            generic_axis_points2 = electrical_axis_meta.get("axis_point_2") if electrical_axis_meta else None
            if use_border_points and generic_axis_points is not None and generic_axis_points2 is not None:
                points=[_clip_point(float(generic_axis_points[0]),float(generic_axis_points[1]),image_width,image_height),_clip_point(float(generic_axis_points2[0]),float(generic_axis_points2[1]),image_width,image_height)]
                axis_angle=float(electrical_axis_meta.get("axis_angle_degrees", electrical_axis_meta.get("electrical_axis_angle",0.0)))
                angle_confidence=float(orientation_confidence);angle_source=str(electrical_axis_meta.get("source","class_geometry"))
            elif use_border_points and orientation == "vertical":
                tx = electrical_axis_meta.get("top_x")
                bx = electrical_axis_meta.get("bottom_x")
                top_point = electrical_axis_meta.get("top_point")
                bottom_point = electrical_axis_meta.get("bottom_point")
                if top_point is not None and bottom_point is not None:
                    points = [
                        _clip_point(float(top_point[0]), float(top_point[1]), image_width, image_height),
                        _clip_point(float(bottom_point[0]), float(bottom_point[1]), image_width, image_height),
                    ]
                    tx, bx = float(top_point[0]), float(bottom_point[0])
                    axis_angle = math.degrees(math.atan2(
                        float(bottom_point[1]) - float(top_point[1]),
                        max(1e-9, float(bottom_point[0]) - float(top_point[0]))
                    ))
                    angle_confidence = float(orientation_confidence)
                    angle_source = "body_axis_endpoints"
                elif tx is not None and bx is not None:
                    points = [
                        _clip_point(float(tx), float(bbox["y1"]), image_width, image_height),
                        _clip_point(float(bx), float(bbox["y2"]), image_width, image_height),
                    ]
                    dx = float(bx) - float(tx)
                    dy = max(1e-9, float(bbox["y2"]) - float(bbox["y1"]))
                    axis_angle = math.degrees(math.atan2(dy, dx))
                    angle_confidence = float(orientation_confidence)
                    angle_source = "opposite_border_electrical_axis"
                else:
                    points, axis_angle, angle_confidence, angle_source = _two_terminal_geometry(
                        image, bbox, orientation, image_width, image_height
                    )
            elif use_border_points and orientation == "horizontal":
                ly = electrical_axis_meta.get("left_y")
                ry = electrical_axis_meta.get("right_y")
                left_point = electrical_axis_meta.get("left_point")
                right_point = electrical_axis_meta.get("right_point")
                if left_point is not None and right_point is not None:
                    points = [
                        _clip_point(float(left_point[0]), float(left_point[1]), image_width, image_height),
                        _clip_point(float(right_point[0]), float(right_point[1]), image_width, image_height),
                    ]
                    axis_angle = math.degrees(math.atan2(
                        float(right_point[1]) - float(left_point[1]),
                        max(1e-9, float(right_point[0]) - float(left_point[0]))
                    ))
                    angle_confidence = float(orientation_confidence)
                    angle_source = "body_axis_endpoints"
                elif ly is not None and ry is not None:
                    points = [
                        _clip_point(float(bbox["x1"]), float(ly), image_width, image_height),
                        _clip_point(float(bbox["x2"]), float(ry), image_width, image_height),
                    ]
                    dx = max(1e-9, float(bbox["x2"]) - float(bbox["x1"]))
                    dy = float(ry) - float(ly)
                    axis_angle = math.degrees(math.atan2(dy, dx))
                    angle_confidence = float(orientation_confidence)
                    angle_source = "opposite_border_electrical_axis"
                else:
                    points, axis_angle, angle_confidence, angle_source = _two_terminal_geometry(
                        image, bbox, orientation, image_width, image_height
                    )
            else:
                points, axis_angle, angle_confidence, angle_source = _two_terminal_geometry(
                    image, bbox, orientation, image_width, image_height
                )
            # Para componentes de dois terminais, a geometria do eixo ainda
            # fornece a orientação/ângulo. O ponto final, porém, é refinado na
            # borda elétrica esperada para não grudar em detalhes internos.
            layout = _focus_layout(tipo, orientation) if tipo in FOCUS_CLASSES else PIN_LIBRARY[tipo]
            point_pin_names = [pin for pin, _nx, _ny in layout]
            if tipo in {"diode", "led"} and electrical_axis_meta and electrical_axis_meta.get("bar_center_x") is not None:
                if orientation == "vertical":
                    bar_pos = float(electrical_axis_meta.get("bar_center_y", (points[0][1] + points[1][1]) / 2.0))
                    d0 = abs(float(points[0][1]) - bar_pos)
                    d1 = abs(float(points[1][1]) - bar_pos)
                else:
                    bar_pos = float(electrical_axis_meta.get("bar_center_x", (points[0][0] + points[1][0]) / 2.0))
                    d0 = abs(float(points[0][0]) - bar_pos)
                    d1 = abs(float(points[1][0]) - bar_pos)
                if d0 <= d1:
                    point_pin_names = ["K", "A"]
                else:
                    point_pin_names = ["A", "K"]
            for index, ((pin_name, nx, ny), (x, y)) in enumerate(zip([(point_pin_names[i], *layout[i][1:]) for i in range(len(layout))], points), start=1):
                if tipo in FOCUS_CLASSES:
                    direct_class_axis = bool(electrical_axis_meta and electrical_axis_meta.get("axis_point_1") is not None and tipo in {"resistor","capacitor","capacitor_polarized"})
                    if direct_class_axis:
                        sx,sy=float(x),float(y);snapped=True;terminal_side,_=_expected_side(nx,ny)
                        csx,csy,connection_snapped,connection_mode=_snap_to_local_connection(image,(sx,sy),orientation,max(6,min(14,radius)))
                        if connection_snapped:sx,sy=csx,csy
                    else:
                        # Para símbolos levemente inclinados, use a própria
                        # interseção do eixo com a bbox como alvo. Assim o snap
                        # preserva a altura correta de cada extremidade.
                        bx1, by1 = float(bbox["x1"]), float(bbox["y1"])
                        bx2, by2 = float(bbox["x2"]), float(bbox["y2"])
                        target_nx = (x - bx1) / max(1e-9, bx2 - bx1)
                        target_ny = (y - by1) / max(1e-9, by2 - by1)
                        sx, sy, snapped, terminal_side = _snap_to_border_ink(
                            image, bbox, target_nx, target_ny, radius
                        )
                        csx, csy, connection_snapped, connection_mode = _snap_to_local_connection(
                            image, (sx, sy), orientation, max(6, min(14, radius))
                        )
                        if connection_snapped:
                            sx, sy = csx, csy
                else:
                    sx, sy, snapped = _snap_to_ink(image, bbox, (x, y), radius)
                    terminal_side, _ = _expected_side(nx, ny)
                    connection_snapped, connection_mode = False, "none"
                sx, sy = _clip_point(sx, sy, image_width, image_height)
                terminals.append(
                    {
                        "pin_id": f"{componente.get('component_id', 'COMP')}.{pin_name}",
                        "pin": pin_name,
                        "electrical_role": PIN_SEMANTICS.get(tipo, {}).get(pin_name, pin_name),
                        "terminal_side": terminal_side,
                        "index": index,
                        "position": {"x": round(sx, 2), "y": round(sy, 2)},
                        "search_radius_px": radius,
                        "source": "border_ink_geometry" if snapped else ("local_image_geometry" if angle_source == "local_hough" else "bbox_geometry"),
                        "connection_snap": {
                            "applied": bool(connection_snapped),
                            "mode": connection_mode,
                        },
                        "confidence": round(max(orientation_confidence, angle_confidence * 0.85), 3),
                    }
                )
        else:
            layout = _focus_layout(tipo, orientation) if tipo in FOCUS_CLASSES else _normalized_layout(tipo, orientation)
            for index, (pin_name, nx, ny) in enumerate(layout, start=1):
                x, y = _pixel_from_normalized(bbox, nx, ny)
                x, y = _clip_point(x, y, image_width, image_height)
                if tipo in FOCUS_CLASSES:
                    sx, sy, snapped, terminal_side = _snap_to_border_ink(
                        image, bbox, nx, ny, radius
                    )
                else:
                    sx, sy, snapped = _snap_to_ink(image, bbox, (x, y), radius)
                    terminal_side, _ = _expected_side(nx, ny)
                sx, sy = _clip_point(sx, sy, image_width, image_height)
                terminals.append(
                    {
                        "pin_id": f"{componente.get('component_id', 'COMP')}.{pin_name}",
                        "pin": pin_name,
                        "electrical_role": PIN_SEMANTICS.get(tipo, {}).get(pin_name, pin_name),
                        "terminal_side": terminal_side,
                        "index": index,
                        "position": {"x": round(sx, 2), "y": round(sy, 2)},
                        "search_radius_px": radius,
                        "source": "border_ink_geometry" if snapped else "bbox_geometry",
                        "confidence": round(orientation_confidence, 3),
                    }
                )

        identity_meta = None
        if tipo in IDENTITY_CLASSES:
            # Camera shadows can erase the small input signs from a global
            # color mask. The opamp owns and reads its original local ink.
            terminals, identity_meta = resolve_terminal_identity(image if tipo=='opamp' else identity_image, componente, terminals)
            for terminal in terminals:
                terminal["electrical_role"] = PIN_SEMANTICS[tipo][terminal["pin"]]
            componente["terminal_identity_geometry"] = identity_meta
            if identity_meta["confirmed"]:
                orientation_confidence = max(orientation_confidence, .93)
                if len(terminals) >= 2:
                    if tipo == "opamp":
                        start = np.mean([[t["position"]["x"], t["position"]["y"]] for t in terminals[:2]], axis=0)
                        end = np.array([terminals[2]["position"]["x"], terminals[2]["position"]["y"]])
                    else:
                        start = np.array([terminals[0]["position"]["x"], terminals[0]["position"]["y"]])
                        end = np.array([terminals[1]["position"]["x"], terminals[1]["position"]["y"]])
                    axis_angle = math.degrees(math.atan2(end[1]-start[1], end[0]-start[0]))
                    orientation = "horizontal" if abs(end[0]-start[0]) >= abs(end[1]-start[1]) else "vertical"
                    angle_source, angle_confidence = "visible_symbol_terminal_identity", .93

        reasons = []
        if orientation_confidence < 0.5 and tipo not in ONE_PIN_CLASSES:
            reasons.append("orientacao_geometrica_ambigua")
        if tipo in POLARITY_SENSITIVE and tipo not in {"diode", "led", "transistor_npn", "transistor_pnp"} and not (identity_meta or {}).get("confirmed"):
            reasons.append("confirmar_identidade_eletrica_dos_lados")
            reasons.append((identity_meta or {}).get("reason", "identidade_visual_nao_confirmada"))
        if tipo in {"transistor_npn", "transistor_pnp"}:
            tg = componente.get("transistor_geometry") or {}
            if tg.get("branch_distinct") is False:
                reasons.append("ramos_coletor_emissor_nao_separados")
            if tg.get("suppressed_unreliable_points"):
                reasons.append("treinar_ou_corrigir_terminais_do_transistor")
            if tg.get("model_review_required"):
                reasons.append("modelo_keypoints_baixa_confianca_revisar")
            if str(tg.get("source") or "") == "trained_terminal_keypoint_model_rejected":
                reasons.append("modelo_keypoints_rejeitou_previsao")
        if tipo in {"diode", "led", "transistor_npn", "transistor_pnp"}:
            if orientation_confidence < 0.5:
                reasons.append("confirmar_orientacao_dos_pinos_polarizados")
        if generic_model_applied and learned_generic.get("review_required") and not (identity_meta or {}).get("confirmed"):
            reasons.append("modelo_multiclasse_baixa_confianca_revisar")
        if tipo in {"transistor_npn", "transistor_pnp"}:
            tg = componente.get("transistor_geometry") or {}
            if image is not None and tg.get("arrow_branch") is None:
                reasons.append("seta_do_emissor_nao_confirmada")
            if image is not None and tg.get("class_arrow_consistent") is False:
                reasons.append("classe_bjt_conflita_com_sentido_da_seta")
            if image is not None and not tg.get("branch_distinct", tg.get("success", False)):
                reasons.append("forquilha_do_transistor_ambigua")

        orientation_source = "bbox_aspect_ratio"
        if (identity_meta or {}).get("confirmed"):
            orientation_source = "visible_symbol_terminal_identity"
        if generic_model_applied:
            orientation_source = "trained_all_component_terminal_model"
        if tipo in {"resistor", "capacitor", "capacitor_polarized", "diode", "led"} and electrical_axis_meta and electrical_axis_meta.get("source") != "bbox_fallback":
            if not generic_model_applied:
                orientation_source = str(electrical_axis_meta.get("source"))
            componente["electrical_axis_geometry"] = electrical_axis_meta
        if tipo == "resistor" and resistor_body_meta:
            componente["resistor_body_geometry"] = resistor_body_meta
        if tipo in {"transistor_npn", "transistor_pnp"} and (componente.get("transistor_geometry") or {}).get("success"):
            tg_source = str((componente.get("transistor_geometry") or {}).get("source") or "")
            orientation_source = "trained_terminal_keypoint_model" if tg_source == "trained_terminal_keypoint_model" else "transistor_fork_geometry"
        if (identity_meta or {}).get("confirmed"):
            orientation_source = "visible_symbol_terminal_identity"
        componente["orientation"] = {
            "axis": orientation,
            "confidence": round(orientation_confidence, 3),
            "source": orientation_source,
            "angle_degrees": round(axis_angle, 2),
            "angle_confidence": round(angle_confidence, 3),
            "angle_source": angle_source,
        }
        componente["terminals"] = terminals
        componente["terminal_detection"] = {
            "status": "estimated",
            "pin_count": len(terminals),
            "review_required": bool(reasons),
            "review_reasons": reasons,
            "method": "visible_symbol_terminal_identity" if (identity_meta or {}).get("confirmed") else "trained_multiclass_pose_with_safe_geometry_fallback" if generic_model_applied else "transistor_fork_or_two_terminal_electrical_axis_plus_local_image_geometry",
        }
        total_pinos += len(terminals)
        if reasons:
            revisoes += 1

    # Connectivity belongs to the wire graph. Merging pins between boxes can
    # move them outside their own symbols and must not alter their coordinates.
    aligned_terminal_count = _align_nearby_terminals(componentes,image) if legacy_wire_context else 0

    result = {
        "schema_version": "stage7-4.1-visible-terminal-identity",
        "terminal_identity_ink_profile": ink_profile,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_schema_version": components_info.get("schema_version"),
        "count": len(componentes),
        "terminal_count": total_pinos,
        "review_count": revisoes,
        "nearby_terminal_snap_count": aligned_terminal_count,
        "method": "class_geometry_plus_model_first_transistor_keypoints",
        "components": componentes,
    }
    if legacy_wire_context:
        return result
    from backend.terminal_wire_contacts import constrain_terminal_positions
    if image is not None:
        return constrain_terminal_positions(result,image)
    # Geometry-only callers still receive bounded coordinates without allocating
    # a page-sized synthetic image or pretending that wire evidence exists.
    for component in result["components"]:
        box=component.get("bbox") or {}
        if not all(k in box for k in ("x1","y1","x2","y2")):
            continue
        for terminal in component.get("terminals") or []:
            p=terminal["position"]
            for axis in ("x","y"):
                lo,hi=float(box[axis+"1"]),float(box[axis+"2"])
                inset=min(1.,max(0.,hi-lo)*.25)
                p[axis]=round(max(lo+inset,min(hi-inset,float(p[axis]))),2)
    result["terminals_inside_component_bbox"]=True
    return result



def salvar_imagem_terminais(image_path: Path, resultado: dict, output_path: Path) -> None:
    imagem = cv2.imread(str(image_path))
    if imagem is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path.name}")

    canvas = imagem.copy()
    for componente in resultado.get("components", []):
        bbox = componente.get("bbox") or {}
        if bbox:
            p1 = (int(round(bbox["x1"])), int(round(bbox["y1"])))
            p2 = (int(round(bbox["x2"])), int(round(bbox["y2"])))
            cv2.rectangle(canvas, p1, p2, (90, 90, 90), 1)

        tipo_comp = str(componente.get("type") or "")
        td = componente.get("terminal_detection") or {}
        tg = componente.get("transistor_geometry") or {}
        if not componente.get('terminals') and td.get('review_required') and bbox:
            cv2.putText(canvas,'REVISAR CONTATOS',(int(bbox['x1'])+3,int(bbox['y1'])+18),
                        cv2.FONT_HERSHEY_SIMPLEX,.45,(0,100,210),1,cv2.LINE_AA)
        if tipo_comp in {"transistor_npn", "transistor_pnp"} and not componente.get("terminals") and tg.get("training_required") and bbox:
            cv2.putText(
                canvas,
                "TREINAR B/C/E",
                (int(round(bbox["x1"])), max(18, int(round(bbox["y1"])) + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 220),
                2,
                cv2.LINE_AA,
            )

        for terminal in componente.get("terminals", []):
            pos = terminal["position"]
            # Clip all terminal graphics to their box, including search rings.
            # Their radius remains available to the connection graph in JSON.
            x1,y1=max(0,int(math.floor(bbox['x1']))+1),max(0,int(math.floor(bbox['y1']))+1)
            x2,y2=min(canvas.shape[1],int(math.ceil(bbox['x2']))),min(canvas.shape[0],int(math.ceil(bbox['y2'])))
            marker=canvas[y1:y2,x1:x2]
            if marker.size==0:
                continue
            p = (int(round(pos["x"]))-x1, int(round(pos["y"]))-y1)
            cv2.circle(marker, p, 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(marker, p, int(terminal.get("search_radius_px",8)), (0, 120, 255), 1, cv2.LINE_AA)
            cv2.putText(
                marker,
                str(terminal["pin"]) + ("?" if terminal.get("identity_uncertain") else ""),
                (max(1,min(marker.shape[1]-26,p[0]+7)), max(14,min(marker.shape[0]-2,p[1]-7))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 180),
                1,
                cv2.LINE_AA,
            )

        rotulo = componente.get("component_id", "")
        tipo = componente.get("type", "")
        valor_info = componente.get("value") or {}
        valor = str(valor_info.get("text") or "").strip()
        orient_data = componente.get("orientation") or {}
        orient = orient_data.get("axis")
        angle = orient_data.get("angle_degrees")
        if bbox and rotulo:
            suffix = f" {angle:+.1f}deg" if isinstance(angle, (int, float)) else ""
            texto_rotulo = f"{rotulo} {tipo}".strip()
            if valor:
                texto_rotulo += f" {valor}"
            texto_rotulo += f" [{orient or '?'}{suffix}]"
            if (componente.get("terminal_detection") or {}).get("review_required"):
                texto_rotulo += " ?"
            cv2.putText(
                canvas,
                texto_rotulo[:90],
                (int(round(bbox["x1"])), max(16, int(round(bbox["y1"])) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (60, 60, 60),
                1,
                cv2.LINE_AA,
            )

    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Não foi possível salvar {output_path.name}.")


def executar_deteccao_terminais(image_path: Path, components_info: dict, output_json_path: Path, output_image_path: Path,
                              *, wires_info: dict | None = None, skeleton: np.ndarray | None = None, save_preview: bool = True) -> dict:
    imagem = cv2.imread(str(image_path))
    if imagem is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path.name}")
    height, width = imagem.shape[:2]

    confirmed_bjt=set();observed_bjt={}
    if wires_info is not None and skeleton is not None:
        from backend.npn_wire_contacts import detect_npn_wire_contacts,detect_bjt_wire_contacts
        color=image_with_component_ink(imagem,estimate_component_ink(imagem,components_info))
        for c in components_info.get('components',[]):
            kind=c.get('type');cid=c.get('detection_id')
            if kind not in {'transistor_npn','transistor_pnp'} or not cid:continue
            if any(str(t.get('source','')).startswith('manual') for t in c.get('terminals',[])):continue
            try:
                if kind=='transistor_npn':found,meta=detect_npn_wire_contacts(imagem,c['bbox'],skeleton,color_reference=color)
                else:found,meta=detect_bjt_wire_contacts(imagem,c['bbox'],skeleton,kind,color_reference=color)
            except Exception:continue
            if found is not None and meta.get('arrow_confirmed') and not meta.get('direction_conflict'):
                confirmed_bjt.add(cid);observed_bjt[cid]=(found,meta)

    component_errors = []
    degraded_mode = False
    try:
        resultado = detectar_terminais_componentes(
            components_info,
            image_width=width,
            image_height=height,
            image=imagem,
            confirmed_wire_bjt=confirmed_bjt,
        )
    except Exception as erro_global:
        # Modo resiliente: um único símbolo difícil não pode derrubar toda a
        # Etapa 7. Reprocessa componente por componente, preservando os bons e
        # marcando somente o componente que falhou para revisão.
        degraded_mode = True
        componentes_saida = []
        total_pinos = 0
        revisoes = 0
        origem = (components_info or {}).get("components") or []
        for index, comp in enumerate(origem, start=1):
            try:
                parcial = detectar_terminais_componentes(
                    {
                        "schema_version": (components_info or {}).get("schema_version"),
                        "components": [copy.deepcopy(comp)],
                    },
                    image_width=width,
                    image_height=height,
                    image=imagem,
                    confirmed_wire_bjt=confirmed_bjt,
                )
                item = parcial["components"][0]
                componentes_saida.append(item)
                total_pinos += int(parcial.get("terminal_count", len(item.get("terminals") or [])))
                revisoes += int(bool((item.get("terminal_detection") or {}).get("review_required")))
            except Exception as erro_comp:
                item = copy.deepcopy(comp)
                item["terminals"] = []
                item["terminal_detection"] = {
                    "status": "geometry_error",
                    "pin_count": 0,
                    "review_required": True,
                    "review_reasons": ["erro_geometria_componente"],
                    "error": str(erro_comp),
                }
                componentes_saida.append(item)
                revisoes += 1
                component_errors.append({
                    "component_id": item.get("component_id") or item.get("detection_id") or f"COMP_{index:03d}",
                    "type": item.get("type"),
                    "error": str(erro_comp),
                })

        resultado = {
            "schema_version": "stage7-resilient-1.0",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source_schema_version": (components_info or {}).get("schema_version"),
            "count": len(componentes_saida),
            "terminal_count": total_pinos,
            "review_count": revisoes,
            "nearby_terminal_snap_count": 0,
            "method": "per_component_resilient_fallback",
            "components": componentes_saida,
            "degraded_mode": True,
            "global_error": str(erro_global),
            "component_errors": component_errors,
        }

    from backend.terminal_wire_contacts import constrain_terminal_positions, refine_all_terminal_contacts
    if wires_info is not None and skeleton is not None:
        from backend.transistor_wire_refinement import refine_transistor_terminals_from_wires
        resultado,_ = refine_transistor_terminals_from_wires(imagem,resultado,wires_info,skeleton,observed_bjt=observed_bjt)
        resultado=refine_all_terminal_contacts(imagem,resultado,wires_info,skeleton)
    else:
        resultado=constrain_terminal_positions(resultado,imagem)
    resultado["source_image"] = image_path.name
    resultado["image_width"] = width
    resultado["image_height"] = height
    resultado["degraded_mode"] = bool(resultado.get("degraded_mode", degraded_mode))
    resultado.setdefault("component_errors", component_errors)
    output_json_path.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if save_preview:salvar_imagem_terminais(image_path, resultado, output_image_path)
    return resultado
