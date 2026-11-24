from __future__ import annotations

from copy import deepcopy
from itertools import combinations, permutations
from math import atan2, degrees, hypot, isfinite
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.terminal_learning import (
    diagnose_transistor_keypoints,
    lookup_known_training_ground_truth,
)
from backend.terminal_detection import _detect_transistor_fork
from backend.ink_color import estimate_component_ink, image_with_component_ink
from backend.npn_wire_contacts import detect_npn_wire_contacts, detect_bjt_wire_contacts

TRANSISTOR_TYPES = {"transistor_npn", "transistor_pnp"}


def _bbox_dims(bbox: dict) -> tuple[float, float]:
    return max(1.0, float(bbox["x2"]) - float(bbox["x1"])), max(1.0, float(bbox["y2"]) - float(bbox["y1"]))


def _center(bbox: dict) -> tuple[float, float]:
    return (float(bbox["x1"]) + float(bbox["x2"])) / 2.0, (float(bbox["y1"]) + float(bbox["y2"])) / 2.0


def _point_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _angle_diff_180(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 90.0
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def _bbox_boundary_distance(x: float, y: float, bbox: dict) -> float:
    x1, y1, x2, y2 = map(float, (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
    if x1 <= x <= x2 and y1 <= y <= y2:
        return min(abs(x - x1), abs(x - x2), abs(y - y1), abs(y - y2))
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return hypot(dx, dy)


def _radial_angle(point: tuple[float, float], center: tuple[float, float]) -> float:
    return (degrees(atan2(point[1] - center[1], point[0] - center[0])) + 360.0) % 360.0


def _unit_from_center(point: tuple[float, float], center: tuple[float, float]) -> np.ndarray:
    v = np.array([point[0] - center[0], point[1] - center[1]], dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros(2, dtype=np.float32)


def _terminal_map(component: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in component.get("terminals") or []:
        pin = str(t.get("pin") or "")
        pos = t.get("position") or {}
        if pin in {"B", "C", "E"} and isinstance(pos, dict):
            try:
                x, y = float(pos["x"]), float(pos["y"])
            except Exception:
                continue
            if isfinite(x) and isfinite(y):
                out[pin] = {**t, "position": {"x": x, "y": y}}
    return out


def _is_manual(component: dict) -> bool:
    td = component.get("terminal_detection") or {}
    if td.get("method") == "user_keypoint_correction" or td.get("status") == "manual_ground_truth":
        return True
    terms = component.get("terminals") or []
    return bool(terms) and all(str(t.get("source") or "") == "manual_ground_truth" for t in terms)


def _symbol_image_variants(image: np.ndarray, bbox: dict) -> list[tuple[str, np.ndarray]]:
    """Cria tentativas locais sem alterar coordenadas globais.

    A seta do BJT costuma ocupar poucos pixels. Em vez de confiar em uma unica
    binarizacao, repetimos a mesma geometria sobre contraste local, nitidez e
    fechamento leve. Todas as imagens preservam tamanho/coordenadas originais,
    portanto os pontos retornados continuam diretamente comparaveis.
    """
    if image is None or image.size == 0:
        return []
    h, w = image.shape[:2]
    try:
        x1, y1, x2, y2 = map(float, (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
    except Exception:
        return [("original", image)]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad = int(max(5, min(28, round(max(bw, bh) * 0.16))))
    xa, ya = max(0, int(x1) - pad), max(0, int(y1) - pad)
    xb, yb = min(w, int(np.ceil(x2)) + pad + 1), min(h, int(np.ceil(y2)) + pad + 1)
    if xb - xa < 8 or yb - ya < 8:
        return [("original", image)]

    roi = image[ya:yb, xa:xb]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    variants: list[tuple[str, np.ndarray]] = [("original", image)]

    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(4, 4)).apply(gray)
    sharp = cv2.addWeighted(clahe, 1.75, cv2.GaussianBlur(clahe, (0, 0), 1.0), -0.75, 0)
    _thr, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)

    for name, processed in (("clahe", clahe), ("clahe_sharp", sharp), ("otsu_closed", closed)):
        canvas = image.copy()
        patch = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR) if canvas.ndim == 3 else processed
        canvas[ya:yb, xa:xb] = patch
        variants.append((name, canvas))
    return variants


def _fork_attempt_score(meta: dict) -> float:
    if not meta.get("success"):
        return -10.0
    upper = float(meta.get("upper_arrow_score") or 0.0)
    lower = float(meta.get("lower_arrow_score") or 0.0)
    dominance = abs(upper - lower) / max(1.0, max(upper, lower))
    return (
        4.0
        + (3.0 if meta.get("arrow_branch") is not None else 0.0)
        + (1.5 if meta.get("branch_distinct") else 0.0)
        + (1.0 if meta.get("class_arrow_consistent") is not False else 0.0)
        + min(0.8, float(meta.get("base_candidates") or 0.0) * 0.08)
        + min(1.2, dominance * 1.2)
    )


def _detect_symbol_role_evidence(image: np.ndarray, component: dict) -> tuple[dict, dict | None]:
    """Lê a seta e a barra no símbolo original, antes de usar o modelo.

    A classe NPN/PNP já vem do detector principal. A seta coerente com essa
    classe fornece exclusivamente o emissor; a barra reta fornece a base.
    """
    bbox = component.get("bbox") or {}
    ctype = str(component.get("type") or "")
    orientation = str((component.get("orientation") or {}).get("axis") or "")
    if orientation not in {"horizontal", "vertical"}:
        bw, bh = _bbox_dims(bbox)
        orientation = "horizontal" if bw >= bh else "vertical"
    h, w = image.shape[:2]
    radius = int(max(6, min(18, round(max(_bbox_dims(bbox)) * 0.10))))
    attempts = []
    for variant_name, variant in _symbol_image_variants(image, bbox):
        try:
            points, meta = _detect_transistor_fork(variant, bbox, ctype, orientation, w, h, radius)
        except Exception as exc:
            attempts.append({"variant": variant_name, "accepted": False, "score": -10.0, "error": str(exc)})
            continue
        accepted_attempt = bool(
            meta.get("success")
            and meta.get("branch_distinct", False)
            and meta.get("arrow_branch") is not None
            and "B" in points
            and "E" in points
        )
        attempts.append({
            "variant": variant_name,
            "accepted": accepted_attempt,
            "score": _fork_attempt_score(meta),
            "points": points,
            "meta": meta,
        })

    accepted_attempts = [a for a in attempts if a.get("accepted")]
    if not accepted_attempts:
        compact = [{k: a.get(k) for k in ("variant", "accepted", "score", "error")} for a in attempts]
        return {"accepted": False, "multi_pass": True, "attempts": compact}, None

    diag = max(1.0, hypot(*_bbox_dims(bbox)))
    # Escolhe o resultado que mais concorda espacialmente com as demais
    # tentativas. O score visual desempata, em vez de uma tentativa isolada
    # substituir silenciosamente as outras.
    for attempt in accepted_attempts:
        ep = attempt["points"]["E"]
        bp = attempt["points"]["B"]
        agreement = 0
        for other in accepted_attempts:
            oe, ob = other["points"]["E"], other["points"]["B"]
            if _point_dist(ep, oe) <= diag * 0.24 and _point_dist(bp, ob) <= diag * 0.24:
                agreement += 1
        attempt["agreement_count"] = agreement
    chosen = max(accepted_attempts, key=lambda a: (a.get("agreement_count", 0), float(a.get("score") or 0.0)))
    points, meta = chosen["points"], chosen["meta"]
    consensus_count = int(chosen.get("agreement_count") or 0)
    consensus_ratio = consensus_count / max(1, len(accepted_attempts))
    multi_pass_uncertain = (
        (len(accepted_attempts) == 1 and float(chosen.get("score") or 0.0) < 9.0)
        or (len(accepted_attempts) > 1 and consensus_ratio < 0.60)
    )

    accepted = bool(
        meta.get("success")
        and meta.get("branch_distinct", False)
        and meta.get("arrow_branch") is not None
        and "B" in points
        and "E" in points
    )
    direction_conflict = meta.get("class_arrow_consistent") is False
    summary = {
        "accepted": accepted,
        "source": meta.get("source"),
        "selected_variant": chosen.get("variant"),
        "multi_pass": True,
        "attempt_count": len(attempts),
        "accepted_attempt_count": len(accepted_attempts),
        "consensus_count": consensus_count,
        "consensus_ratio": round(consensus_ratio, 4),
        "multi_pass_uncertain": multi_pass_uncertain,
        "arrow_branch": meta.get("arrow_branch"),
        "expected_arrow_direction": meta.get("expected_arrow_direction"),
        "detected_arrow_direction": meta.get("detected_arrow_direction"),
        "class_arrow_consistent": meta.get("class_arrow_consistent"),
        "direction_conflict_review": direction_conflict,
        "base_axis": meta.get("base_axis"),
        "fork_side": meta.get("fork_side"),
    }
    if not accepted:
        return summary, None
    hypothesis = {
        "source": "physical_arrow_then_base_bar",
        "points": {
            "B": {"x": float(points["B"][0]), "y": float(points["B"][1])},
            "E": {"x": float(points["E"][0]), "y": float(points["E"][1])},
        },
        # A presença da seta define o ramo E mesmo se a estimativa do SENTIDO
        # entrar em conflito com a classe. Nesse caso a identidade espacial é
        # preservada e o componente fica marcado para revisão da classe/sentido.
        "keypoint_confidences": {"B": 0.86 if direction_conflict else 0.91, "E": 0.90 if direction_conflict else 0.96},
        "weight": 10.0,
        "physical": True,
        "direction_conflict": direction_conflict,
        "multi_pass_uncertain": multi_pass_uncertain,
    }
    return summary, hypothesis


def _ring_candidates(skeleton: np.ndarray, bbox: dict) -> list[dict[str, Any]]:
    """Find distinct wire crossings around a transistor bbox.

    Stage 8 masks the symbol body, so real external leads terminate/cross a thin
    ring around the detector bbox. We cluster skeleton pixels in that ring and
    estimate the external tangent with a small local PCA.
    """
    h, w = skeleton.shape[:2]
    bw, bh = _bbox_dims(bbox)
    scale = max(bw, bh)
    band = int(max(6, min(22, round(scale * 0.085))))
    x1 = max(0, int(round(float(bbox["x1"]) - band)))
    y1 = max(0, int(round(float(bbox["y1"]) - band)))
    x2 = min(w - 1, int(round(float(bbox["x2"]) + band)))
    y2 = min(h - 1, int(round(float(bbox["y2"]) + band)))
    if x2 <= x1 or y2 <= y1:
        return []

    ring = np.zeros_like(skeleton, dtype=np.uint8)
    cv2.rectangle(ring, (x1, y1), (x2, y2), 255, -1)
    ix1 = max(0, int(round(float(bbox["x1"]) + band)))
    iy1 = max(0, int(round(float(bbox["y1"]) + band)))
    ix2 = min(w - 1, int(round(float(bbox["x2"]) - band)))
    iy2 = min(h - 1, int(round(float(bbox["y2"]) - band)))
    if ix2 > ix1 and iy2 > iy1:
        cv2.rectangle(ring, (ix1, iy1), (ix2, iy2), 0, -1)

    ring_skel = ((skeleton > 0) & (ring > 0)).astype(np.uint8)
    # Merge tiny 1px gaps caused by the component mask.
    ring_skel = cv2.dilate(ring_skel, np.ones((3, 3), np.uint8), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ring_skel, 8)
    center = _center(bbox)
    candidates: list[dict[str, Any]] = []

    for idx in range(1, n):
        ys, xs = np.nonzero(labels == idx)
        if len(xs) == 0:
            continue
        # Snap representative back to an actual skeleton pixel in the cluster neighborhood.
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        raw_ys, raw_xs = np.nonzero((skeleton > 0) & (labels == idx))
        if len(raw_xs):
            distances = (raw_xs.astype(float) - mx) ** 2 + (raw_ys.astype(float) - my) ** 2
            j = int(np.argmin(distances))
            px, py = float(raw_xs[j]), float(raw_ys[j])
        else:
            px, py = mx, my

        # PCA on nearby skeleton pixels estimates the real conductor direction.
        radius = int(max(14, min(42, round(scale * 0.22))))
        xa, xb = max(0, int(px) - radius), min(w, int(px) + radius + 1)
        ya, yb = max(0, int(py) - radius), min(h, int(py) + radius + 1)
        ly, lx = np.nonzero(skeleton[ya:yb, xa:xb] > 0)
        angle = None
        linearity = 0.0
        if len(lx) >= 4:
            arr = np.column_stack([lx + xa, ly + ya]).astype(np.float32)
            # Keep points reasonably close to the contact so nearby circuit branches
            # do not dominate the tangent estimate.
            d = np.sqrt((arr[:, 0] - px) ** 2 + (arr[:, 1] - py) ** 2)
            arr = arr[d <= radius]
            if len(arr) >= 4:
                z = arr - arr.mean(axis=0)
                cov = np.cov(z.T)
                vals, vecs = np.linalg.eigh(cov)
                order = np.argsort(vals)[::-1]
                vals = vals[order]
                v = vecs[:, order[0]]
                angle = (degrees(atan2(float(v[1]), float(v[0]))) + 180.0) % 180.0
                linearity = float(vals[0] / max(1e-6, vals.sum()))

        boundary_distance = _bbox_boundary_distance(px, py, bbox)
        radial_distance = _point_dist((px, py), center)
        # Small clusters are legitimate for thin wires, but a long extension is
        # preferred. boundary_distance must remain local to the component.
        if boundary_distance > max(18.0, scale * 0.18):
            continue
        candidates.append({
            "x": px,
            "y": py,
            "point": (px, py),
            "pixels": int(stats[idx, cv2.CC_STAT_AREA]),
            "angle_degrees": angle,
            "linearity": round(linearity, 4),
            "boundary_distance": round(boundary_distance, 3),
            "radial_distance": round(radial_distance, 3),
            "radial_angle": round(_radial_angle((px, py), center), 3),
        })

    # Deduplicate multiple ring intersections produced by the SAME physical
    # conductor. Thick/diagonal hand-drawn wires may touch the bbox ring in two
    # nearby clusters. They are merged only when position, tangent and radial
    # sector all agree; distinct C/E branches therefore remain separate.
    candidates.sort(key=lambda c: (c["boundary_distance"], -c["pixels"]))
    merged: list[dict[str, Any]] = []
    merge_dist = max(10.0, scale * 0.18)
    for cand in candidates:
        duplicate = False
        for old in merged:
            d = _point_dist(cand["point"], old["point"])
            ad = _angle_diff_180(cand.get("angle_degrees"), old.get("angle_degrees"))
            rd = abs(float(cand.get("radial_angle", 0.0)) - float(old.get("radial_angle", 0.0))) % 360.0
            rd = min(rd, 360.0 - rd)
            if d < merge_dist and ad <= 18.0 and rd <= 32.0:
                duplicate = True
                # Keep the contact closest to the actual bbox perimeter.
                if cand["boundary_distance"] < old["boundary_distance"]:
                    old.update(cand)
                break
        if not duplicate:
            merged.append(cand)
    return merged[:10]


def _model_evidence(image: np.ndarray, component: dict) -> dict[str, Any]:
    bbox = component.get("bbox") or {}
    ctype = str(component.get("type") or "")
    evidence: dict[str, Any] = {
        "detector_class": ctype,
        "source": None,
        "points": {},
        "keypoint_confidences": {},
        "accepted": False,
        "hypotheses": [],
    }
    symbol_geometry, physical_hypothesis = _detect_symbol_role_evidence(image, component)
    evidence["symbol_geometry"] = symbol_geometry
    if physical_hypothesis is not None:
        evidence["hypotheses"].append(physical_hypothesis)
    try:
        known = lookup_known_training_ground_truth(image=image, bbox=bbox, component_type=ctype)
    except Exception as exc:
        known = None
        evidence["known_lookup_error"] = str(exc)
    if known and known.get("points"):
        confidences = known.get("keypoint_confidences") or {"B": 1.0, "C": 1.0, "E": 1.0}
        evidence.update({
            "source": "known_training_ground_truth",
            "points": known.get("points") or {},
            "keypoint_confidences": confidences,
            "accepted": bool(known.get("accepted", True)),
            "known_match": {k: known.get(k) for k in ("sample_id", "match_method", "score", "local_similarity", "feature_similarity", "template_similarity") if k in known},
            "hypotheses": ([dict(physical_hypothesis)] if physical_hypothesis is not None else []),
        })
        # A memória ensinada é fallback/validação. A evidência física, quando
        # presente, continua na primeira posição e terá prioridade rígida.
        evidence["hypotheses"].append({
            "source": "known_training_ground_truth",
            "points": known.get("points") or {},
            "keypoint_confidences": confidences,
            "weight": 5.0,
        })
        return evidence

    try:
        diag = diagnose_transistor_keypoints(image=image, bbox=bbox, component_type=ctype)
    except Exception as exc:
        diag = {"accepted": False, "points": {}, "keypoint_confidences": {}, "rejection_reasons": [str(exc)]}
    evidence.update({
        "source": "trained_terminal_keypoint_model",
        "points": diag.get("points") or {},
        "keypoint_confidences": diag.get("keypoint_confidences") or {},
        "accepted": bool(diag.get("accepted")),
        "review_required": bool(diag.get("review_required", True)),
        "box_confidence": float(diag.get("box_confidence") or 0.0),
        "keypoint_confidence": float(diag.get("keypoint_confidence") or 0.0),
        "geometry_sane": bool(diag.get("geometry_sane", False)),
        "rejection_reasons": diag.get("rejection_reasons") or [],
    })
    if evidence["accepted"] and all(pin in evidence["points"] for pin in ("B", "C", "E")):
        evidence["hypotheses"].append({
            "source": "trained_terminal_keypoint_model",
            "points": evidence["points"],
            "keypoint_confidences": evidence["keypoint_confidences"],
            "weight": 2.35,
        })

    return evidence


def _anchor_point(evidence: dict, pin: str) -> tuple[float, float] | None:
    p = (evidence.get("points") or {}).get(pin)
    if not isinstance(p, dict):
        return None
    try:
        return float(p["x"]), float(p["y"])
    except Exception:
        return None


def _choose_three_and_base(candidates: list[dict], bbox: dict, evidence: dict, old_terms: dict[str, dict]) -> tuple[list[dict], int, dict] | None:
    if len(candidates) < 3:
        return None
    center = _center(bbox)
    bw, bh = _bbox_dims(bbox)
    diag = hypot(bw, bh)
    b_anchor = _anchor_point(evidence, "B")

    best = None
    # Consider at most 8 strongest/local candidates.
    pool = candidates[:8]
    for combo in combinations(range(len(pool)), 3):
        tri = [pool[i] for i in combo]
        # Reject contacts that are too close to each other.
        min_sep = min(_point_dist(tri[i]["point"], tri[j]["point"]) for i, j in ((0, 1), (0, 2), (1, 2)))
        if min_sep < max(8.0, min(bw, bh) * 0.10):
            continue
        for base_idx in range(3):
            pair = [i for i in range(3) if i != base_idx]
            c1, c2 = tri[pair[0]], tri[pair[1]]
            base = tri[base_idx]
            # Collector/emitter external wires usually have nearly the same
            # undirected tangent orientation, while the base lead is transverse.
            ce_parallel = _angle_diff_180(c1.get("angle_degrees"), c2.get("angle_degrees"))
            b_to_ce1 = _angle_diff_180(base.get("angle_degrees"), c1.get("angle_degrees"))
            b_to_ce2 = _angle_diff_180(base.get("angle_degrees"), c2.get("angle_degrees"))
            parallel_score = max(0.0, 1.0 - ce_parallel / 65.0)
            transverse_score = min(1.0, (b_to_ce1 + b_to_ce2) / 100.0)

            # Geometric same-side prior: C/E are normally on one side of the
            # symbol and B is on the opposite side. Rotation does not matter.
            u1 = _unit_from_center(c1["point"], center)
            u2 = _unit_from_center(c2["point"], center)
            ub = _unit_from_center(base["point"], center)
            avg = u1 + u2
            avg_n = float(np.linalg.norm(avg))
            same_side_score = 0.0
            opposite_score = 0.0
            if avg_n > 1e-6:
                avg /= avg_n
                same_side_score = max(0.0, float(np.dot(u1, u2)))
                opposite_score = max(0.0, float(np.dot(-avg, ub)))

            b_model_score = 0.0
            if b_anchor is not None:
                b_model_score = max(0.0, 1.0 - _point_dist(base["point"], b_anchor) / max(20.0, diag * 0.65))

            locality = sum(max(0.0, 1.0 - float(c["boundary_distance"]) / max(12.0, diag * 0.18)) for c in tri) / 3.0
            linearity = sum(float(c.get("linearity") or 0.0) for c in tri) / 3.0
            score = (
                parallel_score * 2.6
                + transverse_score * 0.9
                + same_side_score * 0.6
                + opposite_score * 1.1
                + b_model_score * 2.15
                + locality * 0.75
                + linearity * 0.45
            )
            meta = {
                "ce_parallel_angle_diff": round(ce_parallel, 3),
                "base_to_ce_angle_diffs": [round(b_to_ce1, 3), round(b_to_ce2, 3)],
                "parallel_score": round(parallel_score, 4),
                "transverse_score": round(transverse_score, 4),
                "same_side_score": round(same_side_score, 4),
                "opposite_score": round(opposite_score, 4),
                "base_model_score": round(b_model_score, 4),
                "selection_score": round(score, 4),
            }
            if best is None or score > best[0]:
                best = (score, tri, base_idx, meta)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _choose_three_wire_paths(candidates: list[dict], bbox: dict) -> tuple[list[dict], dict] | None:
    """Seleciona três contatos físicos sem atribuir B/C/E nesta etapa."""
    if len(candidates) < 3:
        return None
    bw, bh = _bbox_dims(bbox)
    diag = max(1.0, hypot(bw, bh))
    pool = candidates[:8]
    best = None
    for combo in combinations(range(len(pool)), 3):
        tri = [pool[i] for i in combo]
        distances = [
            _point_dist(tri[i]["point"], tri[j]["point"])
            for i, j in ((0, 1), (0, 2), (1, 2))
        ]
        min_sep = min(distances)
        if min_sep < max(8.0, min(bw, bh) * 0.10):
            continue
        locality = sum(
            max(0.0, 1.0 - float(c.get("boundary_distance") or 0.0) / max(12.0, diag * 0.18))
            for c in tri
        ) / 3.0
        linearity = sum(float(c.get("linearity") or 0.0) for c in tri) / 3.0
        separation = min(1.0, min_sep / max(10.0, min(bw, bh) * 0.45))
        support = sum(min(1.0, float(c.get("pixels") or 0.0) / 20.0) for c in tri) / 3.0
        score = locality * 1.6 + linearity * 1.0 + separation * 1.15 + support * 0.35
        if best is None or score > best[0]:
            best = (score, tri, {
                "method": "three_stage8_wire_paths_before_roles",
                "selection_score": round(score, 4),
                "minimum_path_separation_px": round(min_sep, 3),
                "role_neutral": True,
            })
    if best is None:
        return None
    return best[1], best[2]


def _hypothesis_anchor(evidence: dict, pin: str, *, physical_only: bool = False) -> tuple[tuple[float, float] | None, str | None, float]:
    """Retorna a melhor âncora respeitando física -> memória -> pose."""
    priority = {
        "physical_arrow_then_base_bar": 300,
        "known_training_ground_truth": 200,
        "trained_terminal_keypoint_model": 100,
    }
    best = None
    hypotheses = list(evidence.get("hypotheses") or [])
    if not hypotheses and evidence.get("points"):
        hypotheses.append({
            "source": str(evidence.get("source") or "legacy_model_evidence"),
            "points": evidence.get("points") or {},
            "keypoint_confidences": evidence.get("keypoint_confidences") or {},
            "physical": False,
        })
    for hyp in hypotheses:
        source = str(hyp.get("source") or "")
        if physical_only and not bool(hyp.get("physical")):
            continue
        point = (hyp.get("points") or {}).get(pin)
        if not isinstance(point, dict):
            continue
        try:
            anchor = (float(point["x"]), float(point["y"]))
        except Exception:
            continue
        confidence = float((hyp.get("keypoint_confidences") or {}).get(pin, 0.5))
        rank = priority.get(source, 10) + confidence
        if best is None or rank > best[0]:
            best = (rank, anchor, source, confidence)
    if best is None:
        return None, None, 0.0
    return best[1], best[2], max(0.0, min(1.0, best[3]))


def _nearest_available(
    tri: list[dict],
    anchor: tuple[float, float],
    available: list[int],
    diag: float,
) -> tuple[int, float, float]:
    ranked = sorted((_point_dist(tri[i]["point"], anchor), i) for i in available)
    best_distance, best_idx = ranked[0]
    second_distance = ranked[1][0] if len(ranked) > 1 else best_distance + diag
    margin = max(0.0, (second_distance - best_distance) / max(1.0, diag))
    return best_idx, best_distance, margin


def _assign_roles_arrow_base_leftover(
    tri: list[dict],
    evidence: dict,
    bbox: dict,
) -> tuple[dict[str, int], dict]:
    """Ordem rígida: seta->E, barra->B, restante->C.

    O treinamento entra somente quando a leitura física correspondente não
    existe. Nenhuma hipótese pode escolher C diretamente.
    """
    diag = max(1.0, hypot(*_bbox_dims(bbox)))
    all_indexes = [0, 1, 2]

    e_anchor, e_source, e_anchor_conf = _hypothesis_anchor(evidence, "E", physical_only=True)
    physical_arrow = e_anchor is not None
    if e_anchor is None:
        e_anchor, e_source, e_anchor_conf = _hypothesis_anchor(evidence, "E")

    b_anchor, b_source, b_anchor_conf = _hypothesis_anchor(evidence, "B", physical_only=True)
    physical_base = b_anchor is not None
    if b_anchor is None:
        b_anchor, b_source, b_anchor_conf = _hypothesis_anchor(evidence, "B")

    # 1) A seta escolhe e bloqueia o emissor.
    if e_anchor is not None:
        e_idx, e_distance, e_margin = _nearest_available(tri, e_anchor, all_indexes, diag)
    else:
        # Sem seta nem treino utilizável: usa provisoriamente o contato mais
        # distante do centro e exige revisão. Não promove C a partir de voto.
        center = _center(bbox)
        available = all_indexes
        if b_anchor is not None and evidence.get("source") in {"npn_original_ink_bar_branches_and_external_leads", "bjt_original_ink_bar_branches_and_external_leads"}:
            known_base, _, _ = _nearest_available(tri, b_anchor, all_indexes, diag)
            available = [i for i in all_indexes if i != known_base]
        e_idx = max(available, key=lambda i: _point_dist(tri[i]["point"], center))
        e_distance, e_margin = diag, 0.0

    # 2) A barra escolhe a base apenas entre os dois contatos restantes.
    remaining_after_e = [i for i in all_indexes if i != e_idx]
    if b_anchor is not None:
        b_idx, b_distance, b_margin = _nearest_available(tri, b_anchor, remaining_after_e, diag)
    else:
        # Fallback geométrico apenas para localizar B; permanece revisável.
        b_idx = max(remaining_after_e, key=lambda i: _topology_score_for_base(tri, i, bbox))
        b_distance, b_margin = diag, 0.0

    # 3) Coletor é sempre o único ponto que sobrou.
    c_idx = next(i for i in all_indexes if i not in {e_idx, b_idx})

    physical_complete = physical_arrow and physical_base
    training_complete = e_anchor is not None and b_anchor is not None
    physical_direction_conflict = any(
        bool(hyp.get("physical")) and bool(hyp.get("direction_conflict"))
        for hyp in (evidence.get("hypotheses") or [])
    )
    physical_multi_pass_uncertain = any(
        bool(hyp.get("physical")) and bool(hyp.get("multi_pass_uncertain"))
        for hyp in (evidence.get("hypotheses") or [])
    )
    uncertain = bool(
        not training_complete
        or physical_direction_conflict
        or physical_multi_pass_uncertain
        or (physical_arrow and e_margin < 0.045)
        or (physical_base and b_margin < 0.045)
        or (not physical_complete and (e_margin < 0.08 or b_margin < 0.08))
    )
    confidence = min(
        0.99,
        max(0.20, 0.38 + 0.22 * e_anchor_conf + 0.18 * b_anchor_conf + 0.75 * min(e_margin, 0.20) + 0.65 * min(b_margin, 0.20)),
    )
    return {"B": b_idx, "C": c_idx, "E": e_idx}, {
        "source": "arrow_then_base_bar_then_collector_leftover",
        "sequence": ["stage8_wires", "detector_NPN_PNP", "arrow_emitter", "base_bar", "collector_leftover"],
        "detector_class": evidence.get("detector_class"),
        "emitter": {
            "source": e_source,
            "physical_arrow": physical_arrow,
            "distance_px": round(float(e_distance), 3),
            "margin_normalized": round(float(e_margin), 4),
        },
        "base": {
            "source": b_source,
            "physical_base_bar": physical_base,
            "distance_px": round(float(b_distance), 3),
            "margin_normalized": round(float(b_margin), 4),
        },
        "collector": {"source": "remaining_wire_after_E_and_B"},
        "confidence": round(float(confidence), 4),
        "identity_uncertain": uncertain,
        "class_arrow_direction_conflict": physical_direction_conflict,
        "multi_pass_geometry_uncertain": physical_multi_pass_uncertain,
        "mapping": {"B": b_idx, "C": c_idx, "E": e_idx},
    }


def _topology_score_for_base(tri: list[dict], base_idx: int, bbox: dict) -> float:
    """Pontua a identidade de B sem assumir orientação horizontal/vertical."""
    center = _center(bbox)
    pair = [i for i in range(3) if i != base_idx]
    base, p0, p1 = tri[base_idx], tri[pair[0]], tri[pair[1]]
    u0 = _unit_from_center(p0["point"], center)
    u1 = _unit_from_center(p1["point"], center)
    ub = _unit_from_center(base["point"], center)
    avg = u0 + u1
    avg_n = float(np.linalg.norm(avg))
    same_side = max(0.0, float(np.dot(u0, u1)))
    opposite = 0.0
    if avg_n > 1e-6:
        opposite = max(0.0, float(np.dot(-(avg / avg_n), ub)))
    ce_parallel = _angle_diff_180(p0.get("angle_degrees"), p1.get("angle_degrees"))
    parallel = max(0.0, 1.0 - ce_parallel / 70.0)
    return same_side * 0.85 + opposite * 1.35 + parallel * 0.45


def _assign_all_roles(
    tri: list[dict],
    topology_base_idx: int,
    evidence: dict,
    bbox: dict,
) -> tuple[dict[str, int], dict]:
    """Faz uma associação bijetiva B/C/E -> três fios físicos.

    A versão anterior fixava B por heurística e só depois separava C/E. Se a
    heurística escolhesse o ramo errado, todos os nomes podiam girar. Aqui as
    seis permutações são avaliadas de uma vez com: topologia, ground truth,
    pose treinada e barra/seta do símbolo.
    """
    diag = max(1.0, hypot(*_bbox_dims(bbox)))
    hypotheses = list(evidence.get("hypotheses") or [])
    if not hypotheses and all(pin in (evidence.get("points") or {}) for pin in ("B", "C", "E")):
        hypotheses = [{
            "source": str(evidence.get("source") or "legacy_model_evidence"),
            "points": evidence.get("points") or {},
            "keypoint_confidences": evidence.get("keypoint_confidences") or {},
            "weight": 5.0 if evidence.get("source") == "known_training_ground_truth" else 2.35,
        }]
    ranked: list[tuple[float, dict[str, int], dict]] = []
    for order in permutations(range(3), 3):
        mapping = {"B": order[0], "C": order[1], "E": order[2]}
        topology = _topology_score_for_base(tri, mapping["B"], bbox)
        score = topology * 1.35
        source_scores: dict[str, float] = {}
        evidence_weight = 0.0
        for hyp in hypotheses:
            points = hyp.get("points") or {}
            confs = hyp.get("keypoint_confidences") or {}
            weight = float(hyp.get("weight") or 0.0)
            if weight <= 0 or not all(pin in points for pin in ("B", "C", "E")):
                continue
            weighted_closeness = 0.0
            conf_total = 0.0
            for pin in ("B", "C", "E"):
                try:
                    anchor = (float(points[pin]["x"]), float(points[pin]["y"]))
                except Exception:
                    continue
                conf = max(0.05, min(1.0, float(confs.get(pin, 0.5))))
                distance = _point_dist(tri[mapping[pin]]["point"], anchor)
                closeness = max(-0.45, 1.0 - distance / (diag * 0.72))
                weighted_closeness += conf * closeness
                conf_total += conf
            if conf_total <= 0:
                continue
            agreement = weighted_closeness / conf_total
            source_name = str(hyp.get("source") or "evidence")
            source_scores[source_name] = round(agreement, 4)
            score += weight * agreement
            evidence_weight += weight
        # A topologia anterior continua como desempate leve, nunca como dona
        # absoluta dos nomes.
        if mapping["B"] == topology_base_idx:
            score += 0.18
        ranked.append((score, mapping, {
            "topology_score": round(topology, 4),
            "source_scores": source_scores,
            "evidence_weight": round(evidence_weight, 3),
        }))

    ranked.sort(key=lambda item: item[0], reverse=True)
    best_score, best_mapping, details = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else best_score
    margin = float(best_score - second_score)
    has_emitter_evidence = any(
        "E" in (hyp.get("points") or {}) and float(hyp.get("weight") or 0.0) >= 2.0
        for hyp in hypotheses
    )
    identity_uncertain = bool(not has_emitter_evidence or margin < 0.22)
    confidence = max(0.0, min(0.99, 0.42 + margin * 0.22 + min(0.28, details["evidence_weight"] * 0.035)))
    return best_mapping, {
        "source": "joint_BCE_assignment",
        "confidence": round(confidence, 4),
        "identity_uncertain": identity_uncertain,
        "score": round(float(best_score), 4),
        "margin_to_second": round(margin, 4),
        "mapping": dict(best_mapping),
        **details,
    }


def _assign_ce(tri: list[dict], base_idx: int, evidence: dict, old_terms: dict[str, dict], bbox: dict) -> tuple[int, int, dict]:
    pair = [i for i in range(3) if i != base_idx]
    e_anchor = _anchor_point(evidence, "E")
    c_anchor = _anchor_point(evidence, "C")
    source = evidence.get("source")
    kpconf = evidence.get("keypoint_confidences") or {}

    # If trained/known evidence provides separated C/E, use both anchors.
    confidence = 0.0
    assignment_source = None
    if e_anchor is not None and c_anchor is not None:
        sep = _point_dist(e_anchor, c_anchor)
        bw, bh = _bbox_dims(bbox)
        if sep >= max(6.0, min(bw, bh) * 0.08):
            cost01 = _point_dist(tri[pair[0]]["point"], c_anchor) + _point_dist(tri[pair[1]]["point"], e_anchor)
            cost10 = _point_dist(tri[pair[1]]["point"], c_anchor) + _point_dist(tri[pair[0]]["point"], e_anchor)
            if cost01 <= cost10:
                c_idx, e_idx = pair[0], pair[1]
            else:
                c_idx, e_idx = pair[1], pair[0]
            confidence = max(0.55, min(1.0, float(kpconf.get("E", 0.5) + kpconf.get("C", 0.5)) / 2.0))
            assignment_source = f"{source}_C_and_E_anchors"
            return c_idx, e_idx, {"source": assignment_source, "confidence": round(confidence, 4), "anchor_separation_px": round(sep, 2)}

    # E alone is often the strongest learned feature because it contains the arrow.
    if e_anchor is None and "E" in old_terms:
        p = old_terms["E"]["position"]
        e_anchor = (float(p["x"]), float(p["y"]))
        source = "previous_stage7_emitter_anchor"
        kpconf = {"E": float(old_terms["E"].get("confidence") or 0.35)}
    if e_anchor is not None:
        d0 = _point_dist(tri[pair[0]]["point"], e_anchor)
        d1 = _point_dist(tri[pair[1]]["point"], e_anchor)
        e_idx = pair[0] if d0 <= d1 else pair[1]
        c_idx = pair[1] if e_idx == pair[0] else pair[0]
        bw, bh = _bbox_dims(bbox)
        near = min(d0, d1)
        geom_conf = max(0.0, 1.0 - near / max(20.0, hypot(bw, bh) * 0.55))
        confidence = max(0.25, min(0.95, 0.55 * float(kpconf.get("E", 0.35)) + 0.45 * geom_conf))
        assignment_source = f"{source or 'model'}_E_anchor"
        return c_idx, e_idx, {"source": assignment_source, "confidence": round(confidence, 4), "emitter_anchor_distance_px": round(near, 2)}

    # No arrow/model evidence: keep the line positions but explicitly mark C/E uncertain.
    return pair[0], pair[1], {"source": "wire_pair_without_emitter_evidence", "confidence": 0.0, "identity_uncertain": True}


def _terminal_record(component_id: str, pin: str, point: tuple[float, float], confidence: float, uncertain: bool = False) -> dict:
    roles = {"B": "base", "C": "collector", "E": "emitter"}
    return {
        "pin_id": f"{component_id}.{pin}",
        "pin": pin,
        "electrical_role": roles[pin],
        "terminal_side": "stage8_wire_centerline",
        "index": {"B": 1, "C": 2, "E": 3}[pin],
        "position": {"x": round(float(point[0]), 2), "y": round(float(point[1]), 2)},
        "search_radius_px": 8,
        "source": "stage8_wire_refinement",
        "identity_uncertain": bool(uncertain),
        "confidence": round(float(max(0.0, min(1.0, confidence))), 4),
    }


def refine_transistor_terminals_from_wires(
    image: np.ndarray,
    terminals_info: dict,
    wires_info: dict,
    skeleton: np.ndarray,
    *, observed_bjt: dict | None = None,
) -> tuple[dict, dict]:
    """Refine only transistor terminals after Stage 8 wires are detected.

    Non-transistor components are copied byte-for-byte. Manual transistor
    corrections are preserved. Automatic transistor points are replaced only
    by three distinct Stage-8 wire contacts; otherwise they are cleared and the
    component is explicitly marked for review.
    """
    result = deepcopy(terminals_info or {})
    components = result.get("components") or []
    summary = {
        "method": "stage8_wires_class_arrow_E_base_B_leftover_C",
        "transistor_count": 0,
        "refined_count": 0,
        "manual_preserved_count": 0,
        "review_count": 0,
        "components": [],
    }
    if image is None or skeleton is None:
        return result, summary
    npn_image = image
    if any(c.get("type") in TRANSISTOR_TYPES and not _is_manual(c) for c in components):
        npn_image = image_with_component_ink(image, estimate_component_ink(image, terminals_info))

    for component in components:
        ctype = str(component.get("type") or "")
        if ctype not in TRANSISTOR_TYPES:
            continue
        summary["transistor_count"] += 1
        cid = str(component.get("component_id") or component.get("detection_id") or "TRANSISTOR")
        bbox = component.get("bbox") or {}
        item = {"component_id": cid, "type": ctype, "detector_class": ctype}
        if _is_manual(component):
            item.update({"status": "manual_preserved", "refined": False})
            summary["manual_preserved_count"] += 1
            summary["components"].append(item)
            continue
        if not all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            component["terminals"] = []
            component["review_required"] = True
            component["review_reasons"] = list(dict.fromkeys((component.get("review_reasons") or []) + ["transistor_bbox_missing_for_wire_refinement"]))
            summary["review_count"] += 1
            item.update({"status": "review", "reason": "bbox_missing"})
            summary["components"].append(item)
            continue

        candidates = _ring_candidates(skeleton, bbox)
        old_terms = _terminal_map(component)
        visible_npn, visible_npn_meta = (None, None)
        prior=(observed_bjt or {}).get(component.get('detection_id'))
        if prior is not None:
            visible_npn,visible_npn_meta=deepcopy(prior)
            item['original_ink_geometry']=visible_npn_meta
        elif ctype == "transistor_npn":
            visible_npn, visible_npn_meta = detect_npn_wire_contacts(image, bbox, skeleton, color_reference=npn_image)
            item["original_ink_geometry"] = visible_npn_meta
        elif ctype == "transistor_pnp":
            visible_npn, visible_npn_meta = detect_bjt_wire_contacts(image,bbox,skeleton,ctype,color_reference=npn_image)
            item["original_ink_geometry"] = visible_npn_meta
        if visible_npn is not None:
            tri, evidence = visible_npn
            selected = (tri, {"method":"bjt_original_ink_leads_with_stage8_context", "selection_score":5.,
                "role_neutral":False, "original_ink_geometry":visible_npn_meta})
        else:
            evidence = _model_evidence(image, component)
            selected = _choose_three_wire_paths(candidates, bbox)
        item["wire_candidate_count"] = len(candidates)
        item["model_evidence_source"] = evidence.get("source")
        item["model_class"] = ctype

        if selected is None:
            component["terminals"] = []
            component["review_required"] = True
            component["review_reasons"] = list(dict.fromkeys((component.get("review_reasons") or []) + ["stage8_three_distinct_transistor_wires_not_found"]))
            component["terminal_detection"] = {
                "status": "review_required_after_wires",
                "pin_count": 0,
                "review_required": True,
                "review_reasons": ["stage8_three_distinct_transistor_wires_not_found"],
                "method": "stage8_wire_topology_refinement",
            }
            geom = dict(component.get("transistor_geometry") or {})
            geom.update({
                "source": "stage8_wire_topology_refinement",
                "training_required": False,
                "wire_refinement_failed": True,
                "wire_candidate_count": len(candidates),
                "wire_candidates": [{k: c.get(k) for k in ("x", "y", "angle_degrees", "linearity", "boundary_distance")} for c in candidates],
            })
            component["transistor_geometry"] = geom
            summary["review_count"] += 1
            item.update({"status": "review", "reason": "less_than_three_coherent_wire_paths"})
            summary["components"].append(item)
            continue

        tri, selection_meta = selected
        role_indices, assignment_meta = _assign_roles_arrow_base_leftover(tri, evidence, bbox)
        base_idx = role_indices["B"]
        c_idx = role_indices["C"]
        e_idx = role_indices["E"]
        identity_meta = {
            "source": assignment_meta.get("source"),
            "confidence": assignment_meta.get("confidence"),
            "identity_uncertain": assignment_meta.get("identity_uncertain"),
            "emitter": assignment_meta.get("emitter"),
            "base": assignment_meta.get("base"),
            "collector": assignment_meta.get("collector"),
        }
        selection_meta = {**selection_meta, "ordered_role_assignment": assignment_meta}
        b_point = tri[base_idx]["point"]
        c_point = tri[c_idx]["point"]
        e_point = tri[e_idx]["point"]
        # Hard invariant: all final terminals must be distinct Stage-8 paths.
        bw, bh = _bbox_dims(bbox)
        min_final_sep = max(7.0, min(bw, bh) * 0.09)
        sep_bc = _point_dist(b_point, c_point)
        sep_be = _point_dist(b_point, e_point)
        sep_ce = _point_dist(c_point, e_point)
        distinct = min(sep_bc, sep_be, sep_ce) >= min_final_sep
        identity_uncertain = bool(identity_meta.get("identity_uncertain")) or float(identity_meta.get("confidence") or 0.0) < 0.45

        if not distinct:
            component["terminals"] = []
            component["review_required"] = True
            component["review_reasons"] = list(dict.fromkeys((component.get("review_reasons") or []) + ["stage8_transistor_wire_paths_collapsed"]))
            summary["review_count"] += 1
            item.update({"status": "review", "reason": "wire_paths_collapsed"})
            summary["components"].append(item)
            continue

        base_conf = max(0.55, min(0.98, float(selection_meta.get("selection_score") or 0.0) / 6.0))
        role_conf = float(identity_meta.get("confidence") or 0.0)
        component["terminals"] = [
            _terminal_record(cid, "B", b_point, max(base_conf, role_conf), identity_uncertain),
            _terminal_record(cid, "C", c_point, max(0.35, role_conf), identity_uncertain),
            _terminal_record(cid, "E", e_point, max(0.35, role_conf), identity_uncertain),
        ]
        if visible_npn is not None:
            for terminal,index in zip(component["terminals"],(base_idx,c_idx,e_idx)):
                contact = tri[index]
                terminal["source"] = "stage8_original_ink_bjt_contact"
                terminal["original_ink_confirmed"] = True
                terminal["stage8_context_distance_px"] = contact["stage8_distance_px"]
                terminal["search_radius_px"] = max(8, int(np.ceil(contact["stage8_distance_px"]))+3)
        review_reasons = [r for r in (component.get("review_reasons") or []) if not str(r).startswith("stage8_") and "keypoint" not in str(r).lower() and "transistor" not in str(r).lower()]
        if visible_npn is not None:
            review_reasons = [r for r in review_reasons if not str(r).startswith("aguardando_etapa8_")]
        if identity_uncertain:
            review_reasons.append("confirmar_seta_emissor_e_barra_base")
        if assignment_meta.get("class_arrow_direction_conflict"):
            review_reasons.append("classe_bjt_conflita_com_sentido_estimado_da_seta")
        if assignment_meta.get("multi_pass_geometry_uncertain"):
            review_reasons.append("tentativas_de_imagem_nao_concordaram_totalmente")
        component["review_required"] = bool(review_reasons)
        component["review_reasons"] = list(dict.fromkeys(review_reasons))
        component["provisional_from_detection"] = False
        component["terminal_detection"] = {
            "status": "wire_refined" if not identity_uncertain else "wire_refined_identity_review",
            "pin_count": 3,
            "review_required": bool(identity_uncertain),
            "review_reasons": list(dict.fromkeys(
                (["confirmar_seta_emissor_e_barra_base"] if identity_uncertain else [])
                + (["classe_bjt_conflita_com_sentido_estimado_da_seta"] if assignment_meta.get("class_arrow_direction_conflict") else [])
                + (["tentativas_de_imagem_nao_concordaram_totalmente"] if assignment_meta.get("multi_pass_geometry_uncertain") else [])
            )),
            "method": "stage8_wires_then_arrow_E_then_base_B_then_leftover_C",
            "class_source": "component_detector",
            "class": ctype,
            "emitter_identity_source": (identity_meta.get("emitter") or {}).get("source"),
            "base_identity_source": (identity_meta.get("base") or {}).get("source"),
            "collector_identity_source": "remaining_wire_after_E_and_B",
        }
        geom = {} if visible_npn is not None else dict(component.get("transistor_geometry") or {})
        geom.update({
            "source": "stage8_wire_topology_refinement",
            "wire_refined": True,
            "detector_class": ctype,
            "wire_candidate_count": len(candidates),
            "selected_wire_points": {
                "B": {"x": round(b_point[0], 2), "y": round(b_point[1], 2)},
                "C": {"x": round(c_point[0], 2), "y": round(c_point[1], 2)},
                "E": {"x": round(e_point[0], 2), "y": round(e_point[1], 2)},
            },
            "wire_separations_px": {"B_C": round(sep_bc, 2), "B_E": round(sep_be, 2), "C_E": round(sep_ce, 2)},
            "wire_selection": selection_meta,
            "ordered_identity": identity_meta,
            "trained_evidence": {
                "source": evidence.get("source"),
                "accepted": evidence.get("accepted"),
                "keypoint_confidences": evidence.get("keypoint_confidences"),
                "geometry_sane": evidence.get("geometry_sane"),
                "hypothesis_sources": [str(h.get("source")) for h in (evidence.get("hypotheses") or [])],
                "symbol_geometry": evidence.get("symbol_geometry"),
            },
        })
        if visible_npn is not None:
            bar_angle = visible_npn_meta["base_bar_angle_degrees"] % 180
            geom.update({"source":"stage8_bjt_original_ink_contacts", "training_required":False,
                "npn_emitter_rule_applied":False, "base_axis":"horizontal" if bar_angle<45 or bar_angle>135 else "vertical",
                "branch_distinct":True, "reliable":not identity_uncertain,
                "original_ink_geometry":visible_npn_meta,
                "detected_arrow_direction":visible_npn_meta["detected_arrow_direction"],
                "class_arrow_consistent":not visible_npn_meta["direction_conflict"] if visible_npn_meta["arrow_confirmed"] else None})
        component["transistor_geometry"] = geom
        summary["refined_count"] += 1
        if identity_uncertain:
            summary["review_count"] += 1
        item.update({
            "status": "refined" if not identity_uncertain else "refined_review_BCE",
            "refined": True,
            "B": component["terminals"][0]["position"],
            "C": component["terminals"][1]["position"],
            "E": component["terminals"][2]["position"],
            "wire_selection": selection_meta,
            "ordered_identity": identity_meta,
        })
        summary["components"].append(item)

    # Recompute counts without touching non-transistor data.
    result["terminal_count"] = sum(len(c.get("terminals") or []) for c in components)
    result["review_count"] = sum(
        1 for c in components
        if c.get("review_required") or (c.get("terminal_detection") or {}).get("review_required")
    )
    result["stage8_transistor_refinement"] = summary
    return result, summary


def save_transistor_wire_refinement_preview(
    image: np.ndarray,
    terminals_info: dict,
    wires_info: dict,
    output_path: Path,
) -> None:
    canvas = image.copy()
    # Draw Stage-8 wires faintly first.
    for seg in wires_info.get("segments") or []:
        pts = seg.get("points") or []
        if len(pts) < 2:
            continue
        arr = np.array([[int(round(p["x"])), int(round(p["y"]))] for p in pts], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [arr], False, (60, 180, 60), 1, cv2.LINE_AA)

    for comp in terminals_info.get("components") or []:
        if str(comp.get("type") or "") not in TRANSISTOR_TYPES:
            continue
        bbox = comp.get("bbox") or {}
        if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            cv2.rectangle(canvas, (int(bbox["x1"]), int(bbox["y1"])), (int(bbox["x2"]), int(bbox["y2"])), (90, 90, 90), 1)
        for term in comp.get("terminals") or []:
            pos = term.get("position") or {}
            if "x" not in pos or "y" not in pos:
                continue
            p = (int(round(pos["x"])), int(round(pos["y"])))
            cv2.circle(canvas, p, 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, str(term.get("pin") or "?"), (p[0] + 8, max(14, p[1] - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 190), 2, cv2.LINE_AA)
        if not comp.get("terminals") and bbox:
            cv2.putText(canvas, "REVISAR TRANSISTOR", (int(bbox["x1"]), max(18, int(bbox["y1"]) + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 220), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Não foi possível salvar {output_path.name}.")
