from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import itertools
import math

import cv2
import numpy as np

from backend.ink_color import image_with_component_ink
from backend.notebook_rulings import apply_notebook_rule_profile


JUNCTION_SCHEMA_VERSION = "stage9-junctions-1.1"


def _incident_segments(wires_info: dict) -> dict[str, list[dict]]:
    incident: dict[str, list[dict]] = {}
    for segment in wires_info.get("segments") or []:
        for key in ("start_node", "end_node"):
            node_id = segment.get(key)
            if node_id:
                incident.setdefault(str(node_id), []).append(segment)
    return incident


def _nearest_point_on_segment(point, a, b):
    px, py = map(float, point)
    ax, ay = map(float, a)
    bx, by = map(float, b)
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-9:
        return (ax, ay), math.hypot(px - ax, py - ay), 0.0
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / denom))
    qx, qy = ax + t * vx, ay + t * vy
    return (qx, qy), math.hypot(px - qx, py - qy), t


def _nearest_on_polyline(point, segment: dict):
    points = segment.get("points") or []
    if len(points) < 2:
        return None
    best = None
    for index, (a, b) in enumerate(zip(points, points[1:])):
        q, distance, t = _nearest_point_on_segment(
            point, (a["x"], a["y"]), (b["x"], b["y"])
        )
        if best is None or distance < best["distance"]:
            dx = float(b["x"]) - float(a["x"])
            dy = float(b["y"]) - float(a["y"])
            best = {
                "point": q,
                "distance": float(distance),
                "piece_index": index,
                "piece_t": float(t),
                "angle": float(math.degrees(math.atan2(dy, dx)) % 180.0),
            }
    return best


def _acute_angle_difference(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % 180.0
    return min(delta, 180.0 - delta)


def _ink_bridge_support(image: np.ndarray, a, b) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    length = max(1.0, math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))
    samples = max(5, int(round(length)) + 1)
    hits = 0
    h, w = gray.shape[:2]
    for t in np.linspace(0.0, 1.0, samples):
        x = int(round(float(a[0]) + (float(b[0]) - float(a[0])) * float(t)))
        y = int(round(float(a[1]) + (float(b[1]) - float(a[1])) * float(t)))
        x1, x2 = max(0, x - 1), min(w, x + 2)
        y1, y2 = max(0, y - 1), min(h, y + 2)
        if x1 < x2 and y1 < y2 and int(np.min(gray[y1:y2, x1:x2])) <= 190:
            hits += 1
    return hits / float(samples)


def _discover_missing_t_junctions(image: np.ndarray, wires_info: dict, *, color_filtered: bool = False) -> list[dict]:
    """Recupera T perdido quando a máscara deixou o ramo a poucos pixels do fio."""
    nodes = wires_info.get("nodes") or []
    segments = wires_info.get("segments") or []
    incident = _incident_segments(wires_info)
    by_id = {str(segment.get("wire_id")): segment for segment in segments}
    h, w = image.shape[:2]
    snap = max(10.0, min(24.0, 0.022 * min(h, w)))
    existing_junctions = [
        (float((node.get("position") or {}).get("x", 0)), float((node.get("position") or {}).get("y", 0)))
        for node in nodes
        if int(node.get("segment_degree", 0)) >= 3
    ]
    created = []
    for node in nodes:
        node_id = str(node.get("node_id"))
        attached = incident.get(node_id, [])
        if len(attached) != 1:
            continue
        position = node.get("position") or {}
        p = (float(position.get("x", 0)), float(position.get("y", 0)))
        stem = attached[0]
        stem_angle = _direction_from_node(stem, node_id)
        if stem_angle is None:
            continue
        candidates = []
        for target in segments:
            if target is stem or str(target.get("wire_id")) == str(stem.get("wire_id")):
                continue
            nearest = _nearest_on_polyline(p, target)
            if nearest is None or nearest["distance"] > snap:
                continue
            angle_difference = _acute_angle_difference(stem_angle, nearest["angle"])
            if angle_difference < 32.0:
                continue
            bridge = _ink_bridge_support(image, p, nearest["point"])
            if bridge < (0.80 if color_filtered else 0.70):
                continue
            evidence = _source_branch_evidence(image, {"x": nearest["point"][0], "y": nearest["point"][1]})
            if evidence["branch_count"] < 3:
                continue
            candidates.append((nearest["distance"], -bridge, target, nearest, angle_difference))
        if not candidates:
            continue
        _, neg_bridge, target, nearest, angle_difference = min(candidates, key=lambda item: (item[0], item[1]))
        q = nearest["point"]
        if any(math.hypot(q[0] - x, q[1] - y) <= 7.0 for x, y in existing_junctions):
            continue
        if any(
            math.hypot(q[0] - float((saved.get("position") or {}).get("x", 0)), q[1] - float((saved.get("position") or {}).get("y", 0))) <= 7.0
            for saved in created
        ):
            continue
        wire_ids = [str(stem.get("wire_id")), str(target.get("wire_id"))]
        created.append({
            "node_id": f"S9_T_{len(created) + 1:04d}",
            "position": {"x": round(float(q[0]), 2), "y": round(float(q[1]), 2)},
            "kind_candidate": "junction_candidate",
            "segment_degree": 2,
            "electrical_branch_count": 3,
            "incident_wire_ids": wire_ids,
            "stage9_synthetic_kind": "endpoint_to_segment_t",
            "stage9_source_endpoint": node_id,
            "snap_distance_px": round(float(nearest["distance"]), 3),
            "snap_limit_px": round(float(snap), 3),
            "angle_difference_degrees": round(float(angle_difference), 3),
            "ink_bridge_support": round(float(-neg_bridge), 3),
        })
    return created


def _terminal_points(terminals_info: dict | None) -> list[dict]:
    points = []
    for component in (terminals_info or {}).get("components") or []:
        component_id = str(
            component.get("component_id")
            or component.get("detection_id")
            or component.get("reference")
            or "COMP"
        )
        for index, terminal in enumerate(component.get("terminals") or [], start=1):
            position = terminal.get("position") or {}
            if "x" not in position or "y" not in position:
                continue
            pin = str(
                terminal.get("pin")
                or terminal.get("terminal")
                or terminal.get("name")
                or index
            )
            points.append({
                "terminal_ref": f"{component_id}:{pin}",
                "component_id": component_id,
                "pin": pin,
                "position": {
                    "x": float(position["x"]),
                    "y": float(position["y"]),
                },
            })
    return points


def _discover_terminal_to_wire_t_junctions(
    image: np.ndarray,
    wires_info: dict,
    terminals_info: dict | None,
    occupied: list[dict],
    *,
    color_filtered: bool = False,
) -> list[dict]:
    """Reconhece T formado por fio passante e um terminal de componente."""
    terminals = _terminal_points(terminals_info)
    segments = wires_info.get("segments") or []
    if not terminals or not segments:
        return []
    h, w = image.shape[:2]
    snap = max(11.0, min(28.0, 0.026 * min(h, w)))
    occupied_positions = [
        (
            float((node.get("position") or {}).get("x", 0)),
            float((node.get("position") or {}).get("y", 0)),
        )
        for node in occupied
        if int(node.get("electrical_branch_count", node.get("segment_degree", 0))) >= 3
    ]
    created = []
    for terminal in terminals:
        p = (
            float(terminal["position"]["x"]),
            float(terminal["position"]["y"]),
        )
        candidates = []
        for segment in segments:
            points = segment.get("points") or []
            nearest = _nearest_on_polyline(p, segment)
            if nearest is None or nearest["distance"] > snap or len(points) < 2:
                continue
            q = nearest["point"]
            first = (float(points[0]["x"]), float(points[0]["y"]))
            last = (float(points[-1]["x"]), float(points[-1]["y"]))
            # Para ser T, o fio precisa continuar para os dois lados. Um
            # terminal ligado ao fim normal de um fio não é uma junção T.
            if min(math.hypot(q[0] - first[0], q[1] - first[1]), math.hypot(q[0] - last[0], q[1] - last[1])) < 1.25 * snap:
                continue
            bridge = _ink_bridge_support(image, p, q)
            if bridge < (0.80 if color_filtered else 0.70):
                continue
            evidence = _source_branch_evidence(image, {"x": q[0], "y": q[1]})
            if evidence["branch_count"] < 3:
                continue
            candidates.append((nearest["distance"], -bridge, segment, nearest))
        if not candidates:
            continue
        _, neg_bridge, segment, nearest = min(candidates, key=lambda item: (item[0], item[1]))
        q = nearest["point"]
        if any(math.hypot(q[0] - x, q[1] - y) <= 7.0 for x, y in occupied_positions):
            continue
        if any(
            math.hypot(q[0] - float((node.get("position") or {}).get("x", 0)), q[1] - float((node.get("position") or {}).get("y", 0))) <= 7.0
            for node in created
        ):
            continue
        created.append({
            "node_id": f"S9_TERM_T_{len(created) + 1:04d}",
            "position": {"x": round(float(q[0]), 2), "y": round(float(q[1]), 2)},
            "kind_candidate": "junction_candidate",
            "segment_degree": 1,
            "electrical_branch_count": 3,
            "incident_wire_ids": [str(segment.get("wire_id"))],
            "attached_terminal_refs": [terminal["terminal_ref"]],
            "stage9_synthetic_kind": "terminal_to_wire_t",
            "snap_distance_px": round(float(nearest["distance"]), 3),
            "snap_limit_px": round(float(snap), 3),
            "ink_bridge_support": round(float(-neg_bridge), 3),
        })
    return created


def _direction_from_node(segment: dict, node_id: str) -> float | None:
    points = segment.get("points") or []
    if len(points) < 2:
        return None
    if str(segment.get("start_node")) == str(node_id):
        ordered = points
    elif str(segment.get("end_node")) == str(node_id):
        ordered = list(reversed(points))
    else:
        return None
    origin = np.array([ordered[0]["x"], ordered[0]["y"]], float)
    target = origin.copy()
    traveled = 0.0
    for a, b in zip(ordered, ordered[1:]):
        start = np.array([a["x"], a["y"]], float)
        end = np.array([b["x"], b["y"]], float)
        length = float(np.linalg.norm(end - start))
        target = start + (end - start) * min(1.0, max(0.0, 12.0 - traveled) / max(length, 1e-9))
        traveled += length
        if traveled >= 12.0:
            break
    dx, dy = target - origin
    if abs(dx) + abs(dy) < 1e-6:
        return None
    return float(math.degrees(math.atan2(dy, dx)) % 360.0)


def _opposition_error(a: float, b: float) -> float:
    delta = abs((a - b) % 360.0)
    delta = min(delta, 360.0 - delta)
    return abs(180.0 - delta)


def _pair_opposite_segments(segments: list[dict], node_id: str) -> tuple[list[list[str]], float]:
    if len(segments) not in {3, 4}:
        return [[str(s.get("wire_id"))] for s in segments], 0.0
    directions = {
        str(segment.get("wire_id")): _direction_from_node(segment, node_id)
        for segment in segments
    }
    if any(value is None for value in directions.values()):
        return [[str(s.get("wire_id"))] for s in segments], 0.0
    ids = list(directions)
    if len(ids) == 3:
        a, b = min(itertools.combinations(ids, 2), key=lambda pair:
                   _opposition_error(float(directions[pair[0]]), float(directions[pair[1]])))
        error = _opposition_error(float(directions[a]), float(directions[b]))
        return [[a, b], [wire for wire in ids if wire not in {a, b}]], round(max(0.0, 1.0 - error / 45.0), 4)
    matchings = [
        ((ids[0], ids[1]), (ids[2], ids[3])),
        ((ids[0], ids[2]), (ids[1], ids[3])),
        ((ids[0], ids[3]), (ids[1], ids[2])),
    ]
    scored = []
    for pairs in matchings:
        error = sum(
            _opposition_error(float(directions[a]), float(directions[b]))
            for a, b in pairs
        )
        scored.append((error, pairs))
    error, pairs = min(scored, key=lambda item: item[0])
    confidence = max(0.0, min(1.0, 1.0 - float(error) / 90.0))
    return [list(pair) for pair in pairs], round(confidence, 4)


def _junction_dot_evidence(image: np.ndarray, position: dict) -> dict:
    return _source_branch_evidence(image, position)["dot_evidence"]


def _source_branch_evidence(image: np.ndarray, position: dict) -> dict:
    """Count observed arms outside the junction, excluding nearby disconnected ink.

    A skeleton can split a thick bend into three branches. The original stroke
    must have three outgoing arms before that bend becomes an electrical T.
    A dot is measured against the width of the outgoing strokes, and its center
    can be a few pixels away from the skeleton cluster's arithmetic centroid.
    """
    h, w = image.shape[:2]
    cx = int(round(float(position.get("x", 0))))
    cy = int(round(float(position.get("y", 0))))
    radius = max(28, min(48, int(round(min(h, w) * 0.04))))
    x1, x2 = max(0, cx - radius), min(w, cx + radius + 1)
    y1, y2 = max(0, cy - radius), min(h, cy + radius + 1)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return {"branch_count": 0, "dot_evidence": {"dot_present": False, "reason": "empty_crop"}}
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    from backend.drawing_mode import digital,printed_ink
    if digital():mask=printed_ink(roi)
    else:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    lx, ly = cx - x1, cy - y1
    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
    rr = np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2)
    search = np.where(rr <= 5.0, distance, 0)
    py, px = np.unravel_index(int(np.argmax(search)), search.shape)
    center_radius = float(search[py, px])
    if center_radius <= 0:
        return {"branch_count": 0, "dot_evidence": {"dot_present": False, "reason": "no_ink_at_node"}}
    # Use one connected stroke only; a nearby parallel wire is not another arm.
    _, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    connected = labels == labels[py, px]
    peak_rr = np.hypot(xx - px, yy - py)
    annulus_values = distance[(peak_rr >= max(9.0, center_radius + 4)) &
                              (peak_rr <= radius - 2) & connected]
    line_half = float(np.percentile(annulus_values, 85)) if annulus_values.size else 1.0
    line_half = max(0.8, line_half)
    ratio = center_radius / line_half
    disk = peak_rr <= max(4.0, 1.65 * line_half)
    central_fill = float(np.count_nonzero((mask > 0) & disk)) / max(1.0, float(np.count_nonzero(disk)))
    dot_present = bool(
        center_radius >= max(2.2 if digital() else 4.8, (1.50 if digital() else 2.0) * line_half)
        and ratio >= (1.50 if digital() else 2.0)
        and central_fill >= (0.70 if digital() else 0.92)
    )
    dot = {
        "dot_present": dot_present,
        "center_radius_px": round(center_radius, 3),
        "wire_half_width_px": round(line_half, 3),
        "thickness_ratio": round(ratio, 3),
        "central_fill_ratio": round(central_fill, 3),
        "crop_radius_px": int(radius),
        "observed_center": {"x": int(px + x1), "y": int(py + y1)},
    }
    # The ring lies outside the filled dot, so the dot itself cannot join all
    # outgoing arms into a single blob. Closing one-pixel gaps tolerates JPEG.
    ring_radius = min(radius - 5, max(9.0, 3.0 * line_half, center_radius + 5.0))
    center_rr = peak_rr if dot_present else rr
    ring = (connected & (center_rr >= ring_radius) &
            (center_rr <= ring_radius + max(3.0, line_half))).astype(np.uint8)
    ring = cv2.morphologyEx(ring, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(ring, connectivity=8)
    branches = sum(int(stats[i, cv2.CC_STAT_AREA]) >= max(3, line_half) for i in range(1, count))
    if digital() and branches==4:
        from backend.printed_junction import dot_shape
        from backend.printed_capacitor import plate_ink
        present,shape=dot_shape(plate_ink(roi),(px,py))
        if present is not None:
            dot.update(dot_present=bool(present),printed_shape=shape)
    return {"branch_count": int(branches), "dot_evidence": dot}


def _summary(nodes: list[dict]) -> dict:
    return {
        "connected_junction_count": sum(
            1 for node in nodes if node.get("electrical_status") == "junction_connected"
        ),
        "unconnected_crossing_count": sum(
            1 for node in nodes if node.get("electrical_status") == "crossing_unconnected"
        ),
        "endpoint_count": sum(
            1 for node in nodes if node.get("electrical_status") == "endpoint"
        ),
        "path_vertex_count": sum(
            1 for node in nodes if node.get("electrical_status") == "wire_path"
        ),
        "review_count": sum(1 for node in nodes if node.get("review_required")),
    }


def _merge_dot_junctions(image: np.ndarray, nodes: list[dict], segments: list[dict]) -> list[dict]:
    """One filled dot may produce several small skeleton clusters.

    Keep one electrical junction and gather the interrupted wire ends that
    actually enter that dot. Proximity alone never adds a conductor.
    """
    result = []
    dots = []
    for node in nodes:
        dot = node.get("dot_evidence") or {}
        if node.get("electrical_status") != "junction_connected" or not dot.get("dot_present"):
            result.append(node)
            continue
        p = node["position"]
        existing = next((saved for saved in dots if math.hypot(
            p["x"] - saved["position"]["x"], p["y"] - saved["position"]["y"]) <= 7), None)
        if existing is not None:
            existing["incident_wire_ids"] = list(dict.fromkeys(existing["incident_wire_ids"] + node["incident_wire_ids"]))
            existing["attached_terminal_refs"] = list(dict.fromkeys(
                existing.get("attached_terminal_refs", []) + node.get("attached_terminal_refs", [])))
            existing.setdefault("merged_candidate_ids", []).append(node["node_id"])
        else:
            dots.append(node)
            result.append(node)
    for node in dots:
        p = (node["position"]["x"], node["position"]["y"])
        radius = float(node["dot_evidence"].get("center_radius_px", 0)) + 3
        for segment in segments:
            near = _nearest_on_polyline(p, segment)
            if near and near["distance"] <= radius and _ink_bridge_support(image, p, near["point"]) >= .80:
                wire_id = str(segment["wire_id"])
                if wire_id not in node["incident_wire_ids"]:
                    node["incident_wire_ids"].append(wire_id)
        node["connection_groups"] = [node["incident_wire_ids"]]
    return result


def resolver_juncoes(
    image: np.ndarray,
    wires_info: dict,
    terminals_info: dict | None = None,
) -> dict:
    """Resolve candidatos da Etapa 8 sem inventar conectividade elétrica."""
    if image is None or image.size == 0:
        raise ValueError("Imagem inválida para resolução de junções.")
    if not wires_info or not (wires_info.get("segments") or []):
        raise ValueError("Execute a Etapa 8 de linhas antes da Etapa 9.")

    # Reuse Stage 8's exact reference for bridges and junction dots. A rejected
    # notebook ruling must not reappear as electrical evidence in the raw photo.
    color_profile = wires_info.get("ink_color_profile") or {}
    from backend.paper_ink import wire_analysis_image
    image=wire_analysis_image(image,wires_info)
    image,_ = apply_notebook_rule_profile(image,wires_info.get("notebook_rule_profile"),color_profile)
    image = image_with_component_ink(image, color_profile)
    evidence_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    incident = _incident_segments(wires_info)
    segments_by_id = {
        str(segment.get("wire_id")): segment
        for segment in wires_info.get("segments") or []
    }
    synthesized_t = _discover_missing_t_junctions(evidence_image, wires_info, color_filtered=bool(color_profile.get("active")))
    synthesized_terminal_t = _discover_terminal_to_wire_t_junctions(
        evidence_image,
        wires_info,
        terminals_info,
        list(wires_info.get("nodes") or []) + synthesized_t,
        color_filtered=bool(color_profile.get("active")),
    )
    resolved_nodes = []
    for original in (
        list(wires_info.get("nodes") or [])
        + synthesized_t
        + synthesized_terminal_t
    ):
        node = deepcopy(original)
        node_id = str(node.get("node_id"))
        if node.get("stage9_synthetic_kind") in {
            "endpoint_to_segment_t", "terminal_to_wire_t"
        }:
            segment_ids = list(node.get("incident_wire_ids") or [])
            segments = [segments_by_id[wire_id] for wire_id in segment_ids if wire_id in segments_by_id]
            degree = int(node.get("electrical_branch_count", 3))
        else:
            segments = incident.get(node_id, [])
            segment_ids = [str(segment.get("wire_id")) for segment in segments]
            degree = len(segments)
        node["incident_wire_ids"] = segment_ids
        node["electrical_degree_candidate"] = int(degree)
        node["manual_override"] = False
        node["review_required"] = False
        evidence = _source_branch_evidence(evidence_image, node.get("position") or {}) if degree >= 3 else None
        if evidence:
            node["observed_branch_count"] = evidence["branch_count"]
            node["dot_evidence"] = evidence["dot_evidence"]
            # The graph is still the source of wire IDs. Pixel evidence rejects
            # extra skeleton spurs and identifies four arms around a real dot.
            if degree == 3 and evidence["branch_count"] < 3:
                degree = 2 if evidence["branch_count"] == 2 else 0
                node["electrical_degree_candidate"] = degree
                node["source_rejected_extra_branches"] = True
            elif evidence["dot_evidence"].get("dot_present"):
                dot = evidence["dot_evidence"]
                node["position"] = deepcopy(dot["observed_center"])
                node.update({
                    "electrical_status": "junction_connected",
                    "junction_type": "four_way_with_dot" if evidence["branch_count"] >= 4 else "t_junction_with_dot",
                    "connected": True,
                    "connection_groups": [segment_ids],
                    "confidence": round(min(1.0, max(0.90, float(dot.get("thickness_ratio", 0)) / 2.5)), 4),
                    "decision_reason": "visible_junction_dot",
                })
                resolved_nodes.append(node)
                continue
            elif degree == 3 and evidence["branch_count"] == 4 and not node.get("stage9_synthetic_kind"):
                # Losing one arm in the component mask cannot turn a plain X
                # crossing into an automatically connected T.
                degree = 4
                node["electrical_degree_candidate"] = 4

        if degree <= 1:
            node.update({
                "electrical_status": "endpoint",
                "junction_type": "endpoint",
                "connected": False,
                "connection_groups": ([[wire] for wire in segment_ids] if node.get("source_rejected_extra_branches")
                                      else [segment_ids] if segment_ids else []),
                "confidence": 0.99,
                "decision_reason": "single_incident_wire",
            })
        elif degree == 2:
            node.update({
                "electrical_status": "wire_path",
                "junction_type": "path_vertex",
                "connected": True,
                "connection_groups": [segment_ids],
                "confidence": 0.98,
                "decision_reason": "two_segments_form_same_wire_path",
            })
        elif degree == 3:
            node.update({
                "electrical_status": "junction_connected",
                "junction_type": "t_junction",
                "connected": True,
                "connection_groups": [segment_ids],
                "confidence": 0.97,
                "decision_reason": (
                    "terminal_snapped_to_passing_wire_forms_t"
                    if node.get("stage9_synthetic_kind") == "terminal_to_wire_t"
                    else "endpoint_snapped_to_wire_forms_t"
                    if node.get("stage9_synthetic_kind") == "endpoint_to_segment_t"
                    else "three_branch_t_junction"
                ),
            })
        elif degree == 4:
            dot = _junction_dot_evidence(evidence_image, node.get("position") or {})
            groups, pairing_confidence = _pair_opposite_segments(segments, node_id)
            node["dot_evidence"] = dot
            node["pairing_confidence"] = pairing_confidence
            if dot.get("dot_present"):
                node.update({
                    "electrical_status": "junction_connected",
                    "junction_type": "four_way_with_dot",
                    "connected": True,
                    "connection_groups": [segment_ids],
                    "confidence": round(min(1.0, max(0.90, float(dot.get("thickness_ratio", 0)) / 2.5)), 4),
                    "decision_reason": "visible_junction_dot",
                })
            elif pairing_confidence >= 0.72:
                node.update({
                    "electrical_status": "crossing_unconnected",
                    "junction_type": "four_way_without_dot",
                    "connected": False,
                    "connection_groups": groups,
                    "confidence": round(0.78 + 0.18 * pairing_confidence, 4),
                    "decision_reason": "opposite_wires_cross_without_dot",
                })
            else:
                node.update({
                    "electrical_status": "review_required",
                    "junction_type": "ambiguous_crossing",
                    "connected": False,
                    "connection_groups": groups,
                    "confidence": round(0.45 + 0.25 * pairing_confidence, 4),
                    "decision_reason": "four_way_pairing_ambiguous",
                    "review_required": True,
                })
        else:
            dot = _junction_dot_evidence(evidence_image, node.get("position") or {})
            node["dot_evidence"] = dot
            if dot.get("dot_present"):
                node.update({
                    "electrical_status": "junction_connected",
                    "junction_type": "multiway_with_dot",
                    "connected": True,
                    "connection_groups": [segment_ids],
                    "confidence": 0.90,
                    "decision_reason": "visible_dot_on_multiway_node",
                })
            else:
                node.update({
                    "electrical_status": "review_required",
                    "junction_type": "ambiguous_multiway",
                    "connected": False,
                    "connection_groups": [[wire_id] for wire_id in segment_ids],
                    "confidence": 0.35,
                    "decision_reason": "more_than_four_branches_without_dot",
                    "review_required": True,
                })
        resolved_nodes.append(node)

    resolved_nodes = _merge_dot_junctions(evidence_image, resolved_nodes, list(wires_info.get("segments") or []))
    result = {
        "schema_version": JUNCTION_SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "stage9_electrical_junction_resolution",
        "source_wire_schema": wires_info.get("schema_version"),
        "ink_color_profile": deepcopy(color_profile),
        "notebook_rule_profile": deepcopy(wires_info.get("notebook_rule_profile") or {}),
        "project_id": wires_info.get("project_id"),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "rules": {
            "t_junction_connects": True,
            "four_way_requires_dot_to_connect": True,
            "four_way_without_dot_keeps_opposite_paths_separate": True,
            "ambiguous_cases_require_review": True,
            "automatic_t_requires_three_observed_ink_arms": True,
            "small_gaps_also_require_ink_bridge": True,
        },
        "nodes": resolved_nodes,
        "segments": deepcopy(wires_info.get("segments") or []),
        "synthesized_t_junction_count": len(synthesized_t),
        "synthesized_terminal_t_junction_count": len(synthesized_terminal_t),
    }
    result.update(_summary(resolved_nodes))
    result["ready_for_stage10"] = result["review_count"] == 0
    return result


def analisar_clique_juncao(
    stage9: dict,
    x: float,
    y: float,
    terminals_info: dict | None = None,
    snap_radius_px: float | None = None,
) -> dict:
    """Explica e valida um clique manual sem aplicar a correção.

    A confirmação manual é deliberadamente mais permissiva que a descoberta
    automática: duas pontas de fios, ou um fio e um terminal, já representam
    referências condutoras suficientes para o usuário confirmar a junção.
    """
    width = float(stage9.get("image_width") or 0)
    height = float(stage9.get("image_height") or 0)
    sizes = [value for value in (width, height) if value > 0]
    reference = min(sizes) if sizes else 0
    default_snap = max(24.0, min(42.0, 0.032 * reference)) if reference else 34.0
    snap = max(12.0, min(48.0, float(snap_radius_px or default_snap)))

    nearby = []
    for segment in stage9.get("segments") or []:
        nearest = _nearest_on_polyline((x, y), segment)
        if nearest is None or nearest["distance"] > snap:
            continue
        nearby.append({
            "wire_id": str(segment.get("wire_id")),
            "distance_px": round(float(nearest["distance"]), 2),
            "location": "interior" if 0.08 < nearest["piece_t"] < 0.92 else "endpoint",
            "nearest": nearest,
        })
    nearby.sort(key=lambda item: item["distance_px"])

    selected = []
    seen_wires = set()
    for item in nearby:
        if item["wire_id"] in seen_wires:
            continue
        seen_wires.add(item["wire_id"])
        selected.append(item)
        if len(selected) >= 6:
            break

    terminal_snap = max(28.0, 1.35 * snap)
    nearby_terminals = []
    for item in _terminal_points(terminals_info):
        distance = math.hypot(
            float(x) - float(item["position"]["x"]),
            float(y) - float(item["position"]["y"]),
        )
        if distance <= terminal_snap:
            nearby_terminals.append({
                "terminal_ref": item["terminal_ref"],
                "distance_px": round(distance, 2),
            })
    nearby_terminals.sort(key=lambda item: item["distance_px"])
    nearby_terminals = nearby_terminals[:4]

    automatic_branch_count = sum(
        2 if item["location"] == "interior" else 1 for item in selected
    )
    conductor_reference_count = len(selected) + len(nearby_terminals)
    can_confirm = bool(selected) and conductor_reference_count >= 2
    return {
        "x": round(float(x), 2),
        "y": round(float(y), 2),
        "snap_radius_px": round(snap, 2),
        "wire_count": len(selected),
        "terminal_count": len(nearby_terminals),
        "automatic_branch_count": int(automatic_branch_count),
        "conductor_reference_count": int(conductor_reference_count),
        "can_confirm": can_confirm,
        "nearby_wires": [
            {key: value for key, value in item.items() if key != "nearest"}
            for item in selected
        ],
        "nearby_terminals": nearby_terminals,
        "message": (
            "Clique pronto para confirmação manual."
            if can_confirm
            else "Clique mais perto do encontro: é necessário encontrar dois fios, ou um fio e um terminal."
        ),
        "_selected": selected,
    }


def aplicar_correcoes_juncoes(
    stage9: dict,
    corrections: list[dict],
    terminals_info: dict | None = None,
) -> dict:
    result = deepcopy(stage9)
    nodes = {str(node.get("node_id")): node for node in result.get("nodes") or []}
    allowed = {"junction_connected", "crossing_unconnected", "review_required"}
    changed = 0
    for correction in corrections or []:
        if str(correction.get("action") or "") == "add_junction":
            x = float(correction.get("x"))
            y = float(correction.get("y"))
            analysis = analisar_clique_juncao(
                result,
                x,
                y,
                terminals_info=terminals_info,
                snap_radius_px=correction.get("snap_radius_px"),
            )
            selected = analysis.pop("_selected")
            attached_terminals = [
                item["terminal_ref"] for item in analysis["nearby_terminals"]
            ]
            if not analysis["can_confirm"]:
                raise ValueError(
                    f"{analysis['message']} Encontrados: {analysis['wire_count']} fio(s) e "
                    f"{analysis['terminal_count']} terminal(is), raio {analysis['snap_radius_px']:.0f}px."
                )
            existing_ids = {
                str(node.get("node_id")) for node in result.get("nodes") or []
            }
            index = 1
            while f"MANUAL_JUNCTION_{index:04d}" in existing_ids:
                index += 1
            node_id = f"MANUAL_JUNCTION_{index:04d}"
            wire_ids = [item["wire_id"] for item in selected]
            # A confirmação explícita do usuário resolve inclusive o caso em
            # que o detector quebrou um T em duas pontas e contou só 2 ramos.
            branch_count = max(3, int(analysis["automatic_branch_count"]))
            node = {
                "node_id": node_id,
                "position": {"x": round(x, 2), "y": round(y, 2)},
                "incident_wire_ids": wire_ids,
                "attached_terminal_refs": attached_terminals,
                "electrical_degree_candidate": int(branch_count),
                "electrical_status": "junction_connected",
                "junction_type": "manual_added_junction",
                "connected": True,
                "connection_groups": [wire_ids],
                "confidence": 1.0,
                "decision_reason": "manual_click_added_junction",
                "review_required": False,
                "manual_override": True,
                "manual_snap_radius_px": analysis["snap_radius_px"],
                "manual_click_analysis": analysis,
            }
            result.setdefault("nodes", []).append(node)
            nodes[node_id] = node
            changed += 1
            continue
        node_id = str(correction.get("node_id") or "")
        status = str(correction.get("electrical_status") or "")
        if node_id not in nodes:
            raise ValueError(f"Nó inexistente: {node_id}")
        if status not in allowed:
            raise ValueError(f"Estado elétrico inválido para {node_id}: {status}")
        node = nodes[node_id]
        wires = list(node.get("incident_wire_ids") or [])
        supplied_groups = correction.get("connection_groups")
        if status == "junction_connected":
            groups = [wires]
            connected = True
            kind = "manual_connected_junction"
        elif status == "crossing_unconnected":
            groups = supplied_groups or node.get("connection_groups") or [[wire] for wire in wires]
            connected = False
            kind = "manual_unconnected_crossing"
        else:
            groups = supplied_groups or node.get("connection_groups") or [[wire] for wire in wires]
            connected = False
            kind = "manual_review_pending"
        node.update({
            "electrical_status": status,
            "junction_type": kind,
            "connected": connected,
            "connection_groups": groups,
            "confidence": 1.0 if status != "review_required" else 0.0,
            "decision_reason": "manual_user_correction",
            "review_required": status == "review_required",
            "manual_override": True,
        })
        changed += 1
    result.update(_summary(list(nodes.values())))
    result["ready_for_stage10"] = result["review_count"] == 0
    result["manual_correction_count"] = int(result.get("manual_correction_count", 0)) + changed
    result["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    return result


def salvar_imagem_juncoes(image: np.ndarray, stage9: dict, output_path: Path) -> None:
    canvas = image.copy()
    for segment in stage9.get("segments") or []:
        points = segment.get("points") or []
        if len(points) >= 2:
            arr = np.asarray(
                [[int(point["x"]), int(point["y"])] for point in points],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            cv2.polylines(canvas, [arr], False, (170, 170, 170), 1, cv2.LINE_AA)
    colors = {
        "junction_connected": (20, 180, 20),
        "crossing_unconnected": (190, 30, 190),
        "review_required": (0, 150, 255),
        "endpoint": (30, 30, 220),
        "wire_path": (180, 180, 40),
    }
    for node in stage9.get("nodes") or []:
        pos = node.get("position") or {}
        x, y = int(round(float(pos.get("x", 0)))), int(round(float(pos.get("y", 0))))
        status = str(node.get("electrical_status"))
        color = colors.get(status, (0, 150, 255))
        if status == "crossing_unconnected":
            cv2.line(canvas, (x - 6, y - 6), (x + 6, y + 6), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (x - 6, y + 6), (x + 6, y - 6), color, 2, cv2.LINE_AA)
        elif status == "junction_connected":
            cv2.circle(canvas, (x, y), 6, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (x, y), 5, color, 2, cv2.LINE_AA)
        if status in {"junction_connected", "crossing_unconnected", "review_required"}:
            cv2.putText(
                canvas, str(node.get("node_id", "")), (x + 7, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Não foi possível salvar {output_path}.")
