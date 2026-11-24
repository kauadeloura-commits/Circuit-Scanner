from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math
from typing import Any

import cv2
import numpy as np

from backend.ink_color import component_ink_mask, estimate_component_ink, image_with_component_ink
from backend.notebook_rulings import detect_notebook_rule_profile, apply_notebook_rule_profile, protected_component_ink_mask, _paper_colors


WIRE_SCHEMA_VERSION = "stage8-wires-1.0"


def _clip(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(float(v)))))


def _bbox_tuple(bbox: dict, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = _clip(bbox.get("x1", 0), 0, max(0, w - 1))
    y1 = _clip(bbox.get("y1", 0), 0, max(0, h - 1))
    x2 = _clip(bbox.get("x2", x1 + 1), 0, w)
    y2 = _clip(bbox.get("y2", y1 + 1), 0, h)
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def _background_normalized_ink(image: np.ndarray, components_info: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return grayscale and a dark-ink binary mask.

    Local background normalization handles uneven lighting / paper shadows. The
    Otsu result is intersected with a conservative absolute-darkness mask so
    faint notebook rulings are less likely to become wires.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    sigma = max(7.0, min(image.shape[:2]) / 55.0)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    norm = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    norm = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8)).apply(norm)
    otsu, local = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Keep clearly dark ink even when local contrast is weak; reject pale page
    # lines that adaptive/local thresholding alone tends to retain.
    # Estimate the darkness threshold inside the circuit region rather than the
    # whole photo (which may include a black table/desk). The lower tail is a
    # good proxy for pencil/pen strokes; ruled-paper lines usually sit above it.
    sample = gray
    comps = _collect_components(components_info) if components_info else []
    if comps:
        h0, w0 = gray.shape[:2]
        boxes = [_bbox_tuple(c.get("bbox") or {}, w0, h0) for c in comps]
        x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
        margin = int(max(24, min(max(h0, w0) * 0.08, 110)))
        x1=max(0,x1-margin); y1=max(0,y1-margin); x2=min(w0,x2+margin); y2=min(h0,y2+margin)
        if x2>x1 and y2>y1:
            sample = gray[y1:y2, x1:x2]
    p3 = float(np.percentile(sample, 3))
    dark_limit = int(max(105, min(165, p3 + 6)))
    absolute_dark = (gray <= dark_limit).astype(np.uint8) * 255
    ink = cv2.bitwise_and(local, absolute_dark)
    # Also accept strokes with strong *local* contrast even when illumination
    # makes their absolute gray value slightly too bright. Ruled-paper lines
    # are much smoother and are removed in the following ruling-suppression step.
    strong_local_limit = int(max(0, float(otsu) - 22.0))
    strong_local = (norm <= strong_local_limit).astype(np.uint8) * 255
    ink = cv2.bitwise_or(ink, strong_local)

    # Restore very dark pixels which may disappear under strong shadows.
    very_dark = (gray <= min(135, dark_limit - 20)).astype(np.uint8) * 255
    ink = cv2.bitwise_or(ink, very_dark)
    return gray, ink, {
        "otsu_threshold": round(float(otsu), 2),
        "absolute_dark_limit": dark_limit,
        "darkness_percentile_p3": round(p3, 2),
        "strong_local_limit": strong_local_limit,
        "background_sigma": round(float(sigma), 2),
    }


def _remove_pale_page_rulings(image: np.ndarray, gray: np.ndarray, ink: np.ndarray, *, preserve_reference_color: bool = False) -> tuple[np.ndarray, int]:
    """Suppress notebook rulings while preserving dark circuit strokes.

    Two complementary cues are used:
    1) blue/gray ruled-paper chroma (common in notebook photos), and
    2) very long, pale, near-horizontal Hough lines.
    Dark strokes are explicitly protected so real wires crossing a ruling survive.
    """
    h, w = gray.shape[:2]
    result = ink.copy()
    removed = 0

    if image.ndim == 3 and not preserve_reference_color:
        b, g, r = cv2.split(image.astype(np.int16))
        blue_excess = b - r
        # Notebook lines are usually pale blue. Dilate vertically a little to
        # cover antialiased/compressed edges; keep dark blue/black circuit ink.
        blue = ((blue_excess >= 5) & (gray >= 120)).astype(np.uint8) * 255
        blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((1, 11), np.uint8))
        blue = cv2.dilate(blue, np.ones((3, 1), np.uint8), iterations=1)
        before = int(cv2.countNonZero(result))
        result[blue > 0] = 0
        if before > int(cv2.countNonZero(result)):
            removed += 1

    # Gray/pale rulings: detect very long horizontal lines. Run Hough on the
    # binary ink instead of only Canny(gray), because low-contrast paper lines
    # may not produce a stable grayscale edge.
    gray_edges = cv2.Canny(gray, 30, 85)
    lines = cv2.HoughLinesP(
        gray_edges, 1, np.pi / 180.0,
        threshold=max(35, int(w * 0.055)),
        minLineLength=max(100, int(w * 0.40)),
        maxLineGap=max(24, int(w * 0.12)),
    )
    if lines is None:
        return result, removed

    rows = np.asarray(lines).reshape(-1, 4)
    # Detect a *family* of long, parallel horizontal rulings. Notebook pages
    # produce many such lines at different y positions; circuit wires do not.
    family = []
    for row in rows:
        x1,y1,x2,y2=[int(v) for v in row]
        dx,dy=x2-x1,y2-y1
        length=math.hypot(dx,dy)
        if abs(dx) < 1 or length < w*0.34:
            continue
        angle=abs(math.degrees(math.atan2(dy,dx)))%180.0
        angle=min(angle,180.0-angle)
        if angle > 5.5:
            continue
        slope=dy/float(dx)
        yc=y1 + slope*((w-1)/2.0-x1)
        family.append((float(yc),length,slope,x1,y1,x2,y2))
    family.sort(key=lambda item:item[0])
    clusters=[]
    for item in family:
        if not clusters or abs(item[0]-np.mean([v[0] for v in clusters[-1]]))>8.0:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    if len(clusters) >= 4:
        for cluster in clusters:
            rep=max(cluster,key=lambda item:item[1])
            _yc,_len,slope,x1,y1,x2,y2=rep
            # Only erase a ruling-family member when it spans most of the
            # page. Shorter near-horizontal strokes can be real schematic wires.
            if _len < w*0.65:
                continue
            y_left=int(round(y1 + slope*(0-x1)))
            y_right=int(round(y1 + slope*((w-1)-x1)))
            band=np.zeros_like(result)
            cv2.line(band,(0,y_left),(w-1,y_right),255,13)
            # Estimate the ruling brightness itself and erase only pixels close
            # to that pale level. Pencil/pen wires crossing or running near the
            # ruling are usually 10-30 gray levels darker and must survive.
            vals=[]
            for xx in np.linspace(0, w-1, 64):
                yy = y_left + (y_right-y_left) * (float(xx)/max(1.0,float(w-1)))
                xi=_clip(xx,0,w-1); yi=_clip(yy,0,h-1)
                xa,xb=max(0,xi-1),min(w,xi+2); ya,yb=max(0,yi-1),min(h,yi+2)
                vals.append(float(np.median(gray[ya:yb,xa:xb])))
            ruling_level=float(np.median(vals)) if vals else 145.0
            erase_limit=max(105.0, ruling_level-12.0)
            erase=(band>0) & (gray>=erase_limit)
            before=int(cv2.countNonZero(result))
            result[erase]=0
            if before>int(cv2.countNonZero(result)):
                removed += 1

    for row in rows:
        x1, y1, x2, y2 = [int(v) for v in row]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        # Standalone suppression is reserved for almost page-wide strokes.
        # Real schematic wires often run horizontally for 30-55% of the page.
        if length < w * 0.68:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0
        angle = min(angle, abs(180.0 - angle))
        if angle > 4.5:
            continue
        n = max(24, int(length // 10))
        vals = []
        for t in np.linspace(0.0, 1.0, n):
            x = _clip(x1 + dx * t, 0, w - 1)
            y = _clip(y1 + dy * t, 0, h - 1)
            xa, xb = max(0, x - 2), min(w, x + 3)
            ya, yb = max(0, y - 2), min(h, y + 3)
            vals.append(float(np.percentile(gray[ya:yb, xa:xb], 20)))
        median_dark = float(np.median(vals)) if vals else 255.0
        span_ratio = length / max(1.0, float(w))
        # A line spanning almost the entire page is overwhelmingly likely to
        # be a notebook ruling. Remove it even when the photo made it fairly
        # dark, while preserving very-dark circuit ink crossing the band.
        if span_ratio < 0.75 and median_dark < 115.0:
            continue
        band = np.zeros_like(result)
        cv2.line(band, (x1, y1), (x2, y2), 255, 15)
        # Preserve strokes clearly darker than the detected page line.
        # A fixed threshold (old behavior) erased faint pencil wires too.
        keep_dark_limit = int(max(105, min(165, median_dark - 12.0)))
        pale = (gray >= keep_dark_limit).astype(np.uint8) * 255
        erase = cv2.bitwise_and(band, pale)
        before = int(cv2.countNonZero(result))
        result[erase > 0] = 0
        if before > int(cv2.countNonZero(result)):
            removed += 1
    return result, removed


def _remove_border_frame_artifacts(gray: np.ndarray, ink: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove long page/crop borders near the left/right image edges."""
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 30, 90)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=max(45, int(h * 0.08)),
        minLineLength=max(100, int(h * 0.48)),
        maxLineGap=max(25, int(h * 0.08)),
    )
    if lines is None:
        return ink, 0
    out = ink.copy(); removed = 0
    for row in np.asarray(lines).reshape(-1,4):
        x1,y1,x2,y2=[int(v) for v in row]
        dx,dy=x2-x1,y2-y1
        length=math.hypot(dx,dy)
        angle=abs(math.degrees(math.atan2(dy,dx)))%180.0
        if abs(angle-90.0)>4.5 or length < h*0.48:
            continue
        xm=(x1+x2)/2.0
        if not (xm <= w*0.10 or xm >= w*0.90):
            continue
        band=np.zeros_like(out)
        cv2.line(band,(x1,y1),(x2,y2),255,11)
        out[band>0]=0
        removed += 1
    return out, removed


def _collect_components(info: dict | None) -> list[dict]:
    if not isinstance(info, dict):
        return []
    comps = info.get("components")
    if isinstance(comps, list):
        return [c for c in comps if isinstance(c, dict) and isinstance(c.get("bbox"), dict)]
    dets = info.get("detections")
    if isinstance(dets, list):
        out = []
        for d in dets:
            if not isinstance(d, dict) or not isinstance(d.get("bbox"), dict):
                continue
            out.append({
                "component_id": d.get("detection_id") or d.get("component_id"),
                "type": d.get("class") or d.get("type"),
                "bbox": d.get("bbox"),
                "terminals": [],
            })
        return out
    return []


def _mask_components_and_text(
    ink: np.ndarray,
    components_info: dict | None,
    ocr_info: dict | None,
    *,
    mask_components: bool,
    mask_text: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    h, w = ink.shape[:2]
    cleaned = ink.copy()
    erased = np.zeros_like(ink)
    component_count = 0
    text_count = 0
    preserved_terminal_count = 0

    if mask_components:
        for comp in _collect_components(components_info):
            bbox = comp.get("bbox") or {}
            x1, y1, x2, y2 = _bbox_tuple(bbox, w, h)
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            # Mask almost the entire component body. Only a very thin perimeter
            # is left so external leads can reach the bbox edge. Stage 8 should
            # represent conductors *between* components, not the symbol strokes.
            inset_x = max(1, min(4, int(round(bw * 0.035))))
            inset_y = max(1, min(4, int(round(bh * 0.035))))
            ix1, iy1 = min(x2, x1 + inset_x), min(y2, y1 + inset_y)
            ix2, iy2 = max(x1, x2 - inset_x), max(y1, y2 - inset_y)
            if ix2 > ix1 and iy2 > iy1:
                cv2.rectangle(erased, (ix1, iy1), (ix2, iy2), 255, -1)
                component_count += 1

            # Preserve only tiny edge-local pin regions. This is intentionally
            # conservative: uncertain transistor B/C/E must not reintroduce the
            # internal fork into the wire mask.
            r = 3
            for terminal in comp.get("terminals") or []:
                pos = terminal.get("position") if isinstance(terminal, dict) else None
                if not isinstance(pos, dict):
                    continue
                tx = _clip(pos.get("x", 0), 0, w - 1)
                ty = _clip(pos.get("y", 0), 0, h - 1)
                edge_dist = min(abs(tx-x1), abs(tx-x2), abs(ty-y1), abs(ty-y2))
                if edge_dist <= 5:
                    cv2.circle(erased, (tx, ty), r, 0, -1)
                    preserved_terminal_count += 1

    component_erased=erased.copy()
    crossing_protection=np.zeros_like(ink);crossing_count=0
    if mask_text and isinstance(ocr_info, dict):
        tokens = ocr_info.get("tokens") or []
        for tok in tokens:
            bbox = tok.get("bbox") if isinstance(tok, dict) else None
            if not isinstance(bbox, dict):
                continue
            x1, y1, x2, y2 = _bbox_tuple(bbox, w, h)
            pad = 2
            cv2.rectangle(
                erased,
                (max(0, x1 - pad), max(0, y1 - pad)),
                (min(w - 1, x2 + pad), min(h - 1, y2 + pad)),
                255,
                -1,
            )
            text_count += 1
            from backend.text_wire_preservation import crossing_ink
            evidence=crossing_ink(ink,bbox)
            if evidence:
                (ax,ay,bx,by),observed,count=evidence
                crossing_protection[ay:by,ax:bx]|=observed
                crossing_count+=count

    # Text bounding rectangles are not evidence that every enclosed pixel is
    # writing. Restore only original continuous ink, outside component bodies.
    preserved=(crossing_protection>0)&(component_erased==0)&(erased>0)
    erased[preserved]=0

    cleaned[erased > 0] = 0
    return cleaned, erased, {
        "component_regions_masked": component_count,
        "ocr_regions_masked": text_count,
        "terminal_regions_preserved": preserved_terminal_count,
        "observed_text_crossing_wires_preserved": crossing_count,
        "text_mask_conductor_pixels_preserved": int(np.count_nonzero(preserved)),
    }


def _restrict_to_component_roi(mask: np.ndarray, components_info: dict | None) -> tuple[np.ndarray, dict]:
    comps = _collect_components(components_info)
    if not comps:
        return mask, {"component_roi_applied": False, "component_roi": None}
    h, w = mask.shape[:2]
    boxes = [_bbox_tuple(c.get("bbox") or {}, w, h) for c in comps]
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    margin = int(max(28, min(max(h, w) * 0.10, 140)))
    x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin); y2 = min(h, y2 + margin)
    out = np.zeros_like(mask)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out, {
        "component_roi_applied": True,
        "component_roi": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "margin_px": margin},
    }


def _filter_small_blobs(mask: np.ndarray, min_area: int = 7) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == i] = 255
    return out


def _prepare_wire_mask(mask: np.ndarray) -> np.ndarray:
    # Bridge 1-2 px breaks without thickening the image aggressively.
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return _filter_small_blobs(closed, min_area=6)


def _filter_wire_candidates_by_terminal_seeds(mask: np.ndarray, components_info: dict | None) -> tuple[np.ndarray, dict]:
    """Keep conductor blobs that actually approach a known component terminal.

    Stage 7 terminal estimates are not required to be perfect for this filter: a
    wire normally has two ends, so touching a reliable terminal on *either* end
    is enough to retain the entire connected blob. This is especially useful on
    ruled paper, where page lines may pass near component bounding boxes but do
    not approach the electrical pin locations.

    When no usable terminal positions exist, the caller can fall back to the
    broader component-proximity filter.
    """
    comps = _collect_components(components_info)
    if not comps:
        return mask, {
            "terminal_seed_filter_applied": False,
            "terminal_seed_count": 0,
            "terminal_seed_radius_px": 0,
            "kept_terminal_seed_blobs": 0,
            "removed_nonterminal_blobs": 0,
        }

    h, w = mask.shape[:2]
    seeds: list[tuple[int, int]] = []
    for comp in comps:
        ctype = str(comp.get("type") or "")
        bbox = comp.get("bbox") or {}
        if ctype in {"transistor_npn", "transistor_pnp"} and isinstance(bbox, dict):
            # Transistor terminals are intentionally NOT trusted before Stage 8.
            # Preserve any conductor that reaches the bbox perimeter instead, so
            # the three real paths can be used afterwards to reconstruct B/C/E.
            try:
                x1, y1, x2, y2 = _bbox_tuple(bbox, w, h)
                step_x = max(4, int(max(1, x2-x1) / 8))
                step_y = max(4, int(max(1, y2-y1) / 8))
                for x in range(x1, x2 + 1, step_x):
                    seeds.append((_clip(x, 0, w - 1), _clip(y1, 0, h - 1)))
                    seeds.append((_clip(x, 0, w - 1), _clip(y2, 0, h - 1)))
                for y in range(y1, y2 + 1, step_y):
                    seeds.append((_clip(x1, 0, w - 1), _clip(y, 0, h - 1)))
                    seeds.append((_clip(x2, 0, w - 1), _clip(y, 0, h - 1)))
            except Exception:
                pass
            continue
        for terminal in comp.get("terminals") or []:
            if not isinstance(terminal, dict):
                continue
            pos = terminal.get("position")
            if not isinstance(pos, dict):
                continue
            try:
                x = float(pos.get("x"))
                y = float(pos.get("y"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                seeds.append((_clip(x, 0, w - 1), _clip(y, 0, h - 1)))

    # Require a minimally useful set of Stage-7 anchors. One lonely/incorrect
    # pin should not be allowed to erase the rest of the circuit.
    if len(seeds) < 2:
        return mask, {
            "terminal_seed_filter_applied": False,
            "terminal_seed_count": len(seeds),
            "terminal_seed_radius_px": 0,
            "kept_terminal_seed_blobs": 0,
            "removed_nonterminal_blobs": 0,
        }

    # Radius scales mildly with image size, but remains local enough to reject
    # nearby notebook rulings. A hand-drawn terminal can easily be 5-12 px away
    # from the Stage-7 estimate after masking/thresholding.
    radius = int(max(10, min(22, round(max(h, w) * 0.018))))
    seed_mask = np.zeros_like(mask)
    for x, y in seeds:
        cv2.circle(seed_mask, (x, y), radius, 255, -1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask)
    kept = 0
    removed = 0
    for i in range(1, n):
        component = labels == i
        if not np.any(component):
            continue
        if np.any(component & (seed_mask > 0)):
            out[component] = 255
            kept += 1
        else:
            removed += 1

    # Safety valve: if the filter would erase virtually everything, Stage-7
    # anchors are probably too inaccurate for this project. Keep the original
    # mask and report that terminal seeding was not trusted.
    before = int(cv2.countNonZero(mask))
    after = int(cv2.countNonZero(out))
    if before > 0 and (after < max(20, int(before * 0.08)) or kept == 0):
        return mask, {
            "terminal_seed_filter_applied": False,
            "terminal_seed_filter_fallback": True,
            "terminal_seed_count": len(seeds),
            "terminal_seed_radius_px": radius,
            "kept_terminal_seed_blobs": kept,
            "removed_nonterminal_blobs": removed,
        }

    return out, {
        "terminal_seed_filter_applied": True,
        "terminal_seed_filter_fallback": False,
        "terminal_seed_count": len(seeds),
        "terminal_seed_radius_px": radius,
        "kept_terminal_seed_blobs": kept,
        "removed_nonterminal_blobs": removed,
    }


def _filter_wire_candidates_by_components(mask: np.ndarray, components_info: dict | None) -> tuple[np.ndarray, dict]:
    comps = _collect_components(components_info)
    if not comps:
        return mask, {"component_proximity_filter_applied": False, "removed_candidate_blobs": 0}
    h, w = mask.shape[:2]
    proximity = int(max(12, min(34, max(h, w) * 0.025)))
    near = np.zeros_like(mask)
    for c in comps:
        x1,y1,x2,y2 = _bbox_tuple(c.get("bbox") or {}, w, h)
        cv2.rectangle(
            near,
            (max(0, x1-proximity), max(0, y1-proximity)),
            (min(w-1, x2+proximity), min(h-1, y2+proximity)),
            255, -1,
        )
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask)
    removed = 0
    kept = 0
    for i in range(1, n):
        component = labels == i
        if not np.any(component):
            continue
        if np.any((near > 0) & component):
            out[component] = 255
            kept += 1
        else:
            removed += 1
    return out, {
        "component_proximity_filter_applied": True,
        "component_proximity_px": proximity,
        "kept_candidate_blobs": kept,
        "removed_candidate_blobs": removed,
    }


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Exact Zhang-Suen thinning with precomputed neighborhood rules."""
    from backend.thinning import thin_binary
    return thin_binary(mask)


def _neighbors8(y: int, x: int, h: int, w: int):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                yield yy, xx


def _polyline_length(points: list[tuple[int, int]]) -> float:
    if len(points) < 2:
        return 0.0
    return float(sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(points, points[1:])))


def _simplify(points: list[tuple[int, int]], epsilon: float = 1.8) -> list[tuple[int, int]]:
    if len(points) <= 2:
        return points
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    simp = cv2.approxPolyDP(arr, epsilon, False).reshape(-1, 2)
    return [(int(round(x)), int(round(y))) for x, y in simp]


def _extract_graph(skeleton: np.ndarray, min_segment_length: float = 8.0) -> tuple[list[dict], list[dict], dict]:
    """Convert a 1-pixel skeleton into raw graph segments.

    This is deliberately *topological candidate* output only. Stage 9 remains
    responsible for deciding whether a branch/crossing is an electrical
    junction or a non-connected crossing.
    """
    sk = skeleton.astype(bool)
    h, w = sk.shape[:2]
    ys, xs = np.nonzero(sk)
    if len(xs) == 0:
        return [], [], {"raw_skeleton_pixels": 0, "closed_loop_count": 0}

    # The same eight-neighbor count in one pass, including zero padding at
    # page edges. Do not scan the whole page again for every node cluster.
    degree = cv2.filter2D(sk.astype(np.uint8), cv2.CV_16S,
                        np.ones((3,3),np.int16),borderType=cv2.BORDER_CONSTANT)-sk

    # Pixels that are not simple degree-2 chain pixels are node candidates.
    node_pixels = (sk & (degree != 2)).astype(np.uint8)
    # Adjacent node clusters can be linked by a two-pixel chain at a thick
    # elbow/dot. Discarding that short segment later broke one visible wire
    # into unrelated nets. Absorb only skeleton-connected pixels, preserving
    # blank gaps and the electrical decision for crossings in Stage 9.
    if min_segment_length>=8:
        for _ in range(2):
            node_pixels=cv2.dilate(node_pixels,np.ones((3,3),np.uint8))&sk.astype(np.uint8)
    nlab, labels, stats, cents = cv2.connectedComponentsWithStats(node_pixels, 8)
    node_map: dict[tuple[int, int], str] = {}
    nodes: list[dict] = []
    for i in range(1, nlab):
        bx,by,bw,bh=stats[i,:4]
        coords = np.argwhere(labels[by:by+bh,bx:bx+bw] == i)+[by,bx]
        if coords.size == 0:
            continue
        cy, cx = cents[i][1], cents[i][0]
        node_id = f"NODE_{len(nodes)+1:04d}"
        # External branches touching the cluster; this is candidate degree, not
        # final electrical degree (Stage 9 can split crossings).
        external = set()
        cluster_set = {(int(y), int(x)) for y, x in coords}
        for y, x in cluster_set:
            for yy, xx in _neighbors8(y, x, h, w):
                if sk[yy, xx] and (yy, xx) not in cluster_set:
                    external.add((yy, xx))
            node_map[(y, x)] = node_id
        branch_count = len(external)
        kind = "endpoint_candidate" if branch_count <= 1 else "junction_candidate"
        nodes.append({
            "node_id": node_id,
            "position": {"x": round(float(cx), 2), "y": round(float(cy), 2)},
            "pixel_count": int(len(coords)),
            "raw_branch_count": int(branch_count),
            "kind_candidate": kind,
        })

    node_lookup = {n["node_id"]: n for n in nodes}
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def edge_key(a, b):
        return (a, b) if a <= b else (b, a)

    segments: list[dict] = []
    traced_pixels: set[tuple[int, int]] = set()

    # Trace outward from every node cluster.
    node_clusters: dict[str, set[tuple[int, int]]] = {}
    for pix, nid in node_map.items():
        node_clusters.setdefault(nid, set()).add(pix)

    for nid, cluster in node_clusters.items():
        starts = set()
        for y, x in cluster:
            for yy, xx in _neighbors8(y, x, h, w):
                if sk[yy, xx] and (yy, xx) not in cluster:
                    starts.add(((y, x), (yy, xx)))
        for from_pix, start in starts:
            ek = edge_key(from_pix, start)
            if ek in visited_edges:
                continue
            visited_edges.add(ek)
            path_yx = [from_pix, start]
            prev = from_pix
            cur = start
            end_node = None
            guard = 0
            while guard < (h * w):
                guard += 1
                if cur in node_map:
                    end_node = node_map[cur]
                    break
                nxts = [(yy, xx) for yy, xx in _neighbors8(cur[0], cur[1], h, w) if sk[yy, xx] and (yy, xx) != prev]
                if not nxts:
                    break
                # Degree-2 chains normally have one continuation. If thinning
                # leaves a tiny ambiguity, prefer an unvisited edge.
                nxts.sort(key=lambda p: edge_key(cur, p) in visited_edges)
                nxt = nxts[0]
                visited_edges.add(edge_key(cur, nxt))
                path_yx.append(nxt)
                prev, cur = cur, nxt

            traced_pixels.update(path_yx)
            points = [(x, y) for y, x in path_yx]
            length = _polyline_length(points)
            if length < min_segment_length:
                continue
            simp = _simplify(points, epsilon=1.8)
            xvals = [p[0] for p in simp]
            yvals = [p[1] for p in simp]
            dx = simp[-1][0] - simp[0][0]
            dy = simp[-1][1] - simp[0][1]
            angle = math.degrees(math.atan2(dy, dx)) if (dx or dy) else 0.0
            aa = abs(angle) % 180.0
            aa = min(aa, 180.0 - aa)
            if aa <= 12:
                orientation = "horizontal"
            elif abs(aa - 90.0) <= 12:
                orientation = "vertical"
            else:
                orientation = "diagonal_or_curved"
            segments.append({
                "wire_id": f"WIRE_{len(segments)+1:04d}",
                "start_node": nid,
                "end_node": end_node,
                "start": {"x": simp[0][0], "y": simp[0][1]},
                "end": {"x": simp[-1][0], "y": simp[-1][1]},
                "points": [{"x": int(x), "y": int(y)} for x, y in simp],
                "raw_point_count": len(points),
                "length_px": round(length, 2),
                "bbox": {
                    "x1": int(min(xvals)), "y1": int(min(yvals)),
                    "x2": int(max(xvals)), "y2": int(max(yvals)),
                },
                "angle_degrees": round(float(angle), 2),
                "orientation": orientation,
                "semantic_status": "unresolved_until_stage9",
            })

    # Handle pure closed loops (all pixels degree 2) that never touch a node.
    # These are rare in schematic wires but keeping them makes the representation
    # complete and avoids silently discarding a loop-shaped conductor.
    remaining = set(zip(ys.tolist(), xs.tolist())) - traced_pixels - set(node_map.keys())
    closed_loop_count = 0
    while remaining:
        start = next(iter(remaining))
        path = [start]
        prev = None
        cur = start
        for _ in range(len(remaining) + 2):
            remaining.discard(cur)
            nxts = [(yy, xx) for yy, xx in _neighbors8(cur[0], cur[1], h, w) if sk[yy, xx] and (yy, xx) != prev]
            if not nxts:
                break
            nxt = nxts[0]
            if nxt == start:
                path.append(nxt)
                break
            if nxt not in remaining and nxt in path:
                path.append(nxt)
                break
            path.append(nxt)
            prev, cur = cur, nxt
        points = [(x, y) for y, x in path]
        length = _polyline_length(points)
        if length >= min_segment_length:
            simp = _simplify(points, epsilon=1.8)
            segments.append({
                "wire_id": f"WIRE_{len(segments)+1:04d}",
                "start_node": None,
                "end_node": None,
                "start": {"x": simp[0][0], "y": simp[0][1]},
                "end": {"x": simp[-1][0], "y": simp[-1][1]},
                "points": [{"x": int(x), "y": int(y)} for x, y in simp],
                "raw_point_count": len(points),
                "length_px": round(length, 2),
                "bbox": {
                    "x1": int(min(x for x, _ in simp)), "y1": int(min(y for _, y in simp)),
                    "x2": int(max(x for x, _ in simp)), "y2": int(max(y for _, y in simp)),
                },
                "angle_degrees": 0.0,
                "orientation": "closed_loop_or_curve",
                "semantic_status": "unresolved_until_stage9",
            })
            closed_loop_count += 1

    # Recompute graph degree from accepted segments for a clean debug field.
    counts = {n["node_id"]: 0 for n in nodes}
    for seg in segments:
        if seg.get("start_node") in counts:
            counts[seg["start_node"]] += 1
        if seg.get("end_node") in counts:
            counts[seg["end_node"]] += 1
    for n in nodes:
        n["segment_degree"] = int(counts[n["node_id"]])
        if n["segment_degree"] <= 1:
            n["kind_candidate"] = "endpoint_candidate"
        elif n["segment_degree"] == 2:
            n["kind_candidate"] = "path_vertex_candidate"
        else:
            n["kind_candidate"] = "junction_candidate"

    return nodes, segments, {
        "raw_skeleton_pixels": int(len(xs)),
        "closed_loop_count": closed_loop_count,
    }


def detectar_fios(
    image: np.ndarray,
    *,
    components_info: dict | None = None,
    ocr_info: dict | None = None,
    source_image: str = "original",
    mask_components: bool = True,
    mask_text: bool = True,
    min_segment_length: float = 8.0,
    color_reference_image: np.ndarray | None = None,
    color_components_info: dict | None = None,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    if image is None or image.size == 0:
        raise ValueError("Imagem inválida para detecção de fios.")

    h, w = image.shape[:2]
    gray, ink, prep_meta = _background_normalized_ink(image, components_info)
    reference_image = image if color_reference_image is None else color_reference_image
    reference_components = components_info if color_components_info is None else color_components_info
    from backend.drawing_mode import digital,printed_ink
    is_digital=digital()
    if is_digital:
        ink=printed_ink(image)
        prep_meta.update(drawing_type="digital",method="printed_thin_strokes_without_gap_filling")
    color_profile = dict(active=False,chromatic=False,reason="explicit_digital_input") if is_digital else estimate_component_ink(reference_image, reference_components)
    if reference_image.shape[:2] != image.shape[:2]:
        color_profile.update(active=False, reason="color_reference_dimension_mismatch")
    rule_profile = dict(active=False,reason="explicit_digital_input",removed_pixel_count=0) if is_digital else detect_notebook_rule_profile(reference_image, color_profile)
    if reference_image.shape[:2] != image.shape[:2]:
        rule_profile.update(active=False,reason="color_reference_dimension_mismatch")
    _, ruling_mask = apply_notebook_rule_profile(reference_image, rule_profile, color_profile)
    from backend.paper_ink import fragmentation_policy,stroke_mask
    local_ink_policy=fragmentation_policy(reference_image,reference_components,color_profile,rule_profile)
    color_mask = component_ink_mask(reference_image, color_profile)
    analysis_image = image
    if color_mask is not None:
        # Use the original photo's matching ink, including blue strokes that
        # grayscale darkness thresholds or the old pale-blue heuristic erased.
        ink = color_mask
        analysis_image = image_with_component_ink(reference_image, color_profile)
        gray = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2GRAY)
    if local_ink_policy['active']:
        from backend.paper_ink import conductor_mask
        ink=conductor_mask(reference_image,local_ink_policy)
        analysis_image=cv2.cvtColor(255-ink,cv2.COLOR_GRAY2BGR)
        gray=cv2.cvtColor(analysis_image,cv2.COLOR_BGR2GRAY)
        color_profile=dict(active=False,chromatic=False,reason='paper_relative_strokes_replace_fragmented_global_color',original_profile=color_profile)
    original_color_ink=ink.copy() if rule_profile['active'] and color_mask is not None else None
    if is_digital:rulings_removed=0
    else:
        ink, rulings_removed = _remove_pale_page_rulings(
            analysis_image, gray, ink, preserve_reference_color=color_mask is not None,
        )
    if original_color_ink is not None:
        # The old brightness-only ruling pass can erase a light pencil even
        # when it is darker than the page rule. Restore only the reliable
        # component color, then apply the specific page-color exclusions.
        protect=protected_component_ink_mask(reference_image,color_profile)
        ink[protect]=original_color_ink[protect]
    if is_digital:border_artifacts_removed=0
    else:ink, border_artifacts_removed = _remove_border_frame_artifacts(gray, ink)
    # Apply the shared page-rule exclusions before geometry and component
    # masking. Local paper normalization rejects shadowed blank paper that
    # otherwise falls inside a loose global component-color tolerance.
    if rule_profile['active']:
        _,_,normalized=_paper_colors(reference_image)
        ink[normalized.mean(axis=2)>=242]=0
        ink[ruling_mask>0]=0
    cleaned, erased_mask, mask_meta = _mask_components_and_text(
        ink,
        components_info,
        ocr_info,
        mask_components=mask_components,
        mask_text=mask_text,
    )
    cleaned, roi_meta = _restrict_to_component_roi(cleaned, components_info)
    wire_mask = _filter_small_blobs(cleaned,min_area=2) if is_digital else _prepare_wire_mask(cleaned)
    if is_digital:min_segment_length=min(min_segment_length,2.)
    wire_mask, proximity_meta = _filter_wire_candidates_by_components(wire_mask, components_info)
    wire_mask, terminal_seed_meta = _filter_wire_candidates_by_terminal_seeds(wire_mask, components_info)
    skeleton = _skeletonize(wire_mask)
    nodes, segments, graph_meta = _extract_graph(skeleton, min_segment_length=min_segment_length)

    total_length = round(sum(float(s.get("length_px") or 0.0) for s in segments), 2)
    endpoint_count = sum(1 for n in nodes if n.get("kind_candidate") == "endpoint_candidate")
    branch_candidate_count = sum(1 for n in nodes if n.get("kind_candidate") == "junction_candidate")

    result = {
        "schema_version": WIRE_SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_image": source_image,
        "image_width": int(w),
        "image_height": int(h),
        "method": "opencv_mask_morphology_skeleton_graph",
        "scope": "stage8_raw_wires_only",
        "ink_color_profile": color_profile,
        "conductor_ink_policy": local_ink_policy,
        "notebook_rule_profile": rule_profile,
        "stage9_note": "junction_candidate e cruzamentos ainda não são conexões elétricas confirmadas; isso pertence à Etapa 9.",
        "parameters": {
            "mask_components": bool(mask_components),
            "mask_text": bool(mask_text),
            "min_segment_length_px": float(min_segment_length),
            "connectivity": 8,
        },
        "preprocessing": {
            **prep_meta,
            **mask_meta,
            **roi_meta,
            **proximity_meta,
            **terminal_seed_meta,
            "component_ink_color_filter_active": bool(color_profile["active"]),
            "notebook_rule_color_filter_active": bool(rule_profile["active"]),
            "notebook_rule_color_pixels_removed": int(rule_profile["removed_pixel_count"]),
            "pale_page_rulings_removed": int(rulings_removed),
            "border_frame_artifacts_removed": int(border_artifacts_removed),
            "ink_pixels_before_masking": int(cv2.countNonZero(ink)),
            "wire_mask_pixels": int(cv2.countNonZero(wire_mask)),
            "erased_mask_pixels": int(cv2.countNonZero(erased_mask)),
        },
        "segment_count": len(segments),
        "node_count": len(nodes),
        "endpoint_candidate_count": int(endpoint_count),
        "junction_candidate_count": int(branch_candidate_count),
        "total_wire_length_px": total_length,
        "nodes": nodes,
        "segments": segments,
        "graph_debug": graph_meta,
    }
    return result, wire_mask, skeleton, erased_mask


def salvar_imagem_fios(
    image: np.ndarray,
    result: dict,
    output_path: Path,
) -> None:
    canvas = image.copy()
    # Draw segments first so endpoints/nodes remain visible.
    for seg in result.get("segments") or []:
        pts = seg.get("points") or []
        if len(pts) >= 2:
            arr = np.array([[int(p["x"]), int(p["y"])] for p in pts], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [arr], False, (40, 190, 40), 2, cv2.LINE_AA)
            mid = pts[len(pts) // 2]
            cv2.putText(
                canvas,
                str(seg.get("wire_id", "")),
                (int(mid["x"]) + 3, int(mid["y"]) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (20, 120, 20),
                1,
                cv2.LINE_AA,
            )
    for node in result.get("nodes") or []:
        pos = node.get("position") or {}
        x, y = int(round(pos.get("x", 0))), int(round(pos.get("y", 0)))
        if node.get("kind_candidate") == "endpoint_candidate":
            color = (20, 20, 230)
            r = 4
        else:
            color = (230, 80, 20)
            r = 5
        cv2.circle(canvas, (x, y), r, color, -1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise IOError(f"Não foi possível salvar {output_path}")


def salvar_mascara(mask: np.ndarray, output_path: Path, *, invert_for_view: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = ((mask > 0).astype(np.uint8) * 255)
    if invert_for_view:
        img = 255 - img
    if not cv2.imwrite(str(output_path), img):
        raise IOError(f"Não foi possível salvar {output_path}")
