"""Learn the circuit's ink from reliable symbols, before drawing any overlays."""
from __future__ import annotations

import cv2
import numpy as np
from backend.resistor_geometry import cached_vision


COLOR_SCHEMA_VERSION = "component-ink-color-1.0"
MIN_COMPONENT_CONFIDENCE = 0.80


def _components(info: dict | None) -> list[dict]:
    if not isinstance(info, dict):
        return []
    return [c for c in (info.get("components") or info.get("detections") or [])
            if isinstance(c, dict) and isinstance(c.get("bbox"), dict)]


def _reliable(component: dict) -> bool:
    # Terminal/OCR uncertainty is separate from visual symbol confidence.
    if component.get("manual_reviewed") is True:
        return True
    if component.get("visual_review_required", component.get("review_required", False)):
        return False
    confidence = next((component[k] for k in
                       ("visual_confidence", "class_confidence", "confidence")
                       if component.get(k) is not None), None)
    try:
        return np.isfinite(float(confidence)) and float(confidence) >= MIN_COMPONENT_CONFIDENCE
    except (TypeError, ValueError):
        return False


def _hue_distance(a, b):
    distance = np.abs(a - b)
    return np.minimum(distance, 180.0 - distance)


@cached_vision
def estimate_component_ink(image: np.ndarray, components_info: dict | None) -> dict:
    """Average one robust ink sample per symbol, so a large bbox cannot dominate.

    Paper, long page rulings and antialiased fringes are excluded from samples.
    A consensus is required; incompatible pen colors are never averaged together.
    """
    profile = {"schema_version": COLOR_SCHEMA_VERSION, "active": False,
               "reason": "no_reliable_component_samples", "sampled_component_count": 0,
               "minimum_component_confidence": MIN_COMPONENT_CONFIDENCE}
    from backend.drawing_mode import digital
    if digital():return dict(profile,reason='explicit_digital_input')
    if image.ndim != 3 or image.shape[2] != 3:
        profile["reason"] = "color_image_required"
        return profile
    reliable = [c for c in _components(components_info) if _reliable(c)]
    profile["reliable_component_count"] = len(reliable)
    if not reliable:
        return profile
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Used only to exclude page lines from reference samples, not from wires.
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(7.0, min(h, w) / 55.0))
    foreground = (gray.astype(np.float32) < background.astype(np.float32) - 10).astype(np.uint8) * 255
    rulings = cv2.morphologyEx(foreground, cv2.MORPH_OPEN,
                             np.ones((1, max(40, int(w * 0.55))), np.uint8))
    rulings = cv2.dilate(rulings, np.ones((3, 1), np.uint8))
    vertical = cv2.morphologyEx(foreground, cv2.MORPH_OPEN,
                              np.ones((max(40, int(h * 0.55)), 1), np.uint8))
    rulings |= cv2.dilate(vertical, np.ones((1, 3), np.uint8))
    # Photos may rotate the notebook; rule sampling must handle tilted lines.
    long_lines = cv2.HoughLinesP(foreground, 1, np.pi / 180, threshold=45,
                                minLineLength=max(40, int(min(h, w) * 0.55)), maxLineGap=20)
    if long_lines is not None:
        for x1, y1, x2, y2 in np.asarray(long_lines).reshape(-1, 4):
            if abs(x2 - x1) >= 0.55 * w or abs(y2 - y1) >= 0.55 * h:
                cv2.line(rulings, (int(x1), int(y1)), (int(x2), int(y2)), 255, 7)
    samples = []
    for component in reliable:
        bbox = component["bbox"]
        try:
            values = [float(bbox[k]) for k in ("x1", "y1", "x2", "y2")]
            if not all(np.isfinite(v) for v in values):
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in values]
        except (KeyError, TypeError, ValueError):
            continue
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        roi = image[y1:y2, x1:x2]
        roi_gray = gray[y1:y2, x1:x2]
        paper = float(np.percentile(roi_gray, 90))
        stroke = ((roi_gray <= min(215.0, paper - 25.0))
                  & (rulings[y1:y2, x1:x2] == 0)).astype(np.uint8) * 255
        # Prefer the cores of strokes; thin pen strokes still have a fallback.
        core = cv2.erode(stroke, np.ones((3, 3), np.uint8))
        if cv2.countNonZero(core) >= 12:
            stroke = core
        pixels = roi[stroke > 0].astype(np.float32)
        colors = hsv[y1:y2, x1:x2][stroke > 0].astype(np.float32)
        if len(pixels) < 12:
            continue
        chromatic = (colors[:, 1] >= 30) & (np.ptp(pixels, axis=1) >= 12)
        # Find a local dominant hue instead of averaging blue with a black ruling.
        counts = np.bincount((colors[chromatic, 0] // 10).astype(int), minlength=18)
        smooth = counts + np.roll(counts, 1) + np.roll(counts, -1)
        hue_bin = int(np.argmax(smooth))
        colored = chromatic & (_hue_distance(colors[:, 0], hue_bin * 10 + 5) <= 17)
        neutral = ~chromatic
        use_color = int(np.count_nonzero(colored)) > int(np.count_nonzero(neutral))
        chosen = colored if use_color else neutral
        if np.count_nonzero(chosen) < max(12, len(pixels) * 0.55):
            continue
        selected = pixels[chosen]
        median = np.median(selected, axis=0)
        deviation = np.max(np.abs(selected - median), axis=1)
        selected = selected[deviation <= max(12.0, float(np.percentile(deviation, 75)))]
        center = np.mean(selected, axis=0)
        center_hsv = cv2.cvtColor(np.uint8([[np.rint(center)]]), cv2.COLOR_BGR2HSV)[0, 0]
        samples.append({"component_id": str(component.get("component_id") or component.get("detection_id") or ""),
                        "bgr": center, "hsv": center_hsv.astype(float), "chromatic": use_color,
                        "spread": float(np.median(np.max(np.abs(selected - center), axis=1)))})
    profile["sampled_component_count"] = len(samples)
    if not samples:
        return profile
    # Select the consensus of symbol colors and reject atypical components.
    groups = []
    for anchor in samples:
        group = [s for s in samples if s["chromatic"] == anchor["chromatic"]
                 and np.max(np.abs(s["bgr"] - anchor["bgr"])) <= 48
                 and (not s["chromatic"] or _hue_distance(s["hsv"][0], anchor["hsv"][0]) <= 14)]
        groups.append(group)
    selected = max(groups, key=len)
    if len(selected) < len(samples) * 0.60:
        profile["reason"] = "component_colors_without_consensus"
        return profile
    centers = np.array([s["bgr"] for s in selected])
    median = np.median(centers, axis=0)
    deviations = np.max(np.abs(centers - median), axis=1)
    robust = deviations <= max(12.0, float(np.median(deviations)) * 3.0)
    selected = [s for s, keep in zip(selected, robust) if keep]
    centers = centers[robust]
    if len(selected) < len(samples) * 0.60:
        profile["reason"] = "component_colors_without_consensus"
        return profile
    bgr = np.mean(centers, axis=0)
    color = np.uint8([[np.rint(bgr)]])
    center_hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0, 0].astype(float)
    between = float(np.median(np.max(np.abs(centers - bgr), axis=1)))
    spread = float(np.median([s["spread"] for s in selected]))
    chromatic = selected[0]["chromatic"]
    tolerance = min(65.0 if chromatic else 35.0,
                    max(32.0 if chromatic else 22.0, 3 * (between + spread) + 12))
    rgb = [int(v) for v in np.rint(bgr[::-1])]
    profile.update({"active": True, "reason": "reliable_component_color_consensus",
                    "reference_rgb": rgb, "reference_hex": "#" + "".join(f"{v:02X}" for v in rgb),
                    "reference_hsv": [round(float(v), 2) for v in center_hsv],
                    "chromatic": bool(chromatic), "channel_tolerance": round(tolerance, 2),
                    "hue_tolerance": 12.0, "consensus_component_count": len(selected),
                    "consensus_component_ids": [s["component_id"] for s in selected],
                    "consensus_ratio": round(len(selected) / len(samples), 4),
                    "averaging": "equal_weight_per_component_trimmed_stroke_mean"})
    return profile


@cached_vision
def component_ink_mask(image: np.ndarray, profile: dict) -> np.ndarray | None:
    """Compare each pixel to the saved RGB code with tolerance for camera noise."""
    if not profile.get("active") or image.ndim != 3 or image.shape[2] != 3:
        return None
    bgr = np.array(profile["reference_rgb"][::-1], dtype=np.float32)
    values = image.astype(np.float32)
    match = np.max(np.abs(values - bgr), axis=2) <= float(profile["channel_tolerance"])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    if profile["chromatic"]:
        hue, saturation, _ = profile["reference_hsv"]
        match &= (_hue_distance(hsv[:, :, 0], hue) <= profile["hue_tolerance"])
        match &= (hsv[:, :, 1] >= max(25.0, saturation - 65.0))
    else:
        # Also distinguish near-black blue ink from achromatic black rulings.
        match &= (hsv[:, :, 1] < 30) | (np.ptp(values, axis=2) < 12)
    return match.astype(np.uint8) * 255


@cached_vision
def image_with_component_ink(image: np.ndarray, profile: dict) -> np.ndarray:
    mask = component_ink_mask(image, profile)
    if mask is None:
        return image
    out = image.copy()
    out[mask == 0] = 255
    return out
