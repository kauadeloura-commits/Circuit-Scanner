"""Retain photographed pen hue when exposure changes its absolute RGB value."""
import cv2
import numpy as np
from backend.notebook_rulings import _paper_colors


def local_color_mask(image, profile, ruling_mask):
    """Return observed pixels only; never bridge a gap or extend a stroke."""
    if image.ndim != 3 or not profile.get('active') or not profile.get('chromatic'):
        return np.zeros(image.shape[:2], np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = float(profile['reference_hsv'][0])
    delta = np.abs(hsv[:, :, 0].astype(float) - hue)
    same_hue = np.minimum(delta, 180 - delta) <= float(profile.get('hue_tolerance', 12))
    _, _, normalized = _paper_colors(image)
    # Camera exposure can desaturate the middle of a blue/violet stroke.
    # Retain such neutral pixels only on an observed continuous ink island
    # that also contains the sampled pen hue. Unrelated gray marks and page
    # rules cannot seed a recovered conductor.
    colored = same_hue & (hsv[:, :, 1] >= 20)
    weak = (colored | (hsv[:, :, 1] < 25)) & (normalized.mean(axis=2) < 242) & (ruling_mask == 0)
    count, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    seeds = np.unique(labels[colored & weak & (normalized.mean(axis=2) < 210)])
    keep = np.zeros(count, np.uint8)
    keep[seeds] = 1
    keep[0] = 0
    return keep[labels]
