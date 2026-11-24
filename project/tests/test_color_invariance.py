from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.recognition import (  # noqa: E402
    _deteccoes_gnd_geometrico,
    _deteccoes_triangulos_preenchidos,
)


def _canvas(background, foreground):
    image = np.full((520, 760, 3), background, dtype=np.uint8)
    for y, direction in ((80, 1), (190, -1), (300, 1), (410, -1)):
        if direction == 1:
            triangle = np.array([[180, y - 25], [180, y + 25], [230, y]])
            cv2.line(image, (235, y - 31), (235, y + 31), foreground, 5)
        else:
            triangle = np.array([[230, y - 25], [230, y + 25], [180, y]])
            cv2.line(image, (175, y - 31), (175, y + 31), foreground, 5)
        cv2.fillPoly(image, [triangle], foreground)
    return image


def test_four_filled_diodes_survive_color_and_background_changes():
    for background, foreground in (
        ((255, 255, 255), (35, 110, 65)),
        ((28, 46, 76), (245, 188, 55)),
        ((228, 214, 240), (125, 35, 185)),
    ):
        detections = _deteccoes_triangulos_preenchidos(
            _canvas(background, foreground)
        )
        assert len(detections) == 4
        assert {item["class_id"] for item in detections} == {3}


def test_gnd_is_recognized_by_line_geometry():
    image = np.full((300, 300, 3), (240, 225, 205), dtype=np.uint8)
    color = (120, 35, 175)
    cv2.line(image, (150, 60), (150, 125), color, 5)
    cv2.line(image, (105, 125), (195, 125), color, 5)
    cv2.line(image, (120, 145), (180, 145), color, 5)
    cv2.line(image, (137, 165), (163, 165), color, 5)
    detections = _deteccoes_gnd_geometrico(image)
    assert detections
    assert {item["class_id"] for item in detections} == {8}
