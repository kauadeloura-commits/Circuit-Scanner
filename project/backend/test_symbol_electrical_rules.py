import random
import unittest

import cv2
import numpy as np

from backend.dataset_generator import _bbox_from_mask, _draw_symbol
from backend.terminal_detection import detectar_terminais_componentes


class SymbolElectricalRuleRegressionTests(unittest.TestCase):
    def test_vertical_diode_bar_defines_cathode_bottom(self):
        gray, mask = _draw_symbol("diode", 256, random.Random(4411))
        gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        x1, y1, x2, y2 = _bbox_from_mask(mask)
        bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": x2 - x1, "height": y2 - y1}
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        data = {"schema_version": "test", "components": [{
            "component_id": "D1", "type": "diode", "bbox": bbox,
        }]}
        comp = detectar_terminais_componentes(data, 256, 256, image)["components"][0]
        pins = {p["pin"]: p["position"] for p in comp["terminals"]}
        self.assertEqual(comp["orientation"]["axis"], "vertical")
        self.assertLess(pins["A"]["y"], pins["K"]["y"])

    def _transistor_with_arrow(self, arrow_branch):
        img = np.full((220, 220, 3), 245, np.uint8)
        cv2.line(img, (95, 65), (95, 155), (10, 10, 10), 4)
        cv2.line(img, (45, 110), (95, 110), (10, 10, 10), 3)
        cv2.line(img, (95, 90), (150, 50), (10, 10, 10), 3)
        cv2.line(img, (95, 130), (150, 170), (10, 10, 10), 3)
        if arrow_branch == "upper":
            cv2.line(img, (132, 63), (120, 66), (10, 10, 10), 3)
            cv2.line(img, (132, 63), (126, 75), (10, 10, 10), 3)
        else:
            cv2.line(img, (132, 157), (120, 154), (10, 10, 10), 3)
            cv2.line(img, (132, 157), (126, 145), (10, 10, 10), 3)
        bbox = {"x1": 40, "y1": 45, "x2": 160, "y2": 180,
                "width": 120, "height": 135}
        data = {"schema_version": "test", "components": [{
            "component_id": "Q1", "type": "transistor_pnp", "bbox": bbox,
        }]}
        return detectar_terminais_componentes(data, 220, 220, img)["components"][0]

    def test_upper_arrow_branch_is_emitter(self):
        comp = self._transistor_with_arrow("upper")
        pins = {p["pin"]: p["position"] for p in comp["terminals"]}
        self.assertEqual(comp["transistor_geometry"].get("arrow_branch"), "upper")
        self.assertLess(pins["E"]["y"], pins["C"]["y"])

    def test_lower_arrow_branch_is_emitter(self):
        comp = self._transistor_with_arrow("lower")
        pins = {p["pin"]: p["position"] for p in comp["terminals"]}
        self.assertEqual(comp["transistor_geometry"].get("arrow_branch"), "lower")
        self.assertGreater(pins["E"]["y"], pins["C"]["y"])


if __name__ == "__main__":
    unittest.main()
