import unittest

import cv2
import numpy as np

from backend import terminal_detection as td


class NpnEmitterRuleTests(unittest.TestCase):
    @staticmethod
    def _draw_npn(fork_side: str):
        image = np.full((260, 260, 3), 255, dtype=np.uint8)
        bbox = {"x1": 70, "y1": 35, "x2": 180, "y2": 225, "width": 110, "height": 190}
        base_x = 112 if fork_side == "right" else 138
        cv2.rectangle(image, (base_x - 5, 86), (base_x + 5, 174), (20, 20, 20), -1)
        if fork_side == "right":
            cv2.line(image, (76, 130), (base_x - 5, 130), (20, 20, 20), 4, cv2.LINE_AA)
            cv2.line(image, (base_x + 5, 108), (166, 70), (20, 20, 20), 4, cv2.LINE_AA)
            cv2.line(image, (base_x + 5, 152), (166, 190), (20, 20, 20), 4, cv2.LINE_AA)
        else:
            cv2.line(image, (base_x + 5, 130), (176, 130), (20, 20, 20), 4, cv2.LINE_AA)
            cv2.line(image, (base_x - 5, 108), (84, 70), (20, 20, 20), 4, cv2.LINE_AA)
            cv2.line(image, (base_x - 5, 152), (84, 190), (20, 20, 20), 4, cv2.LINE_AA)
        data = {"schema_version": "test", "components": [{
            "component_id": "Q1", "type": "transistor_npn", "bbox": bbox,
            "review_required": False, "review_reasons": [],
        }]}
        component = td.detectar_terminais_componentes(data, 260, 260, image)["components"][0]
        return component

    def test_right_facing_npn_uses_lower_right_as_emitter(self):
        component = self._draw_npn("right")
        pins = {item["pin"]: item["position"] for item in component["terminals"]}
        self.assertEqual(component["transistor_geometry"]["npn_emitter_side"], "lower_right")
        self.assertGreater(pins["E"]["x"], pins["B"]["x"])
        self.assertGreater(pins["E"]["y"], pins["C"]["y"])

    def test_mirrored_npn_uses_lower_left_as_emitter(self):
        component = self._draw_npn("left")
        pins = {item["pin"]: item["position"] for item in component["terminals"]}
        self.assertEqual(component["transistor_geometry"]["npn_emitter_side"], "lower_left")
        self.assertLess(pins["E"]["x"], pins["B"]["x"])
        self.assertGreater(pins["E"]["y"], pins["C"]["y"])

    def test_learned_horizontal_prediction_cannot_invert_collector_emitter(self):
        bbox = {"x1": 50, "y1": 40, "x2": 190, "y2": 210, "width": 140, "height": 170}
        inverted = {"B": (100.0, 125.0), "C": (170.0, 180.0), "E": (168.0, 70.0)}
        corrected, audit = td._enforce_horizontal_npn_emitter_rule(inverted, bbox)
        self.assertTrue(audit["applied"])
        self.assertTrue(audit["swapped"])
        self.assertGreater(corrected["E"][1], corrected["C"][1])


if __name__ == "__main__":
    unittest.main()
