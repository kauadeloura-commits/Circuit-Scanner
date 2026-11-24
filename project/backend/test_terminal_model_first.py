import unittest
from unittest.mock import patch

import numpy as np

from backend import terminal_detection as td


class PhysicalEvidenceThenModelFallbackTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((220, 260, 3), 245, np.uint8)
        self.info = {
            "schema_version": "test",
            "components": [{
                "component_id": "Q1",
                "type": "transistor_pnp",
                "bbox": {"x1": 70, "y1": 50, "x2": 180, "y2": 180, "width": 110, "height": 130},
            }],
        }

    def test_loadable_model_is_used_when_physical_fork_is_not_confirmed(self):
        diag = {
            "accepted": True,
            "review_required": False,
            "points": {
                "B": {"x": 101.25, "y": 112.75, "confidence": 0.88},
                "C": {"x": 161.50, "y": 72.25, "confidence": 0.83},
                "E": {"x": 164.75, "y": 158.50, "confidence": 0.91},
            },
            "box_confidence": 0.77,
            "keypoint_confidence": 0.873,
            "keypoint_confidences": {"B": 0.88, "C": 0.83, "E": 0.91},
        }
        runtime = {"model_exists": True, "model_loadable": True, "model_load_error": None}
        with patch.object(td, "terminal_model_runtime_status", return_value=runtime), \
             patch.object(td, "diagnose_transistor_keypoints", return_value=diag), \
             patch.object(td, "_detect_transistor_fork", return_value=({}, {
                 "success": False, "branch_distinct": False, "source": "no_physical_fork"
             })):
            out = td.detectar_terminais_componentes(self.info, 260, 220, self.image)
        comp = out["components"][0]
        pins = {p["pin"]: p for p in comp["terminals"]}
        self.assertEqual(pins["B"]["position"], {"x": 101.25, "y": 112.75})
        self.assertEqual(pins["C"]["position"], {"x": 161.5, "y": 72.25})
        self.assertEqual(pins["E"]["position"], {"x": 164.75, "y": 158.5})
        self.assertTrue(all(p["source"] == "trained_terminal_keypoint_model" for p in pins.values()))
        self.assertEqual(comp["transistor_geometry"]["source"], "trained_terminal_keypoint_model")

    def test_rejected_model_does_not_invent_points_when_physical_fork_failed(self):
        diag = {
            "accepted": False,
            "review_required": True,
            "rejection_reasons": ["media_keypoints_muito_baixa"],
            "points": {},
        }
        runtime = {"model_exists": True, "model_loadable": True, "model_load_error": None}
        with patch.object(td, "terminal_model_runtime_status", return_value=runtime), \
             patch.object(td, "diagnose_transistor_keypoints", return_value=diag), \
             patch.object(td, "_detect_transistor_fork", return_value=({}, {
                 "success": False, "branch_distinct": False, "source": "no_physical_fork"
             })):
            out = td.detectar_terminais_componentes(self.info, 260, 220, self.image)
        comp = out["components"][0]
        self.assertEqual(comp["terminals"], [])
        self.assertEqual(comp["transistor_geometry"]["source"], "trained_terminal_keypoint_model_rejected")
        self.assertTrue(comp["terminal_detection"]["review_required"])
        self.assertIn("modelo_keypoints_rejeitou_previsao", comp["terminal_detection"]["review_reasons"])


if __name__ == "__main__":
    unittest.main()
