import unittest

import cv2
import numpy as np

from backend.wire_detection import detectar_fios


class WireDetectionStage8Tests(unittest.TestCase):
    def test_straight_wire_produces_segment_and_endpoints(self):
        img = np.full((180, 280, 3), 255, np.uint8)
        cv2.line(img, (30, 90), (250, 90), (20, 20, 20), 4)
        result, mask, skeleton, _ = detectar_fios(img, components_info={}, ocr_info={}, min_segment_length=10)
        self.assertGreaterEqual(result["segment_count"], 1)
        self.assertGreaterEqual(result["endpoint_candidate_count"], 2)
        self.assertGreater(int(np.count_nonzero(skeleton)), 100)

    def test_t_branch_is_only_a_stage9_candidate(self):
        img = np.full((240, 320, 3), 255, np.uint8)
        cv2.line(img, (30, 100), (280, 100), (20, 20, 20), 4)
        cv2.line(img, (160, 100), (160, 205), (20, 20, 20), 4)
        result, *_ = detectar_fios(img, components_info={}, ocr_info={}, min_segment_length=10)
        self.assertGreaterEqual(result["segment_count"], 3)
        self.assertGreaterEqual(result["junction_candidate_count"], 1)
        self.assertTrue(all(s["semantic_status"] == "unresolved_until_stage9" for s in result["segments"]))

    def test_component_body_is_masked_but_external_leads_remain(self):
        img = np.full((180, 320, 3), 255, np.uint8)
        # external leads + resistor-like body
        cv2.line(img, (20, 90), (120, 90), (20, 20, 20), 4)
        cv2.rectangle(img, (120, 70), (200, 110), (20, 20, 20), 4)
        cv2.line(img, (200, 90), (300, 90), (20, 20, 20), 4)
        components = {
            "components": [{
                "component_id": "R1",
                "type": "resistor",
                "bbox": {"x1": 115, "y1": 65, "x2": 205, "y2": 115},
                "terminals": [],
            }]
        }
        result, mask, *_ = detectar_fios(img, components_info=components, ocr_info={}, min_segment_length=10)
        self.assertEqual(int(mask[90, 160]), 0, "corpo do resistor não deve virar fio")
        self.assertGreater(int(np.count_nonzero(mask[:, :110])), 20)
        self.assertGreater(int(np.count_nonzero(mask[:, 210:])), 20)
        self.assertGreaterEqual(result["segment_count"], 2)

    def test_blue_notebook_ruling_is_suppressed_and_dark_wire_survives(self):
        img = np.full((220, 360, 3), 235, np.uint8)
        # BGR pale blue ruling, full width.
        cv2.line(img, (0, 70), (359, 70), (210, 190, 185), 2)
        # dark schematic wire crossing it vertically
        cv2.line(img, (180, 25), (180, 190), (25, 25, 25), 4)
        result, mask, *_ = detectar_fios(img, components_info={}, ocr_info={}, min_segment_length=10)
        row_nonzero = int(np.count_nonzero(mask[70]))
        col_nonzero = int(np.count_nonzero(mask[:, 180]))
        self.assertLess(row_nonzero, 40, "linha pautada não deve permanecer como fio longo")
        self.assertGreater(col_nonzero, 100, "fio escuro deve sobreviver à remoção da pauta")
        self.assertGreaterEqual(result["segment_count"], 1)


if __name__ == "__main__":
    unittest.main()
