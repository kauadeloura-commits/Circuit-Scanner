import unittest
from unittest.mock import patch

from backend import component_diagnostics
from backend.recognition import (
    consolidar_multiview,
    remover_caixas_englobantes_multicomponente,
)


def detection(class_id, name, confidence, x1, y1, x2, y2, source):
    return {
        "detection_id": "DET_001",
        "class_id": class_id,
        "class": name,
        "confidence": confidence,
        "bbox": {
            "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
            "width": float(x2 - x1), "height": float(y2 - y1),
            "center_x": float((x1 + x2) / 2),
            "center_y": float((y1 + y2) / 2),
        },
        "inference_sources": [source],
    }


class ComponentDiagnosticTests(unittest.TestCase):
    def test_large_box_does_not_merge_two_small_components(self):
        raw = [
            detection(3, "diode", 0.86, 10, 10, 210, 110, "original_color"),
            detection(0, "resistor", 0.82, 25, 35, 75, 75, "original_color"),
            detection(1, "capacitor", 0.80, 145, 35, 185, 75, "original_color"),
        ]
        merged = consolidar_multiview(raw)
        self.assertEqual(len(merged), 3)

    def test_large_box_containing_two_detections_is_rejected(self):
        boxes = [
            detection(3, "diode", 0.86, 10, 10, 210, 110, "original_color"),
            detection(0, "resistor", 0.82, 25, 35, 75, 75, "original_color"),
            detection(1, "capacitor", 0.80, 145, 35, 185, 75, "original_color"),
        ]
        kept, rejected = remover_caixas_englobantes_multicomponente(boxes)
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(
            rejected[0]["rejection_reason"],
            "oversized_box_contains_multiple_components",
        )

    def test_annotation_audit_reports_large_and_duplicate_boxes(self):
        records = [{
            "image_id": "REAL_0001",
            "width": 1000,
            "height": 800,
            "annotations_reviewed": True,
            "annotations": [
                {"class_id": 0, "x1": 20, "y1": 20, "x2": 800, "y2": 500},
                {"class_id": 0, "x1": 22, "y1": 22, "x2": 798, "y2": 498},
            ],
        }]
        classes = [{"id": 0, "key": "resistor", "name": "Resistor"}]
        with (
            patch.object(component_diagnostics, "list_real_images", return_value=records),
            patch.object(component_diagnostics, "load_classes", return_value=classes),
        ):
            report = component_diagnostics.audit_real_annotations()
        self.assertGreaterEqual(report["issue_count"], 3)
        self.assertEqual(report["possible_duplicate_annotation_pairs"], 1)
        reasons = {
            reason
            for issue in report["issues"]
            for reason in issue["reasons"]
        }
        self.assertIn("box_covers_more_than_12_percent_of_image", reasons)


if __name__ == "__main__":
    unittest.main()
