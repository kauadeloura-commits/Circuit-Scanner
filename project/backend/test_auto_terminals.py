import json
import shutil
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.main import (
    PROJECTS_DIR,
    _components_info_from_detections,
    _gerar_terminais_automaticamente,
)


class AutomaticTerminalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.project_id = "TEST_AUTO_TERMINALS"
        self.project_dir = PROJECTS_DIR / self.project_id
        shutil.rmtree(self.project_dir, ignore_errors=True)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        image = np.full((220, 420, 3), 255, dtype=np.uint8)
        cv2.line(image, (100, 110), (320, 110), (0, 0, 0), 3)
        cv2.rectangle(image, (170, 85), (250, 135), (0, 0, 0), 3)
        cv2.imwrite(str(self.project_dir / "original.png"), image)

    def tearDown(self):
        shutil.rmtree(self.project_dir, ignore_errors=True)


    def test_detection_edit_overwrites_old_terminal_result(self):
        projeto = {"stored_filename": "original.png"}
        first = {
            "schema_version": "stage5-test",
            "detections": [
                {
                    "detection_id": "DET_001",
                    "class": "resistor",
                    "confidence": 0.9,
                    "bbox": {"x1": 165, "y1": 80, "x2": 255, "y2": 140, "width": 90, "height": 60},
                }
            ],
        }
        first_result = _gerar_terminais_automaticamente(
            self.project_id,
            projeto,
            _components_info_from_detections(first),
            trigger="automatic_detection",
        )
        self.assertEqual(first_result["count"], 1)
        self.assertEqual(first_result["terminal_count"], 2)

        corrected = {
            "schema_version": "stage5-test",
            "detections": [
                first["detections"][0],
                {
                    "detection_id": "DET_002",
                    "class": "capacitor",
                    "confidence": 1.0,
                    "bbox": {"x1": 280, "y1": 80, "x2": 340, "y2": 140, "width": 60, "height": 60},
                },
            ],
        }
        corrected_result = _gerar_terminais_automaticamente(
            self.project_id,
            projeto,
            _components_info_from_detections(corrected),
            trigger="manual_detection_edit",
        )
        self.assertEqual(corrected_result["count"], 2)
        # Only one observed wire reaches the added box. Retain that contact
        # without inventing its opposite terminal or erasing the first one.
        self.assertEqual(corrected_result["terminal_count"], 3)
        added=corrected_result['components'][1]['terminals']
        self.assertEqual(len(added),1)
        self.assertEqual(added[0]['position']['y'],110)
        self.assertTrue(added[0]['wire_contact_confirmed'])
        self.assertEqual(corrected_result["generation_trigger"], "manual_detection_edit")
        self.assertEqual(corrected_result["generated_from"], "detections")
        self.assertIn("automatic_stage8", corrected_result)

        saved = json.loads((self.project_dir / "terminals.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["count"], 2)
        self.assertEqual(saved["terminal_count"], 3)
        self.assertEqual(
            {item["detection_id"] for item in saved["components"]},
            {"DET_001", "DET_002"},
        )

    def test_ocr_trigger_generates_json_and_preview(self):
        projeto = {"stored_filename": "original.png"}
        components = {
            "schema_version": "stage6-test",
            "components": [
                {
                    "component_id": "R1",
                    "type": "resistor",
                    "bbox": {"x1": 165, "y1": 80, "x2": 255, "y2": 140, "width": 90, "height": 60},
                    "reference": {"text": "R1", "source": "ocr"},
                    "value": {"text": "1k", "source": "ocr"},
                }
            ],
        }
        result = _gerar_terminais_automaticamente(
            self.project_id,
            projeto,
            components,
            trigger="ocr",
        )
        self.assertTrue(result["auto_generated_after_ocr"])
        self.assertEqual(result["generation_trigger"], "ocr")
        self.assertEqual(result["terminal_count"], 2)
        self.assertIn("automatic_stage8", result)
        self.assertTrue((self.project_dir / "terminals.png").exists())
        self.assertTrue((self.project_dir / "terminals.json").exists())
        saved = json.loads((self.project_dir / "terminals.json").read_text(encoding="utf-8"))
        self.assertTrue(saved["auto_generated_after_ocr"])


if __name__ == "__main__":
    unittest.main()
