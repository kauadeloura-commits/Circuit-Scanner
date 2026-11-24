import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend import generic_terminal_learning as gtl
from backend import main
from backend.dataset_generator import CLASSES


class GenericTerminalLearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_data = gtl.GENERIC_TERMINAL_DATA_DIR
        self.old_raw = gtl.GENERIC_RAW_DIR
        self.old_registry = gtl.GENERIC_REGISTRY_PATH
        gtl.GENERIC_TERMINAL_DATA_DIR = self.root / "terminal_training_all_components"
        gtl.GENERIC_RAW_DIR = gtl.GENERIC_TERMINAL_DATA_DIR / "raw"
        gtl.GENERIC_REGISTRY_PATH = gtl.GENERIC_TERMINAL_DATA_DIR / "classes.json"
        image = np.full((160, 240, 3), 255, np.uint8)
        cv2.rectangle(image, (80, 60), (160, 100), (0, 0, 0), 2)
        self.image_path = self.root / "circuit.png"
        cv2.imwrite(str(self.image_path), image)

    def tearDown(self):
        gtl.GENERIC_TERMINAL_DATA_DIR = self.old_data
        gtl.GENERIC_RAW_DIR = self.old_raw
        gtl.GENERIC_REGISTRY_PATH = self.old_registry
        self.tmp.cleanup()

    def test_every_dataset_class_has_default_editable_pin_schema(self):
        dataset_classes = {key for _class_id, key, _name in CLASSES}
        self.assertEqual(dataset_classes - set(main.MANUAL_PIN_SCHEMAS), set())

    def test_all_component_sample_is_saved_and_recorrection_replaces_it(self):
        component = {
            "component_id": "R1",
            "type": "resistor",
            "bbox": {"x1": 75, "y1": 50, "x2": 165, "y2": 110},
        }
        schema = [{"pin": "1", "role": "terminal_1"}, {"pin": "2", "role": "terminal_2"}]
        first = gtl.save_generic_terminal_training_sample(
            image_path=self.image_path,
            project_id="PRJ_TEST",
            component=component,
            points={"1": {"x": 60, "y": 80}, "2": {"x": 180, "y": 80}},
            pin_schema=schema,
        )
        second = gtl.save_generic_terminal_training_sample(
            image_path=self.image_path,
            project_id="PRJ_TEST",
            component=component,
            points={"1": {"x": 58, "y": 80}, "2": {"x": 182, "y": 80}},
            pin_schema=schema,
        )
        self.assertEqual(first["sample_id"], second["sample_id"])
        status = gtl.generic_terminal_learning_status()
        self.assertEqual(status["sample_count"], 1)
        self.assertEqual(status["samples_by_class"]["resistor"]["sample_count"], 1)
        metadata = json.loads(Path(second["metadata"]).read_text(encoding="utf-8"))
        self.assertEqual([p["pin"] for p in metadata["keypoints"]], ["1", "2"])
        self.assertTrue(Path(second["image"]).exists())

    def test_future_unknown_class_can_use_dynamic_pin_schema(self):
        component = {"type": "new_sensor", "terminals": []}
        schema = main._schema_for_component(
            component,
            requested_schema=[{"pin": "V+", "role": "power"}, {"pin": "SIG", "role": "signal"}],
        )
        self.assertEqual(schema, [("V+", "power"), ("SIG", "signal")])

    def test_dynamic_class_correction_is_reapplied_after_regeneration(self):
        old_projects = main.PROJECTS_DIR
        try:
            main.PROJECTS_DIR = self.root / "projects"
            project_dir = main.PROJECTS_DIR / "PRJ_DYNAMIC"
            project_dir.mkdir(parents=True)
            correction = {
                "schema_version": "terminal-corrections-2.0",
                "components": [{
                    "component_id": "X1",
                    "type": "new_sensor",
                    "bbox": {"x1": 10, "y1": 10, "x2": 60, "y2": 60},
                    "pin_schema": [{"pin": "V+", "role": "power"}, {"pin": "SIG", "role": "signal"}],
                    "points": {"V+": {"x": 12, "y": 20}, "SIG": {"x": 58, "y": 40}},
                }],
            }
            (project_dir / "terminal_corrections.json").write_text(json.dumps(correction), encoding="utf-8")
            result = {"components": [{
                "component_id": "X1", "type": "new_sensor",
                "bbox": {"x1": 10, "y1": 10, "x2": 60, "y2": 60},
                "terminals": [],
            }]}
            applied = main._apply_saved_terminal_corrections("PRJ_DYNAMIC", result)
            self.assertEqual(applied, 1)
            terminals = result["components"][0]["terminals"]
            self.assertEqual([t["pin"] for t in terminals], ["V+", "SIG"])
            self.assertTrue(all(t["source"] == "manual_ground_truth_persisted" for t in terminals))
        finally:
            main.PROJECTS_DIR = old_projects

    def test_workflow_javascript_keeps_newlines_escaped(self):
        with patch.object(main, "buscar_projeto", return_value={"stored_filename": "original.png"}):
            html = main.abrir_fluxo_do_projeto("PRJ_TEST").body.decode("utf-8")
        self.assertIn("message+'\\n'", html)
        self.assertNotIn("message+'\n'", html)
        self.assertIn("refresh().catch", html)


if __name__ == "__main__":
    unittest.main()
