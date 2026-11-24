from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from backend import component_learning
from backend import real_dataset
from backend.component_trainer_page import component_trainer_html


class ComponentTrainerTests(unittest.TestCase):
    def test_component_trainer_has_complete_no_docs_flow(self):
        html = component_trainer_html()
        expected = (
            "Adicionar circuito",
            "Adicionar circuito CAD",
            "Somente feitas à mão",
            "Somente CAD",
            "source_type",
            "Desenhar caixa",
            "Selecionar/mover",
            "Salvar anotações",
            "Montar dataset misto",
            "Iniciar treinamento",
            "Finalizar após esta época e validar",
            "Restaurar modelo anterior",
            "Fixar modelo atual como base estável",
            "/component-learning/train",
            "/component-learning/stop-and-validate",
            "/component-learning/rollback",
            "/component-learning/foundation-from-active",
            "Auditar anotações",
            "Testar modelo e cores",
            "/component-learning/dataset-audit",
            "/component-learning/diagnostic",
        )
        for item in expected:
            self.assertIn(item, html)

    def test_cad_upload_keeps_source_and_group(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            images = root / "images"
            annotations = root / "annotations"
            meta = root / "meta"
            for directory in (images, annotations, meta):
                directory.mkdir()

            image = np.full((80, 120, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (20, 25), (100, 55), (0, 0, 0), 2)
            ok, encoded = cv2.imencode(".png", image)
            self.assertTrue(ok)
            with (
                patch.object(real_dataset, "REAL_IMAGES_DIR", images),
                patch.object(real_dataset, "REAL_ANNOTATIONS_DIR", annotations),
                patch.object(real_dataset, "REAL_META_DIR", meta),
            ):
                first = real_dataset.upload_real_image(
                    BytesIO(encoded.tobytes()), "cad_a.png", "Fonte CAD",
                    source_type="cad", source_group="Projeto Fonte 01",
                )
                second = real_dataset.upload_real_image(
                    BytesIO(encoded.tobytes()), "cad_b.png", "Fonte CAD página 2",
                    source_type="cad", source_group="Projeto Fonte 01",
                )
            self.assertEqual(first["source_type"], "cad")
            self.assertEqual(first["source_group"], "Projeto Fonte 01")
            self.assertEqual(first["source_group_id"], second["source_group_id"])

    def test_related_cad_pages_never_cross_splits(self):
        records = [
            {
                "image_id": f"CAD_{group}_{page}",
                "source_type": "cad",
                "source_group_id": f"CAD_GROUP_{group}",
            }
            for group in range(6)
            for page in range(2)
        ]
        mapping = real_dataset._assign_real_splits(records, seed=4411)
        for group in range(6):
            splits = {mapping[f"CAD_{group}_{page}"] for page in range(2)}
            self.assertEqual(len(splits), 1)

    def test_training_status_reports_readiness(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            status_path = root / "status.json"
            summary_path = root / "summary.json"
            model_path = root / "model.pt"
            mixed_dir = root / "mixed"
            model_path.write_bytes(b"model")
            (mixed_dir / "dataset.yaml").parent.mkdir(parents=True)
            (mixed_dir / "dataset.yaml").write_text("names: {}", encoding="utf-8")
            with (
                patch.object(component_learning, "STATUS_PATH", status_path),
                patch.object(component_learning, "SUMMARY_PATH", summary_path),
                patch.object(component_learning, "MODEL_PATH", model_path),
                patch.object(component_learning, "BASE_DIR", root),
                patch.object(
                    component_learning,
                    "active_dataset_dirs",
                    lambda: (mixed_dir, root / "real_eval"),
                ),
            ):
                result = component_learning.component_training_status()
            self.assertEqual(result["state"], "idle")
            self.assertTrue(result["model_exists"])
            self.assertTrue(result["dataset_ready"])

    def test_training_parameter_guards(self):
        cases = (
            ({"epochs": 0}, "epochs"),
            ({"imgsz": 123}, "imgsz"),
            ({"batch": 0}, "batch"),
            ({"epochs": 10, "patience": 11}, "patience"),
        )
        with TemporaryDirectory() as root:
            root = Path(root)
            with (
                patch.object(
                    component_learning,
                    "active_dataset_dirs",
                    lambda: (root, root / "real_eval"),
                ),
                patch.object(component_learning, "MODEL_PATH", root / "model.pt"),
            ):
                for kwargs, message in cases:
                    with self.subTest(kwargs=kwargs):
                        with self.assertRaisesRegex(ValueError, message):
                            component_learning.start_component_training(**kwargs)

    def test_quality_gate_rejects_false_positive_regression(self):
        result = component_learning._candidate_quality_gate(
            {"precision": 0.82, "map50": 0.87, "recall": 0.75},
            {"precision": 0.80, "map50": 0.85},
            {"image_count": 40, "false_boxes_per_image": 2.4},
            {"image_count": 40, "false_boxes_per_image": 0.1},
        )
        self.assertFalse(result["passed"])
        self.assertIn(
            "false_positives_on_negative_images_increased", result["reasons"]
        )

    def test_quality_gate_accepts_real_improvement(self):
        result = component_learning._candidate_quality_gate(
            {"precision": 0.86, "map50": 0.90, "recall": 0.78},
            {"precision": 0.79, "map50": 0.84},
            {"image_count": 40, "false_boxes_per_image": 0.05},
            {"image_count": 40, "false_boxes_per_image": 0.12},
        )
        self.assertTrue(result["passed"], result["reasons"])

    def test_real_page_tiles_create_focused_positives_and_background_negatives(self):
        image = np.full((720, 720, 3), 245, dtype=np.uint8)
        for y in range(30, 720, 36):
            cv2.line(image, (0, y), (719, y), (190, 190, 190), 1)
        # Dois símbolos pequenos e bastante área com pautas/fios sem símbolo.
        cv2.rectangle(image, (80, 90), (150, 145), (0, 0, 0), 4)
        cv2.circle(image, (580, 560), 35, (0, 0, 0), 4)
        record = {
            "annotations": [
                {"class_id": 0, "x1": 78, "y1": 88, "x2": 152, "y2": 147},
                {"class_id": 6, "x1": 543, "y1": 523, "x2": 617, "y2": 597},
            ]
        }
        import random
        positives, negatives = real_dataset._real_page_tiles(
            record, image, random.Random(4411)
        )
        self.assertGreaterEqual(len(positives), 2)
        self.assertTrue(all(tile["label"].strip() for tile in positives))
        self.assertGreaterEqual(len(negatives), 1)
        self.assertTrue(all(not tile["label"] for tile in negatives))


if __name__ == "__main__":
    unittest.main()
