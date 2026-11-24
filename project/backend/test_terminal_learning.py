import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend import terminal_learning as tl


class TerminalLearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.old = {
            "TERMINAL_DATA_DIR": tl.TERMINAL_DATA_DIR,
            "RAW_IMAGES_DIR": tl.RAW_IMAGES_DIR,
            "RAW_LABELS_DIR": tl.RAW_LABELS_DIR,
            "RAW_META_DIR": tl.RAW_META_DIR,
            "POSE_DATASET_DIR": tl.POSE_DATASET_DIR,
            "MODELS_DIR": tl.MODELS_DIR,
            "TERMINAL_MODEL_PATH": tl.TERMINAL_MODEL_PATH,
            "TRAINING_SUMMARY_PATH": tl.TRAINING_SUMMARY_PATH,
            "BASE_DIR": tl.BASE_DIR,
            "PROJECTS_DIR": tl.PROJECTS_DIR,
        }
        tl.BASE_DIR = root
        tl.PROJECTS_DIR = root / "projects"
        tl.TERMINAL_DATA_DIR = root / "terminal_keypoints"
        tl.RAW_IMAGES_DIR = tl.TERMINAL_DATA_DIR / "raw" / "images"
        tl.RAW_LABELS_DIR = tl.TERMINAL_DATA_DIR / "raw" / "labels"
        tl.RAW_META_DIR = tl.TERMINAL_DATA_DIR / "raw" / "meta"
        tl.POSE_DATASET_DIR = tl.TERMINAL_DATA_DIR / "pose_dataset"
        tl.MODELS_DIR = root / "models"
        tl.TERMINAL_MODEL_PATH = tl.MODELS_DIR / "terminal_keypoints.pt"
        tl.TRAINING_SUMMARY_PATH = tl.MODELS_DIR / "terminal_keypoints_summary.json"
        for d in (tl.RAW_IMAGES_DIR, tl.RAW_LABELS_DIR, tl.RAW_META_DIR, tl.MODELS_DIR):
            d.mkdir(parents=True, exist_ok=True)
        tl._MODEL_CACHE.update({"signature": None, "model": None, "load_error": None, "loaded_at": None})
        tl._VISUAL_PRIOR_CACHE.update({"signature": None, "entries": []})
        self.image = np.full((200, 200, 3), 255, np.uint8)
        cv2.line(self.image, (50, 100), (100, 100), (0, 0, 0), 3)
        cv2.line(self.image, (100, 80), (150, 40), (0, 0, 0), 3)
        cv2.line(self.image, (100, 120), (150, 160), (0, 0, 0), 3)
        self.image_path = root / "source.png"
        cv2.imwrite(str(self.image_path), self.image)
        self.component = {
            "component_id": "Q1",
            "type": "transistor_npn",
            "bbox": {"x1": 40, "y1": 30, "x2": 160, "y2": 170},
        }
        self.points = {
            "B": {"x": 100, "y": 100},
            "C": {"x": 150, "y": 40},
            "E": {"x": 150, "y": 160},
        }

    def tearDown(self):
        for key, value in self.old.items():
            setattr(tl, key, value)
        self.tmp.cleanup()

    def test_manual_correction_creates_pose_sample(self):
        meta = tl.save_transistor_training_sample(
            image_path=self.image_path,
            project_id="PRJ_0001",
            component=self.component,
            points=self.points,
        )
        self.assertTrue((tl.RAW_IMAGES_DIR / meta["image_file"]).exists())
        label = (tl.RAW_LABELS_DIR / meta["label_file"]).read_text().strip().split()
        self.assertEqual(label[0], "0")
        self.assertEqual(len(label), 14)  # class + bbox4 + 3*(x,y,vis)

    def test_recorrecting_same_transistor_overwrites_sample(self):
        first = tl.save_transistor_training_sample(
            image_path=self.image_path, project_id="PRJ_0001", component=self.component, points=self.points
        )
        second_points = dict(self.points)
        second_points["B"] = {"x": 102, "y": 101}
        second = tl.save_transistor_training_sample(
            image_path=self.image_path, project_id="PRJ_0001", component=self.component, points=second_points
        )
        self.assertEqual(first["sample_id"], second["sample_id"])
        self.assertEqual(tl.terminal_learning_status()["sample_count"], 1)

    def test_training_crop_contains_manual_points_without_edge_clipping(self):
        component = {
            "component_id": "Q_EDGE",
            "type": "transistor_npn",
            "bbox": {"x1": 80, "y1": 75, "x2": 120, "y2": 125},
        }
        points = {
            "B": {"x": 82, "y": 100},
            "C": {"x": 178, "y": 24},
            "E": {"x": 182, "y": 176},
        }
        meta = tl.save_transistor_training_sample(
            image_path=self.image_path, project_id="PRJ_EDGE", component=component, points=points
        )
        fields = (tl.RAW_LABELS_DIR / meta["label_file"]).read_text().strip().split()
        for i in range(3):
            x = float(fields[5 + i*3]); y = float(fields[6 + i*3])
            self.assertGreater(x, 0.015); self.assertLess(x, 0.985)
            self.assertGreater(y, 0.015); self.assertLess(y, 0.985)
        quality = tl.terminal_learning_status()["dataset_quality"]
        self.assertEqual(quality["edge_clipped_keypoint_count"], 0)

    def test_pose_dataset_keeps_same_project_in_single_split(self):
        for project_id in ("PRJ_A", "PRJ_B", "PRJ_C", "PRJ_D"):
            for suffix, typ in (("N", "transistor_npn"), ("P", "transistor_pnp")):
                component = dict(self.component)
                component["component_id"] = f"Q_{suffix}"
                component["type"] = typ
                tl.save_transistor_training_sample(
                    image_path=self.image_path, project_id=project_id, component=component, points=self.points
                )
        manifest = tl.build_pose_dataset(val_fraction=0.25)
        self.assertEqual(manifest["split_strategy"], "grouped_by_project_id")
        train_names = {p.name for p in (tl.POSE_DATASET_DIR / "images" / "train").glob("*.png")}
        val_names = {p.name for p in (tl.POSE_DATASET_DIR / "images" / "val").glob("*.png")}
        for project_id in ("PRJ_A", "PRJ_B", "PRJ_C", "PRJ_D"):
            in_train = any(name.startswith(project_id + "_") for name in train_names)
            in_val = any(name.startswith(project_id + "_") for name in val_names)
            self.assertNotEqual(in_train, in_val)

    def test_repair_old_clipped_sample_from_metadata(self):
        project_dir = tl.BASE_DIR / "projects" / "PRJ_REPAIR"
        project_dir.mkdir(parents=True, exist_ok=True)
        source = project_dir / "original.png"
        cv2.imwrite(str(source), self.image)
        meta = tl.save_transistor_training_sample(
            image_path=source, project_id="PRJ_REPAIR", component=self.component, points=self.points
        )
        label_path = tl.RAW_LABELS_DIR / meta["label_file"]
        fields = label_path.read_text().strip().split()
        fields[5] = "0.000000"  # simula clipping legado no B.x
        label_path.write_text(" ".join(fields) + "\n")
        self.assertGreater(tl.terminal_learning_status()["dataset_quality"]["edge_clipped_keypoint_count"], 0)
        repaired = tl.repair_training_samples_from_metadata()
        self.assertEqual(repaired["repaired"], 1)
        self.assertEqual(repaired["dataset_quality_after_repair"]["edge_clipped_keypoint_count"], 0)

    def test_known_training_image_reuses_manual_ground_truth(self):
        project_dir = tl.PROJECTS_DIR / "PRJ_KNOWN"
        project_dir.mkdir(parents=True, exist_ok=True)
        source = project_dir / "original.png"
        cv2.imwrite(str(source), self.image)
        tl.save_transistor_training_sample(
            image_path=source, project_id="PRJ_KNOWN", component=self.component, points=self.points
        )
        # Mesma foto, bbox levemente diferente como pode ocorrer numa nova detecção.
        current_bbox = {"x1": 43, "y1": 31, "x2": 158, "y2": 169}
        match = tl.lookup_known_training_ground_truth(
            image=self.image, bbox=current_bbox, component_type="transistor_npn"
        )
        self.assertIsNotNone(match)
        self.assertEqual(match["source"], "known_training_ground_truth")
        self.assertAlmostEqual(match["points"]["B"]["x"], 100.0, places=1)
        self.assertAlmostEqual(match["points"]["C"]["y"], 40.0, places=1)
        self.assertAlmostEqual(match["points"]["E"]["y"], 160.0, places=1)


    def test_known_training_local_memory_survives_resize_and_canvas_change(self):
        project_dir = tl.PROJECTS_DIR / "PRJ_LOCAL_MEMORY"
        project_dir.mkdir(parents=True, exist_ok=True)
        source = project_dir / "original.png"
        cv2.imwrite(str(source), self.image)
        tl.save_transistor_training_sample(
            image_path=source, project_id="PRJ_LOCAL_MEMORY", component=self.component, points=self.points
        )
        # Mesma região visual em outro canvas/resolução, simulando reupload/recorte.
        current = np.full((260, 280, 3), 248, np.uint8)
        scaled = cv2.resize(self.image, (180, 180), interpolation=cv2.INTER_AREA)
        current[35:215, 45:225] = scaled
        # bbox do mesmo transistor após escala 0.9 + deslocamento.
        current_bbox = {
            "x1": 45 + 40*0.9, "y1": 35 + 30*0.9,
            "x2": 45 + 160*0.9, "y2": 35 + 170*0.9,
        }
        match = tl.lookup_known_training_ground_truth(
            image=current, bbox=current_bbox, component_type="transistor_npn"
        )
        self.assertIsNotNone(match)
        self.assertEqual(match["match_method"], "local_visual_memory")
        self.assertGreaterEqual(match["local_similarity"], 0.86)
        # O mapeamento relativo à bbox deve acompanhar a nova posição.
        self.assertAlmostEqual(match["points"]["B"]["x"], 45 + 100*0.9, delta=2.0)
        self.assertAlmostEqual(match["points"]["B"]["y"], 35 + 100*0.9, delta=2.0)


    def test_known_training_template_memory_survives_bad_bbox(self):
        project_dir = tl.PROJECTS_DIR / "PRJ_TEMPLATE_MEMORY"
        project_dir.mkdir(parents=True, exist_ok=True)
        source = project_dir / "original.png"
        cv2.imwrite(str(source), self.image)
        tl.save_transistor_training_sample(
            image_path=source, project_id="PRJ_TEMPLATE_MEMORY", component=self.component, points=self.points
        )
        # Mesma imagem, mas bbox deslocada e menor: o contexto por bbox sozinho
        # fica ruim. A memória por template deve localizar o crop ensinado.
        bad_bbox = {"x1": 72, "y1": 55, "x2": 142, "y2": 145}
        match = tl.lookup_known_training_ground_truth(
            image=self.image, bbox=bad_bbox, component_type="transistor_npn"
        )
        self.assertIsNotNone(match)
        self.assertIn(match["point_mapping"], {"full_image_scale", "multiscale_template_transform"})
        self.assertAlmostEqual(match["points"]["B"]["x"], 100.0, delta=4.0)
        self.assertAlmostEqual(match["points"]["E"]["y"], 160.0, delta=5.0)

    def test_known_training_feature_memory_survives_rotation_and_scale(self):
        project_dir = tl.PROJECTS_DIR / "PRJ_FEATURE_MEMORY"
        project_dir.mkdir(parents=True, exist_ok=True)
        source = project_dir / "original.png"
        # adiciona textura local para dar features estáveis sem mudar o transistor
        textured = self.image.copy()
        cv2.putText(textured, "Q1", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30,30,30), 1, cv2.LINE_AA)
        cv2.circle(textured, (175, 25), 5, (20,20,20), 1)
        cv2.line(textured, (20, 175), (55, 165), (40,40,40), 1)
        cv2.imwrite(str(source), textured)
        tl.save_transistor_training_sample(
            image_path=source, project_id="PRJ_FEATURE_MEMORY", component=self.component, points=self.points
        )
        center = (100.0, 100.0)
        M = cv2.getRotationMatrix2D(center, 11.0, 0.88)
        M[:, 2] += np.array([48.0, 37.0])
        current = cv2.warpAffine(textured, M, (300, 280), flags=cv2.INTER_LINEAR, borderValue=(250,250,250))
        def tr(x, y):
            out = M @ np.array([x, y, 1.0], dtype=np.float64)
            return float(out[0]), float(out[1])
        corners = [tr(40,30), tr(160,30), tr(160,170), tr(40,170)]
        current_bbox = {
            "x1": min(x for x,_ in corners)-4, "y1": min(y for _,y in corners)-4,
            "x2": max(x for x,_ in corners)+4, "y2": max(y for _,y in corners)+4,
        }
        match = tl.lookup_known_training_ground_truth(
            image=current, bbox=current_bbox, component_type="transistor_npn"
        )
        self.assertIsNotNone(match)
        self.assertEqual(match["point_mapping"], "feature_affine_transform")
        bx, by = tr(100, 100)
        cx, cy = tr(150, 40)
        ex, ey = tr(150, 160)
        self.assertAlmostEqual(match["points"]["B"]["x"], bx, delta=6.0)
        self.assertAlmostEqual(match["points"]["B"]["y"], by, delta=6.0)
        self.assertAlmostEqual(match["points"]["C"]["x"], cx, delta=8.0)
        self.assertAlmostEqual(match["points"]["E"]["y"], ey, delta=8.0)



    def test_visual_prior_uses_three_similar_labeled_neighbors(self):
        # Três amostras rotuladas da mesma classe alimentam o prior.
        for idx in range(3):
            img = self.image.copy()
            if idx:
                img = cv2.convertScaleAbs(img, alpha=1.0, beta=idx * 2)
            path = tl.BASE_DIR / f"prior_{idx}.png"
            cv2.imwrite(str(path), img)
            comp = dict(self.component)
            comp["component_id"] = f"Q{idx}"
            tl.save_transistor_training_sample(
                image_path=path, project_id=f"PRJ_PRIOR_{idx}", component=comp, points=self.points
            )
        prior = tl.predict_transistor_visual_prior(
            image=self.image, bbox=self.component["bbox"], component_type="transistor_npn"
        )
        self.assertIsNotNone(prior)
        self.assertTrue(prior["accepted"])
        self.assertEqual(prior["source"], "training_visual_prior")
        self.assertAlmostEqual(prior["points"]["B"]["x"], 100.0, delta=6.0)
        self.assertAlmostEqual(prior["points"]["C"]["y"], 40.0, delta=8.0)
        self.assertAlmostEqual(prior["points"]["E"]["y"], 160.0, delta=8.0)


if __name__ == "__main__":
    unittest.main()
