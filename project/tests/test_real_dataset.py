from pathlib import Path
from io import BytesIO
from tempfile import TemporaryDirectory
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.real_dataset import (
    REAL_DATA_DIR,
    MIXED_DATASET_DIR,
    REAL_EVAL_DIR,
    upload_real_image,
    save_annotations,
    build_mixed_dataset,
    _generate_text_negatives,
)


def test_textos_negativos_possuem_rotulos_yolo_vazios():
    with TemporaryDirectory() as pasta:
        root = Path(pasta)
        for split in ("train", "val", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        counts = _generate_text_negatives(root, 12, seed=4411)
        assert sum(counts.values()) == 12
        labels = list((root / "labels").rglob("*.txt"))
        assert len(labels) == 12
        assert all(path.read_text(encoding="utf-8") == "" for path in labels)


def main():
    backups = []
    for path in (REAL_DATA_DIR, MIXED_DATASET_DIR, REAL_EVAL_DIR):
        backup = path.with_name(path.name + "_test_backup")
        if backup.exists():
            shutil.rmtree(backup)
        if path.exists():
            path.rename(backup)
            backups.append((path, backup))

    try:
        for sub in (REAL_DATA_DIR / "images", REAL_DATA_DIR / "annotations", REAL_DATA_DIR / "meta"):
            sub.mkdir(parents=True, exist_ok=True)

        image = np.full((300, 500, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (80, 100), (420, 200), (0, 0, 0), 4)
        ok, buffer = cv2.imencode(".jpg", image)
        assert ok

        record = upload_real_image(BytesIO(buffer.tobytes()), "real.jpg", "teste")
        save_annotations(
            record["image_id"],
            [{"class_id": 0, "x1": 80, "y1": 100, "x2": 420, "y2": 200}],
        )
        manifest = build_mixed_dataset(
            real_train_augments=1,
            seed=4411,
            text_negative_count=100,
        )
        assert manifest["real_original_counts"]["train"] == 1
        assert (MIXED_DATASET_DIR / "dataset.yaml").exists()
        assert (REAL_EVAL_DIR / "dataset.yaml").exists()
        print("TESTE REAL DATASET OK")
    finally:
        for path in (REAL_DATA_DIR, MIXED_DATASET_DIR, REAL_EVAL_DIR):
            if path.exists():
                shutil.rmtree(path)
        for path, backup in backups:
            if backup.exists():
                backup.rename(path)


if __name__ == "__main__":
    main()
