from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.dataset_generator import CLASSES, generate_dataset


def main():
    temp = ROOT / "tests" / "_dataset_test"
    if temp.exists():
        shutil.rmtree(temp)

    manifest = generate_dataset(
        per_class=6,
        seed=123,
        root=temp,
        clear=True,
    )

    expected = 6 * len(CLASSES)
    assert manifest["total"] == expected, (manifest["total"], expected)

    images = list((temp / "images").rglob("*.jpg"))
    labels = list((temp / "labels").rglob("*.txt"))
    assert len(images) == expected
    assert len(labels) == expected

    for label in labels[:20]:
        parts = label.read_text(encoding="utf-8").strip().split()
        assert len(parts) == 5
        class_id = int(parts[0])
        nums = [float(x) for x in parts[1:]]
        assert 0 <= class_id < len(CLASSES)
        assert all(0 <= x <= 1 for x in nums)

    shutil.rmtree(temp)
    print("TESTE DATASET OK")
    print("Classes:", len(CLASSES))
    print("Amostras validadas:", expected)


if __name__ == "__main__":
    main()
