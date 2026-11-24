from collections import Counter
from datetime import datetime
from pathlib import Path
import argparse
import json
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.recognition import MODEL_PATH, validar_mapeamento_classes


def load_classes():
    data = json.loads(
        (ROOT / "dataset" / "classes.json").read_text(encoding="utf-8")
    )
    return {int(item["id"]): item["key"] for item in data}


def count_split(dataset_dir, split):
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split
    images = [p for p in images_dir.glob("*.*") if p.is_file()]
    labels = list(labels_dir.glob("*.txt"))
    boxes = Counter()
    invalid = []

    for label in labels:
        for line_number, line in enumerate(
            label.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            parts = line.split()
            if len(parts) != 5:
                invalid.append(f"{label.name}:{line_number}:campos")
                continue
            try:
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
            except ValueError:
                invalid.append(f"{label.name}:{line_number}:numero")
                continue
            if any(value < 0 or value > 1 for value in values):
                invalid.append(f"{label.name}:{line_number}:limites")
            boxes[class_id] += 1

    return {
        "images": len(images),
        "labels": len(labels),
        "boxes": sum(boxes.values()),
        "boxes_by_class_id": dict(sorted(boxes.items())),
        "invalid_labels": invalid,
    }


def count_real_annotations(classes):
    counts = Counter()
    annotated_images = 0
    total_images = 0
    invalid = []
    for path in sorted((ROOT / "real_data" / "annotations").glob("REAL_*.json")):
        total_images += 1
        payload = json.loads(path.read_text(encoding="utf-8"))
        boxes = payload.get("boxes", [])
        if boxes:
            annotated_images += 1
        for index, box in enumerate(boxes, start=1):
            class_id = int(box.get("class_id", -1))
            if class_id not in classes:
                invalid.append(f"{path.name}:caixa_{index}:classe_{class_id}")
            else:
                counts[class_id] += 1

    return {
        "images": total_images,
        "annotated_images": annotated_images,
        "boxes": sum(counts.values()),
        "boxes_by_class": {
            classes[class_id]: counts[class_id]
            for class_id in sorted(classes)
        },
        "invalid_annotations": invalid,
    }


def inspect_model():
    result = {
        "path": str(MODEL_PATH.relative_to(ROOT)),
        "exists": MODEL_PATH.exists(),
        "class_mapping_valid": None,
        "names": None,
        "error": None,
    }
    if not MODEL_PATH.exists():
        return result
    try:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        valid, _, names = validar_mapeamento_classes(model.names)
        result["class_mapping_valid"] = valid
        result["names"] = names
    except Exception as error:
        result["error"] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Audita estrutura, classes, labels e modelo da Etapa 5."
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "models" / "stage5_audit.json"),
    )
    args = parser.parse_args()

    classes = load_classes()
    datasets = {}
    for name in ("dataset", "dataset_mixed", "dataset_real_eval"):
        directory = ROOT / name
        if not directory.exists():
            continue
        datasets[name] = {
            split: count_split(directory, split)
            for split in ("train", "val", "test")
        }

    real = count_real_annotations(classes)
    model = inspect_model()
    warnings = []
    errors = []

    missing_real = [
        name for name, count in real["boxes_by_class"].items()
        if count == 0
    ]
    scarce_real = [
        name for name, count in real["boxes_by_class"].items()
        if 0 < count < 10
    ]
    if missing_real:
        warnings.append(
            "Classes sem exemplos reais: " + ", ".join(missing_real)
        )
    if scarce_real:
        warnings.append(
            "Classes com menos de 10 caixas reais: " + ", ".join(scarce_real)
        )
    if not model["exists"]:
        errors.append("Modelo ativo não encontrado.")
    elif model["class_mapping_valid"] is False:
        errors.append("Ordem/nome das classes do modelo não corresponde ao dataset.")

    for dataset_name, splits in datasets.items():
        for split, info in splits.items():
            if info["images"] != info["labels"]:
                errors.append(
                    f"{dataset_name}/{split}: quantidade de imagens e labels difere."
                )
            if info["invalid_labels"]:
                errors.append(
                    f"{dataset_name}/{split}: há labels inválidos."
                )

    manifest_check = None
    manifest_path = ROOT / "dataset_mixed" / "manifest.json"
    if manifest_path.exists() and "dataset_mixed" in datasets:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("total_counts", {})
        actual = {
            split: datasets["dataset_mixed"][split]["images"]
            for split in ("train", "val", "test")
        }
        manifest_check = {
            "expected": expected,
            "actual": actual,
            "matches": expected == actual,
        }
        if expected != actual:
            warnings.append(
                "manifest.json do dataset misto não corresponde aos arquivos atuais."
            )

    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "classes": classes,
        "model": model,
        "datasets": datasets,
        "real_annotations": real,
        "mixed_manifest_check": manifest_check,
        "errors": errors,
        "warnings": warnings,
        "status": "error" if errors else "ok_with_warnings" if warnings else "ok",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nRelatório salvo em:", output)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
