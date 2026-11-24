from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODEL = ROOT / "models" / "circuit_scanner_yolo.pt"
YAML = ROOT / "dataset_mixed" / "dataset.yaml"

from scripts.yolo_utils import create_runtime_yaml


def main():
    if not MODEL.exists():
        raise SystemExit("Modelo final não encontrado.")
    if not (ROOT / "dataset_mixed" / "images").exists():
        raise SystemExit("Dataset misto não encontrado.")
    yaml_path = create_runtime_yaml(
        ROOT / "dataset_mixed",
        ROOT / "models" / "dataset_mixed_runtime.yaml",
    )
    from ultralytics import YOLO
    model = YOLO(str(MODEL))
    metrics = model.val(data=str(yaml_path), imgsz=640, verbose=True)
    box = getattr(metrics, "box", None)
    summary = {
        "mAP50": round(float(getattr(box, "map50")), 6) if box is not None else None,
        "mAP50_95": round(float(getattr(box, "map")), 6) if box is not None else None,
        "mAP75": round(float(getattr(box, "map75")), 6) if box is not None else None,
    }
    path = ROOT / "models" / "validation_summary_mixed.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("VALIDAÇÃO MISTA CONCLUÍDA")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
