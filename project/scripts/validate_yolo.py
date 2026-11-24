from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODEL_PATH = ROOT / "models" / "circuit_scanner_yolo.pt"
DATASET_RUNTIME = ROOT / "models" / "dataset_runtime.yaml"

from scripts.yolo_utils import create_runtime_yaml


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(
            "Modelo não encontrado. Execute primeiro:\n"
            r"python scripts\train_yolo.py"
        )

    create_runtime_yaml(ROOT / "dataset", DATASET_RUNTIME)

    try:
        from ultralytics import YOLO
    except Exception as erro:
        raise SystemExit(
            "Ultralytics não instalada. Execute pip install -r requirements.txt"
        ) from erro

    model = YOLO(str(MODEL_PATH))
    metrics = model.val(
        data=str(DATASET_RUNTIME),
        imgsz=640,
        verbose=True,
    )

    box = getattr(metrics, "box", None)

    resumo = {
        "mAP50": (
            round(float(getattr(box, "map50")), 6)
            if box is not None and hasattr(box, "map50")
            else None
        ),
        "mAP50_95": (
            round(float(getattr(box, "map")), 6)
            if box is not None and hasattr(box, "map")
            else None
        ),
        "mAP75": (
            round(float(getattr(box, "map75")), 6)
            if box is not None and hasattr(box, "map75")
            else None
        ),
    }

    caminho = ROOT / "models" / "validation_summary.json"
    caminho.write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("VALIDAÇÃO CONCLUÍDA")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print("Salvo em:", caminho)


if __name__ == "__main__":
    main()
