from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODEL = ROOT / "models" / "circuit_scanner_yolo.pt"
YAML = ROOT / "dataset_real_eval" / "dataset.yaml"

from scripts.yolo_utils import create_runtime_yaml


def count_images(split):
    folder = ROOT / "dataset_real_eval" / "images" / split
    return len(list(folder.glob("*.*"))) if folder.exists() else 0


def metrics_to_dict(metrics):
    box = getattr(metrics, "box", None)
    if box is None:
        return {"mAP50": None, "mAP50_95": None, "mAP75": None}
    return {
        "mAP50": round(float(getattr(box, "map50")), 6),
        "mAP50_95": round(float(getattr(box, "map")), 6),
        "mAP75": round(float(getattr(box, "map75")), 6),
    }


def main():
    if not MODEL.exists():
        raise SystemExit("Modelo final não encontrado.")
    if not (ROOT / "dataset_real_eval" / "images").exists():
        raise SystemExit("Gere o dataset misto primeiro; dataset_real_eval ainda não existe.")

    yaml_path = create_runtime_yaml(
        ROOT / "dataset_real_eval",
        ROOT / "models" / "dataset_real_eval_runtime.yaml",
    )

    from ultralytics import YOLO
    model = YOLO(str(MODEL))
    summary = {"counts": {s: count_images(s) for s in ("train", "val", "test")}}

    if summary["counts"]["val"]:
        print("\nVALIDAÇÃO SOMENTE EM IMAGENS REAIS")
        summary["val"] = metrics_to_dict(model.val(data=str(yaml_path), split="val", imgsz=640, verbose=True))
    else:
        summary["val"] = None
        print("Sem imagens reais em val. Adicione mais fontes reais.")

    if summary["counts"]["test"]:
        print("\nTESTE FINAL SOMENTE EM IMAGENS REAIS")
        summary["test"] = metrics_to_dict(model.val(data=str(yaml_path), split="test", imgsz=640, verbose=True))
    else:
        summary["test"] = None
        print("Sem imagens reais em test. Adicione mais fontes reais.")

    path = ROOT / "models" / "validation_summary_real.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nRESUMO REAL")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
