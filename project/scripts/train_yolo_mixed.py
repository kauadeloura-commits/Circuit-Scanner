from pathlib import Path
from datetime import datetime
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATASET = ROOT / "dataset_mixed"
MODELS = ROOT / "models"
RUNS = MODELS / "training_mixed"
FINAL_MODEL = MODELS / "circuit_scanner_yolo.pt"
BACKUPS = MODELS / "backups"

from scripts.yolo_utils import create_runtime_yaml


def choose_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 0, f"GPU CUDA: {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    return "cpu", "CPU"


def main():
    parser = argparse.ArgumentParser(description="Treina YOLO com dataset sintético + real.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--base-model",
        default="auto",
        help="auto continua do modelo ativo; informe outro .pt para trocar.",
    )
    args = parser.parse_args()

    yaml_path = DATASET / "dataset.yaml"
    manifest_path = DATASET / "manifest.json"
    if not yaml_path.exists() or not manifest_path.exists():
        raise SystemExit(
            "Dataset misto não existe. Abra a página de anotações e clique em 'Gerar dataset misto'."
        )

    yaml_path = create_runtime_yaml(
        DATASET,
        MODELS / "dataset_mixed_runtime.yaml",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print("Dataset misto:", manifest["total_counts"])
    print("Imagens reais originais:", manifest["real_original_counts"])
    print("Aumentos reais no treino:", manifest["real_augmented_train"])
    print(
        "Negativos de texto gerados:",
        manifest.get("generated_text_negative_counts", {}),
    )
    print(
        "Negativos coletados pelo OCR:",
        manifest.get("ocr_text_negative_counts", {}),
    )

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit("Ultralytics não instalada. Rode pip install -r requirements.txt") from exc

    device, device_name = choose_device()
    MODELS.mkdir(exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    BACKUPS.mkdir(exist_ok=True)

    if FINAL_MODEL.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = BACKUPS / f"circuit_scanner_yolo_before_mixed_{timestamp}.pt"
        shutil.copy2(FINAL_MODEL, backup)
        print("Backup do modelo atual:", backup)

    base_model = (
        str(FINAL_MODEL)
        if args.base_model == "auto" and FINAL_MODEL.exists()
        else "yolo26n.pt"
        if args.base_model == "auto"
        else args.base_model
    )
    print("Modelo usado como base:", base_model)
    model = YOLO(base_model)
    result = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        seed=4411,
        project=str(RUNS),
        name="circuit_scanner_mixed",
        exist_ok=True,
        patience=max(10, min(30, args.epochs)),
        hsv_h=0.15,
        hsv_s=0.80,
        hsv_v=0.50,
        plots=True,
        verbose=True,
    )

    save_dir = Path(getattr(result, "save_dir", RUNS / "circuit_scanner_mixed"))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        last = save_dir / "weights" / "last.pt"
        if last.exists():
            best = last
        else:
            raise SystemExit("Treino terminou, mas não encontrei best.pt nem last.pt.")

    shutil.copy2(best, FINAL_MODEL)
    summary = {
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "mixed_synthetic_real",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device_name,
        "base_model": base_model,
        "source_weight": str(best.relative_to(ROOT)),
        "final_model": str(FINAL_MODEL.relative_to(ROOT)),
        "dataset_manifest": manifest,
        "color_augmentation": {
            "hsv_h": 0.15,
            "hsv_s": 0.80,
            "hsv_v": 0.50,
        },
    }
    (MODELS / "training_summary_mixed.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nTREINO MISTO CONCLUÍDO")
    print("Modelo final:", FINAL_MODEL)


if __name__ == "__main__":
    main()
