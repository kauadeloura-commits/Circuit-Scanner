from pathlib import Path
from datetime import datetime
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
TRAINING_DIR = MODELS_DIR / "training"
FINAL_MODEL = MODELS_DIR / "circuit_scanner_yolo.pt"
SUMMARY_PATH = MODELS_DIR / "training_summary.json"

from scripts.yolo_utils import create_runtime_yaml


def contar_imagens():
    return {
        split: len(list((DATASET_DIR / "images" / split).glob("*.*")))
        for split in ("train", "val", "test")
    }


def criar_yaml_runtime():
    return create_runtime_yaml(
        DATASET_DIR,
        MODELS_DIR / "dataset_runtime.yaml",
    )


def escolher_device():
    try:
        import torch

        if torch.cuda.is_available():
            return 0, f"GPU CUDA: {torch.cuda.get_device_name(0)}"
    except Exception:
        pass

    return "cpu", "CPU"


def main():
    parser = argparse.ArgumentParser(
        description="Treina o detector YOLO do Circuit Scanner."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Quantidade de épocas. Padrão: 30.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamanho das imagens usado no treino. Padrão: 640.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch. Se faltar memória, tente 4 ou 2.",
    )
    parser.add_argument(
        "--base-model",
        default="yolo26n.pt",
        help="Modelo base da Ultralytics. Padrão: yolo26n.pt.",
    )
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="Treina do zero usando yolo26n.yaml em vez de pesos pré-treinados.",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise SystemExit("--epochs deve ser maior que zero.")

    MODELS_DIR.mkdir(exist_ok=True)
    TRAINING_DIR.mkdir(exist_ok=True)

    contagem = contar_imagens()
    total = sum(contagem.values())

    print("=" * 62)
    print("CIRCUIT SCANNER — ETAPA 5 — TREINAMENTO YOLO")
    print("=" * 62)
    print("Dataset:")
    print("  treino    :", contagem["train"])
    print("  validação :", contagem["val"])
    print("  teste     :", contagem["test"])
    print("  total     :", total)

    if contagem["train"] < 100:
        print()
        print("ATENÇÃO: há poucas imagens de treino.")
        print(
            "Antes do treinamento final, gere na Etapa 4 pelo menos "
            "150 imagens por classe."
        )

    yaml_path = criar_yaml_runtime()
    device, device_nome = escolher_device()

    print()
    print("Dispositivo:", device_nome)
    print("Épocas:", args.epochs)
    print("Imagem:", args.imgsz)
    print("Batch:", args.batch)
    print("Config dataset:", yaml_path)
    print()

    try:
        from ultralytics import YOLO
    except Exception as erro:
        raise SystemExit(
            "Ultralytics não está instalada.\n"
            "Ative o venv e execute:\n"
            "pip install -r requirements.txt"
        ) from erro

    modelo_inicial = "yolo26n.yaml" if args.scratch else args.base_model

    print("Carregando modelo base:", modelo_inicial)
    print(
        "Na primeira execução, o peso pré-treinado pode precisar "
        "ser obtido pela Ultralytics."
    )

    model = YOLO(modelo_inicial)

    resultado_treino = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(TRAINING_DIR),
        name="circuit_scanner",
        exist_ok=True,
        patience=max(10, min(30, args.epochs)),
        hsv_h=0.15,
        hsv_s=0.80,
        hsv_v=0.50,
        plots=True,
        verbose=True,
    )

    save_dir_attr = getattr(resultado_treino, "save_dir", None)

    if save_dir_attr is None:
        trainer = getattr(model, "trainer", None)
        save_dir_attr = getattr(trainer, "save_dir", None)

    if save_dir_attr is None:
        save_dir = TRAINING_DIR / "circuit_scanner"
    else:
        save_dir = Path(save_dir_attr)

    best = save_dir / "weights" / "best.pt"

    if not best.exists():
        last = save_dir / "weights" / "last.pt"
        if last.exists():
            best = last
        else:
            raise SystemExit(
                "O treinamento terminou, mas nenhum peso best.pt/last.pt foi encontrado."
            )

    shutil.copy2(best, FINAL_MODEL)

    resumo = {
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_model": modelo_inicial,
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device_nome,
        "dataset_counts": contagem,
        "training_run": str(save_dir.relative_to(ROOT)),
        "source_weight": str(best.relative_to(ROOT)),
        "final_model": str(FINAL_MODEL.relative_to(ROOT)),
        "color_augmentation": {
            "hsv_h": 0.15,
            "hsv_s": 0.80,
            "hsv_v": 0.50,
        },
    }

    SUMMARY_PATH.write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 62)
    print("TREINAMENTO CONCLUÍDO")
    print("=" * 62)
    print("Modelo final:")
    print(FINAL_MODEL)
    print()
    print("Agora o site pode usar o botão:")
    print("Reconhecer componentes com YOLO")


if __name__ == "__main__":
    main()
