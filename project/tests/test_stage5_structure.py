from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    obrigatorios = [
        "backend/recognition.py",
        "scripts/train_yolo.py",
        "scripts/validate_yolo.py",
        "scripts/detect_image.py",
        "scripts/audit_stage5.py",
        "scripts/stage5_acceptance.py",
        "scripts/yolo_utils.py",
        "frontend/yolo.html",
        "frontend/js/yolo.js",
        "models/README.md",
        "dataset/dataset.yaml",
        "dataset/classes.json",
        "models/active_model_summary.json",
    ]

    faltando = [
        caminho
        for caminho in obrigatorios
        if not (ROOT / caminho).exists()
    ]

    assert not faltando, f"Arquivos faltando: {faltando}"

    print("TESTE OK — estrutura da Etapa 5")


if __name__ == "__main__":
    main()
