from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.recognition import detectar_componentes


def main():
    parser = argparse.ArgumentParser(
        description="Testa o modelo YOLO em uma imagem avulsa."
    )
    parser.add_argument("image", help="Caminho da imagem.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    image = Path(args.image).resolve()

    output_dir = ROOT / "models" / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    resultado = detectar_componentes(
        image_path=image,
        output_image_path=output_dir / "detections_test.png",
        output_json_path=output_dir / "detections_test.json",
        confidence=args.conf,
        imgsz=args.imgsz,
    )

    print("Detecções:", resultado["count"])
    for item in resultado["detections"]:
        print(
            item["detection_id"],
            item["class"],
            f'{item["confidence"]:.3f}',
            item["bbox"],
        )

    print("Resultado visual:", output_dir / "detections_test.png")
    print("JSON:", output_dir / "detections_test.json")


if __name__ == "__main__":
    main()
