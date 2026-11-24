from pathlib import Path
import json
import shutil
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.recognition import detectar_componentes


def main():
    project = ROOT / "projects" / "PRJ_0047"
    originals = list(project.glob("original.*"))
    assert originals, "Imagem de regressão PRJ_0047 não encontrada."

    output = ROOT / "tests" / "_stage5_inference"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    try:
        result = detectar_componentes(
            image_path=originals[0],
            output_image_path=output / "detections.png",
            output_json_path=output / "detections.json",
            confidence=0.25,
            imgsz=640,
            strategy="auto",
        )

        assert result["source_image"].startswith("original.")
        assert result["count"] >= 5, result["detections"]
        assert any(
            view["name"] == "original_grayscale"
            for view in result["inference_views"]
        )
        assert (output / "detections.png").exists()
        saved = json.loads(
            (output / "detections.json").read_text(encoding="utf-8")
        )
        assert saved["count"] == result["count"]
        print("TESTE OK — inferência original + escala de cinza")
        print("Detecções:", result["count"])
    finally:
        if output.exists():
            shutil.rmtree(output)


if __name__ == "__main__":
    main()
