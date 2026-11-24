from pathlib import Path
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.ocr import executar_ocr_componentes, ocr_status


OUTPUT = ROOT / "models" / "stage6_acceptance"


def deteccao(detection_id, classe, x1, y1, x2, y2):
    return {
        "detection_id": detection_id,
        "class": classe,
        "confidence": 0.95,
        "review_required": False,
        "bbox": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2,
        },
    }


def criar_fixture():
    imagem = np.full((500, 1000, 3), 255, dtype=np.uint8)
    cv2.putText(imagem, "R1", (150, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    cv2.putText(imagem, "10k", (245, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    cv2.rectangle(imagem, (170, 190), (330, 260), (0, 0, 0), 3)
    cv2.putText(imagem, "C1", (620, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    cv2.putText(imagem, "100uF", (715, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    cv2.rectangle(imagem, (640, 190), (800, 260), (0, 0, 0), 3)
    caminho = OUTPUT / "fixture_texto.png"
    if not cv2.imwrite(str(caminho), imagem):
        raise OSError("Não foi possível criar a imagem de teste.")
    return caminho, [
        deteccao("DET_001", "resistor", 170, 190, 330, 260),
        deteccao("DET_002", "capacitor", 640, 190, 800, 260),
    ]


def main():
    status = ocr_status()
    if not status["ready"]:
        raise SystemExit(status["message"])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fixture, deteccoes = criar_fixture()
    impresso = executar_ocr_componentes(
        fixture,
        {"detections": deteccoes},
        OUTPUT / "fixture_components.json",
        OUTPUT / "fixture_ocr.png",
    )

    atual_dir = ROOT / "projects" / "PRJ_0047"
    atual_deteccoes = json.loads(
        (atual_dir / "detections.json").read_text(encoding="utf-8")
    )
    atual = executar_ocr_componentes(
        atual_dir / "original.jpg",
        atual_deteccoes,
        atual_dir / "components.json",
        atual_dir / "ocr.png",
    )

    assert impresso["reference_ocr_count"] == 2
    assert impresso["value_ocr_count"] == 2
    assert atual["count"] == atual_deteccoes["count"]
    assert atual["value_ocr_count"] == 0

    resumo = {
        "status": "passed",
        "engine": status,
        "printed_fixture": {
            "components": impresso["count"],
            "references_read": impresso["reference_ocr_count"],
            "values_read": impresso["value_ocr_count"],
            "review_count": impresso["review_count"],
        },
        "current_circuit_PRJ_0047": {
            "components": atual["count"],
            "references_read": atual["reference_ocr_count"],
            "values_read": atual["value_ocr_count"],
            "review_count": atual["review_count"],
            "interpretation": (
                "A imagem não contém referências ou valores escritos. "
                "O sistema gerou identificadores provisórios e não inventou valores."
            ),
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
