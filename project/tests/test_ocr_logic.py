from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.ocr import (
    aplicar_edicoes_componentes,
    classificar_token,
    montar_componentes_logicos,
)


def bbox(x1, y1, x2, y2):
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": x2 - x1,
        "height": y2 - y1,
        "center_x": (x1 + x2) / 2,
        "center_y": (y1 + y2) / 2,
    }


def token(text, x1, y1, x2, y2, confidence=90):
    return {
        "text": text,
        "kind": classificar_token(text),
        "confidence": confidence,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "center_x": (x1 + x2) / 2,
        "center_y": (y1 + y2) / 2,
    }


def deteccoes_exemplo():
    return [
        {
            "detection_id": "DET_001",
            "class": "resistor",
            "confidence": 0.91,
            "bbox": bbox(100, 100, 220, 150),
        },
        {
            "detection_id": "DET_002",
            "class": "capacitor_polarized",
            "confidence": 0.88,
            "bbox": bbox(400, 100, 470, 180),
        },
    ]


def main():
    assert classificar_token("R1") == "reference"
    assert classificar_token("10k") == "value"
    assert classificar_token("100uF") == "value"
    assert classificar_token("1N4007") == "value"
    assert classificar_token("texto") is None

    tokens = [
        token("R1", 105, 65, 140, 90),
        token("10k", 155, 65, 205, 90),
        token("C1", 400, 65, 435, 90),
        token("100uF", 440, 65, 515, 90),
    ]

    componentes = montar_componentes_logicos(
        deteccoes_exemplo(),
        tokens,
    )

    assert componentes[0]["id"] == "R1"
    assert componentes[0]["value"] == "10k"
    assert componentes[0]["needs_review"] is False
    assert componentes[1]["id"] == "C1"
    assert componentes[1]["value"] == "100uF"

    sem_texto = montar_componentes_logicos(
        deteccoes_exemplo(),
        [],
    )
    assert sem_texto[0]["id"] == "R1"
    assert sem_texto[0]["value"] is None
    assert sem_texto[0]["reference_source"] == "provisional"
    assert sem_texto[0]["needs_review"] is True

    payload = {
        "count": 2,
        "review_count": 2,
        "components": sem_texto,
    }
    revisado = aplicar_edicoes_componentes(
        payload,
        [
            {
                "component_uid": "DET_001",
                "id": "R7",
                "type": "resistor",
                "value": "4k7",
            },
            {
                "component_uid": "DET_002",
                "id": "C3",
                "type": "capacitor_polarized",
                "value": "100uF",
            },
        ],
    )
    assert revisado["review_count"] == 0
    assert revisado["components"][0]["id"] == "R7"
    assert revisado["components"][0]["value"] == "4k7"
    assert revisado["components"][0]["confirmed_by_user"] is True

    print("TESTE OK - lógica de OCR, IDs provisórios e revisão manual")


if __name__ == "__main__":
    main()
