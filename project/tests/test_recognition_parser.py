from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.recognition import (
    consolidar_multiview,
    montar_deteccoes,
    validar_mapeamento_classes,
)


def main():
    xyxy = np.array(
        [
            [10, 20, 110, 80],
            [200, 100, 280, 190],
        ],
        dtype=np.float32,
    )

    confs = np.array([0.91, 0.77], dtype=np.float32)
    classes = np.array([0, 4], dtype=np.float32)

    deteccoes = montar_deteccoes(
        xyxy=xyxy,
        confs=confs,
        class_ids=classes,
        names={0: "resistor", 4: "led"},
    )

    assert len(deteccoes) == 2
    assert deteccoes[0]["class"] == "resistor"
    assert deteccoes[1]["class"] == "led"
    assert deteccoes[0]["bbox"]["width"] == 100.0
    assert deteccoes[0]["bbox"]["height"] == 60.0
    assert deteccoes[0]["detection_id"] == "DET_001"

    valido, _, _ = validar_mapeamento_classes(
        {
            0: "resistor",
            1: "capacitor",
            2: "capacitor_polarized",
            3: "diode",
            4: "led",
            5: "inductor",
            6: "source_dc",
            7: "battery",
            8: "gnd",
            9: "vcc",
            10: "transistor_npn",
            11: "transistor_pnp",
            12: "mosfet_n",
            13: "mosfet_p",
            14: "opamp",
        }
    )
    assert valido

    conflito = montar_deteccoes(
        xyxy=np.array([[10, 20, 110, 80]], dtype=np.float32),
        confs=np.array([0.72], dtype=np.float32),
        class_ids=np.array([4], dtype=np.float32),
        names={4: "led"},
        inference_source="original_grayscale",
    )[0]
    deteccoes[0]["inference_sources"] = ["original_color"]
    consolidadas = consolidar_multiview([deteccoes[0], conflito])
    assert len(consolidadas) == 1
    assert consolidadas[0]["class"] == "resistor"
    assert consolidadas[0]["review_required"] is True
    assert consolidadas[0]["alternatives"][0]["class"] == "led"

    print("TESTE OK — parser de detecções YOLO")


if __name__ == "__main__":
    main()
