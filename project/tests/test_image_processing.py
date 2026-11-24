from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.image_processing import processar_imagem


def gerar_imagem_teste(caminho):
    imagem = np.full((700, 1000, 3), 255, dtype=np.uint8)

    cv2.rectangle(imagem, (60, 60), (940, 640), (220, 220, 220), 3)

    # Linhas que lembram um esquema simples.
    cv2.line(imagem, (180, 350), (350, 350), (0, 0, 0), 4)
    cv2.rectangle(imagem, (350, 315), (520, 385), (0, 0, 0), 4)
    cv2.line(imagem, (520, 350), (750, 350), (0, 0, 0), 4)

    cv2.circle(imagem, (180, 350), 8, (0, 0, 0), -1)
    cv2.circle(imagem, (750, 350), 8, (0, 0, 0), -1)

    cv2.putText(
        imagem,
        "R1 1k",
        (380, 285),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # Pequena rotação para testar deskew.
    centro = (500, 350)
    matriz = cv2.getRotationMatrix2D(centro, 4.0, 1.0)
    imagem = cv2.warpAffine(
        imagem,
        matriz,
        (1000, 700),
        borderValue=(255, 255, 255),
    )

    ok, buffer = cv2.imencode(".png", imagem)
    assert ok
    buffer.tofile(str(caminho))


def main():
    pasta = ROOT / "tests" / "_saida_teste"
    pasta.mkdir(exist_ok=True)

    original = pasta / "original_teste.png"
    processada = pasta / "processed_teste.png"

    gerar_imagem_teste(original)

    resultado = processar_imagem(
        original,
        processada,
    )

    assert processada.exists()
    assert processada.stat().st_size > 0
    assert "operacoes" in resultado

    print("TESTE OK")
    print(resultado)


if __name__ == "__main__":
    main()
