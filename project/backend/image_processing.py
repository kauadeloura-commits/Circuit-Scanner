from pathlib import Path
import cv2
import numpy as np


def _ler_imagem(caminho: Path):
    """
    Lê a imagem usando bytes + imdecode.
    Isso evita problemas comuns do cv2.imread com caminhos
    que possuem acentos no Windows.
    """
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)

    if imagem is None:
        raise ValueError("O OpenCV não conseguiu abrir a imagem.")

    return imagem


def _salvar_png(caminho: Path, imagem):
    ok, buffer = cv2.imencode(".png", imagem)

    if not ok:
        raise ValueError("O OpenCV não conseguiu gerar o arquivo PNG.")

    buffer.tofile(str(caminho))


def _ordenar_pontos(pontos):
    pontos = np.asarray(pontos, dtype=np.float32).reshape(4, 2)

    soma = pontos.sum(axis=1)
    diferenca = np.diff(pontos, axis=1).reshape(-1)

    superior_esquerdo = pontos[np.argmin(soma)]
    inferior_direito = pontos[np.argmax(soma)]
    superior_direito = pontos[np.argmin(diferenca)]
    inferior_esquerdo = pontos[np.argmax(diferenca)]

    return np.array(
        [
            superior_esquerdo,
            superior_direito,
            inferior_direito,
            inferior_esquerdo,
        ],
        dtype=np.float32,
    )


def _corrigir_perspectiva(imagem):
    """
    Procura um grande contorno quadrilateral, como uma folha fotografada.
    Só aplica a transformação se o quadrilátero ocupar uma parte relevante
    da imagem. Caso contrário, mantém a imagem original.
    """
    altura, largura = imagem.shape[:2]
    area_total = altura * largura

    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(cinza, (5, 5), 0)
    bordas = cv2.Canny(suavizada, 50, 150)

    contornos, _ = cv2.findContours(
        bordas,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contornos = sorted(
        contornos,
        key=cv2.contourArea,
        reverse=True,
    )[:20]

    for contorno in contornos:
        area = cv2.contourArea(contorno)

        if area < area_total * 0.25:
            continue

        perimetro = cv2.arcLength(contorno, True)
        aproximado = cv2.approxPolyDP(
            contorno,
            0.02 * perimetro,
            True,
        )

        if len(aproximado) != 4:
            continue

        pontos = _ordenar_pontos(aproximado)

        tl, tr, br, bl = pontos

        largura_superior = np.linalg.norm(tr - tl)
        largura_inferior = np.linalg.norm(br - bl)
        nova_largura = int(max(largura_superior, largura_inferior))

        altura_esquerda = np.linalg.norm(bl - tl)
        altura_direita = np.linalg.norm(br - tr)
        nova_altura = int(max(altura_esquerda, altura_direita))

        if nova_largura < 100 or nova_altura < 100:
            continue

        destino = np.array(
            [
                [0, 0],
                [nova_largura - 1, 0],
                [nova_largura - 1, nova_altura - 1],
                [0, nova_altura - 1],
            ],
            dtype=np.float32,
        )

        matriz = cv2.getPerspectiveTransform(pontos, destino)

        corrigida = cv2.warpPerspective(
            imagem,
            matriz,
            (nova_largura, nova_altura),
            borderValue=(255, 255, 255),
        )

        return corrigida, True

    return imagem, False


def _estimar_inclinacao(cinza):
    """
    Estima pequenas inclinações usando segmentos quase horizontais.
    Limitamos a correção para não girar indevidamente circuitos
    que possuem muitas linhas verticais.
    """
    bordas = cv2.Canny(cinza, 50, 150)

    linhas = cv2.HoughLinesP(
        bordas,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=max(40, cinza.shape[1] // 8),
        maxLineGap=20,
    )

    if linhas is None:
        return 0.0

    angulos = []

    # HoughLinesP normalmente retorna (N, 1, 4), mas dependendo da
    # imagem/versao do OpenCV o array pode chegar com outra forma.
    # Normalizamos para linhas [x1, y1, x2, y2] antes de percorrer.
    linhas_array = np.asarray(linhas, dtype=np.int32)

    if linhas_array.size < 4 or linhas_array.size % 4 != 0:
        return 0.0

    linhas_normalizadas = linhas_array.reshape(-1, 4)

    for x1, y1, x2, y2 in linhas_normalizadas:

        if x2 == x1:
            continue

        angulo = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Normaliza para a faixa -90..90.
        if angulo > 90:
            angulo -= 180
        elif angulo < -90:
            angulo += 180

        # Para deskew, usamos apenas linhas razoavelmente horizontais.
        if -20 <= angulo <= 20:
            angulos.append(angulo)

    if len(angulos) < 2:
        return 0.0

    mediana = float(np.median(angulos))

    # Correções minúsculas não valem a interpolação.
    if abs(mediana) < 0.25:
        return 0.0

    return float(np.clip(mediana, -15.0, 15.0))


def _rotacionar(imagem, angulo):
    if abs(angulo) < 0.01:
        return imagem

    altura, largura = imagem.shape[:2]
    centro = (largura / 2, altura / 2)

    matriz = cv2.getRotationMatrix2D(
        centro,
        angulo,
        1.0,
    )

    cos = abs(matriz[0, 0])
    sin = abs(matriz[0, 1])

    nova_largura = int((altura * sin) + (largura * cos))
    nova_altura = int((altura * cos) + (largura * sin))

    matriz[0, 2] += (nova_largura / 2) - centro[0]
    matriz[1, 2] += (nova_altura / 2) - centro[1]

    return cv2.warpAffine(
        imagem,
        matriz,
        (nova_largura, nova_altura),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def processar_imagem(caminho_original: Path, caminho_processado: Path):
    imagem = _ler_imagem(caminho_original)

    altura_original, largura_original = imagem.shape[:2]

    # 1. Correção de perspectiva quando uma "folha" é detectada.
    imagem, perspectiva_aplicada = _corrigir_perspectiva(imagem)

    # 2. Escala de cinza.
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # 3. Correção automática de pequena rotação/inclinação.
    angulo_detectado = _estimar_inclinacao(cinza)

    if abs(angulo_detectado) > 0.01:
        # Se as linhas estão inclinadas +X graus, giramos -X.
        imagem = _rotacionar(imagem, -angulo_detectado)
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # 4. Aumento de contraste local.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    contraste = clahe.apply(cinza)

    # 5. Redução de ruído preservando linhas.
    filtrada = cv2.bilateralFilter(
        contraste,
        d=7,
        sigmaColor=45,
        sigmaSpace=45,
    )

    # 6. Binarização adaptativa.
    binaria = cv2.adaptiveThreshold(
        filtrada,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    # Garante, na maioria dos desenhos, fundo branco e traços escuros.
    borda = np.concatenate(
        [
            binaria[0, :],
            binaria[-1, :],
            binaria[:, 0],
            binaria[:, -1],
        ]
    )

    if float(np.mean(borda)) < 127:
        binaria = cv2.bitwise_not(binaria)

    # Remove pequenos pontos isolados sem destruir linhas do esquema.
    kernel = np.ones((2, 2), np.uint8)
    processada = cv2.morphologyEx(
        binaria,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    _salvar_png(caminho_processado, processada)

    altura_final, largura_final = processada.shape[:2]

    return {
        "operacoes": [
            "correcao_de_perspectiva_automatica",
            "escala_de_cinza",
            "correcao_de_rotacao_automatica",
            "aumento_de_contraste_CLAHE",
            "reducao_de_ruido_bilateral",
            "binarizacao_adaptativa",
            "limpeza_morfologica_leve",
        ],
        "perspectiva_aplicada": bool(perspectiva_aplicada),
        "angulo_detectado_graus": round(float(angulo_detectado), 3),
        "angulo_corrigido_graus": round(float(-angulo_detectado), 3),
        "largura_original": int(largura_original),
        "altura_original": int(altura_original),
        "largura_processada": int(largura_final),
        "altura_processada": int(altura_final),
        "arquivo_saida": caminho_processado.name,
    }
