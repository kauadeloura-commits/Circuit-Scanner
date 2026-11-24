from pathlib import Path
from datetime import datetime
import hashlib
import itertools
import json
import math
import threading
import copy
from time import perf_counter

import cv2
import numpy as np
from backend.detection_boxes import remove_nested_boxes
from backend.resistor_geometry import symbol_analysis_session,cached_vision

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "circuit_scanner_yolo.pt"
CLASSES_PATH = BASE_DIR / "dataset" / "classes.json"
SUMMARY_PATHS = (
    MODELS_DIR / "active_model_summary.json",
    MODELS_DIR / "training_summary_mixed.json",
    MODELS_DIR / "training_summary.json",
)

_MODEL_CACHE = {"signature": None, "model": None}
_MODEL_LOCK = threading.Lock()


def _ultralytics_info():
    try:
        import ultralytics
        return True, getattr(ultralytics, "__version__", "desconhecida")
    except Exception:
        return False, None


def _normalizar_names(names):
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def _classes_esperadas():
    if not CLASSES_PATH.exists():
        return {}
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    return {int(item["id"]): str(item["key"]) for item in classes}


def validar_mapeamento_classes(names):
    esperado = _classes_esperadas()
    recebido = _normalizar_names(names)
    return esperado == recebido, esperado, recebido


def _carregar_resumo_treino():
    existentes = sorted(
        (caminho for caminho in SUMMARY_PATHS if caminho.exists()),
        key=lambda caminho: caminho.stat().st_mtime_ns,
        reverse=True,
    )
    for caminho in existentes:
        try:
            resumo = json.loads(caminho.read_text(encoding="utf-8"))
            resumo["summary_file"] = caminho.name
            return resumo
        except Exception:
            continue
    return None


def _hash_modelo():
    if not MODEL_PATH.exists():
        return None
    digest = hashlib.sha256()
    with MODEL_PATH.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _carregar_modelo():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modelo YOLO ainda não foi treinado.")
    try:
        from backend.cpu_inference import configure_cpu_inference
        configure_cpu_inference()
        from ultralytics import YOLO
    except Exception as erro:
        raise RuntimeError(
            "A biblioteca ultralytics não está instalada. "
            "Execute: pip install -r requirements.txt"
        ) from erro

    stat = MODEL_PATH.stat()
    assinatura = (stat.st_mtime_ns, stat.st_size)
    with _MODEL_LOCK:
        if _MODEL_CACHE["signature"] != assinatura:
            modelo = YOLO(str(MODEL_PATH))
            valido, esperado, recebido = validar_mapeamento_classes(
                modelo.names
            )
            if not valido:
                raise RuntimeError(
                    "As classes gravadas no modelo não correspondem a "
                    f"dataset/classes.json. Esperado: {esperado}. "
                    f"Modelo: {recebido}."
                )
            _MODEL_CACHE["signature"] = assinatura
            _MODEL_CACHE["model"] = modelo
    return _MODEL_CACHE["model"]


def model_status():
    instalado, versao = _ultralytics_info()
    resumo = _carregar_resumo_treino()
    mapping_valid = None
    model_names = None
    model_error = None

    if instalado and MODEL_PATH.exists():
        try:
            modelo = _carregar_modelo()
            mapping_valid, _, recebido = validar_mapeamento_classes(
                modelo.names
            )
            model_names = recebido
        except Exception as erro:
            mapping_valid = False
            model_error = str(erro)

    return {
        "ultralytics_installed": instalado,
        "ultralytics_version": versao,
        "model_ready": MODEL_PATH.exists() and model_error is None,
        "model_path": str(MODEL_PATH.relative_to(BASE_DIR)),
        "model_size_mb": (
            round(MODEL_PATH.stat().st_size / (1024 * 1024), 2)
            if MODEL_PATH.exists()
            else None
        ),
        "model_sha256": _hash_modelo(),
        "class_mapping_valid": mapping_valid,
        "model_names": model_names,
        "model_error": model_error,
        "training_summary": resumo,
        "default_inference_strategy": "ai_first_precision_with_geometry_validation",
    }


def montar_deteccoes(
    xyxy,
    confs,
    class_ids,
    names,
    inference_source=None,
):
    names = _normalizar_names(names)
    deteccoes = []

    for indice, (box, conf, class_id) in enumerate(
        zip(xyxy, confs, class_ids),
        start=1,
    ):
        x1, y1, x2, y2 = [float(v) for v in box]
        cid = int(class_id)
        deteccao = {
            "detection_id": f"DET_{indice:03d}",
            "class_id": cid,
            "class": names.get(cid, f"class_{cid}"),
            "confidence": round(float(conf), 6),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "width": round(max(0.0, x2 - x1), 2),
                "height": round(max(0.0, y2 - y1), 2),
                "center_x": round((x1 + x2) / 2.0, 2),
                "center_y": round((y1 + y2) / 2.0, 2),
            },
        }
        if inference_source:
            deteccao["inference_sources"] = [str(inference_source)]
        deteccoes.append(deteccao)
    return deteccoes


def _area_bbox(bbox):
    return max(0.0, bbox["x2"] - bbox["x1"]) * max(
        0.0, bbox["y2"] - bbox["y1"]
    )


def _sobreposicao(d1, d2):
    a = d1["bbox"]
    b = d2["bbox"]
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = _area_bbox(a)
    area_b = _area_bbox(b)
    if inter <= 0 or area_a <= 0 or area_b <= 0:
        return 0.0, 0.0
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    cobertura_menor = inter / min(area_a, area_b)
    return iou, cobertura_menor


def _mesma_regiao_de_componente(
    d1,
    d2,
    iou_threshold=0.45,
    cobertura_threshold=0.80,
):
    """Evita que uma caixa enorme engula dois componentes vizinhos.

    Cobertura alta, sozinha, não prova duplicidade: uma caixa errada que cobre
    metade do circuito contém várias caixas corretas com cobertura 1.0. Para
    usar cobertura como equivalência, exigimos também tamanho e centro
    compatíveis.
    """
    iou, cobertura = _sobreposicao(d1, d2)
    if iou >= iou_threshold:
        return True
    if cobertura < cobertura_threshold:
        return False
    a, b = d1["bbox"], d2["bbox"]
    area_a, area_b = _area_bbox(a), _area_bbox(b)
    size_ratio = max(area_a, area_b) / max(1.0, min(area_a, area_b))
    center_distance = math.hypot(
        float(a["center_x"]) - float(b["center_x"]),
        float(a["center_y"]) - float(b["center_y"]),
    )
    reference_diagonal = math.hypot(
        min(float(a["width"]), float(b["width"])),
        min(float(a["height"]), float(b["height"])),
    )
    return bool(
        size_ratio <= 3.20
        and center_distance <= 0.42 * max(1.0, reference_diagonal)
    )


def _reordenar_deteccoes(deteccoes):
    deteccoes.sort(
        key=lambda d: (d["bbox"]["y1"], d["bbox"]["x1"])
    )
    for indice, det in enumerate(deteccoes, start=1):
        det["detection_id"] = f"DET_{indice:03d}"
    return deteccoes


def remover_duplicadas(
    deteccoes,
    iou_threshold=0.30,
    cobertura_threshold=0.75,
):
    """Remove caixas repetidas da mesma classe."""
    ordenadas = sorted(
        deteccoes,
        key=lambda d: float(d["confidence"]),
        reverse=True,
    )
    mantidas = []
    for atual in ordenadas:
        duplicada = False
        for salva in mantidas:
            if atual["class_id"] != salva["class_id"]:
                continue
            if _mesma_regiao_de_componente(
                atual,
                salva,
                iou_threshold=iou_threshold,
                cobertura_threshold=cobertura_threshold,
            ):
                duplicada = True
                break
        if not duplicada:
            mantidas.append(atual)
    return _reordenar_deteccoes(mantidas)


def _pixels_borda(imagem, faixa=None):
    altura, largura = imagem.shape[:2]
    faixa = faixa or max(2, min(altura, largura) // 40)
    return np.concatenate(
        (
            imagem[:faixa].reshape(-1, imagem.shape[2]),
            imagem[-faixa:].reshape(-1, imagem.shape[2]),
            imagem[:, :faixa].reshape(-1, imagem.shape[2]),
            imagem[:, -faixa:].reshape(-1, imagem.shape[2]),
        ),
        axis=0,
    )


def criar_vista_cinza_normalizada(imagem):
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    if float(np.median(_pixels_borda(imagem))) < 127:
        cinza = 255 - cinza
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(cinza), cv2.COLOR_GRAY2BGR)


def _imagem_quase_monocromatica(imagem):
    """Retorna True para scans/desenhos essencialmente preto e branco.

    Não exige que os pixels sejam 0/255 puros: uma foto de folha em tons de
    cinza também deve entrar aqui. A decisão usa saturação, então uma imagem
    colorida normal continua usando o fluxo anterior.
    """
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    saturacao = hsv[:, :, 1]
    return float(np.percentile(saturacao, 95)) <= 18.0


def _remover_pautas_longas(imagem):
    """Remove pautas longas e claras sem apagar fios/simbolos escuros.

    Fotografias de caderno costumam conter linhas quase completas, paralelas
    e bem mais claras que o grafite do circuito. Uma abertura morfologica
    encontra somente essas estruturas muito longas; em seguida removemos
    apenas os pixels de baixo contraste. Trechos escuros que cruzam a pauta
    (fios, corpos e terminais) sao preservados pixel a pixel.
    """
    if imagem is None or imagem.size == 0:
        return imagem

    colorida = imagem.ndim == 3
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) if colorida else imagem.copy()
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    invertida = float(np.median(border)) < 127.0
    if invertida:
        gray = 255 - gray

    h, w = gray.shape[:2]
    if min(h, w) < 40:
        limpa = 255 - gray if invertida else gray
        return cv2.cvtColor(limpa, cv2.COLOR_GRAY2BGR) if colorida else limpa

    # O fechamento grande estima o papel por baixo dos tracos. A diferenca
    # resultante e robusta a sombras e a iluminacao desigual da fotografia.
    fundo_kernel = max(21, int(round(min(h, w) * 0.16)))
    if fundo_kernel % 2 == 0:
        fundo_kernel += 1
    background_scale=min(1.,384/min(h,w))
    background_gray=cv2.resize(gray,None,fx=background_scale,fy=background_scale,interpolation=cv2.INTER_AREA) if background_scale<1 else gray
    background_kernel=max(7,round(fundo_kernel*background_scale));background_kernel+=int(background_kernel%2==0)
    fundo = cv2.morphologyEx(
        cv2.GaussianBlur(background_gray, (3, 3), 0),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (background_kernel, background_kernel)
        ),
    )
    if background_scale<1:fundo=cv2.resize(fundo,(w,h),interpolation=cv2.INTER_LINEAR)
    contraste = cv2.subtract(fundo, gray)
    candidatos = np.where(contraste >= 6, 255, 0).astype(np.uint8)

    comprimento_v = max(25, int(round(h * 0.52)))
    comprimento_h = max(25, int(round(w * 0.52)))
    verticais = cv2.morphologyEx(
        candidatos,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, comprimento_v)),
    )
    horizontais = cv2.morphologyEx(
        candidatos,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (comprimento_h, 1)),
    )
    pautas = cv2.bitwise_or(verticais, horizontais)
    if not np.any(pautas):
        limpa = 255 - gray if invertida else gray
        return cv2.cvtColor(limpa, cv2.COLOR_GRAY2BGR) if colorida else limpa

    largura_pauta = max(2, int(round(min(h, w) * 0.006)))
    pautas = cv2.dilate(
        pautas,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * largura_pauta + 1, 2 * largura_pauta + 1),
        ),
        iterations=1,
    )

    valores_pauta = contraste[(pautas > 0) & (contraste > 0)]
    contraste_tipico = (
        float(np.median(valores_pauta)) if valores_pauta.size else 0.0
    )
    # Uma pauta e removida, mas grafite/caneta nitidamente mais escuro que ela
    # continua intacto. O piso tambem protege tracos finos de baixo contraste.
    # O teto impede que uma unica linha eletrica preta e muito longa seja
    # tomada como a propria pauta e apagada por inteiro.
    limite_traco_forte = max(30.0, min(90.0, contraste_tipico * 1.8))
    remover = (pautas > 0) & (contraste < limite_traco_forte)
    limpa = gray.copy()
    limpa[remover] = fundo[remover]
    limpa = cv2.GaussianBlur(limpa, (3, 3), 0)
    if invertida:
        limpa = 255 - limpa
    return cv2.cvtColor(limpa, cv2.COLOR_GRAY2BGR) if colorida else limpa


def criar_vista_binaria_normalizada(imagem):
    """Cria uma vista P&B limpa para o YOLO.

    Essa vista é adicional: ela não substitui a imagem original. O objetivo é
    aproximar scans, PDFs rasterizados e esquemas já binarizados das formas
    geométricas usadas no treinamento, preservando traços finos.
    """
    sem_pautas = _remover_pautas_longas(imagem)
    cinza = cv2.cvtColor(sem_pautas, cv2.COLOR_BGR2GRAY)

    # Normaliza a polaridade: fundo claro, tinta escura.
    borda = np.concatenate((cinza[0, :], cinza[-1, :], cinza[:, 0], cinza[:, -1]))
    if float(np.median(borda)) < 127:
        cinza = 255 - cinza

    # Um blur mínimo reduz serrilhado/JPEG, mas não usa abertura morfológica
    # porque ela pode apagar fios e detalhes finos de diodo/transistor.
    suavizada = cv2.GaussianBlur(cinza, (3, 3), 0)
    _, binaria = cv2.threshold(
        suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    borda_bin = np.concatenate((
        binaria[0, :], binaria[-1, :], binaria[:, 0], binaria[:, -1]
    ))
    if float(np.mean(borda_bin)) < 127:
        binaria = cv2.bitwise_not(binaria)

    return cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)


def criar_vistas_monocromaticas_universais(imagem):
    """Gera vistas P&B complementares sem conhecer a classe do componente.

    Todas as classes registradas no dataset passam pelas mesmas vistas. Isso
    evita a antiga dependência de listas fixas (resistor, diodo etc.) e cobre
    também bateria, fonte, indutor, MOSFET, amp-op e classes adicionadas no
    futuro, desde que o modelo tenha exemplos anotados delas.
    """
    sem_pautas = _remover_pautas_longas(imagem)
    gray = cv2.cvtColor(sem_pautas, cv2.COLOR_BGR2GRAY)
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    if float(np.median(border)) < 127.0:
        gray = 255 - gray

    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (3, 3), 0)
    _, otsu = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    min_dim = max(3, min(gray.shape[:2]))
    block = max(15, min(61, int(round(min_dim / 12.0))))
    if block % 2 == 0:
        block += 1
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        9,
    )

    # Uma vista engrossa apenas um pouco os traços muito finos de PDF/scan.
    ink = cv2.bitwise_not(otsu)
    thick_ink = cv2.dilate(ink, np.ones((2, 2), np.uint8), iterations=1)
    thick = cv2.bitwise_not(thick_ink)

    outputs = {
        "monochrome_clahe": clahe,
        "monochrome_otsu": otsu,
        "monochrome_adaptive": adaptive,
        "monochrome_stroke_normalized": thick,
    }
    normalized = {}
    for name, view in outputs.items():
        edge = np.concatenate((view[0, :], view[-1, :], view[:, 0], view[:, -1]))
        if float(np.mean(edge)) < 127.0:
            view = cv2.bitwise_not(view)
        normalized[name] = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    return normalized


def criar_vista_invariante_cor(imagem):
    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB).astype(np.float32)
    borda_lab = cv2.cvtColor(
        _pixels_borda(imagem).reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_BGR2LAB,
    ).reshape(-1, 3).astype(np.float32)
    cor_fundo = np.median(borda_lab, axis=0)
    distancia_cor = np.linalg.norm(lab - cor_fundo, axis=2)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sigma = max(3.0, min(imagem.shape[:2]) / 45.0)
    fundo_local = cv2.GaussianBlur(cinza, (0, 0), sigmaX=sigma)
    tinta = np.maximum(distancia_cor, np.abs(cinza - fundo_local) * 1.8)
    baixo = float(np.percentile(tinta, 15))
    alto = float(np.percentile(tinta, 99))
    if alto <= baixo + 1e-6:
        escala = np.zeros_like(tinta, dtype=np.uint8)
    else:
        escala = np.clip(
            (tinta - baixo) * (255.0 / (alto - baixo)), 0, 255
        ).astype(np.uint8)
    return cv2.cvtColor(cv2.medianBlur(255 - escala, 3), cv2.COLOR_GRAY2BGR)


def _deteccoes_triangulos_preenchidos(imagem):
    """Detecta diodos preenchidos pela forma, sem usar OCR ou letras D1-D4."""
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB).astype(np.float32)
    borda_lab = cv2.cvtColor(
        _pixels_borda(imagem).reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_BGR2LAB,
    ).reshape(-1, 3).astype(np.float32)
    fundo = np.median(borda_lab, axis=0)
    primeiro_plano = (
        np.linalg.norm(lab - fundo, axis=2) > 25.0
    ).astype(np.uint8) * 255
    caixas = []

    for inicio_matiz in range(0, 180, 10):
        mascara = cv2.inRange(
            hsv,
            np.array([inicio_matiz, 20, 25], dtype=np.uint8),
            np.array([min(179, inicio_matiz + 14), 255, 255], dtype=np.uint8),
        )
        mascara = cv2.bitwise_and(mascara, primeiro_plano)
        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contorno in contornos:
            area = float(cv2.contourArea(contorno))
            x, y, largura, altura = cv2.boundingRect(contorno)
            if not (
                150 <= area <= 6000
                and 18 <= largura <= 140
                and 18 <= altura <= 140
                and 0.45 <= largura / max(1, altura) <= 2.0
            ):
                continue
            vertices = cv2.approxPolyDP(
                contorno, 0.04 * cv2.arcLength(contorno, True), True
            ).reshape(-1, 2)
            if len(vertices) != 3:
                continue
            pares = [
                (abs(int(vertices[i, 0]) - int(vertices[j, 0])), i, j)
                for i in range(3)
                for j in range(i + 1, 3)
            ]
            _, indice_a, indice_b = min(pares)
            indice_ponta = ({0, 1, 2} - {indice_a, indice_b}).pop()
            base = vertices[[indice_a, indice_b]]
            ponta = vertices[indice_ponta]
            if abs(int(base[0, 1]) - int(base[1, 1])) < 0.45 * altura:
                continue
            if abs(float(ponta[1]) - float(base[:, 1].mean())) > 0.30 * altura:
                continue
            base_x = float(base[:, 0].mean())
            direcao = 1 if float(ponta[0]) > base_x else -1
            if abs(float(ponta[0]) - base_x) < 0.45 * largura:
                continue
            inicio_x = int(ponta[0] + direcao * 2)
            fim_x = int(ponta[0] + direcao * 22)
            menor_x = max(0, min(inicio_x, fim_x))
            maior_x = min(primeiro_plano.shape[1] - 1, max(inicio_x, fim_x))
            menor_y = max(0, y - int(0.10 * altura))
            maior_y = min(primeiro_plano.shape[0], y + int(1.10 * altura))
            tamanho_barra = max(
                (
                    int(np.count_nonzero(primeiro_plano[menor_y:maior_y, coluna]))
                    for coluna in range(menor_x, maior_x + 1)
                ),
                default=0,
            )
            if tamanho_barra < 0.45 * altura:
                continue
            caixas.append([
                max(0, x - 6 - (18 if direcao < 0 else 0)),
                max(0, y - 6),
                min(imagem.shape[1] - 1, x + largura + 6 + (18 if direcao > 0 else 0)),
                min(imagem.shape[0] - 1, y + altura + 6),
            ])

    deteccoes = montar_deteccoes(
        np.asarray(caixas, dtype=np.float32).reshape(-1, 4),
        np.full(len(caixas), 0.92, dtype=np.float32),
        np.full(len(caixas), 3, dtype=np.float32),
        {3: "diode"},
        inference_source="filled_triangle_geometry",
    )
    return remover_duplicadas(
        deteccoes, iou_threshold=0.30, cobertura_threshold=0.70
    )


def _deteccoes_resistores_geometricos(imagem):
    from backend.resistor_geometry import zigzag_paths
    candidates=zigzag_paths(imagem)
    boxes=[[c['bbox'][k] for k in ('x1','y1','x2','y2')] for c in candidates]
    detections=montar_deteccoes(np.asarray(boxes,dtype=np.float32).reshape(-1,4),
        np.asarray([c['score'] for c in candidates],np.float32),np.zeros(len(candidates),np.float32),
        {0:'resistor'},inference_source='resistor_zigzag_geometry')
    for detection,candidate in zip(detections,candidates):detection['resistor_shape_evidence']=candidate['evidence']
    return remover_duplicadas(detections,iou_threshold=.25,cobertura_threshold=.70)




def _deteccoes_resistores_retangulares(imagem):
    from backend.resistor_geometry import iec_bodies
    candidates=iec_bodies(imagem)
    boxes=[[c['bbox'][k] for k in ('x1','y1','x2','y2')] for c in candidates]
    detections=montar_deteccoes(np.asarray(boxes,dtype=np.float32).reshape(-1,4),
        np.asarray([c['score'] for c in candidates],np.float32),np.zeros(len(candidates),np.float32),
        {0:'resistor'},inference_source='resistor_iec_geometry')
    for detection,candidate in zip(detections,candidates):detection['resistor_shape_evidence']=candidate['evidence']
    return remover_duplicadas(detections,iou_threshold=.25,cobertura_threshold=.70)




def _deteccoes_capacitores_geometricos(imagem):
    """Propõe capacitor por duas placas paralelas e terminais opostos."""
    h, w = imagem.shape[:2]
    mind = float(max(1, min(h, w)))
    lines = _linhas_estruturais(imagem)
    plates = [
        line for line in lines
        if line["length"] >= 0.035 * mind
        and line["length"] <= 0.26 * mind
        and (
            abs(line["angle"]) <= 12.0
            or abs(abs(line["angle"]) - 90.0) <= 12.0
        )
    ]
    leads = [
        line for line in lines
        if line["length"] >= 0.025 * mind
        and (
            abs(line["angle"]) <= 16.0
            or abs(abs(line["angle"]) - 90.0) <= 16.0
        )
    ]
    unique = []
    for line in sorted(plates, key=lambda item: item["length"], reverse=True):
        if any(
            abs(line["angle"] - saved["angle"]) <= 5.0
            and _distancia_pontos(
                (line["mx"], line["my"]), (saved["mx"], saved["my"])
            ) <= max(4.0, 0.008 * mind)
            for saved in unique
        ):
            continue
        unique.append(line)

    boxes = []
    scores = []
    for index, first in enumerate(unique):
        first_vertical = abs(abs(first["angle"]) - 90.0) <= 12.0
        for second in unique[index + 1:]:
            second_vertical = abs(abs(second["angle"]) - 90.0) <= 12.0
            if first_vertical != second_vertical:
                continue
            length = (first["length"] + second["length"]) / 2.0
            ratio = first["length"] / max(1.0, second["length"])
            if not 0.55 <= ratio <= 1.80:
                continue
            if first_vertical:
                separation = abs(first["mx"] - second["mx"])
                along = abs(first["my"] - second["my"])
            else:
                separation = abs(first["my"] - second["my"])
                along = abs(first["mx"] - second["mx"])
            if not (
                0.10 * length <= separation <= 0.80 * length
                and along <= 0.25 * length
            ):
                continue

            tolerance = max(7.0, 0.25 * length)
            if first_vertical:
                left, right = sorted((first, second), key=lambda item: item["mx"])
                left_ok = any(
                    abs(lead["angle"]) <= 16.0
                    and min(
                        _distancia_pontos(point, (left["mx"], left["my"]))
                        for point in _endpoints(lead)
                    ) <= tolerance
                    and min(point[0] for point in _endpoints(lead))
                    <= left["mx"] - 0.12 * length
                    for lead in leads
                )
                right_ok = any(
                    abs(lead["angle"]) <= 16.0
                    and min(
                        _distancia_pontos(point, (right["mx"], right["my"]))
                        for point in _endpoints(lead)
                    ) <= tolerance
                    and max(point[0] for point in _endpoints(lead))
                    >= right["mx"] + 0.12 * length
                    for lead in leads
                )
                # Capacitor parcialmente cortado pela borda: a placa externa
                # pode não mostrar seu terminal, mas a placa interna ainda
                # precisa ter a ligação perpendicular completa.
                left_ok = left_ok or left["mx"] <= 0.025 * w
                right_ok = right_ok or right["mx"] >= 0.975 * w
            else:
                top, bottom = sorted((first, second), key=lambda item: item["my"])
                left_ok = any(
                    abs(abs(lead["angle"]) - 90.0) <= 16.0
                    and min(
                        _distancia_pontos(point, (top["mx"], top["my"]))
                        for point in _endpoints(lead)
                    ) <= tolerance
                    and min(point[1] for point in _endpoints(lead))
                    <= top["my"] - 0.12 * length
                    for lead in leads
                )
                right_ok = any(
                    abs(abs(lead["angle"]) - 90.0) <= 16.0
                    and min(
                        _distancia_pontos(point, (bottom["mx"], bottom["my"]))
                        for point in _endpoints(lead)
                    ) <= tolerance
                    and max(point[1] for point in _endpoints(lead))
                    >= bottom["my"] + 0.12 * length
                    for lead in leads
                )
                left_ok = left_ok or top["my"] <= 0.025 * h
                right_ok = right_ok or bottom["my"] >= 0.975 * h
            if not (left_ok and right_ok):
                continue

            points = [point for plate in (first, second) for point in _endpoints(plate)]
            pad = max(7.0, 0.30 * length)
            candidate_box = [
                max(0.0, min(point[0] for point in points) - pad),
                max(0.0, min(point[1] for point in points) - pad),
                min(float(w - 1), max(point[0] for point in points) + pad),
                min(float(h - 1), max(point[1] for point in points) + pad),
            ]
            if max(
                candidate_box[2] - candidate_box[0],
                candidate_box[3] - candidate_box[1],
            ) > 0.50 * mind:
                continue
            boxes.append(candidate_box)
            scores.append(0.91)

    detections = montar_deteccoes(
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(scores, dtype=np.float32),
        np.ones(len(boxes), dtype=np.float32),
        {1: "capacitor"},
        inference_source="capacitor_parallel_plate_geometry",
    )
    return remover_duplicadas(
        detections, iou_threshold=0.25, cobertura_threshold=0.70
    )


def _deteccoes_transistores_geometricos(imagem, deteccoes_existentes=None):
    """Propõe BJT por círculo + base + ramos, independente do resultado YOLO.

    Uma caixa fraca do YOLO não pode bloquear a proposta geométrica: ela pode
    ser descartada mais tarde ou perder a classe no consenso. As duas propostas
    são consolidadas normalmente quando apontam para o mesmo transistor.
    """
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    h, w = gray.shape[:2]
    mind = max(1, min(h, w))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=max(28, int(0.14 * mind)),
        param1=120,
        param2=24,
        minRadius=max(14, int(0.055 * mind)),
        maxRadius=max(24, int(0.24 * mind)),
    )
    if circles is None:
        return []
    ink = gray < min(210, int(np.percentile(gray, 82)))

    def circle_support(cx, cy, radius):
        # Um círculo Hough isolado pode surgir sobre letras e arcos. O corpo do
        # BJT, porém, tem tinta ao longo de quase toda a circunferência. Medimos
        # a cobertura com tolerância de 2 px para antialiasing/traço grosso.
        hits = 0
        samples = 144
        for angle in np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False):
            x = int(round(float(cx) + float(radius) * np.cos(angle)))
            y = int(round(float(cy) + float(radius) * np.sin(angle)))
            x1, x2 = max(0, x - 2), min(w, x + 3)
            y1, y2 = max(0, y - 2), min(h, y + 3)
            if x1 < x2 and y1 < y2 and np.any(ink[y1:y2, x1:x2]):
                hits += 1
        return hits / float(samples)

    proposals = []
    existing_bjt = [
        det for det in (deteccoes_existentes or [])
        if str(det.get("class")) in {"transistor_npn", "transistor_pnp"}
    ]
    for cx, cy, radius in np.asarray(circles[0]).reshape(-1, 3):
        if circle_support(cx, cy, radius) < 0.72:
            continue
        pad = radius * 1.18
        bbox = {
            "x1": max(0.0, float(cx - pad)),
            "y1": max(0.0, float(cy - pad)),
            "x2": min(float(w - 1), float(cx + pad)),
            "y2": min(float(h - 1), float(cy + pad)),
        }
        valid, score, _ = _validar_geometria_transistor(
            imagem, bbox, "transistor_npn"
        )
        if valid and score >= 0.72:
            probe = {"bbox": bbox}
            overlaps = [
                det for det in existing_bjt
                if max(_sobreposicao(probe, det)) >= 0.35
            ]
            subtype = max(
                overlaps,
                key=lambda det: float(det.get("confidence", 0.0)),
                default=None,
            )
            class_name = (
                str(subtype["class"]) if subtype is not None
                else "transistor_npn"
            )
            class_id = 11 if class_name == "transistor_pnp" else 10
            proposals.append((bbox, class_id, subtype is None))

    detections = []
    for bbox, class_id, defaulted in proposals:
        detection = montar_deteccoes(
            np.asarray([[bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]], dtype=np.float32),
            np.asarray([0.88], dtype=np.float32),
            np.asarray([class_id], dtype=np.float32),
            {10: "transistor_npn", 11: "transistor_pnp"},
            inference_source="bjt_circle_fork_geometry",
        )[0]
        detection["bjt_subtype_source"] = (
            "overlapping_yolo_candidate" if not defaulted
            else "geometry_fallback_default_npn"
        )
        detections.append(detection)
    for detection in detections:
        detection["review_required"] = True
        detection["review_reasons"] = [
            "bjt_arrow_confirmation_required"
        ]
        detection["bjt_type_requires_arrow_confirmation"] = True
    return remover_duplicadas(
        detections, iou_threshold=0.30, cobertura_threshold=0.70
    )


def _deteccoes_opamp_geometricos(imagem):
    """Propõe amp-op pela base vertical e duas diagonais até o ápice."""
    h, w = imagem.shape[:2]
    mind = float(max(1, min(h, w)))
    lines = list(_linhas_estruturais(imagem))

    # Desenhos a lápis podem ter contraste bem menor que as pautas. Uma
    # segunda passagem com CLAHE e Canny suave recupera as bordas longas do
    # triângulo sem depender da cor/espessura usada no treinamento.
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 15, 70)
    faint_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(12, int(0.022 * mind)),
        minLineLength=max(24, int(0.055 * mind)),
        maxLineGap=max(8, int(0.040 * mind)),
    )
    if faint_lines is not None:
        for x1, y1, x2, y2 in np.asarray(faint_lines).reshape(-1, 4):
            dx, dy = float(x2 - x1), float(y2 - y1)
            length = float(np.hypot(dx, dy))
            angle = float(np.degrees(np.arctan2(dy, dx)))
            if angle > 90.0:
                angle -= 180.0
            elif angle < -90.0:
                angle += 180.0
            candidate = {
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "mx": (float(x1) + float(x2)) / 2.0,
                "my": (float(y1) + float(y2)) / 2.0,
                "length": length,
                "angle": angle,
            }
            if not any(
                _distancia_pontos(
                    (candidate["mx"], candidate["my"]),
                    (saved["mx"], saved["my"]),
                ) <= max(5.0, 0.012 * mind)
                and abs(candidate["angle"] - saved["angle"]) <= 7.0
                for saved in lines
            ):
                lines.append(candidate)

    def merge_collinear(candidates):
        """Une fragmentos do mesmo traço apagado antes de formar triângulos."""
        clusters = []
        gap_limit = max(18.0, 0.12 * mind)
        perpendicular_limit = max(7.0, 0.018 * mind)
        for line in sorted(candidates, key=lambda item: item["length"], reverse=True):
            assigned = False
            for cluster in clusters:
                reference = cluster[0]
                angle_delta = abs(line["angle"] - reference["angle"])
                angle_delta = min(angle_delta, abs(180.0 - angle_delta))
                if angle_delta > 8.0:
                    continue
                p1 = np.asarray([reference["x1"], reference["y1"]], dtype=np.float32)
                p2 = np.asarray([reference["x2"], reference["y2"]], dtype=np.float32)
                direction = p2 - p1
                norm = float(np.linalg.norm(direction))
                if norm <= 1e-6:
                    continue
                direction /= norm
                normal = np.asarray([-direction[1], direction[0]])
                midpoint = np.asarray([line["mx"], line["my"]], dtype=np.float32)
                perpendicular = abs(float(np.dot(midpoint - p1, normal)))
                if perpendicular > perpendicular_limit:
                    continue
                ref_projection = sorted(
                    float(np.dot(np.asarray(point) - p1, direction))
                    for item in cluster for point in _endpoints(item)
                )
                line_projection = sorted(
                    float(np.dot(np.asarray(point) - p1, direction))
                    for point in _endpoints(line)
                )
                gap = max(
                    0.0,
                    line_projection[0] - ref_projection[-1],
                    ref_projection[0] - line_projection[-1],
                )
                if gap <= gap_limit:
                    cluster.append(line)
                    assigned = True
                    break
            if not assigned:
                clusters.append([line])

        merged = []
        for cluster in clusters:
            points = np.asarray(
                [point for item in cluster for point in _endpoints(item)],
                dtype=np.float32,
            )
            vx, vy, x0, y0 = [
                float(value) for value in cv2.fitLine(
                    points, cv2.DIST_L2, 0, 0.01, 0.01
                ).reshape(-1)
            ]
            direction = np.asarray([vx, vy], dtype=np.float32)
            origin = np.asarray([x0, y0], dtype=np.float32)
            projections = (points - origin) @ direction
            start = origin + direction * float(np.min(projections))
            end = origin + direction * float(np.max(projections))
            dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
            length = float(np.hypot(dx, dy))
            angle = float(np.degrees(np.arctan2(dy, dx)))
            if angle > 90.0:
                angle -= 180.0
            elif angle < -90.0:
                angle += 180.0
            merged.append({
                "x1": float(start[0]), "y1": float(start[1]),
                "x2": float(end[0]), "y2": float(end[1]),
                "mx": float((start[0] + end[0]) / 2.0),
                "my": float((start[1] + end[1]) / 2.0),
                "length": length, "angle": angle,
                "fragment_count": len(cluster),
            })
        return merged

    merge_candidates = [
        line for line in lines
        if line["length"] >= 0.055 * mind
        and abs(line["angle"]) >= 17.0
    ]
    structural_lines = list(lines) + merge_collinear(merge_candidates)
    marker_lines = []
    fine_markers = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(5, int(0.008 * mind)),
        minLineLength=max(5, int(0.010 * mind)),
        maxLineGap=max(2, int(0.008 * mind)),
    )
    if fine_markers is not None:
        for x1, y1, x2, y2 in np.asarray(fine_markers).reshape(-1, 4):
            dx, dy = float(x2 - x1), float(y2 - y1)
            length = float(np.hypot(dx, dy))
            angle = float(np.degrees(np.arctan2(dy, dx)))
            if angle > 90.0:
                angle -= 180.0
            elif angle < -90.0:
                angle += 180.0
            if 0.010 * mind <= length <= 0.13 * mind:
                marker_lines.append({
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "mx": (float(x1) + float(x2)) / 2.0,
                    "my": (float(y1) + float(y2)) / 2.0,
                    "length": length, "angle": angle,
                })

    short_horizontal = [
        line for line in marker_lines if abs(line["angle"]) <= 14.0
    ]
    short_vertical = [
        line for line in marker_lines
        if abs(abs(line["angle"]) - 90.0) <= 14.0
    ]
    mark_tolerance = max(6.0, 0.016 * mind)
    plus_centers = [
        (
            (horizontal_line["mx"] + vertical_line["mx"]) / 2.0,
            (horizontal_line["my"] + vertical_line["my"]) / 2.0,
        )
        for horizontal_line in short_horizontal
        for vertical_line in short_vertical
        if _distancia_pontos(
            (horizontal_line["mx"], horizontal_line["my"]),
            (vertical_line["mx"], vertical_line["my"]),
        ) <= mark_tolerance
    ]
    polarity_pairs = []
    for plus in plus_centers:
        for minus in short_horizontal:
            minus_center = (minus["mx"], minus["my"])
            vertical_distance = abs(plus[1] - minus_center[1])
            if (
                abs(plus[0] - minus_center[0]) <= 0.13 * mind
                and 0.05 * mind <= vertical_distance <= 0.38 * mind
            ):
                polarity_pairs.append((plus, minus_center))

    # Sem os sinais + e - não fazemos proposta geométrica autônoma. O YOLO
    # ainda pode reconhecer amp-op normalmente, mas pautas/retângulos nunca
    # viram um amp-op apenas por formarem diagonais acidentais.
    if not polarity_pairs:
        return []

    positive = [
        line for line in structural_lines
        if 18.0 <= line["angle"] <= 68.0
        and line["length"] >= 0.10 * mind
    ]
    negative = [
        line for line in structural_lines
        if -68.0 <= line["angle"] <= -18.0
        and line["length"] >= 0.10 * mind
    ]
    vertical = [
        line for line in structural_lines
        if abs(abs(line["angle"]) - 90.0) <= 16.0
        and line["length"] >= 0.12 * mind
    ]
    horizontal = [
        line for line in structural_lines
        if abs(line["angle"]) <= 15.0
        and line["length"] >= 0.045 * mind
    ]
    positive = sorted(
        positive, key=lambda line: line["length"], reverse=True
    )[:40]
    negative = sorted(
        negative, key=lambda line: line["length"], reverse=True
    )[:40]
    vertical = sorted(
        vertical, key=lambda line: line["length"], reverse=True
    )[:30]
    horizontal = sorted(
        horizontal, key=lambda line: line["length"], reverse=True
    )[:80]
    tolerance = max(10.0, 0.055 * mind)
    boxes = []

    for upper in positive:
        for lower in negative:
            endpoint_pairs = [
                (_distancia_pontos(a, b), a, b)
                for a in _endpoints(upper)
                for b in _endpoints(lower)
            ]
            apex_distance, apex_a, apex_b = min(endpoint_pairs, key=lambda item: item[0])
            if apex_distance > tolerance:
                continue
            apex = (
                (apex_a[0] + apex_b[0]) / 2.0,
                (apex_a[1] + apex_b[1]) / 2.0,
            )
            upper_base = max(
                _endpoints(upper), key=lambda point: _distancia_pontos(point, apex)
            )
            lower_base = max(
                _endpoints(lower), key=lambda point: _distancia_pontos(point, apex)
            )
            base_span = _distancia_pontos(upper_base, lower_base)
            if not (
                abs(upper_base[0] - lower_base[0]) <= 0.10 * mind
                and 0.12 * mind <= abs(upper_base[1] - lower_base[1]) <= 0.65 * mind
                and base_span >= 0.14 * mind
            ):
                continue
            base_center = (
                (upper_base[0] + lower_base[0]) / 2.0,
                (upper_base[1] + lower_base[1]) / 2.0,
            )
            base_line = next((
                line for line in vertical
                if abs(line["mx"] - base_center[0]) <= tolerance
                and min(upper_base[1], lower_base[1]) - tolerance
                <= line["my"]
                <= max(upper_base[1], lower_base[1]) + tolerance
            ), None)
            if base_line is None:
                continue

            vertices = (upper_base, lower_base, apex)
            lead_count = 0
            for lead in horizontal:
                if min(
                    _distancia_pontos(endpoint, vertex)
                    for endpoint in _endpoints(lead)
                    for vertex in vertices
                ) <= tolerance:
                    lead_count += 1
            if lead_count < 2:
                continue

            xs = [point[0] for point in vertices]
            ys = [point[1] for point in vertices]
            major = max(max(xs) - min(xs), max(ys) - min(ys))
            if major < 0.18 * mind:
                continue
            marker_margin = max(10.0, 0.10 * mind)
            matching_marks = any(
                all(
                    min(xs) - marker_margin <= mark[0] <= max(xs) + marker_margin
                    and min(ys) - marker_margin <= mark[1] <= max(ys) + marker_margin
                    for mark in pair
                )
                for pair in polarity_pairs
            )
            if not matching_marks:
                continue
            pad = max(8.0, 0.12 * major)
            boxes.append([
                max(0.0, min(xs) - pad),
                max(0.0, min(ys) - pad),
                min(float(w - 1), max(xs) + pad),
                min(float(h - 1), max(ys) + pad),
            ])

    detections = montar_deteccoes(
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.full(len(boxes), 0.91, dtype=np.float32),
        np.full(len(boxes), 14, dtype=np.float32),
        {14: "opamp"},
        inference_source="opamp_triangle_geometry",
    )
    return remover_duplicadas(
        detections, iou_threshold=0.25, cobertura_threshold=0.65
    )


def _deteccoes_gnd_geometrico(imagem):
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    bordas = cv2.Canny(cinza, 40, 140)
    linhas = cv2.HoughLinesP(
        bordas, 1, np.pi / 180, threshold=10,
        minLineLength=6, maxLineGap=5,
    )
    if linhas is None:
        return []
    horizontais = []
    verticais = []
    for x1, y1, x2, y2 in linhas.reshape(-1, 4):
        dx = abs(int(x2) - int(x1))
        dy = abs(int(y2) - int(y1))
        if dy <= 2 and 6 <= dx <= 150:
            esquerda, direita = sorted((int(x1), int(x2)))
            horizontais.append({
                "x1": esquerda, "x2": direita,
                "y": (int(y1) + int(y2)) / 2.0,
                "len": direita - esquerda,
                "cx": (esquerda + direita) / 2.0,
            })
        elif dx <= 3 and dy >= 10:
            verticais.append({
                "x": (int(x1) + int(x2)) / 2.0,
                "y1": min(int(y1), int(y2)),
                "y2": max(int(y1), int(y2)),
            })
    horizontais.sort(key=lambda item: item["len"], reverse=True)
    unicas = []
    for linha in horizontais:
        if not any(
            abs(linha["y"] - salva["y"]) <= 4
            and abs(linha["cx"] - salva["cx"]) <= 7
            and min(linha["x2"], salva["x2"]) > max(linha["x1"], salva["x1"])
            for salva in unicas
        ):
            unicas.append(linha)
    unicas.sort(key=lambda item: item["y"])
    caixas = []
    for superior, meio, inferior in itertools.combinations(unicas, 3):
        espaco_a = meio["y"] - superior["y"]
        espaco_b = inferior["y"] - meio["y"]
        centros = [superior["cx"], meio["cx"], inferior["cx"]]
        if not (5 <= espaco_a <= 25 and 5 <= espaco_b <= 25):
            continue
        if max(centros) - min(centros) > 10:
            continue
        if not (
            20 <= superior["len"] <= 120
            and superior["len"] > meio["len"] * 1.15
            and meio["len"] > inferior["len"] * 1.15
        ):
            continue
        if not (
            0.25 <= meio["len"] / superior["len"] <= 0.85
            and 0.12 <= inferior["len"] / superior["len"] <= 0.65
        ):
            continue
        centro_x = float(np.mean(centros))
        if not any(
            abs(vertical["x"] - centro_x) <= 9
            and vertical["y1"] < superior["y"] - 8
            and abs(vertical["y2"] - superior["y"]) <= 12
            for vertical in verticais
        ):
            continue
        caixas.append([
            max(0, min(superior["x1"], meio["x1"], inferior["x1"]) - 6),
            max(0, int(superior["y"] - 35)),
            min(imagem.shape[1] - 1, max(superior["x2"], meio["x2"], inferior["x2"]) + 6),
            min(imagem.shape[0] - 1, int(inferior["y"] + 7)),
        ])
    deteccoes = montar_deteccoes(
        np.asarray(caixas, dtype=np.float32).reshape(-1, 4),
        np.full(len(caixas), 0.86, dtype=np.float32),
        np.full(len(caixas), 8, dtype=np.float32),
        {8: "gnd"},
        inference_source="gnd_line_geometry",
    )
    return remover_duplicadas(
        deteccoes, iou_threshold=0.25, cobertura_threshold=0.65
    )



def _recortar_bbox_validacao(imagem, bbox, padding_frac=0.10):
    """Recorta uma detecção com pequena margem para validar a geometria real."""
    h_img, w_img = imagem.shape[:2]
    x1 = float(bbox["x1"]); y1 = float(bbox["y1"])
    x2 = float(bbox["x2"]); y2 = float(bbox["y2"])
    bw = max(1.0, x2 - x1); bh = max(1.0, y2 - y1)
    pad = max(3, int(round(min(bw, bh) * float(padding_frac))))
    ix1 = max(0, int(np.floor(x1)) - pad)
    iy1 = max(0, int(np.floor(y1)) - pad)
    ix2 = min(w_img, int(np.ceil(x2)) + pad + 1)
    iy2 = min(h_img, int(np.ceil(y2)) + pad + 1)
    return imagem[iy1:iy2, ix1:ix2]


def _linhas_estruturais(roi):
    """Retorna segmentos Hough com ângulo normalizado em [-90, 90]."""
    if roi is None or roi.size == 0 or min(roi.shape[:2]) < 8:
        return []
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    if float(np.median(border)) < 127.0:
        gray = 255 - gray
    edges = cv2.Canny(gray, 35, 120)
    h, w = gray.shape[:2]
    base = max(8, int(min(h, w) * 0.11))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=max(8, int(min(h, w) * 0.07)),
        minLineLength=base,
        maxLineGap=max(3, int(min(h, w) * 0.05)),
    )
    if lines is None:
        return []
    result = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        dx = float(x2 - x1); dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < base:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0
        result.append({
            "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
            "mx": (float(x1) + float(x2)) / 2.0,
            "my": (float(y1) + float(y2)) / 2.0,
            "length": length,
            "angle": angle,
        })
    return result


def _distancia_pontos(a, b):
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _endpoints(seg):
    return ((seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]))


def _mascara_tinta_local(roi):
    """Gera máscara binária de traços/símbolos para análise topológica local."""
    if roi is None or roi.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    if float(np.median(border)) < 127.0:
        gray = 255 - gray

    # O limiar por percentil antigo classificava quase toda uma fotografia de
    # papel cinza como tinta. No caso real analisado, um indutor legítimo ficou
    # com 79% de "tinta" e foi rejeitado. Estimamos o fundo local por fechamento
    # morfológico: textura, sombra e pauta suave são absorvidas pelo fundo,
    # enquanto lápis/caneta escuros permanecem no mapa de contraste.
    h, w = gray.shape[:2]
    kernel_size = max(15, int(round(min(h, w) / 3.0)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    background = cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        ),
    )
    dark_contrast = cv2.subtract(background, gray)
    _, mask = cv2.threshold(
        dark_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Otsu pode escolher zero em recortes quase vazios. O piso remove ruído
    # fotográfico sem apagar os traços finos do símbolo.
    contrast_floor = max(
        10,
        min(35, int(round(float(np.percentile(dark_contrast, 92)) * 0.45))),
    )
    mask[dark_contrast < contrast_floor] = 0
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((2, 2), np.uint8),
        iterations=1,
    )


def _encontrar_no_preenchido(imagem, bbox):
    """Procura uma 'bolinha' de nó perto da detecção VCC.

    Retorna centro global, raio estimado, máscara local e offsets. A bolinha é
    identificada por espessura local maior que a espessura típica dos fios.
    """
    h_img, w_img = imagem.shape[:2]
    x1 = float(bbox['x1']); y1 = float(bbox['y1'])
    x2 = float(bbox['x2']); y2 = float(bbox['y2'])
    bw = max(1.0, x2 - x1); bh = max(1.0, y2 - y1)

    contexto = int(round(max(24.0, min(90.0, 0.9 * max(bw, bh)))))
    cx_box = (x1 + x2) / 2.0
    cy_box = (y1 + y2) / 2.0
    ix1 = max(0, int(np.floor(cx_box - contexto)))
    iy1 = max(0, int(np.floor(cy_box - contexto)))
    ix2 = min(w_img, int(np.ceil(cx_box + contexto + 1)))
    iy2 = min(h_img, int(np.ceil(cy_box + contexto + 1)))
    roi = imagem[iy1:iy2, ix1:ix2]
    if roi.size == 0:
        return None

    mask = _mascara_tinta_local(roi)
    if mask.size == 0 or np.count_nonzero(mask) == 0:
        return None

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Restringe a busca ao entorno da bounding box original. Isso evita pegar
    # resistor/texto grosso que esteja próximo, mas fora da detecção VCC.
    sx1 = max(0, int(np.floor(x1)) - ix1 - 4)
    sy1 = max(0, int(np.floor(y1)) - iy1 - 4)
    sx2 = min(mask.shape[1], int(np.ceil(x2)) - ix1 + 5)
    sy2 = min(mask.shape[0], int(np.ceil(y2)) - iy1 + 5)
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    sub = dist[sy1:sy2, sx1:sx2]
    _, maxval, _, maxloc = cv2.minMaxLoc(sub)
    if float(maxval) < 2.4:
        return None

    cx = sx1 + int(maxloc[0])
    cy = sy1 + int(maxloc[1])
    dot_radius = float(maxval)

    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    anel = (radial >= dot_radius + 3.0) & (radial <= dot_radius + 18.0) & (mask > 0)
    line_vals = dist[anel]
    line_half = float(np.median(line_vals)) if line_vals.size else 1.0
    line_half = max(0.8, line_half)

    # Uma bolinha real precisa ser claramente mais grossa que um fio comum.
    thickness_ratio = dot_radius / line_half
    if dot_radius < 3.0 or thickness_ratio < 1.45:
        return None

    # Remove traços com a espessura típica dos fios. Uma junção/ápice formado
    # apenas por linhas (por exemplo o chevron do VCC clássico) desaparece; uma
    # bolinha preenchida continua existindo. Isso impede que o ápice do símbolo
    # seja confundido com um nó circular.
    opening_radius = max(3, int(np.ceil(2.5 * line_half)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * opening_radius + 1, 2 * opening_radius + 1),
    )
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if opened[cy, cx] == 0:
        return None

    return {
        'center_local': (float(cx), float(cy)),
        'center_global': (float(cx + ix1), float(cy + iy1)),
        'dot_radius': float(dot_radius),
        'line_half_thickness': float(line_half),
        'thickness_ratio': float(thickness_ratio),
        'mask': mask,
        'offset': (ix1, iy1),
    }


def _contar_ramos_do_no(mask, center, dot_radius, line_half_thickness=1.5):
    """Conta quantos condutores saem de uma bolinha de conexão.

    A bolinha é removida da máscara. Cada componente conectado que toca o anel
    imediatamente externo representa um ramo: endpoint=1, fio passante=2,
    junção em T=3, cruzamento=4.
    """
    if mask is None or mask.size == 0:
        return 0, []
    cx, cy = map(float, center)
    h, w = mask.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    corte = max(float(dot_radius) + 2.0, float(dot_radius) * 1.18)
    probe = corte + max(6.0, 3.0 * float(line_half_thickness))
    longe = probe + max(7.0, 3.0 * float(line_half_thickness))

    work = (mask > 0).astype(np.uint8)
    work[radial <= corte] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(work, 8)

    ramos = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(6, int(round(3.0 * line_half_thickness))):
            continue
        pixels = labels == label
        toca_probe = np.any(pixels & (radial >= corte - 1.0) & (radial <= probe))
        estende = np.any(pixels & (radial >= longe))
        if not (toca_probe and estende):
            continue
        ys, xs = np.nonzero(pixels & (radial >= corte - 1.0) & (radial <= probe))
        if len(xs):
            ang = float(np.degrees(np.arctan2(np.mean(ys) - cy, np.mean(xs) - cx)))
        else:
            ang = 0.0
        ramos.append({'label': int(label), 'area': area, 'angle_deg': round(ang, 1)})
    return len(ramos), ramos


def _mascara_primeiro_plano_local(roi):
    """Separa tinta/fios do fundo sem depender da cor do diagrama."""
    # A mesma máscara usada pelos validadores deve ser usada na contagem de
    # ramos de VCC. A implementação anterior incluía pautas inteiras por causa
    # da diferença de cor do papel e transformava um terminal em junção T/X.
    return _mascara_tinta_local(roi)


def _analisar_topologia_no(imagem, bbox):
    """Localiza uma bolinha/nó e conta quantos ramos elétricos saem dela.

    A bolinha é procurada pelo pico da transformada de distância: um nó
    preenchido é mais grosso que os fios. Depois retiramos a região central e
    contamos os componentes que atravessam um anel ao redor do nó. Assim:
    endpoint/VCC -> 1 ramo; fio passante -> 2; T -> 3; cruzamento -> 4.
    """
    h_img, w_img = imagem.shape[:2]
    x1 = float(bbox["x1"]); y1 = float(bbox["y1"])
    x2 = float(bbox["x2"]); y2 = float(bbox["y2"])
    bw = max(1.0, x2 - x1); bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0

    # Janela propositalmente maior que a caixa YOLO para enxergar os ramos que
    # saem do nó. Isto é essencial para diferenciar VCC de uma junção em T.
    half = int(max(70.0, 3.5 * max(bw, bh)))
    rx1 = max(0, int(round(cx)) - half)
    ry1 = max(0, int(round(cy)) - half)
    rx2 = min(w_img, int(round(cx)) + half + 1)
    ry2 = min(h_img, int(round(cy)) + half + 1)
    roi = imagem[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return {"node_found": False, "reason": "empty_crop"}

    mask = _mascara_primeiro_plano_local(roi)
    if mask.size == 0 or np.count_nonzero(mask) < 8:
        return {"node_found": False, "reason": "no_foreground"}

    binary = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # O pico deve estar dentro/próximo da caixa proposta pelo YOLO, não em um
    # texto ou componente vizinho da janela expandida.
    pad_x = 0.25 * bw; pad_y = 0.25 * bh
    sx1 = max(0, int(np.floor(x1 - rx1 - pad_x)))
    sy1 = max(0, int(np.floor(y1 - ry1 - pad_y)))
    sx2 = min(roi.shape[1], int(np.ceil(x2 - rx1 + pad_x)) + 1)
    sy2 = min(roi.shape[0], int(np.ceil(y2 - ry1 + pad_y)) + 1)
    search = dist[sy1:sy2, sx1:sx2]
    if search.size == 0:
        return {"node_found": False, "reason": "empty_search"}

    _, peak, _, peak_loc = cv2.minMaxLoc(search)
    if not np.isfinite(peak) or float(peak) <= 0.0 or float(peak) > 1e5:
        return {"node_found": False, "reason": "invalid_distance_peak"}
    px = sx1 + int(peak_loc[0]); py = sy1 + int(peak_loc[1])

    nonzero_dist = dist[binary > 0]
    stroke_ref = float(np.percentile(nonzero_dist, 70)) if nonzero_dist.size else 0.0
    thickness_ratio = float(peak) / max(1.0, stroke_ref)

    # Uma simples linha possui pico parecido com a própria espessura; uma
    # bolinha preenchida produz um pico claramente maior.
    node_found = bool(float(peak) >= 4.6 and thickness_ratio >= 1.55)
    if not node_found:
        return {
            "node_found": False,
            "reason": "no_filled_node",
            "distance_peak": round(float(peak), 3),
            "stroke_reference": round(stroke_ref, 3),
            "thickness_ratio": round(thickness_ratio, 3),
        }

    node_radius = float(peak)
    inner = max(4.0, 1.35 * node_radius)
    outer = max(inner + 5.0, 3.10 * node_radius)
    # A janela pode ser cortada na borda da imagem; não deixe o anel ultrapassar
    # demais a área disponível.
    max_possible = max(
        8.0,
        min(
            max(px, roi.shape[1] - 1 - px),
            max(py, roi.shape[0] - 1 - py),
        ) * 0.92,
    )
    outer = min(outer, max_possible)
    if outer <= inner + 2.0:
        return {
            "node_found": True,
            "branch_count": 0,
            "reason": "insufficient_context",
            "node_radius": round(node_radius, 2),
            "node_center": {"x": int(px + rx1), "y": int(py + ry1)},
        }

    yy, xx = np.mgrid[:roi.shape[0], :roi.shape[1]]
    rr = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
    annulus = (
        (rr >= inner) & (rr <= outer) & (binary > 0)
    ).astype(np.uint8)

    count, labels, _, _ = cv2.connectedComponentsWithStats(annulus, 8)
    branches = 0
    branch_sizes = []
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        if xs.size < 3:
            continue
        radii = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)
        # Um ramo verdadeiro atravessa praticamente todo o anel; pequenas
        # letras/ruído que só aparecem em uma parte do anel são ignoradas.
        if (
            float(radii.min()) <= inner + 2.8
            and float(radii.max()) >= outer - 2.8
        ):
            branches += 1
            branch_sizes.append(int(xs.size))

    return {
        "node_found": True,
        "branch_count": int(branches),
        "node_radius": round(node_radius, 2),
        "node_center": {"x": int(px + rx1), "y": int(py + ry1)},
        "inner_radius": round(float(inner), 2),
        "outer_radius": round(float(outer), 2),
        "distance_peak": round(float(peak), 3),
        "stroke_reference": round(stroke_ref, 3),
        "thickness_ratio": round(thickness_ratio, 3),
        "branch_sizes": branch_sizes,
    }


def _validar_vcc_chevron(imagem, bbox):
    """Fallback para o símbolo VCC clássico (seta/chevron + haste)."""
    bw = max(1.0, float(bbox["x2"]) - float(bbox["x1"]))
    bh = max(1.0, float(bbox["y2"]) - float(bbox["y1"]))
    aspect = bh / bw
    roi = _recortar_bbox_validacao(imagem, bbox, 0.12)
    if roi.size == 0:
        return False, 0.0, {"reason": "empty_crop"}
    h, w = roi.shape[:2]
    lines = _linhas_estruturais(roi)

    verticals = [
        l for l in lines
        if abs(abs(l["angle"]) - 90.0) <= 22.0
        and l["length"] >= 0.26 * h
        and 0.25 * w <= l["mx"] <= 0.75 * w
        and l["my"] >= 0.38 * h
    ]
    diag_pos = [
        l for l in lines
        if 24.0 <= l["angle"] <= 72.0
        and l["length"] >= 0.14 * min(w, h)
        and l["my"] <= 0.48 * h
    ]
    diag_neg = [
        l for l in lines
        if -72.0 <= l["angle"] <= -24.0
        and l["length"] >= 0.14 * min(w, h)
        and l["my"] <= 0.48 * h
    ]

    pair_ok = False
    alignment_ok = False
    best_apex_distance = None
    head_bbox=None
    if verticals and diag_pos and diag_neg:
        stem = max(verticals, key=lambda x: x["length"])
        stem_top = min(_endpoints(stem), key=lambda p: p[1])
        for a in diag_pos:
            for b in diag_neg:
                pairs = [
                    (pa, pb, _distancia_pontos(pa, pb))
                    for pa in _endpoints(a) for pb in _endpoints(b)
                ]
                pa, pb, dist = min(pairs, key=lambda t: t[2])
                if best_apex_distance is None or dist < best_apex_distance:
                    best_apex_distance = dist
                apex = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
                other_a = max(_endpoints(a), key=lambda p: _distancia_pontos(p, pa))
                other_b = max(_endpoints(b), key=lambda p: _distancia_pontos(p, pb))
                lower_y = (other_a[1] + other_b[1]) / 2.0
                if dist <= max(10.0, 0.22 * min(w, h)) and apex[1] <= 0.38 * h:
                    pair_ok = True
                    if (
                        abs(stem_top[0] - apex[0]) <= 0.20 * w
                        and abs(stem_top[1] - lower_y) <= 0.24 * h
                    ):
                        alignment_ok = True
                        pad=max(3,int(round(min(bw,bh)*.12)))
                        ox=max(0,int(np.floor(bbox['x1']))-pad);oy=max(0,int(np.floor(bbox['y1']))-pad)
                        head_points=np.array(_endpoints(a)+_endpoints(b),float)
                        lo=head_points.min(axis=0);hi=head_points.max(axis=0)
                        head_bbox=dict(x1=float(lo[0]+ox-2),y1=float(lo[1]+oy-2),x2=float(hi[0]+ox+2),y2=float(hi[1]+oy+2))
                        break
            if alignment_ok:
                break

    score = 0.0
    if aspect >= 1.18:
        score += 0.15
    if verticals:
        score += 0.30
    if diag_pos and diag_neg:
        score += 0.30
    if pair_ok:
        score += 0.15
    if alignment_ok:
        score += 0.10
    valid = bool(verticals and diag_pos and diag_neg and pair_ok and alignment_ok and score >= 0.72)
    return valid, round(score, 3), {
        "aspect_h_over_w": round(aspect, 3),
        "vertical_lines": len(verticals),
        "positive_diagonals": len(diag_pos),
        "negative_diagonals": len(diag_neg),
        "chevron_pair": bool(pair_ok),
        "stem_alignment": bool(alignment_ok),
        "apex_distance": None if best_apex_distance is None else round(best_apex_distance, 2),
        "power_head_bbox":head_bbox,
    }


def _validar_geometria_vcc(imagem,bbox):
    from backend.vcc_topology import external_leads,hollow_power_body
    from backend.resistor_geometry import symbol_mask
    page_mask=symbol_mask(imagem)
    body=hollow_power_body(imagem,bbox,page_mask)
    if body:
        count=external_leads(imagem,body,page_mask)
        if count==1:return True,.94,{'validation_mode':'hollow_power_terminal','external_lead_count':1,'connection_branch_count':1,'junction_rejected':False,'power_head_bbox':body}
    leads=external_leads(imagem,bbox,page_mask)
    topo=_analisar_topologia_no(imagem,bbox)
    center=topo.get('node_center') or {}
    node_inside=bool(topo.get('node_found') and bbox['x1']-2<=center.get('x',-999)<=bbox['x2']+2 and bbox['y1']-2<=center.get('y',-999)<=bbox['y2']+2)
    if leads>1 or (node_inside and topo.get('branch_count')!=1):
        count=max(leads,int(topo.get('branch_count',0)))
        return False,.05,{**topo,'validation_mode':'connection_dot_topology','connection_branch_count':count,'external_lead_count':leads,'junction_rejected':True,'hard_topology_rejection':True}
    body=hollow_power_body(imagem,bbox,page_mask)
    if body:
        count=external_leads(imagem,body,page_mask)
        return count==1,.94 if count==1 else .05,{'validation_mode':'hollow_power_terminal','external_lead_count':count,'connection_branch_count':count,'junction_rejected':count>1,'hard_topology_rejection':count!=1}
    if node_inside:
        return True,.97,{**topo,'validation_mode':'connection_dot_topology','connection_branch_count':1,'external_lead_count':leads,'junction_rejected':False,'chevron_pair':False}
    ok,score,evidence=_validar_vcc_chevron(imagem,bbox)
    if ok and evidence.get('power_head_bbox'):
        count=external_leads(imagem,evidence['power_head_bbox'],page_mask)
        ok=count==1;evidence.update(external_lead_count=count,hard_topology_rejection=count!=1,junction_rejected=count>1)
    evidence.update(validation_mode='chevron_fallback',node_topology=topo)
    return ok,score,evidence



def _ponto_para_segmento_distancia(p, seg):
    px, py = map(float, p)
    x1, y1, x2, y2 = map(float, (seg["x1"], seg["y1"], seg["x2"], seg["y2"]))
    vx, vy = x2 - x1, y2 - y1
    denom = vx * vx + vy * vy
    if denom <= 1e-9:
        return _distancia_pontos((px, py), (x1, y1))
    t = max(0.0, min(1.0, ((px - x1) * vx + (py - y1) * vy) / denom))
    qx, qy = x1 + t * vx, y1 + t * vy
    return _distancia_pontos((px, py), (qx, qy))


@cached_vision
def _bjt_connected_fork(imagem,bbox):
    from backend.terminal_identity import _crop
    from backend.npn_wire_contacts import _external_contacts,_traced_body_geometries
    mask,origin=_crop(imagem,bbox)
    # Recognition precedes wire extraction. These contacts require actual ink
    # continuing through the box edge; no inferred stage-8 graph is consulted.
    contacts=_external_contacts(mask,origin,bbox,None)
    if not 3<=len(contacts)<=4:return None
    shapes=_traced_body_geometries(mask,origin,bbox,contacts)
    if not shapes:return None
    _,bar,a,b,lead=shapes[0]
    return {'connected_fork_trace':True,'external_lead_count':len(contacts),
            'base_lead_connected':True,'fork_connected':True,'base_bar_angle_degrees':bar['angle'],
            'base_vertical_lines':1,'metric':'three_ink_leads_to_base_and_two_distinct_branches'}

def _remove_compound_bjt_proposals(image,detections):
    """Do not let a large prediction vote for two adjacent physical BJTs.

    Partial overlap is normal. Reject only a box containing the centers of
    two independent, smaller predictions whose complete forks are verified.
    This runs before consensus, which otherwise merges the two symbols.
    """
    bjts=[d for d in detections if d['class'] in {'transistor_npn','transistor_pnp'}]
    kept=[];rejected=[]
    def area(b):return max(1,(b['x2']-b['x1'])*(b['y2']-b['y1']))
    for outer in detections:
        if outer not in bjts:kept.append(outer);continue
        box=outer['bbox'];size=area(box)
        smaller=[]
        for inner in bjts:
            b=inner['bbox']
            if inner is outer or inner['confidence']<.45 or size<1.35*area(b):continue
            x=(b['x1']+b['x2'])/2;y=(b['y1']+b['y2'])/2
            coverage=max(0,min(box['x2'],b['x2'])-max(box['x1'],b['x1']))*max(0,min(box['y2'],b['y2'])-max(box['y1'],b['y1']))/area(b)
            if box['x1']<=x<=box['x2'] and box['y1']<=y<=box['y2'] and coverage>=.35:smaller.append(inner)
        compound=False
        for i,a in enumerate(smaller):
            for b in smaller[i+1:]:
                ab=a['bbox'];bb=b['bbox']
                intersection=max(0,min(ab['x2'],bb['x2'])-max(ab['x1'],bb['x1']))*max(0,min(ab['y2'],bb['y2'])-max(ab['y1'],bb['y1']))
                if intersection/(area(ab)+area(bb)-intersection)>.20:continue
                if (_validar_geometria_transistor(image,ab,a['class'])[0]
                    and _validar_geometria_transistor(image,bb,b['class'])[0]):compound=True;break
            if compound:break
        if compound:rejected.append(dict(outer,rejection_reason='two_verified_bjt_forks_in_one_prediction'))
        else:kept.append(outer)
    return kept,rejected

@cached_vision
def _validar_geometria_transistor(imagem, bbox, classe):
    """Exige a forquilha BJT completa; uma diagonal isolada é rejeitada.

    A validação não decide NPN/PNP pela seta. A subclasse continua vindo do
    detector, mas só é aceita como transistor se houver: base vertical, chegada
    da base e os dois ramos diagonais de coletor/emissor.
    """
    bw = max(1.0, float(bbox["x2"]) - float(bbox["x1"]))
    bh = max(1.0, float(bbox["y2"]) - float(bbox["y1"]))
    aspect = bw / bh
    roi = _recortar_bbox_validacao(imagem, bbox, 0.10)
    if roi.size == 0:
        return False, 0.0, {"reason": "empty_crop"}
    h, w = roi.shape[:2]
    lines = _linhas_estruturais(roi)
    mind = float(min(w, h))

    base_verticals = [
        l for l in lines
        if abs(abs(l["angle"]) - 90.0) <= 24.0
        and l["length"] >= 0.18 * h
        and 0.07 * w <= l["mx"] <= 0.93 * w
        and 0.25 * h <= l["my"] <= 0.75 * h
    ]
    base_horiz = [
        l for l in lines
        if abs(l["angle"]) <= 22.0
        and l["length"] >= 0.14 * w
        and l["my"] >= 0.33 * h and l["my"] <= 0.67 * h
    ]
    # A orientação do segmento retornado pelo Hough não é garantida. Para
    # aceitar BJT normal e espelhado, classificamos os ramos pela posição
    # vertical e pelo módulo do ângulo, não pelo sinal do ângulo.
    upper_diag = [
        l for l in lines
        if 18.0 <= abs(l["angle"]) <= 72.0
        and l["length"] >= 0.14 * mind
        and l["my"] <= 0.58 * h
    ]
    lower_diag = [
        l for l in lines
        if 18.0 <= abs(l["angle"]) <= 72.0
        and l["length"] >= 0.14 * mind
        and l["my"] >= 0.42 * h
    ]

    fork_ok = False
    base_lead_ok = False
    chosen_base = None
    orientation = None
    for base in sorted(base_verticals, key=lambda x: x["length"], reverse=True):
        tol = max(8.0, 0.14 * mind)
        for direction in ("right", "left"):
            def branch_reaches(segment):
                endpoints = _endpoints(segment)
                touches = min(
                    _ponto_para_segmento_distancia(p, base) for p in endpoints
                ) <= tol
                if direction == "right":
                    extends = max(p[0] for p in endpoints) >= base["mx"] + 0.10 * w
                else:
                    extends = min(p[0] for p in endpoints) <= base["mx"] - 0.10 * w
                return touches and extends

            up_ok = any(branch_reaches(d) for d in upper_diag)
            down_ok = any(branch_reaches(d) for d in lower_diag)
            lead_ok = any(
                min(
                    _ponto_para_segmento_distancia(p, base)
                    for p in _endpoints(line)
                ) <= tol
                and (
                    min(p[0] for p in _endpoints(line)) <= base["mx"] - 0.10 * w
                    if direction == "right"
                    else max(p[0] for p in _endpoints(line)) >= base["mx"] + 0.10 * w
                )
                for line in base_horiz
            )
            if up_ok and down_ok:
                fork_ok = True
                base_lead_ok = bool(lead_ok)
                chosen_base = base
                orientation = direction
                break
        if fork_ok:
            break

    # Símbolos BJT do conjunto atual são aproximadamente quadrados/largos.
    aspect_ok = 0.55 <= aspect <= 2.10
    score = 0.0
    if aspect_ok:
        score += 0.10
    if base_verticals:
        score += 0.20
    if base_horiz:
        score += 0.15
    if upper_diag:
        score += 0.20
    if lower_diag:
        score += 0.20
    if fork_ok:
        score += 0.10
    if base_lead_ok:
        score += 0.05

    # Uma diagonal isolada, um V ou um X não passam: exigimos haste + dois
    # ramos e pelo menos indício da chegada da base.
    valid = bool(
        aspect_ok and chosen_base is not None and fork_ok and base_horiz
        and score >= 0.72
    )
    connected=None if valid else _bjt_connected_fork(imagem,bbox)
    if connected and not connected.get('multiple_bjt_bases'):
        return True,.92,{'transistor_class':str(classe),'aspect_w_over_h':round(aspect,3),
                         **connected,'subtype_source':'yolo_class_after_connected_fork_validation'}
    return valid, round(score, 3), {
        "transistor_class": str(classe),
        "aspect_w_over_h": round(aspect, 3),
        "base_vertical_lines": len(base_verticals),
        "base_horizontal_lines": len(base_horiz),
        "upper_fork_lines": len(upper_diag),
        "lower_fork_lines": len(lower_diag),
        "fork_connected": bool(fork_ok),
        "base_lead_connected": bool(base_lead_ok),
        "symbol_orientation": orientation,
        "subtype_source": "yolo_class_after_bjt_geometry_validation",
    }


def _validar_geometria_led(imagem, bbox):
    """Rejeita linhas pautadas sem corpo de LED.

    Uma linha de caderno gera quase somente segmentos horizontais. Um LED real
    possui barra/corpo e setas, portanto precisa de tinta e de pelo menos dois
    segmentos não horizontais dentro da caixa proposta pelo detector.
    """
    roi = _recortar_bbox_validacao(imagem, bbox, 0.06)
    if roi.size == 0:
        return False, 0.0, {"reason": "empty_crop"}
    mask = _mascara_tinta_local(roi)
    lines = _linhas_estruturais(roi)
    h, w = roi.shape[:2]
    ink_ratio = float(np.count_nonzero(mask)) / max(1.0, float(h * w))
    horizontal = [line for line in lines if abs(line["angle"]) <= 15.0]
    non_horizontal = [line for line in lines if abs(line["angle"]) > 18.0]
    diagonal = [
        line for line in lines if 20.0 <= abs(line["angle"]) <= 72.0
    ]
    vertical = [
        line for line in lines if abs(abs(line["angle"]) - 90.0) <= 20.0
    ]
    structural_length = sum(line["length"] for line in non_horizontal)
    min_dim = float(max(1, min(w, h)))
    complexity_ok = bool(
        len(non_horizontal) >= 2
        and structural_length >= 0.42 * min_dim
        and (diagonal or vertical)
    )
    # Caixas que contêm somente uma ou duas pautas longas são o padrão de
    # falso positivo observado após o treino.
    ruled_only = bool(
        horizontal
        and len(non_horizontal) < 2
        and max((line["length"] for line in horizontal), default=0.0) >= 0.55 * w
    )
    valid = bool(ink_ratio >= 0.012 and complexity_ok and not ruled_only)
    score = min(
        1.0,
        0.20 * min(1.0, ink_ratio / 0.04)
        + 0.18 * min(3, len(non_horizontal))
        + 0.16 * bool(diagonal)
        + 0.10 * bool(vertical),
    )
    return valid, round(float(score), 3), {
        "ink_ratio": round(ink_ratio, 4),
        "horizontal_lines": len(horizontal),
        "non_horizontal_lines": len(non_horizontal),
        "diagonal_lines": len(diagonal),
        "vertical_lines": len(vertical),
        "ruled_line_only": ruled_only,
    }


def _evidencia_setas_emissao_led(imagem, bbox):
    """Detecta o par de setas luminosas que distingue LED de diodo.

    As setas normalmente ficam desconectadas do corpo do diodo e formam dois
    pequenos grupos de tinta próximos, de tamanho semelhante, no mesmo lado
    do símbolo. A análise por componentes conectados é mais estável em desenho
    a lápis do que tentar inferir a classe apenas pelos ângulos do Hough.
    """
    roi = _recortar_bbox_validacao(imagem, bbox, 0.10)
    if roi.size == 0:
        return False, {"reason": "empty_crop"}
    mask = _mascara_tinta_local(roi)
    h, w = mask.shape[:2]
    area_roi = float(max(1, h * w))
    count, _, stats, centers = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    components = []
    for index in range(1, count):
        x, y, bw, bh, area = [int(value) for value in stats[index]]
        cx, cy = map(float, centers[index])
        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        if not (0.0015 * area_roi <= area <= 0.035 * area_roi):
            continue
        if min(bw, bh) < 3:
            continue
        if aspect > 4.5:
            # Pauta/fio fino atravessando a caixa, não uma seta.
            continue
        components.append({
            "area": area,
            "cx": cx,
            "cy": cy,
            "width": bw,
            "height": bh,
        })

    min_dim = float(max(1, min(w, h)))
    pairs = []
    for first, second in itertools.combinations(components, 2):
        distance = _distancia_pontos(
            (first["cx"], first["cy"]),
            (second["cx"], second["cy"]),
        )
        area_ratio = first["area"] / max(1.0, float(second["area"]))
        if not (0.35 <= area_ratio <= 2.85):
            continue
        if not (0.06 * min_dim <= distance <= 0.34 * min_dim):
            continue
        # As duas setas aparecem juntas em uma região periférica, enquanto
        # pequenos defeitos do corpo tendem a ficar espalhados no centro.
        pair_cx = (first["cx"] + second["cx"]) / 2.0
        pair_cy = (first["cy"] + second["cy"]) / 2.0
        away_from_crop_edge = bool(
            0.08 * w <= pair_cx <= 0.92 * w
            and 0.08 * h <= pair_cy <= 0.92 * h
        )
        peripheral = bool(
            pair_cx <= 0.34 * w or pair_cx >= 0.66 * w
            or pair_cy <= 0.34 * h or pair_cy >= 0.66 * h
        )
        if not (away_from_crop_edge and peripheral):
            continue
        pairs.append({
            "distance": round(distance, 2),
            "areas": [int(first["area"]), int(second["area"])],
            "center": [round(pair_cx, 2), round(pair_cy, 2)],
        })

    return bool(pairs), {
        "candidate_small_components": len(components),
        "emission_ray_pairs": pairs,
        "method": "paired_disconnected_dark_strokes",
    }


def _validar_geometria_diodo(imagem, bbox):
    """Exige triângulo, barra e ligação; pautas isoladas são rejeitadas."""
    roi = _recortar_bbox_validacao(imagem, bbox, 0.14)
    if roi.size == 0:
        return False, 0.0, {"reason": "empty_crop"}
    lines = _linhas_estruturais(roi)
    mask = _mascara_tinta_local(roi)
    h, w = roi.shape[:2]
    mind = float(max(1, min(h, w)))
    positive = [
        line for line in lines
        if 18.0 <= line["angle"] <= 72.0
        and line["length"] >= 0.14 * mind
    ]
    negative = [
        line for line in lines
        if -72.0 <= line["angle"] <= -18.0
        and line["length"] >= 0.14 * mind
    ]
    horizontal = [
        line for line in lines
        if abs(line["angle"]) <= 16.0
        and line["length"] >= 0.16 * mind
    ]
    vertical = [
        line for line in lines
        if abs(abs(line["angle"]) - 90.0) <= 16.0
        and line["length"] >= 0.16 * mind
    ]
    tolerance = max(7.0, 0.16 * mind)
    triangle_pair = any(
        min(
            _distancia_pontos(a, b)
            for a in _endpoints(up)
            for b in _endpoints(down)
        ) <= tolerance
        for up in positive
        for down in negative
    )

    filled_triangle = False
    if mask.size:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        roi_area = float(max(1, h * w))
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not 0.015 * roi_area <= area <= 0.55 * roi_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            vertices = cv2.approxPolyDP(
                contour, 0.055 * perimeter, True
            ).reshape(-1, 2)
            if len(vertices) == 3:
                filled_triangle = True
                break

    # Diagrama pode estar horizontal ou rotacionado em 90 graus.
    bar_and_lead = bool(
        (vertical and horizontal) or (horizontal and vertical)
    )
    ink_ratio = float(np.count_nonzero(mask)) / max(1.0, float(h * w))
    valid = bool(
        (triangle_pair or filled_triangle)
        and bar_and_lead
        and ink_ratio >= 0.018
    )
    score = min(
        1.0,
        0.38 * bool(triangle_pair)
        + 0.30 * bool(filled_triangle)
        + 0.18 * bool(horizontal)
        + 0.14 * bool(vertical),
    )
    return valid, round(float(score), 3), {
        "positive_diagonals": len(positive),
        "negative_diagonals": len(negative),
        "horizontal_lines": len(horizontal),
        "vertical_lines": len(vertical),
        "triangle_pair": bool(triangle_pair),
        "filled_triangle": bool(filled_triangle),
        "bar_and_lead": bool(bar_and_lead),
        "ink_ratio": round(ink_ratio, 4),
    }


def _validar_geometria_resistor(imagem,bbox,page_mask=None):
    from backend.resistor_geometry import iec_bodies,zigzag_paths,symbol_mask
    # A tight crop can cut off the confirming leads or change the scale limit
    # of an otherwise verified body. Match it in the original page first.
    area=max(1,(bbox['x2']-bbox['x1'])*(bbox['y2']-bbox['y1']))
    for body in iec_bodies(imagem):
        b=body['bbox'];cx,cy=body['center'];body_area=(b['x2']-b['x1'])*(b['y2']-b['y1'])
        overlap=max(0,min(b['x2'],bbox['x2'])-max(b['x1'],bbox['x1']))*max(0,min(b['y2'],bbox['y2'])-max(b['y1'],bbox['y1']))
        if bbox['x1']<=cx<=bbox['x2'] and bbox['y1']<=cy<=bbox['y2'] and overlap>=.70*body_area and body_area>=.15*area:
            return True,body['score'],{**body['evidence'],'validation_context':'original_page_body_and_leads'}
    roi=_recortar_bbox_validacao(imagem,bbox,.35)
    if roi.size==0:return False,0.,{'reason':'empty_crop'}
    if page_mask is None:page_mask=symbol_mask(imagem)
    pad=max(3,int(round(min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])*.35)))
    ix1=max(0,int(np.floor(bbox['x1']))-pad);iy1=max(0,int(np.floor(bbox['y1']))-pad)
    mask=page_mask[iy1:iy1+roi.shape[0],ix1:ix1+roi.shape[1]]
    candidates=iec_bodies(roi,mask)
    if not candidates:candidates=zigzag_paths(roi,mask)
    # A resistor must account for the proposed symbol, not a small unrelated
    # letter/component inside an oversized prediction.
    bw=max(1,bbox['x2']-bbox['x1']);bh=max(1,bbox['y2']-bbox['y1'])
    candidates=[c for c in candidates if
        (c['bbox']['x2']-c['bbox']['x1'])*(c['bbox']['y2']-c['bbox']['y1'])>=.15*bw*bh]
    if not candidates:return False,0.,{'reason':'no_body_with_two_opposed_leads','text_like_without_zigzag':True}
    best=max(candidates,key=lambda c:c['score'])
    return True,best['score'],best['evidence']




def _linhas_unicas_por_posicao(lines, orientation, tolerance=4.0):
    selected = []
    for line in sorted(lines, key=lambda item: item["length"], reverse=True):
        position = line["mx"] if orientation == "vertical" else line["my"]
        if any(
            abs(position - (
                saved["mx"] if orientation == "vertical" else saved["my"]
            )) <= tolerance
            for saved in selected
        ):
            continue
        selected.append(line)
    return selected


def _validar_geometria_indutor(imagem, bbox, page_mask=None):
    """Exige corpo curvo repetitivo; um fio ou pauta isolada não passa."""
    roi = _recortar_bbox_validacao(imagem, bbox, 0.08)
    if roi.size == 0:
        return False, 0.0, {"reason": "empty_crop"}
    mask = _mascara_tinta_local(roi)
    h, w = mask.shape[:2]
    aspect = max(w, h) / max(1.0, min(w, h))
    ink = int(np.count_nonzero(mask))
    ink_ratio = ink / max(1.0, float(w * h))
    major = max(w, h)
    if w >= h:
        straight_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(7, int(0.32 * major)), 1)
        )
    else:
        straight_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(7, int(0.32 * major)))
        )
    straight = cv2.morphologyEx(mask, cv2.MORPH_OPEN, straight_kernel)
    curved = cv2.bitwise_and(mask, cv2.bitwise_not(straight))
    curved_ratio = float(np.count_nonzero(curved)) / max(1.0, float(ink))
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (curved > 0).astype(np.uint8), 8
    )
    curved_parts = sum(
        1 for idx in range(1, count)
        if int(stats[idx, cv2.CC_STAT_AREA]) >= max(4, int(0.002 * w * h))
    )
    lines = _linhas_estruturais(roi)
    non_axis = [
        line for line in lines
        if 15.0 < abs(line["angle"]) < 75.0
    ]
    from backend.symbol_text_guard import coil_path_evidence
    connected_coil, path_evidence = False,{}
    if imagem.ndim==3:
        colored=_coil_color_mask(imagem)
        region=_recortar_bbox_validacao(colored,bbox,.08)
        if region.size and np.count_nonzero(region)>30:
            connected_coil,path_evidence=coil_path_evidence(imagem,bbox,colored)
    if not connected_coil:connected_coil,path_evidence=coil_path_evidence(imagem,bbox,page_mask)
    valid = bool(
        connected_coil
        and 0.02 <= ink_ratio <= 0.48
        and curved_ratio >= 0.28
    )
    score = min(1.0, 0.35 * min(1.0, curved_ratio / 0.45)
                + 0.20 * min(3, curved_parts)
                + 0.12 * min(3, len(non_axis)))
    return valid, round(float(score), 3), {
        "aspect_major_minor": round(float(aspect), 3),
        "ink_ratio": round(float(ink_ratio), 4),
        "curved_ink_ratio": round(float(curved_ratio), 3),
        "curved_parts": int(curved_parts),
        "non_axis_lines": len(non_axis),
        **path_evidence,
    }

@cached_vision
def _coil_color_mask(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    return ((hsv[:,:,1]>35)&(hsv[:,:,2]<235)).astype(np.uint8)*255


def _validar_geometria_bateria(imagem, bbox):
    """Bateria precisa de placas paralelas e dois terminais opostos."""
    from backend.vcc_topology import external_leads
    leads=external_leads(imagem,bbox)
    # A box including both lead ends has zero *external* contacts. Its
    # internal plate/lead pattern still needs the checks below.
    if leads==1:
        return False,0.,{'reason':'parallel_glyphs_without_two_battery_leads','external_lead_count':leads,'hard_text_rejection':True}
    roi = _recortar_bbox_validacao(imagem, bbox, 0.10)
    if roi.size == 0:
        return False, 0.0, {"reason": "empty_crop"}
    h, w = roi.shape[:2]
    lines = _linhas_estruturais(roi)
    vertical = _linhas_unicas_por_posicao([
        line for line in lines
        if abs(abs(line["angle"]) - 90.0) <= 15.0
        and line["length"] >= 0.18 * h
    ], "vertical", max(3.0, 0.025 * w))
    horizontal = _linhas_unicas_por_posicao([
        line for line in lines
        if abs(line["angle"]) <= 15.0
        and line["length"] >= 0.18 * w
    ], "horizontal", max(3.0, 0.025 * h))

    def plate_pattern(plates, orientation):
        if len(plates) < 3:
            return False
        centers = [p["my"] if orientation == "vertical" else p["mx"] for p in plates]
        lengths = [p["length"] for p in plates]
        aligned = max(centers) - min(centers) <= 0.32 * (
            h if orientation == "vertical" else w
        )
        length_variation = max(lengths) >= 1.22 * max(1.0, min(lengths))
        if not(aligned and length_variation):return False
        if leads>=2:return True
        # Lead ends can lie inside a generous detector box. In that case
        # require perpendicular wires reaching the outermost plates from
        # opposite sides; the vertical strokes of '4K7' are insufficient.
        along='mx' if orientation=='vertical' else 'my';cross='my' if orientation=='vertical' else 'mx'
        low,high=min(p[along] for p in plates),max(p[along] for p in plates)
        middle=float(np.median([p[cross] for p in plates]));tol=max(5,min(w,h)*.13)
        connectors=horizontal if orientation=='vertical' else vertical
        left=right=False
        for line in connectors:
            ends=[p[0] if orientation=='vertical' else p[1] for p in _endpoints(line)]
            if abs(line[cross]-middle)>max(tol,min(w,h)*.20):continue
            left|=min(ends)<low-tol and min(abs(e-low) for e in ends)<=tol
            right|=max(ends)>high+tol and min(abs(e-high) for e in ends)<=tol
        return bool(left and right)

    vertical_ok = plate_pattern(vertical, "vertical")
    horizontal_ok = plate_pattern(horizontal, "horizontal")
    valid = bool(vertical_ok or horizontal_ok)
    return valid, round(0.92 if valid else 0.12, 3), {
        "vertical_plates": len(vertical),
        "horizontal_plates": len(horizontal),
        "vertical_pattern": vertical_ok,
        "horizontal_pattern": horizontal_ok,
    }


def _analisar_gnd_relaxado_roi(roi):
    """Analisa GND na orientação canônica: haste vertical e três barras."""
    lines = _linhas_estruturais(roi)
    h, w = roi.shape[:2]
    horizontal = _linhas_unicas_por_posicao([
        line for line in lines
        if abs(line["angle"]) <= 18.0
        and 0.10 * w <= line["length"] <= 0.95 * w
    ], "horizontal", max(3.0, 0.035 * h))
    horizontal.sort(key=lambda line: line["my"])
    vertical = [
        line for line in lines
        if abs(abs(line["angle"]) - 90.0) <= 20.0
        and line["length"] >= 0.14 * h
    ]
    relaxed_triplet = False
    for top, middle, bottom in itertools.combinations(horizontal, 3):
        lengths = [top["length"], middle["length"], bottom["length"]]
        gaps = [middle["my"] - top["my"], bottom["my"] - middle["my"]]
        centers = [top["mx"], middle["mx"], bottom["mx"]]
        descending = bool(
            lengths[0] >= 0.85 * lengths[1]
            and lengths[1] >= 0.85 * lengths[2]
            and lengths[0] >= 1.22 * lengths[2]
        )
        spacing = bool(
            min(gaps) >= 2.0
            and max(gaps) <= 0.34 * h
            and max(gaps) / max(1.0, min(gaps)) <= 2.8
        )
        centered = max(centers) - min(centers) <= 0.28 * w
        stem = any(
            abs(line["mx"] - float(np.mean(centers))) <= 0.20 * w
            and line["my"] <= top["my"]
            for line in vertical
        )
        if descending and spacing and centered and stem:
            relaxed_triplet = True
            break

    return bool(relaxed_triplet), {
        "relaxed_handdrawn_triplet": bool(relaxed_triplet),
        "horizontal_bar_candidates": len(horizontal),
        "vertical_stem_candidates": len(vertical),
    }


def _validar_geometria_gnd(imagem, bbox):
    roi = _recortar_bbox_validacao(imagem, bbox, 0.16)
    if roi.size == 0:
        return False, 0.0, {"reason": "empty_crop"}
    proposals = _deteccoes_gnd_geometrico(roi)
    orientations = (
        ("upright", roi),
        ("rotated_cw", cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)),
        ("rotated_ccw", cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    )
    relaxed_results = {}
    detected_orientation = None
    for orientation, oriented_roi in orientations:
        orientation_ok, orientation_evidence = _analisar_gnd_relaxado_roi(
            oriented_roi
        )
        relaxed_results[orientation] = orientation_evidence
        if orientation_ok and detected_orientation is None:
            detected_orientation = orientation

    valid = bool(proposals or detected_orientation is not None)
    return valid, round(0.92 if proposals else 0.78 if valid else 0.08, 3), {
        "gnd_geometry_proposals": len(proposals),
        "three_descending_bars_with_stem": valid,
        "relaxed_handdrawn_triplet": detected_orientation is not None,
        "detected_orientation": detected_orientation,
        "orientation_evidence": relaxed_results,
        # Mantidos no nível superior para compatibilidade com a lógica de
        # resgate parcial usada antes da correção rotacional.
        "horizontal_bar_candidates": max(
            item["horizontal_bar_candidates"]
            for item in relaxed_results.values()
        ),
        "vertical_stem_candidates": max(
            item["vertical_stem_candidates"]
            for item in relaxed_results.values()
        ),
    }


def _pares_placas_por_projecao(mask):
    """Recupera placas desenhadas à mão que o Hough fragmentou."""
    if mask is None or mask.size == 0:
        return False, False, {"vertical_groups": 0, "horizontal_groups": 0}
    h, w = mask.shape[:2]

    def groups(values, threshold):
        indices = [index for index, value in enumerate(values) if value >= threshold]
        result = []
        for index in indices:
            if not result or index > result[-1][-1] + 1:
                result.append([index])
            else:
                result[-1].append(index)
        return [group for group in result if group]

    vertical_groups = groups(
        np.count_nonzero(mask, axis=0), max(5, int(round(0.28 * h)))
    )
    horizontal_groups = groups(
        np.count_nonzero(mask, axis=1), max(5, int(round(0.28 * w)))
    )

    vertical_centers = [float(np.mean(group)) for group in vertical_groups]
    horizontal_centers = [float(np.mean(group)) for group in horizontal_groups]
    vertical_pair = any(
        0.06 * w <= abs(first - second) <= 0.55 * w
        for first, second in itertools.combinations(vertical_centers, 2)
    )
    horizontal_pair = any(
        0.06 * h <= abs(first - second) <= 0.55 * h
        for first, second in itertools.combinations(horizontal_centers, 2)
    )
    return bool(vertical_pair), bool(horizontal_pair), {
        "vertical_groups": len(vertical_groups),
        "horizontal_groups": len(horizontal_groups),
        "vertical_centers": [round(value, 2) for value in vertical_centers],
        "horizontal_centers": [round(value, 2) for value in horizontal_centers],
    }


def _validar_geometria_capacitor(imagem, bbox, polarized=False):
    """Exige duas placas paralelas separadas e indício de ligações."""
    roi = _recortar_bbox_validacao(imagem, bbox, 0.12)
    if roi.size == 0:
        return False, 0.0, {"reason": "empty_crop"}
    h, w = roi.shape[:2]
    lines = _linhas_estruturais(roi)
    vertical = _linhas_unicas_por_posicao([
        line for line in lines
        if abs(abs(line["angle"]) - 90.0) <= 14.0
        and line["length"] >= 0.22 * h
    ], "vertical", max(4.0, 0.05 * w))
    horizontal = _linhas_unicas_por_posicao([
        line for line in lines
        if abs(line["angle"]) <= 14.0
        and line["length"] >= 0.22 * w
    ], "horizontal", max(4.0, 0.05 * h))

    vertical_pair = any(
        0.06 * w <= abs(a["mx"] - b["mx"]) <= 0.55 * w
        and abs(a["my"] - b["my"]) <= 0.24 * h
        for a, b in itertools.combinations(vertical, 2)
    )
    horizontal_pair = any(
        0.06 * h <= abs(a["my"] - b["my"]) <= 0.55 * h
        and abs(a["mx"] - b["mx"]) <= 0.24 * w
        for a, b in itertools.combinations(horizontal, 2)
    )
    mask = _mascara_tinta_local(roi)
    projection_vertical_pair, projection_horizontal_pair, projection_evidence = (
        _pares_placas_por_projecao(mask)
    )
    vertical_pair = bool(vertical_pair or projection_vertical_pair)
    horizontal_pair = bool(horizontal_pair or projection_horizontal_pair)

    # Duas placas verticais precisam de ao menos uma ligação horizontal e
    # vice-versa. Isso elimina a maioria das pautas e fios únicos.
    pair_with_lead = bool(
        (vertical_pair and horizontal)
        or (horizontal_pair and vertical)
    )
    aspect = w / max(1.0, h)
    aspect_ok = 0.22 <= aspect <= 4.5
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    small_marks = sum(
        1 for contour in contours
        if 4 <= cv2.contourArea(contour) <= 0.025 * w * h
    )
    polarity_evidence = small_marks >= 1
    curved_polarized_structure = bool(
        polarized
        and polarity_evidence
        and ((vertical and horizontal) or (horizontal and vertical))
    )
    from backend.capacitor_geometry import capacitor_body_evidence
    body_ok,body_evidence=capacitor_body_evidence(imagem,bbox)
    valid = bool(aspect_ok and body_ok)
    score = max(0.85 * body_ok, 0.62 * pair_with_lead + 0.25 * polarity_evidence)
    return valid, round(float(score), 3), {
        "vertical_plates": len(vertical),
        "horizontal_plates": len(horizontal),
        "parallel_plate_pair": bool(vertical_pair or horizontal_pair),
        "aspect_w_over_h": round(float(aspect), 3),
        "aspect_ok": aspect_ok,
        "pair_with_lead": pair_with_lead,
        "curved_polarized_structure": curved_polarized_structure,
        "polarity_mark_candidates": int(small_marks),
        "projection_plate_evidence": projection_evidence,
        **body_evidence,
    }


def _validar_geometria_mosfet(imagem, bbox, classe):
    """Exige gate/canal paralelos e pelo menos dois ramos de source/drain."""
    roi = _recortar_bbox_validacao(imagem, bbox, 0.10)
    if roi.size == 0:
        return False, 0.0, {"reason": "empty_crop"}
    h, w = roi.shape[:2]
    lines = _linhas_estruturais(roi)
    vertical = _linhas_unicas_por_posicao([
        line for line in lines
        if abs(abs(line["angle"]) - 90.0) <= 18.0
        and line["length"] >= 0.20 * h
    ], "vertical", max(4.0, 0.04 * w))
    horizontal = _linhas_unicas_por_posicao([
        line for line in lines
        if abs(line["angle"]) <= 18.0
        and line["length"] >= 0.20 * w
    ], "horizontal", max(4.0, 0.04 * h))
    diagonal = [
        line for line in lines
        if 18.0 < abs(line["angle"]) < 78.0
        and line["length"] >= 0.12 * min(w, h)
    ]
    vertical_body = len(vertical) >= 2 and (len(diagonal) >= 2 or len(horizontal) >= 2)
    horizontal_body = len(horizontal) >= 2 and (len(diagonal) >= 2 or len(vertical) >= 2)
    aspect = w / max(1.0, h)
    valid = bool(0.45 <= aspect <= 2.6 and (vertical_body or horizontal_body))
    score = min(1.0, 0.22 * min(3, len(vertical))
                + 0.18 * min(3, len(horizontal))
                + 0.14 * min(3, len(diagonal)))
    return valid, round(float(score), 3), {
        "mosfet_class": str(classe),
        "vertical_structures": len(vertical),
        "horizontal_structures": len(horizontal),
        "diagonal_branches": len(diagonal),
        "aspect_w_over_h": round(float(aspect), 3),
        "gate_channel_branch_structure": bool(vertical_body or horizontal_body),
    }


def _validar_geometria_opamp(imagem, bbox):
    """Confirma o corpo triangular e conexões mínimas de um amp-op."""
    roi = _recortar_bbox_validacao(imagem, bbox, 0.08)
    if roi.size == 0:
        return False, 0.0, {"reason": "empty_crop"}
    lines = _linhas_estruturais(roi)
    h, w = roi.shape[:2]
    diag_up = [
        line for line in lines
        if 20.0 <= abs(line["angle"]) <= 72.0 and line["my"] <= 0.62 * h
    ]
    diag_down = [
        line for line in lines
        if 20.0 <= abs(line["angle"]) <= 72.0 and line["my"] >= 0.38 * h
    ]
    horizontal = [
        line for line in lines
        if abs(line["angle"]) <= 18.0 and line["length"] >= 0.10 * w
    ]
    pair_ok = False
    apex_distance = None
    tolerance = max(9.0, 0.16 * min(w, h))
    for upper in diag_up:
        for lower in diag_down:
            distance = min(
                _distancia_pontos(a, b)
                for a in _endpoints(upper)
                for b in _endpoints(lower)
            )
            if apex_distance is None or distance < apex_distance:
                apex_distance = distance
            if distance <= tolerance:
                pair_ok = True
                break
        if pair_ok:
            break
    valid = bool(pair_ok and len(horizontal) >= 2 and 0.55 <= w / max(1, h) <= 2.8)
    score = (
        0.35 * bool(diag_up)
        + 0.35 * bool(diag_down)
        + 0.20 * pair_ok
        + 0.10 * (len(horizontal) >= 2)
    )
    return valid, round(float(score), 3), {
        "upper_diagonals": len(diag_up),
        "lower_diagonals": len(diag_down),
        "horizontal_connections": len(horizontal),
        "triangle_pair": pair_ok,
        "apex_distance": None if apex_distance is None else round(apex_distance, 2),
    }


def _validar_classe_geometrica_upright(imagem, bbox, classe):
    """Despacha a validação de uma classe; None significa sem regra geométrica."""
    classe = str(classe)
    if classe == "source_dc":
        from backend.symbol_text_guard import source_ring_evidence
        ok, evidence = source_ring_evidence(imagem, bbox)
        return ok, .92 if ok else 0., evidence
    if classe == "vcc":
        return _validar_geometria_vcc(imagem, bbox)
    if classe in {"transistor_npn", "transistor_pnp"}:
        return _validar_geometria_transistor(imagem, bbox, classe)
    if classe == "led":
        return _validar_geometria_led(imagem, bbox)
    if classe == "diode":
        return _validar_geometria_diodo(imagem, bbox)
    if classe == "opamp":
        return _validar_geometria_opamp(imagem, bbox)
    if classe == "resistor":
        return _validar_geometria_resistor(imagem, bbox)
    if classe == "inductor":
        return _validar_geometria_indutor(imagem, bbox)
    if classe == "battery":
        return _validar_geometria_bateria(imagem, bbox)
    if classe == "gnd":
        return _validar_geometria_gnd(imagem, bbox)
    if classe in {"capacitor", "capacitor_polarized"}:
        return _validar_geometria_capacitor(
            imagem, bbox, polarized=classe == "capacitor_polarized"
        )
    if classe in {"mosfet_n", "mosfet_p"}:
        return _validar_geometria_mosfet(imagem, bbox, classe)
    return None


def _validar_classe_geometrica(imagem,bbox,classe):
    """Validate the symbol in all quarter-turn orientations, in original pixels."""
    first=_validar_classe_geometrica_upright(imagem,bbox,classe)
    if first is None or first[0] or first[2].get('hard_text_rejection') or (classe=='vcc' and first[2].get('hard_topology_rejection')):
        return first
    h,w=imagem.shape[:2]
    if classe=='vcc':
        corners=np.array([[x,y] for x in (bbox['x1'],bbox['x2']) for y in (bbox['y1'],bbox['y2'])])
        for turn in (1,2,3):
            points=corners.copy()
            for step in range(turn):
                height,width=(h,w) if step%2==0 else (w,h)
                points=np.column_stack((points[:,1],width-1-points[:,0]))
            box=dict(x1=float(points[:,0].min()),y1=float(points[:,1].min()),x2=float(points[:,0].max()),y2=float(points[:,1].max()))
            candidate=_validar_geometria_vcc(np.ascontiguousarray(np.rot90(imagem,turn)),box)
            if candidate[0]:return candidate[0],candidate[1],{**candidate[2],'validation_rotation_degrees':turn*90}
            if candidate[2].get('hard_topology_rejection'):return candidate
        return first
    pad=max(3,int(min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])*.18))
    x1,y1=max(0,int(bbox['x1'])-pad),max(0,int(bbox['y1'])-pad)
    x2,y2=min(w,int(math.ceil(bbox['x2']))+pad),min(h,int(math.ceil(bbox['y2']))+pad)
    roi=imagem[y1:y2,x1:x2]
    if roi.size==0:
        return first
    local={k:float(bbox[k])-(x1 if k.startswith('x') else y1) for k in ('x1','y1','x2','y2')}
    rh,rw=roi.shape[:2]
    for k in (1,2,3):
        corners=np.array([[x,y] for x in (local['x1'],local['x2']) for y in (local['y1'],local['y2'])])
        for turn in range(k):
            height,width=(rh,rw) if turn%2==0 else (rw,rh)
            corners=np.column_stack((corners[:,1],width-corners[:,0]))
        rotated_box=dict(x1=float(corners[:,0].min()),y1=float(corners[:,1].min()),
                         x2=float(corners[:,0].max()),y2=float(corners[:,1].max()))
        candidate=_validar_classe_geometrica_upright(np.ascontiguousarray(np.rot90(roi,k)),rotated_box,classe)
        if candidate is not None and candidate[0]:
            return candidate[0],candidate[1],{**candidate[2],'validation_rotation_degrees':k*90}
    return first


def _falha_geometrica_tem_evidencia_parcial(classe, evidence):
    """Distingue forma incompleta de pauta/fundo sem estrutura do símbolo."""
    classe = str(classe)
    evidence = evidence or {}
    if classe == "inductor":
        return bool(
            int(evidence.get("non_axis_lines", 0)) >= 1
            and int(evidence.get("curved_parts", 0)) >= 1
        )
    if classe in {"capacitor", "capacitor_polarized"}:
        projection = evidence.get("projection_plate_evidence") or {}
        return bool(
            int(evidence.get("vertical_plates", 0))
            + int(evidence.get("horizontal_plates", 0)) >= 2
            or int(projection.get("vertical_groups", 0)) >= 2
            or int(projection.get("horizontal_groups", 0)) >= 2
        )
    if classe == "battery":
        return bool(
            int(evidence.get("vertical_plates", 0))
            + int(evidence.get("horizontal_plates", 0)) >= 3
        )
    if classe == "gnd":
        return bool(
            int(evidence.get("horizontal_bar_candidates", 0)) >= 3
            and int(evidence.get("vertical_stem_candidates", 0)) >= 1
        )
    if classe == "resistor":
        return bool(
            int(evidence.get("distinct_zigzag_segments", 0)) >= 3
            or evidence.get("iec_line_body")
        )
    if classe in {"mosfet_n", "mosfet_p"}:
        return bool(
            int(evidence.get("vertical_structures", 0)) >= 2
            and (
                int(evidence.get("horizontal_structures", 0)) >= 2
                or int(evidence.get("diagonal_branches", 0)) >= 1
            )
        )
    return False


def filtrar_falsos_positivos_geometricos(imagem, deteccoes):
    """Filtro pós-YOLO para classes que geram falsos positivos perigosos.

    A confiança do modelo não substitui a forma física: depois de um treino
    ruim, linhas do caderno chegaram a receber confiança 1.00 como indutor,
    bateria, GND e MOSFET. Por isso as classes estruturais são validadas mesmo
    quando a confiança do YOLO é alta.
    """
    mantidas = []
    rejeitadas = []
    from backend.resistor_geometry import symbol_mask
    resistor_page_mask = symbol_mask(imagem) if deteccoes else None
    for det in deteccoes:
        classe = str(det.get("class", ""))
        if det.get('manual_reviewed'):
            mantidas.append(det)
            continue
        if classe in {'transistor_npn','transistor_pnp'}:
            from backend.symbol_text_guard import dc_source_topology
            source_ok,source_evidence=dc_source_topology(imagem,det['bbox'],resistor_page_mask)
            if source_ok:
                # The ring arcs are not a base and collector/emitter fork.
                det={**det,'class':'source_dc','class_id':6,'original_yolo_class':classe,
                     'class_resolution':'dc_source_ring_topology','source_topology':source_evidence}
                classe='source_dc'
                det['alternatives']=[a for a in det.get('alternatives',[]) if not a['class'].startswith('transistor_')]
                det['review_reasons']=[r for r in det.get('review_reasons',[]) if r not in
                                       {'visually_similar_classes','class_conflict_between_views','close_consensus'}]
                det['review_required']=bool(det['review_reasons'])
        from backend.symbol_text_guard import isolated_text_run
        text_only, text_evidence = isolated_text_run(imagem, det['bbox'], resistor_page_mask)
        if not text_only:
            from backend.text_component_classifier import text_evidence as supervised_text_evidence
            text_only, text_evidence = supervised_text_evidence(imagem, det['bbox'], resistor_page_mask)
            if text_only and classe=='inductor' and text_evidence.get('reason')=='supervised_text_in_notebook_context':
                # Repeated curved letters and a photographed coil can share
                # texture. An independently traced coil on its OWN connected
                # ink component with two leads is stronger evidence than that
                # classifier. Disconnected glyphs cannot supply those leads.
                coil_ok,_,coil_proof=_validar_geometria_indutor(imagem,det['bbox'],resistor_page_mask)
                if coil_ok and coil_proof.get('connected_curve_with_two_leads'):
                    text_only=False
        if text_only:
            validation = False, 0., text_evidence
        elif classe == 'resistor':
            validation = _validar_geometria_resistor(imagem,det['bbox'],resistor_page_mask)
        elif classe == 'inductor':
            validation = _validar_geometria_indutor(imagem,det['bbox'],resistor_page_mask)
        elif classe == 'source_dc':
            from backend.symbol_text_guard import source_ring_evidence
            source_ok, source_evidence = source_ring_evidence(imagem, det['bbox'], resistor_page_mask)
            validation = source_ok, .92 if source_ok else 0., source_evidence
        else:
            validation = _validar_classe_geometrica(imagem,det['bbox'],classe)
        if validation is None:
            mantidas.append(det)
            continue
        ok, score, evidence = validation

        novo = dict(det)
        novo["geometry_validation"] = {
            "valid": bool(ok),
            "score": float(score),
            "evidence": evidence,
        }
        partial_evidence = _falha_geometrica_tem_evidencia_parcial(
            classe, evidence
        )
        votes = int(novo.get("consensus_votes", 1))
        sources = int(novo.get("consensus_source_count", 1))
        best = float(novo.get("best_view_confidence", novo.get("confidence", 0.0)))
        preserve_for_review = bool(
            not ok
            and not evidence.get('hard_text_rejection')
            and partial_evidence
            and sources >= 2
            and (
                (votes >= 2 and best >= 0.55)
                or (votes >= 3 and best >= 0.45)
            )
        )
        if ok or preserve_for_review:
            if preserve_for_review:
                novo["geometry_validation"]["soft_preserved"] = True
                novo["review_required"] = True
                motivos = list(novo.get("review_reasons", []))
                if "geometry_partial_multiview_consensus" not in motivos:
                    motivos.append("geometry_partial_multiview_consensus")
                novo["review_reasons"] = motivos
            mantidas.append(novo)
        else:
            rejeitado = dict(novo)
            rejeitado["rejection_reason"] = (
                "vcc_shape_missing" if classe == "vcc"
                else "bjt_fork_missing" if classe.startswith("transistor_")
                else "led_symbol_missing" if classe == "led"
                else "diode_triangle_bar_missing" if classe == "diode"
                else "resistor_zigzag_missing" if classe == "resistor"
                else "inductor_coil_missing" if classe == "inductor"
                else "battery_parallel_plates_missing" if classe == "battery"
                else "gnd_three_bars_missing" if classe == "gnd"
                else "capacitor_parallel_plates_missing" if classe.startswith("capacitor")
                else "mosfet_gate_channel_missing" if classe.startswith("mosfet_")
                else "opamp_triangle_missing"
            )
            if evidence.get('hard_text_rejection'):
                rejeitado['rejection_reason'] = evidence['reason']
            rejeitadas.append(rejeitado)
    return _reordenar_deteccoes(mantidas), _reordenar_deteccoes(rejeitadas)

def consolidar_multiview(
    deteccoes,
    iou_threshold=0.45,
    cobertura_threshold=0.80,
    review_threshold=0.40,
):
    """Agrupa caixas e escolhe a classe pelo consenso entre as vistas."""
    grupos = []
    for atual in sorted(
        deteccoes,
        key=lambda d: float(d["confidence"]),
        reverse=True,
    ):
        grupo_encontrado = None
        for grupo in grupos:
            # O primeiro elemento é a âncora mais confiante. Comparar contra
            # qualquer membro causava agrupamento em cadeia: A juntava B, B
            # juntava C, e dois componentes separados terminavam como um só.
            if _mesma_regiao_de_componente(
                atual,
                grupo[0],
                iou_threshold=iou_threshold,
                cobertura_threshold=cobertura_threshold,
            ):
                grupo_encontrado = grupo
                break
        if grupo_encontrado is None:
            grupos.append([atual])
        else:
            grupo_encontrado.append(atual)

    mantidas = []
    for indice_grupo, grupo in enumerate(grupos):
        por_classe = {}
        for indice_det, det in enumerate(grupo):
            classe = int(det["class_id"])
            fontes = det.get("inference_sources") or [
                f"unspecified_{indice_grupo}_{indice_det}"
            ]
            candidato = por_classe.setdefault(
                classe,
                {
                    "class_id": classe,
                    "class": det["class"],
                    "por_fonte": {},
                    "deteccoes": [],
                },
            )
            candidato["deteccoes"].append(det)
            for fonte in fontes:
                anterior = candidato["por_fonte"].get(str(fonte))
                if anterior is None or float(det["confidence"]) > anterior:
                    candidato["por_fonte"][str(fonte)] = float(det["confidence"])

        ranking = []
        for candidato in por_classe.values():
            # Vistas derivadas da mesma imagem não são votos independentes.
            # Sem este agrupamento, quatro previsões fracas de uma pauta do
            # caderno viravam uma confiança artificialmente alta para LED.
            por_familia = {}
            for fonte, confianca in candidato["por_fonte"].items():
                fonte = str(fonte)
                if (
                    fonte.startswith("normalized_binary_bw")
                    or fonte.startswith("monochrome_")
                ):
                    familia = "monochrome"
                elif fonte.startswith("normalized_grayscale"):
                    familia = "grayscale"
                elif fonte == "color_invariant_ink":
                    familia = "ink_invariant"
                elif fonte == "original_color":
                    familia = "original"
                else:
                    familia = fonte
                por_familia[familia] = max(
                    float(confianca), float(por_familia.get(familia, 0.0))
                )
            confiancas = sorted(por_familia.values(), reverse=True)
            votos = len(confiancas)
            candidato["votes"] = votos
            candidato["source_count"] = len(candidato["por_fonte"])
            monochrome_variants = set()
            for source_name in candidato["por_fonte"]:
                if source_name.startswith("monochrome_adaptive"):
                    monochrome_variants.add("adaptive")
                elif source_name.startswith("monochrome_otsu"):
                    monochrome_variants.add("otsu")
                elif source_name.startswith("monochrome_clahe"):
                    monochrome_variants.add("clahe")
                elif source_name.startswith("monochrome_stroke"):
                    monochrome_variants.add("stroke")
                elif source_name.startswith("normalized_binary_bw"):
                    monochrome_variants.add("normalized_binary")
            candidato["monochrome_variants"] = sorted(monochrome_variants)
            candidato["view_families"] = sorted(por_familia)
            best = confiancas[0]
            adicionais = confiancas[1:]
            pesos = (0.20, 0.10, 0.05, 0.03)
            consenso = best + sum(
                valor * (pesos[i] if i < len(pesos) else 0.02)
                for i, valor in enumerate(adicionais)
            )
            candidato["consensus_score"] = round(
                consenso + 0.02 * max(0, votos - 1), 6
            )
            # O consenso serve para escolher entre classes concorrentes, mas
            # não pode fabricar uma confiança maior do que a produzida pela
            # rede. Todas as vistas são transformações da MESMA imagem e não
            # observações independentes. Antes desta correção, quatro cópias
            # do mesmo erro de 0.70 podiam aparecer na tela como 1.00.
            candidato["consensus_confidence"] = round(
                min(1.0, consenso), 6
            )
            candidato["best_confidence"] = best
            ranking.append(candidato)

        ranking.sort(
            key=lambda item: (
                item["consensus_score"],
                item["votes"],
                item["best_confidence"],
            ),
            reverse=True,
        )
        vencedor = ranking[0]
        melhor_caixa = max(
            vencedor["deteccoes"],
            key=lambda det: float(det["confidence"]),
        )
        final = dict(melhor_caixa)
        final["best_view_confidence"] = round(vencedor["best_confidence"], 6)
        final["confidence"] = round(vencedor["best_confidence"], 6)
        final["derived_consensus_confidence"] = vencedor[
            "consensus_confidence"
        ]
        final["consensus_score"] = vencedor["consensus_score"]
        final["consensus_votes"] = vencedor["votes"]
        final["consensus_source_count"] = vencedor["source_count"]
        final["monochrome_variants"] = vencedor["monochrome_variants"]
        final["monochrome_variant_count"] = len(
            vencedor["monochrome_variants"]
        )
        final["consensus_view_families"] = vencedor["view_families"]
        final["inference_sources"] = sorted(vencedor["por_fonte"])
        final["alternatives"] = [
            {
                "class_id": candidato["class_id"],
                "class": candidato["class"],
                "confidence": round(candidato["best_confidence"], 6),
                "consensus_score": candidato["consensus_score"],
                "votes": candidato["votes"],
            }
            for candidato in ranking[1:]
        ]
        final["consensus_margin"] = round(
            vencedor["consensus_score"]
            - (ranking[1]["consensus_score"] if len(ranking) > 1 else 0.0),
            6,
        )
        mantidas.append(final)

    grupos_ambiguos = (
        {"diode", "led"},
        {"capacitor", "capacitor_polarized"},
        {"transistor_npn", "transistor_pnp"},
        {"mosfet_n", "mosfet_p"},
    )

    for det in mantidas:
        motivos = list(det.get("review_reasons", []))
        if float(det["confidence"]) < review_threshold:
            motivos.append("low_confidence")
        if (
            det.get("alternatives")
            and float(det.get("consensus_margin", 0.0)) < 0.30
        ):
            motivos.append("class_conflict_between_views")
            if float(det.get("consensus_margin", 1.0)) < 0.20:
                motivos.append("close_consensus")

        classes_disputadas = {det["class"]}
        classes_disputadas.update(
            item["class"] for item in det.get("alternatives", [])
        )
        if any(
            len(classes_disputadas.intersection(grupo)) > 1
            for grupo in grupos_ambiguos
        ):
            motivos.append("visually_similar_classes")

        det["review_required"] = bool(motivos)
        det["review_reasons"] = motivos

    return _reordenar_deteccoes(mantidas)


def corrigir_diodo_led_por_setas(imagem, deteccoes):
    """Resolve both directions from visible signs, including marks outside a box."""
    from backend.symbol_semantics import resolve_marked_families
    class_ids={name:i for i,name in _classes_esperadas().items()}
    corrected,counts=resolve_marked_families(imagem,deteccoes,class_ids)
    for det in corrected:
        if det.get('manual_reviewed'):continue
        if det.get('class') in {'led','diode','capacitor','capacitor_polarized'}:
            det['review_reasons']=[r for r in det.get('review_reasons',[]) if r not in
                                   {'visually_similar_classes','class_conflict_between_views','close_consensus'}]
            det['review_required']=bool(det['review_reasons'])
        if det.get('class')=='led':det['class_resolution']='led_emission_arrows'
    return _reordenar_deteccoes(corrected),counts['led']


def remover_caixas_englobantes_multicomponente(deteccoes):
    """Remove uma caixa grande quando ela contém duas caixas independentes."""
    mantidas = []
    rejeitadas = []
    for outer in deteccoes:
        area_outer = _area_bbox(outer["bbox"])
        internas = []
        for inner in deteccoes:
            if inner is outer:
                continue
            area_inner = _area_bbox(inner["bbox"])
            if area_inner <= 0 or area_outer < 2.20 * area_inner:
                continue
            _, coverage = _sobreposicao(outer, inner)
            if coverage < 0.88:
                continue
            if float(inner.get("confidence", 0.0)) < max(
                0.20, 0.50 * float(outer.get("confidence", 0.0))
            ):
                continue
            internas.append(inner)
        independentes = []
        for inner in sorted(
            internas, key=lambda item: float(item.get("confidence", 0.0)), reverse=True
        ):
            if all(_sobreposicao(inner, saved)[0] < 0.18 for saved in independentes):
                independentes.append(inner)
        if len(independentes) >= 2:
            rejected = dict(outer)
            rejected["rejection_reason"] = "oversized_box_contains_multiple_components"
            rejected["contained_detection_ids"] = [
                item.get("detection_id") for item in independentes
            ]
            rejeitadas.append(rejected)
        else:
            mantidas.append(outer)
    return _reordenar_deteccoes(mantidas), _reordenar_deteccoes(rejeitadas)


def _ler_imagem(caminho: Path):
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if imagem is None:
        raise RuntimeError(
            f"Não foi possível abrir a imagem: {caminho.name}"
        )
    return imagem


def _executar_predicao(
    model,
    imagem,
    confidence,
    imgsz,
    inference_source,
):
    results = model.predict(
        source=imagem,
        conf=float(confidence),
        iou=0.55,
        agnostic_nms=False,
        max_det=300,
        imgsz=int(imgsz),
        verbose=False,
    )
    if not results:
        return []
    return _deteccoes_resultado_yolo(results[0], inference_source)


def _deteccoes_resultado_yolo(result, inference_source):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    return montar_deteccoes(
        xyxy=boxes.xyxy.detach().cpu().numpy(),
        confs=boxes.conf.detach().cpu().numpy(),
        class_ids=boxes.cls.detach().cpu().numpy(),
        names=result.names,
        inference_source=inference_source,
    )




def _deslocar_deteccao(det, dx, dy):
    novo = dict(det)
    bbox = dict(det["bbox"])
    bbox["x1"] = round(float(bbox["x1"]) + dx, 2)
    bbox["x2"] = round(float(bbox["x2"]) + dx, 2)
    bbox["y1"] = round(float(bbox["y1"]) + dy, 2)
    bbox["y2"] = round(float(bbox["y2"]) + dy, 2)
    bbox["center_x"] = round((bbox['x1']+bbox['x2'])/2, 2)
    bbox["center_y"] = round((bbox['y1']+bbox['y2'])/2, 2)
    novo["bbox"] = bbox
    return novo


def _restaurar_deteccoes_rotacionadas(deteccoes, original_shape, rotation):
    """Converte caixas de uma vista a 90 graus para a imagem original."""
    original_h, original_w = original_shape[:2]
    restored = []
    for det in deteccoes:
        box = det["bbox"]
        rx1, ry1 = float(box["x1"]), float(box["y1"])
        rx2, ry2 = float(box["x2"]), float(box["y2"])
        if rotation == "cw":
            x1, x2 = ry1, ry2
            y1, y2 = original_h - rx2, original_h - rx1
        elif rotation == "ccw":
            x1, x2 = original_w - ry2, original_w - ry1
            y1, y2 = rx1, rx2
        elif rotation == 'half':
            x1,x2=original_w-rx2,original_w-rx1
            y1,y2=original_h-ry2,original_h-ry1
        else:
            raise ValueError("rotation deve ser cw, ccw ou half")
        x1, x2 = sorted((max(0.0, x1), min(float(original_w), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(original_h), y2)))
        novo = dict(det)
        novo["bbox"] = {
            "x1": round(x1, 2), "y1": round(y1, 2),
            "x2": round(x2, 2), "y2": round(y2, 2),
            "width": round(max(0.0, x2 - x1), 2),
            "height": round(max(0.0, y2 - y1), 2),
            "center_x": round((x1 + x2) / 2.0, 2),
            "center_y": round((y1 + y2) / 2.0, 2),
        }
        novo["rotation_restored"] = str(rotation)
        restored.append(novo)
    return restored


def _executar_predicao_em_blocos(
    model,
    imagem,
    confidence,
    imgsz,
    inference_source,
    tile_size=960,
    overlap=0.20,
):
    """Executa YOLO em blocos sobre imagens grandes.

    Em scans A4, redimensionar a pagina inteira para 640/960 px pode fazer
    simbolos pequenos desaparecerem. Os blocos mantem o simbolo grande e
    convertem as caixas locais de volta para coordenadas da imagem original.
    """
    h, w = imagem.shape[:2]
    tile_size = int(max(128, min(tile_size, max(h, w))))
    if h <= tile_size and w <= tile_size:
        return _executar_predicao(
            model, imagem, confidence, imgsz, inference_source
        )

    passo = max(1, int(round(tile_size * (1.0 - float(overlap)))))

    def posicoes(tamanho):
        if tamanho <= tile_size:
            return [0]
        vals = list(range(0, max(1, tamanho - tile_size + 1), passo))
        ultimo = tamanho - tile_size
        if vals[-1] != ultimo:
            vals.append(ultimo)
        return vals

    tiles = []
    for y in posicoes(h):
        for x in posicoes(w):
            recorte = imagem[y:y + tile_size, x:x + tile_size]
            tiles.append((recorte, x, y))

    deteccoes = []
    # Ultralytics aceita uma lista de imagens e executa o lote com muito menos
    # overhead do que dezenas de chamadas individuais. Isso mantém a pirâmide
    # monocromática responsiva e evita a antiga tela presa em "Carregando".
    if hasattr(model, "predict") and len(tiles) > 1:
        batch_size = 8 if int(imgsz)<=640 else 4
        try:
            for start in range(0, len(tiles), batch_size):
                batch = tiles[start:start + batch_size]
                results = model.predict(
                    source=[item[0] for item in batch],
                    conf=float(confidence),
                    iou=0.55,
                    agnostic_nms=False,
                    max_det=300,
                    imgsz=int(imgsz),
                    verbose=False,
                )
                for result, (crop, x, y) in zip(results, batch):
                    locais = _deteccoes_resultado_yolo(
                        result, inference_source
                    )
                    if inference_source=='dense_component_tiles':
                        from backend.crowded_components import clipped_on_internal_edge
                        locais=[d for d in locais if not clipped_on_internal_edge(d,x,y,crop.shape,imagem.shape)]
                    deteccoes.extend(_deslocar_deteccao(det,x,y) for det in locais)
        except Exception:
            # Compatibilidade com backends/modelos antigos sem inferência em
            # lote. A lógica continua correta, apenas fica menos rápida.
            deteccoes = []
            for recorte, x, y in tiles:
                locais = _executar_predicao(
                    model, recorte, confidence, imgsz, inference_source
                )
                deteccoes.extend(
                    _deslocar_deteccao(det, x, y) for det in locais
                )
    else:
        for recorte, x, y in tiles:
            locais = _executar_predicao(
                model, recorte, confidence, imgsz, inference_source
            )
            deteccoes.extend(
                _deslocar_deteccao(det, x, y) for det in locais
            )

    return remover_duplicadas(
        deteccoes, iou_threshold=0.35, cobertura_threshold=0.75
    )


def _tamanhos_piramide_monocromatica(imagem):
    """Escolhe blocos multiescala para símbolos grandes e pequenos."""
    h, w = imagem.shape[:2]
    maior = max(h, w)
    menor = min(h, w)
    if maior <= 1200:
        candidates = (
            max(128, int(round(maior * 0.38))),
            max(160, int(round(maior * 0.55))),
            max(192, int(round(maior * 0.72))),
        )
    else:
        candidates = (512, 768, 1024)
    return sorted({
        int(min(value, maior))
        for value in candidates
        if value < 0.94 * maior and value >= min(128, menor)
    })

def _desenhar_deteccoes(imagem, deteccoes):
    saida = imagem.copy()
    for det in deteccoes:
        bbox = det["bbox"]
        x1 = int(round(bbox["x1"]))
        y1 = int(round(bbox["y1"]))
        x2 = int(round(bbox["x2"]))
        y2 = int(round(bbox["y2"]))
        classe = det["class"]
        conf = float(det["confidence"])
        revisar = bool(det.get("review_required"))
        texto = f"{classe} {conf:.2f}" + (" ?" if revisar else "")
        cor = (0, 165, 255) if revisar else (0, 180, 0)

        cv2.rectangle(saida, (x1, y1), (x2, y2), cor, 2)
        (tw, th), baseline = cv2.getTextSize(
            texto,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        topo = max(0, y1 - th - baseline - 6)
        cv2.rectangle(saida, (x1, topo), (x1 + tw + 6, y1), cor, -1)
        cv2.putText(
            saida,
            texto,
            (x1 + 3, max(th + 2, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return saida


def _salvar_png(caminho: Path, imagem):
    ok, buffer = cv2.imencode(".png", imagem)
    if not ok:
        raise RuntimeError("Não foi possível gerar a imagem de detecções.")
    buffer.tofile(str(caminho))


def salvar_imagem_deteccoes(
    image_path: Path,
    deteccoes: list[dict],
    output_path: Path,
) -> None:
    imagem = _ler_imagem(image_path)
    _salvar_png(output_path, _desenhar_deteccoes(imagem, deteccoes))


def _needs_additional_views(image,detections):
    """Extra views remain mandatory for weak symbols or any unexplained ink."""
    if not detections or any(float(d.get('confidence',0))<.75 or d.get('review_required') for d in detections):
        return True
    accepted,_=filtrar_falsos_positivos_geometricos(image,consolidar_multiview(detections))
    if len(accepted)!=len(consolidar_multiview(detections)):return True
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ink=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,41,15)
    for d in accepted:
        b=d['bbox'];cv2.rectangle(ink,(max(0,int(b['x1'])-2),max(0,int(b['y1'])-2)),(int(b['x2'])+2,int(b['y2'])+2),0,-1)
    length=max(48,int(min(gray.shape)*.12))
    wires=cv2.bitwise_or(cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,length),np.uint8)),cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((length,1),np.uint8)))
    ink[cv2.dilate(wires,np.ones((3,3),np.uint8))>0]=0
    _,_,stats,_=cv2.connectedComponentsWithStats(ink,8)
    return any(s[cv2.CC_STAT_AREA]>=5 and max(s[cv2.CC_STAT_WIDTH],s[cv2.CC_STAT_HEIGHT])>=3 for s in stats[1:])


@symbol_analysis_session
def detectar_componentes(
    image_path: Path,
    output_image_path: Path,
    output_json_path: Path,
    confidence: float = 0.25,
    imgsz: int = 640,
    strategy: str = "auto",
    adaptive: bool = True,
):
    if not image_path.exists():
        raise FileNotFoundError(
            f"Imagem para reconhecimento não encontrada: {image_path.name}"
        )

    strategy = str(strategy).strip().lower()
    if strategy not in {"auto", "color", "grayscale"}:
        raise ValueError("strategy deve ser auto, color ou grayscale.")

    started=perf_counter();prediction_cache={};reused_views=[];view_timings=[]
    model = _carregar_modelo()
    def predict(model,image,threshold,size,source):
        at=perf_counter();key=(image.shape,image.dtype.str,hashlib.sha256(image.tobytes()).hexdigest(),float(threshold),int(size))
        if key in prediction_cache:
            detections=copy.deepcopy(prediction_cache[key]);reused_views.append(source)
            for d in detections:d['inference_sources']=[source]
        else:
            detections=_executar_predicao(model,image,threshold,size,source)
            prediction_cache[key]=copy.deepcopy(detections)
        view_timings.append(dict(view=source,duration_seconds=round(perf_counter()-at,4),reused=source in reused_views))
        return detections
    original = _ler_imagem(image_path)
    cinza_bgr = criar_vista_cinza_normalizada(original)
    binaria_bgr = criar_vista_binaria_normalizada(original)
    invariante_cor = criar_vista_invariante_cor(original)
    quase_monocromatica = _imagem_quase_monocromatica(original)
    vistas_mono = (
        criar_vistas_monocromaticas_universais(original)
        if quase_monocromatica and strategy in {"auto", "grayscale"}
        else {}
    )
    brutas = []
    views = []
    # A busca interna antiga usava 20% do limiar solicitado (0.05 quando a
    # tela pedia 0.25). Em várias vistas da mesma foto, ruídos muito fracos
    # sobreviviam e viravam falsos positivos por consenso. Mantemos alguma
    # margem para combinar vistas, mas nunca abaixo de 0.12.
    candidate_confidence = max(
        0.12,
        min(float(confidence), float(confidence) * 0.70),
    )

    coloridas = []
    if strategy in {"auto", "color"}:
        coloridas = predict(
            model,
            original,
            candidate_confidence,
            imgsz,
            "original_color",
        )
        brutas.extend(coloridas)
        views.append({"name": "original_color", "imgsz": int(imgsz), "raw_detections": len(coloridas)})

    if strategy in {"auto", "grayscale"}:
        em_cinza = predict(
            model,
            cinza_bgr,
            candidate_confidence,
            imgsz,
            "normalized_grayscale",
        )
        brutas.extend(em_cinza)
        views.append({"name": "normalized_grayscale", "imgsz": int(imgsz), "raw_detections": len(em_cinza)})

        # Vista dedicada a diagramas preto/branco. Para imagens quase
        # monocromáticas, usamos 960 px automaticamente para não perder
        # traços finos quando o esquema é reduzido pelo YOLO.
        binary_imgsz = max(int(imgsz), 960) if quase_monocromatica else int(imgsz)
        binarias = predict(
            model,
            binaria_bgr,
            candidate_confidence,
            binary_imgsz,
            "normalized_binary_bw",
        )
        brutas.extend(binarias)
        views.append({
            "name": "normalized_binary_bw",
            "imgsz": int(binary_imgsz),
            "preferred_for_monochrome": bool(quase_monocromatica),
            "raw_detections": len(binarias),
        })

        # Duas representações complementares bastam. CLAHE e stroke eram
        # quase duplicatas da original e aumentavam o custo sem acrescentar
        # evidência independente. Otsu e adaptativa cobrem fundo uniforme e
        # iluminação irregular, respectivamente.
        need_extra=not adaptive or strategy!='auto' or _needs_additional_views(original,brutas)
        for nome_vista in ("monochrome_otsu", "monochrome_adaptive") if need_extra else ():
            if nome_vista not in vistas_mono:
                continue
            imagem_vista = vistas_mono[nome_vista]
            deteccoes_vista = predict(
                model,
                imagem_vista,
                candidate_confidence,
                max(int(imgsz), 960),
                nome_vista,
            )
            brutas.extend(deteccoes_vista)
            views.append({
                "name": nome_vista,
                "imgsz": max(int(imgsz), 960),
                "universal_dataset_classes": True,
                "raw_detections": len(deteccoes_vista),
            })

        # All trained classes share the rotation policy; coordinates are always
        # restored before consensus and physical symbol validation.
        if strategy == 'auto':
            from backend.ink_color import estimate_component_ink, image_with_component_ink
            color_candidates=consolidar_multiview(brutas)
            color_candidates,_=filtrar_falsos_positivos_geometricos(original,color_candidates)
            pen_profile=estimate_component_ink(original,{'detections':color_candidates})
            rotation_base=binaria_bgr
            if pen_profile.get('active'):
                pen_view=image_with_component_ink(original,pen_profile)
                pen_detections=predict(model,pen_view,candidate_confidence,max(int(imgsz),960),'component_ink_normalized')
                brutas.extend(pen_detections)
                views.append({'name':'component_ink_normalized','imgsz':max(int(imgsz),960),
                    'raw_detections':len(pen_detections),'ink_profile':pen_profile})
                rotation_base=criar_vista_binaria_normalizada(pen_view)
            for rotation, rotate_code in (
                ("cw", cv2.ROTATE_90_CLOCKWISE),
                ("ccw", cv2.ROTATE_90_COUNTERCLOCKWISE),
                ('half',cv2.ROTATE_180),
            ) if need_extra else ():
                degrees=180 if rotation=='half' else 90 if rotation=='cw' else -90
                source_name = f"rotated_{abs(degrees)}_{rotation}_all_classes"
                rotated_image = cv2.rotate(rotation_base, rotate_code)
                rotated_raw = predict(
                    model,
                    rotated_image,
                    candidate_confidence,
                    max(int(imgsz), 960),
                    source_name,
                )
                targeted = rotated_raw
                restored = _restaurar_deteccoes_rotacionadas(
                    targeted, original.shape, rotation
                )
                brutas.extend(restored)
                views.append({
                    "name": source_name,
                    "imgsz": max(int(imgsz), 960),
                    "rotation_degrees": degrees,
                    "accepted_classes": "all_trained_classes_with_geometry_validation",
                    "raw_detections": len(restored),
                    "discarded_other_classes": len(rotated_raw) - len(targeted),
                })

        # Para scans/PDFs rasterizados grandes, analisar a pagina inteira pode
        # encolher demais os simbolos. Em P&B fazemos tambem inferencia por
        # blocos, preservando o tamanho visual de cada componente.
        h_img, w_img = original.shape[:2]
        from backend.crowded_components import tile_policy
        dense_plan=tile_policy(original.shape,consolidar_multiview(brutas),int(imgsz))
        usar_blocos = bool(dense_plan) or max(h_img, w_img) > 1200 and need_extra
        if usar_blocos:
            tiled = _executar_predicao_em_blocos(
                model,
                rotation_base if strategy=='auto' else binaria_bgr,
                candidate_confidence,
                dense_plan['imgsz'] if dense_plan else 960,
                'dense_component_tiles' if dense_plan else 'normalized_binary_bw_tiled',
                tile_size=dense_plan['tile_size'] if dense_plan else 960,
                overlap=dense_plan['overlap'] if dense_plan else .20,
            )
            brutas.extend(tiled)
            views.append({
                "name": 'dense_component_tiles' if dense_plan else 'normalized_binary_bw_tiled',
                "imgsz": dense_plan['imgsz'] if dense_plan else 960,
                "tile_size": dense_plan['tile_size'] if dense_plan else 960,
                "overlap": dense_plan['overlap'] if dense_plan else .20,
                "density_policy": dense_plan,
                "raw_detections": len(tiled),
            })

        # Não fazemos mais a antiga pirâmide em imagens pequenas. Ela criava
        # dezenas de recortes sobrepostos da mesma página e era a principal
        # fonte de caixas inventadas. Páginas grandes continuam cobertas por
        # um único passe em blocos, no mesmo regime usado pelo novo dataset.

    if strategy == "auto":
        # Recover missed resistors only from a closed IEC body or one
        # continuous alternating ANSI path, each with two opposed leads.
        iec_resistors = _deteccoes_resistores_retangulares(original)
        ansi_resistors = _deteccoes_resistores_geometricos(original)
        brutas.extend(ansi_resistors)
        views.append({'name':'resistor_zigzag_geometry','detections':len(ansi_resistors),'continuous_alternating_path_and_two_leads':True})
        brutas.extend(iec_resistors)
        views.append({
            "name": "resistor_iec_geometry",
            "detections": len(iec_resistors),
            "strict_closed_body_and_two_leads": True,
        })

        # As demais geometrias não inventam componentes. Elas são usadas
        # abaixo somente para confirmar/rejeitar caixas propostas pela IA.
        # Assim, três fios que lembram um triângulo não criam um amp-op novo.
        views.append({
            "name": "geometry_validation_only",
            "detections": 0,
        })

    imagem_grande = max(original.shape[:2]) > int(imgsz) * 1.5
    if strategy == "auto" and imagem_grande and int(imgsz) < 960 and need_extra:
        alta_resolucao = predict(
            model,
            cinza_bgr,
            candidate_confidence,
            960,
            "normalized_grayscale_high_resolution",
        )
        brutas.extend(alta_resolucao)
        views.append(
            {
                "name": "normalized_grayscale_high_resolution",
                "imgsz": 960,
                "raw_detections": len(alta_resolucao),
            }
        )

    # A tight uploaded crop can fill the entire model frame, unlike symbols
    # surrounded by paper during training. Add one context view without
    # altering source pixels or the geometry validators' original context.
    if strategy=='auto' and max(original.shape[:2])<=512:
        h,w=original.shape[:2];margin=max(32,round(max(h,w)*.5))
        corners=np.concatenate([original[:4,:4].reshape(-1,3),original[-4:,-4:].reshape(-1,3)])
        color=tuple(int(v) for v in np.median(corners,axis=0))
        context=cv2.copyMakeBorder(original,margin,margin,margin,margin,cv2.BORDER_CONSTANT,value=color)
        restored=[]
        for d in predict(model,context,candidate_confidence,imgsz,'paper_context_padding'):
            d=_deslocar_deteccao(d,-margin,-margin);b=d['bbox'];area=max(1,(b['x2']-b['x1'])*(b['y2']-b['y1']))
            x1,y1=max(0,b['x1']),max(0,b['y1']);x2,y2=min(w,b['x2']),min(h,b['y2'])
            if x2<=x1 or y2<=y1 or (x2-x1)*(y2-y1)<.8*area:continue
            b.update(x1=x1,y1=y1,x2=x2,y2=y2,width=x2-x1,height=y2-y1,center_x=(x1+x2)/2,center_y=(y1+y2)/2)
            restored.append(d)
        brutas.extend(restored)
        views.append(dict(name='paper_context_padding',raw_detections=len(restored),margin_px=margin,imgsz=int(imgsz)))

    # Remove combined BJT proposals before class votes merge two neighbors.
    # Keep tight independent boxes so a broad NPN prediction cannot absorb PNP.
    brutas,compound_bjt_rejections=_remove_compound_bjt_proposals(original,brutas)
    consolidadas = consolidar_multiview(
        brutas,
        iou_threshold=0.45,
        cobertura_threshold=0.80,
        review_threshold=max(0.40, float(confidence)),
    )
    consolidadas, led_arrow_correction_count = corrigir_diodo_led_por_setas(
        original, consolidadas
    )
    consolidadas, rejeitadas_englobantes = (
        remover_caixas_englobantes_multicomponente(consolidadas)
    )
    # VCC, BJT, LED e amp-op recebem validação pela geometria do símbolo. Isso
    # impede que pontos/junções virem VCC, pautas virem LED e diagonais isoladas
    # sejam aceitas como transistor ou amplificador operacional.
    consolidadas, rejeitadas_geometria = filtrar_falsos_positivos_geometricos(
        original, consolidadas
    )
    # Em P&B o mesmo símbolo costuma receber confiança um pouco menor,
    # especialmente quando os traços são muito finos. Mantemos o limiar final
    # pedido pelo usuário, mas não descartamos uma detecção válida só porque
    # uma vista individual ficou logo abaixo de 0.18.
    confianca_minima_vista = (
        max(0.12, float(confidence) * 0.65)
        if quase_monocromatica
        else max(0.18, float(confidence) * 0.70)
    )
    votos_consenso_minimos = 2
    # A lista vem do próprio dataset. Portanto uma classe adicionada depois
    # recebe automaticamente o mesmo tratamento monocromático, sem alterar o
    # código. As classes com validação geométrica continuam sendo filtradas
    # antes desta etapa.
    classes_dataset = set(_classes_esperadas().values())
    deteccoes_finais = []
    basic_rescue_count = 0
    universal_monochrome_rescue_count = 0
    geometry_consensus_rescue_count = 0
    rejeitadas_confianca = []
    for det in consolidadas:
        conf_consenso = float(det["confidence"])
        best = float(det.get("best_view_confidence", conf_consenso))
        votos = int(det.get("consensus_votes", 1))
        fontes_distintas = int(det.get("consensus_source_count", 1))
        variantes_mono = int(det.get("monochrome_variant_count", 0))
        fontes = set(det.get("inference_sources", []))
        classe = str(det.get("class", ""))
        tem_fonte_piramide = any("_pyramid_" in fonte for fonte in fontes)
        tem_fonte_completa = any(
            "_pyramid_" not in fonte for fonte in fontes
        )

        passou_normal = (
            best >= float(confidence)
            and not (tem_fonte_piramide and not tem_fonte_completa)
            and (
                best >= confianca_minima_vista
                or votos >= votos_consenso_minimos
                or bool(fontes.intersection({
                    "filled_triangle_geometry",
                    "gnd_line_geometry",
                    "resistor_zigzag_geometry",
                    "resistor_iec_geometry",
                    "capacitor_parallel_plate_geometry",
                    "bjt_circle_fork_geometry",
                    "opamp_triangle_geometry",
                    "manual_review",
                }))
            )
        )

        # A symbol seen only after rotation still needs a real shape (when the
        # class has a validator) and the requested model confidence. Repeated
        # weak rotational predictions cannot inflate the model's confidence.
        rotation_only=bool(fontes) and all(f.startswith('rotated_') for f in fontes)
        if rotation_only:
            geometry=det.get('geometry_validation')
            passou_normal=bool(passou_normal and (geometry is None or geometry.get('valid')))

        fonte_monocromatica = any(
            fonte.startswith("monochrome_")
            or fonte.startswith("normalized_binary_bw")
            for fonte in fontes
        )
        resgate_monocromatico_universal = bool(
            not passou_normal
            and quase_monocromatica
            and classe in classes_dataset
            and fonte_monocromatica
            and tem_fonte_completa
            and (
                best >= max(0.32, float(confidence))
                or (
                    variantes_mono >= 2
                    and fontes_distintas >= 2
                    and best >= max(0.26, float(confidence))
                )
            )
        )

        geometria = det.get("geometry_validation") or {}
        resgate_geometrico_por_consenso = bool(
            not passou_normal
            and bool(geometria.get("valid"))
            and tem_fonte_completa
            and votos >= 2
            and fontes_distintas >= 2
            and best >= candidate_confidence
        )

        if (
            passou_normal
            or resgate_monocromatico_universal
            or resgate_geometrico_por_consenso
            or (classe in {'vcc','inductor'} and bool(geometria.get('valid'))
                and float(geometria.get('score',0))>=.88 and best>=candidate_confidence)
        ):
            novo = dict(det)
            if resgate_monocromatico_universal:
                novo["review_required"] = True
                motivos = list(novo.get("review_reasons", []))
                if "universal_monochrome_dataset_rescue" not in motivos:
                    motivos.append("universal_monochrome_dataset_rescue")
                novo["review_reasons"] = motivos
                novo["universal_monochrome_rescue"] = True
                basic_rescue_count += 1
                universal_monochrome_rescue_count += 1
            if resgate_geometrico_por_consenso:
                novo["review_required"] = True
                motivos = list(novo.get("review_reasons", []))
                if "geometry_validated_multiview_rescue" not in motivos:
                    motivos.append("geometry_validated_multiview_rescue")
                novo["review_reasons"] = motivos
                novo["geometry_consensus_rescue"] = True
                basic_rescue_count += 1
                geometry_consensus_rescue_count += 1
            deteccoes_finais.append(novo)
        else:
            rejeitada = dict(det)
            rejeitada["rejection_reason"] = "below_final_confidence_without_safe_rescue"
            rejeitadas_confianca.append(rejeitada)

    if strategy=='auto':
        from backend.coil_recovery import recover_coils
        class_ids={v:int(k) for k,v in _classes_esperadas().items()}
        coils=recover_coils(original,deteccoes_finais,_validar_geometria_indutor,class_ids['inductor'])
        deteccoes_finais.extend(coils)
        views.append(dict(name='continuous_coil_geometry',detections=len(coils)))
        from backend.ground_recovery import recover_grounds
        grounds=recover_grounds(original,deteccoes_finais,class_ids['gnd'])
        for ground in grounds:
            g=ground['bbox']
            for d in deteccoes_finais:
                b=d['bbox']
                if d.get('class') in {'mosfet_n','mosfet_p'} and not d.get('manual_reviewed') and b['x1']<=g['center_x']<=b['x2'] and b['y1']<g['y1']<b['y2']:
                    b=dict(b,y2=g['y1']-1);b.update(height=b['y2']-b['y1'],center_y=(b['y1']+b['y2'])/2);d['bbox']=b
                    d['ground_symbol_excluded_from_body_box']=True
        deteccoes_finais.extend(grounds)
        views.append(dict(name='three_descending_bars_one_lead',detections=len(grounds)))
        from backend.capacitor_recovery import recover_capacitors
        capacitors=recover_capacitors(original,deteccoes_finais,class_ids)
        deteccoes_finais.extend(capacitors)
        views.append(dict(name='separate_plates_opposed_lead_geometry',detections=len(capacitors)))
        from backend.power_recovery import recover_power
        power=recover_power(original,deteccoes_finais,class_ids['vcc'])
        deteccoes_finais.extend(power)
        views.append(dict(name='one_lead_hollow_power_geometry',detections=len(power)))
        from backend.resistor_contacts import extend_boxes
        deteccoes_finais=extend_boxes(original,deteccoes_finais)
        # Recovered plates/diodes need the same visible +/arrow checks as
        # model proposals; otherwise recovery silently loses their subtype.
        deteccoes_finais,corrections=corrigir_diodo_led_por_setas(original,deteccoes_finais)
        led_arrow_correction_count+=corrections

    # A plate/gate inside a validated MOSFET, or a second resistor box inside
    # the first, is part of that symbol, regardless of class, size ratio or IoU.
    deteccoes, rejeitadas_internas = remove_nested_boxes(deteccoes_finais)

    # Removido o resgate global de 0.10/0.12. Uma tela vazia é um resultado
    # melhor do que dezenas de componentes inventados; exemplos difíceis
    # agora são resolvidos pelo treino focado, não por redução silenciosa do
    # limiar.
    rescue_mode = False

    anotada = _desenhar_deteccoes(original, deteccoes)
    _salvar_png(output_image_path, anotada)
    altura, largura = original.shape[:2]

    payload = {
        "elapsed_seconds": round(perf_counter()-started,4),
        "performance": {"adaptive":bool(adaptive),"extra_views_needed": bool(locals().get("need_extra",True)),"identical_views_reused":reused_views,"view_timings":view_timings},
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": str(MODEL_PATH.relative_to(BASE_DIR)),
        "model_sha256": _hash_modelo(),
        "source_image": image_path.name,
        "source_policy": "original_by_default",
        "color_invariance": {
            "enabled": strategy in {"auto", "grayscale"},
            "method": (
                "ai_first_component_ink_all_classes_rotations_grayscale_binary_tiles"
            ),
            "decision": (
                "trained_model_confidence_plus_geometry_validation"
            ),
        },
        "monochrome_input_detected": bool(quase_monocromatica),
        "inference_strategy": strategy,
        "inference_views": views,
        "confidence_threshold": float(confidence),
        "candidate_confidence_threshold": candidate_confidence,
        "yolo_nms_iou": 0.55,
        "yolo_agnostic_nms": False,
        "yolo_max_detections": 300,
        "minimum_single_view_confidence": confianca_minima_vista,
        "imgsz": int(imgsz),
        "image_width": int(largura),
        "image_height": int(altura),
        "raw_count": len(brutas),
        "raw_count_by_source": {
            view["name"]: int(view.get("raw_detections", view.get("detections", 0)))
            for view in views
        },
        "count": len(deteccoes),
        "geometry_rejected_count": len(rejeitadas_geometria)+len(compound_bjt_rejections),
        "geometry_rejections": rejeitadas_geometria+compound_bjt_rejections,
        "symbol_text_guard": {
            "policy": "aligned_glyphs_and_class_body_connected_leads_before_consensus_rescue",
            "ocr_required": False,
            "rejected_count": sum(bool(d.get('geometry_validation', {}).get('evidence', {}).get('hard_text_rejection'))
                                  for d in rejeitadas_geometria),
        },
        "confidence_rejected_count": len(rejeitadas_confianca),
        "confidence_rejections": rejeitadas_confianca,
        "led_arrow_correction_count": int(led_arrow_correction_count),
        "oversized_enclosing_rejected_count": len(rejeitadas_englobantes),
        "oversized_enclosing_rejections": rejeitadas_englobantes,
        "nested_box_rejected_count": len(rejeitadas_internas),
        "nested_box_rejections": rejeitadas_internas,
        "basic_component_rescue_count": int(basic_rescue_count),
        "universal_monochrome_rescue_count": int(
            universal_monochrome_rescue_count
        ),
        "geometry_consensus_rescue_count": int(
            geometry_consensus_rescue_count
        ),
        "dataset_classes_with_monochrome_runtime": sorted(classes_dataset),
        "rescue_mode": bool(rescue_mode),
        "review_count": sum(
            1 for det in deteccoes if det.get("review_required")
        ),
        "duplicate_filter": {
            "cross_class_conflicts": True,
            "class_consensus": True,
            "fully_contained_boxes_allowed": False,
            "containment_edges_inclusive": True,
            "iou_threshold": 0.45,
            "coverage_threshold": 0.80,
        },
        "detections": deteccoes,
    }

    output_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload
