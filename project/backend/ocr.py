from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import time

import cv2
import numpy as np
from backend.component_models import enrich_models, model_edit, model_classes, model_name, model_ocr_reading, CATALOG
from backend.resistor_geometry import symbol_analysis_session

from backend.measurement_text import parse_measurement, unit_anchor, unit_suffix, DIMENSIONS, SYMBOLS, clean_unicode


class OCRTokenList(list):
    """Association tokens with a separate archive of all engine readings."""
    raw_tokens = None


_EASYOCR_READER = None
_EASYOCR_ERROR = None

BASE_DIR = Path(__file__).resolve().parent.parent
OCR_SAMPLES_DIR = BASE_DIR / "ocr_samples"
OCR_SAMPLES_INDEX = OCR_SAMPLES_DIR / "index.json"
OCR_HARD_NEGATIVE_DIR = BASE_DIR / "text_negatives"
OCR_SAMPLE_WIDTH = 128
OCR_SAMPLE_HEIGHT = 64


def easyocr_disponivel() -> bool:
    return (
        os.environ.get("CIRCUIT_SCANNER_DISABLE_EASYOCR") != "1"
        and importlib.util.find_spec("easyocr") is not None
    )


REFERENCE_PREFIXES = {
    "resistor": ("R",),
    "capacitor": ("C",),
    "capacitor_polarized": ("C", "CP"),
    "diode": ("D",),
    "led": ("LED", "D"),
    "inductor": ("L",),
    "source_dc": ("V", "VS"),
    "battery": ("BAT", "BT", "V"),
    "gnd": ("GND",),
    "vcc": ("VCC",),
    "transistor_npn": ("Q", "T"),
    "transistor_pnp": ("Q", "T"),
    "mosfet_n": ("M", "Q"),
    "mosfet_p": ("M", "Q"),
    "opamp": ("U", "IC"),
}

FALLBACK_PREFIXES = {
    "resistor": "R",
    "capacitor": "C",
    "capacitor_polarized": "C",
    "diode": "D",
    "led": "LED",
    "inductor": "L",
    "source_dc": "V",
    "battery": "BAT",
    "gnd": "GND",
    "vcc": "VCC",
    "transistor_npn": "Q",
    "transistor_pnp": "Q",
    "mosfet_n": "M",
    "mosfet_p": "M",
    "opamp": "U",
}

VALUE_REQUIRED = {
    "resistor",
    "capacitor",
    "capacitor_polarized",
    "inductor",
    "source_dc",
    "battery",
}

NO_VALUE_CLASSES = {
    "led",
    "gnd",
    "transistor_npn",
    "transistor_pnp",
    "mosfet_n",
    "mosfet_p",
    "opamp",
}

_REFERENCE_GENERIC = re.compile(
    r"^(?:(?:R|C|CP|D|LED|L|V|VS|BAT|BT|GND|VCC|Q|M|U|IC)\d*|R(?:L|S|B|E|C|IN|OUT)\d*|C(?:IN|OUT)\d*|[+-]?V(?:CC|EE|DD|SS)\d*)$",
    re.IGNORECASE,
)


def localizar_tesseract() -> str | None:
    configurado = os.environ.get("TESSERACT_CMD", "").strip().strip('"')
    candidatos = [
        configurado,
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidato in candidatos:
        if candidato and Path(candidato).is_file():
            return str(Path(candidato))
    return None


def ocr_status() -> dict:
    from backend.handwriting_reference import library,READER_VERSION,BANK
    from backend.glyph_classifier import status
    manifest,refs=library()
    classifier=status(BANK)
    return dict(ready=bool(refs),engine='Leitura da sua escrita e texto impresso local',printed_reader_available=bool(localizar_tesseract()),training=classifier['ready'],glyph_classifier=classifier,
                reference_count=len(refs),version=manifest.get('version'),reader_version=READER_VERSION,
                message='Escrita pronta para leitura.' if refs else 'Adicione referências manuscritas para ler o OCR.')


def _corrigir_confusoes_comuns(texto: str) -> str:
    # Correct digits only behind an explicit reference prefix. Never turn 81
    # into R1 or change the case of a milli/mega prefix in a value.
    compact=re.sub(r'\s+','',texto).upper()
    from backend.drawing_mode import digital
    if digital():
        from backend.reference_labels import printed_reference_classes
        if printed_reference_classes(compact):return compact
    match=re.fullmatch(r'(LED|CP|VS|IC|R|C|Q|T|M|U|L|V)([0-9OIL|]+)',compact)
    if match:
        number=match[2].translate(str.maketrans({'O':'0','I':'1','L':'1','|':'1'}))
        return match[1]+number
    return texto


def _limpar_texto(texto: str) -> str:
    texto=clean_unicode(texto).strip()
    if re.match(r'^(?:hfe|beta|β|B|vgs|vth|gm|aol|avol|dcr|esr|isat)\s*[:=(]',texto,re.IGNORECASE):
        return re.sub(r'[^A-Za-z0-9β.,+\-/µμΩΩω():=\s]', '', texto)[:80]
    texto=re.sub(r'[^A-Za-z0-9β.,+\-/µμΩΩω|():=\s]', '', texto)[:80]
    # A physical unit takes precedence over reference heuristics.
    if parse_measurement(texto):return texto
    return _corrigir_confusoes_comuns(texto)


def _texto_chave(texto: str) -> str:
    rail=re.fullmatch(r'\s*([+-])\s*(V(?:CC|EE|DD|SS)\d*)\s*',texto,re.I)
    if rail:return rail[1]+rail[2].upper()
    return re.sub(r"[^A-Z0-9]", "", texto.upper())


def _chave_leitura(texto: str) -> str:
    # Values must retain unit, decimal punctuation and milli/mega distinctions.
    parsed=parse_measurement(texto) or parse_measurement(texto,'resistor')
    if parsed:
        return f"value:{parsed['unit']}:{parsed['normalized']}:{parsed['unit_explicit']}"
    if unit_anchor(texto):return 'unit:'+re.sub(r'\s+','',clean_unicode(texto))
    return _texto_chave(texto)


def _normalizar_amostra_ocr(imagem: np.ndarray) -> np.ndarray | None:
    """Converte uma palavra manuscrita para um molde binário comparável."""
    if imagem is None or imagem.size == 0:
        return None
    cinza = (
        cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        if imagem.ndim == 3
        else imagem.copy()
    )
    cinza = cv2.GaussianBlur(cinza, (3, 3), 0)
    _, binaria = cv2.threshold(
        cinza,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    # Remove linhas longas do caderno e fios que atravessam o recorte.
    kernel_horizontal = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(12, binaria.shape[1] * 2 // 3), 1),
    )
    kernel_vertical = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(12, binaria.shape[0] * 2 // 3)),
    )
    linhas = cv2.bitwise_or(
        cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_horizontal),
        cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_vertical),
    )
    binaria = cv2.subtract(binaria, linhas)
    quantidade, rotulos, estatisticas, _ = cv2.connectedComponentsWithStats(
        binaria,
        connectivity=8,
    )
    mascara = np.zeros_like(binaria)
    area_minima = max(2, int(binaria.size * 0.0008))
    for indice in range(1, quantidade):
        if estatisticas[indice, cv2.CC_STAT_AREA] >= area_minima:
            mascara[rotulos == indice] = 255
    pontos = cv2.findNonZero(mascara)
    if pontos is None:
        return None
    x, y, largura, altura = cv2.boundingRect(pontos)
    conteudo = mascara[y:y + altura, x:x + largura]
    margem = 6
    limite_largura = OCR_SAMPLE_WIDTH - margem * 2
    limite_altura = OCR_SAMPLE_HEIGHT - margem * 2
    escala = min(
        limite_largura / max(1, largura),
        limite_altura / max(1, altura),
    )
    novo_tamanho = (
        max(1, int(round(largura * escala))),
        max(1, int(round(altura * escala))),
    )
    redimensionada = cv2.resize(
        conteudo,
        novo_tamanho,
        interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_CUBIC,
    )
    tela = np.zeros((OCR_SAMPLE_HEIGHT, OCR_SAMPLE_WIDTH), dtype=np.uint8)
    inicio_x = (OCR_SAMPLE_WIDTH - novo_tamanho[0]) // 2
    inicio_y = (OCR_SAMPLE_HEIGHT - novo_tamanho[1]) // 2
    tela[
        inicio_y:inicio_y + novo_tamanho[1],
        inicio_x:inicio_x + novo_tamanho[0],
    ] = redimensionada
    return tela


def _recortar_token(imagem: np.ndarray, bbox: dict, margem: int = 4) -> np.ndarray | None:
    altura, largura = imagem.shape[:2]
    x1 = max(0, int(math.floor(float(bbox["x1"]))) - margem)
    y1 = max(0, int(math.floor(float(bbox["y1"]))) - margem)
    x2 = min(largura, int(math.ceil(float(bbox["x2"]))) + margem)
    y2 = min(altura, int(math.ceil(float(bbox["y2"]))) + margem)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return imagem[y1:y2, x1:x2]


def _carregar_indice_amostras() -> list[dict]:
    if not OCR_SAMPLES_INDEX.is_file():
        return []
    try:
        dados = json.loads(OCR_SAMPLES_INDEX.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return dados if isinstance(dados, list) else []


def _salvar_indice_amostras(amostras: list[dict]) -> None:
    OCR_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    OCR_SAMPLES_INDEX.write_text(
        json.dumps(amostras[-500:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _token_para_campo(
    componente: dict,
    campo: str,
) -> dict | None:
    tokens = componente.get("ocr_tokens") or []
    anterior = componente.get(campo) or {}
    texto_anterior = _chave_leitura(str(anterior.get("text", "")))
    candidatos = []
    for token in tokens:
        classes, tipo = _classes_sugeridas_por_texto(token.get("text", ""))
        if campo == 'model':
            classes = model_ocr_reading(token.get('text',''))['classes']
            tipo = 'model' if classes else None
        if tipo != campo:
            continue
        prioridade = 0 if _chave_leitura(token.get("text", "")) == texto_anterior else 1
        compatibilidade = 0 if componente.get("type") in classes else 1
        candidatos.append(
            (
                prioridade,
                compatibilidade,
                -float(token.get("confidence", 0.0)),
                token,
            )
        )
    if not candidatos:
        return None
    return min(candidatos, key=lambda item: item[:3])[-1]


def salvar_amostras_ocr(
    image_path: Path,
    resultado: dict,
    edicoes: list[dict],
) -> int:
    """Guarda recortes confirmados para adaptar o OCR à escrita do usuário."""
    imagem = cv2.imread(str(image_path))
    if imagem is None:
        return 0
    edicoes_por_id = {item.get("detection_id"): item for item in edicoes}
    indice = _carregar_indice_amostras()
    por_hash = {item.get("sha256"): item for item in indice if item.get("sha256")}
    adicionadas = 0

    for componente in resultado.get("components", []):
        edicao = edicoes_por_id.get(componente.get("detection_id"))
        if not edicao:
            continue
        classe = componente.get("type", "")
        referencia_nova = _texto_chave(str(edicao.get("reference", "")))
        referencia_anterior = _texto_chave(
            str((componente.get("reference") or {}).get("text", ""))
        )
        campos = []
        if 'model' in edicao:
            modelo_novo = model_name(edicao['model'])
            modelo_anterior = (componente.get('model') or {}).get('text','')
            if modelo_novo and modelo_novo != modelo_anterior:
                campos.append(('model', modelo_novo))
        if referencia_nova and referencia_nova != referencia_anterior:
            campos.append(("reference", referencia_nova))
        if classe not in NO_VALUE_CLASSES:
            valor_novo = _limpar_texto(str(edicao.get("value", "")))
            valor_anterior = _limpar_texto(
                str((componente.get("value") or {}).get("text", ""))
            )
            if valor_novo and _chave_leitura(valor_novo) != _chave_leitura(valor_anterior):
                campos.append(("value", valor_novo))

        for campo, rotulo in campos:
            if not rotulo:
                continue
            token = _token_para_campo(componente, campo)
            if token is None:
                continue
            recorte = _recortar_token(imagem, token["bbox"])
            normalizada = _normalizar_amostra_ocr(recorte)
            if normalizada is None:
                continue
            sucesso, dados_png = cv2.imencode(".png", normalizada)
            if not sucesso:
                continue
            bytes_png = dados_png.tobytes()
            digest = hashlib.sha256(bytes_png).hexdigest()
            nome_arquivo = f"{digest[:20]}.png"
            registro = {
                "sha256": digest,
                "file": nome_arquivo,
                "template_file": f"{digest[:20]}_context.png",
                "label": rotulo,
                "kind": campo,
                "component_type": classe,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            existente = por_hash.get(digest)
            if existente:
                existente.update(registro)
                continue
            OCR_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
            (OCR_SAMPLES_DIR / nome_arquivo).write_bytes(bytes_png)
            sucesso_contexto, contexto_png = cv2.imencode(".png", recorte)
            if sucesso_contexto:
                (OCR_SAMPLES_DIR / registro["template_file"]).write_bytes(
                    contexto_png.tobytes()
                )
            indice.append(registro)
            por_hash[digest] = registro
            adicionadas += 1

    if adicionadas or indice:
        _salvar_indice_amostras(indice)
    return adicionadas


def salvar_negativos_texto_yolo(
    imagem: np.ndarray,
    tokens: list[dict],
    limite: int = 800,
) -> int:
    """Salva textos reais como imagens sem rótulos para o próximo treino YOLO."""
    OCR_HARD_NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)
    existentes = {
        caminho.stem.removeprefix("text_")
        for caminho in OCR_HARD_NEGATIVE_DIR.glob("text_*.jpg")
    }
    adicionadas = 0
    for token in tokens:
        if len(existentes) >= limite:
            break
        chave = _texto_chave(token.get("text", ""))
        if len(chave) < 2 or not re.search(r"[A-Z0-9]", chave):
            continue
        recorte = _recortar_token(imagem, token.get("bbox", {}), margem=7)
        if recorte is None or recorte.size == 0:
            continue
        altura, largura = recorte.shape[:2]
        escala = min(220 / max(1, largura), 220 / max(1, altura))
        tamanho = (
            max(1, int(round(largura * escala))),
            max(1, int(round(altura * escala))),
        )
        ampliado = cv2.resize(recorte, tamanho, interpolation=cv2.INTER_CUBIC)
        fundo = int(np.median(recorte))
        canvas = np.full((256, 256, 3), fundo, dtype=np.uint8)
        x = (256 - tamanho[0]) // 2
        y = (256 - tamanho[1]) // 2
        canvas[y:y + tamanho[1], x:x + tamanho[0]] = ampliado
        sucesso, jpg = cv2.imencode(
            ".jpg",
            canvas,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        if not sucesso:
            continue
        digest = hashlib.sha256(jpg.tobytes()).hexdigest()[:20]
        if digest in existentes:
            continue
        (OCR_HARD_NEGATIVE_DIR / f"text_{digest}.jpg").write_bytes(jpg.tobytes())
        existentes.add(digest)
        adicionadas += 1
    return adicionadas


def _referencia_compativel(texto: str, classe: str) -> bool:
    from backend.reference_labels import reference_classes
    return classe in reference_classes(texto)


def _parece_valor(texto: str, classe: str) -> bool:
    if classe in DIMENSIONS:
        parsed=parse_measurement(texto,classe)
        if not parsed:return False
        if parsed['unit_explicit']:return True
        if classe in {'source_dc','battery','vcc'}:return False
        return bool(parsed['prefix'] or len(re.findall(r'\d',texto))>=2)
    # Model-like texts are kept as text; enrich_models owns their assignment.
    return classe in model_ocr_reading(texto)['classes'] if len(texto)<=80 else False


def _valor_com_confianca(token: dict, classe: str) -> bool:
    if not _parece_valor(token['text'],classe):return False
    parsed=parse_measurement(token['text'],classe) if classe in DIMENSIONS else None
    easy=token.get('engine')=='easyocr'
    if parsed:
        minimum=(.45 if easy else .55) if parsed['corrections'] else (.18 if easy else .30) if parsed['unit_explicit'] or parsed['prefix'] else (.75 if easy else .80)
    else:minimum=.35
    return float(token.get('confidence',0))>=minimum


def _token_relevante(token: dict) -> bool:
    engine=token.get('engine');confidence=float(token.get('confidence',0))
    minimum=.70 if engine=='template' else .075 if engine=='easyocr' else .22
    if confidence<minimum:return False
    text=token['text'];key=_texto_chave(text)
    from backend.drawing_mode import digital
    from backend.reference_labels import reference_classes
    if digital() and reference_classes(text):return True
    if unit_anchor(text):return True
    if _REFERENCE_GENERIC.fullmatch(key):
        return bool(re.search(r'\d',key)) or key in {'GND','VCC','LED'} or bool(re.fullmatch(r'R(?:L|S|B|E|C|IN|OUT)|C(?:IN|OUT)|[+-]?V(?:CC|EE|DD|SS)',key))
    if any(_valor_com_confianca(token,kind) for kind in REFERENCE_PREFIXES):return True
    if re.match(r'^(?:hfe|beta|β|B|vgs|vth|gm|aol|avol|dcr|esr|isat)\s*[:=(]',text,re.I):return confidence>=.35
    if _line_fields(text):return True
    if 2<=len(key)<=80 and re.search(r'[A-Z]',key) and re.search(r'\d',key):
        return confidence>=(.18 if engine=='easyocr' else .38)
    return False


def _normalizar_valor(texto: str, classe: str) -> dict:
    parsed=parse_measurement(texto,classe)
    if parsed:return parsed
    return {'text':texto.strip(),'raw_text':texto.strip(),'normalized':texto.strip(),
            'unit':DIMENSIONS.get(classe),'needs_review':True,'reading_method':'unparsed'}


def _bbox_iou(a: dict, b: dict) -> float:
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    uniao = area_a + area_b - inter
    return inter / uniao if uniao > 0 else 0.0


def _preparar_variantes(imagem: np.ndarray) -> tuple[list[tuple[str, np.ndarray]], float]:
    altura, largura = imagem.shape[:2]
    maior = max(altura, largura)
    escala = min(4.0, max(1.0, 1800.0 / max(maior, 1)))
    ampliada = cv2.resize(
        imagem,
        None,
        fx=escala,
        fy=escala,
        interpolation=cv2.INTER_CUBIC,
    )
    cinza = cv2.cvtColor(ampliada, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(cinza)
    suave = cv2.GaussianBlur(clahe, (3, 3), 0)
    binaria = cv2.adaptiveThreshold(
        suave,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        13,
    )
    return [("gray", clahe), ("binary", binaria)], escala


def _remover_linhas_longas_para_ocr(imagem: np.ndarray) -> np.ndarray:
    """Atenua linhas do caderno e fios longos apenas na imagem enviada ao OCR."""
    if imagem.size == 0:
        return imagem
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    _, tinta = cv2.threshold(
        cinza,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    comprimento = max(24, imagem.shape[1] // 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (comprimento, 1))
    linhas = cv2.morphologyEx(tinta, cv2.MORPH_OPEN, kernel)
    linhas = cv2.dilate(
        linhas,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    if cv2.countNonZero(linhas) == 0:
        return imagem
    return cv2.inpaint(imagem, linhas, 3, cv2.INPAINT_TELEA)


def preparar_imagem_texto_ao_redor(
    imagem: np.ndarray,
    deteccoes: list[dict],
) -> np.ndarray:
    """Mantém apenas vizinhanças dos componentes e mascara seus núcleos."""
    altura_imagem, largura_imagem = imagem.shape[:2]
    contexto = np.full_like(imagem, 255)

    for deteccao in deteccoes:
        bbox = deteccao["bbox"]
        largura = max(1.0, bbox["x2"] - bbox["x1"])
        altura = max(1.0, bbox["y2"] - bbox["y1"])
        margem = min(220.0, max(55.0, max(largura, altura) * 1.35))
        x1 = max(0, int(round(bbox["x1"] - margem)))
        y1 = max(0, int(round(bbox["y1"] - margem)))
        x2 = min(largura_imagem, int(round(bbox["x2"] + margem)))
        y2 = min(altura_imagem, int(round(bbox["y2"] + margem)))
        contexto[y1:y2, x1:x2] = imagem[y1:y2, x1:x2]

    for deteccao in deteccoes:
        bbox = deteccao["bbox"]
        largura = max(1.0, bbox["x2"] - bbox["x1"])
        altura = max(1.0, bbox["y2"] - bbox["y1"])
        x1 = max(0, int(round(bbox["x1"] + largura * 0.22)))
        y1 = max(0, int(round(bbox["y1"] + altura * 0.22)))
        x2 = min(largura_imagem, int(round(bbox["x2"] - largura * 0.22)))
        y2 = min(altura_imagem, int(round(bbox["y2"] - altura * 0.22)))
        if x2 > x1 and y2 > y1:
            contexto[y1:y2, x1:x2] = 255

    return contexto


def _mapear_bbox_rotacionada(
    bbox: dict,
    orientacao: str,
    largura_original: int,
    altura_original: int,
) -> dict:
    pontos = [
        (bbox["x1"], bbox["y1"]),
        (bbox["x2"], bbox["y1"]),
        (bbox["x1"], bbox["y2"]),
        (bbox["x2"], bbox["y2"]),
    ]
    convertidos = []
    for x, y in pontos:
        if orientacao == "cw":
            convertidos.append((y, altura_original - x))
        elif orientacao == "ccw":
            convertidos.append((largura_original - y, x))
        else:
            convertidos.append((x, y))
    xs = [p[0] for p in convertidos]
    ys = [p[1] for p in convertidos]
    return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}


def _executar_tesseract_tsv(
    imagem: np.ndarray,
    comando: str,
    variante: str,
    orientacao: str,
    escala: float,
    largura_ampliada: int,
    altura_ampliada: int,
    *, digits_only: bool = False, timeout: float = 60,
) -> list[dict]:
    ok, png = cv2.imencode(".png", imagem)
    if not ok:
        return []

    args = [
        comando,
        "stdin",
        "stdout",
        "-l",
        "eng",
        "--oem",
        "1",
        "--psm",
        "7" if digits_only else "11",
        "-c",
        "preserve_interword_spaces=1",
        "tsv",
    ]
    if digits_only:
        args[-1:-1]=['-c','tessedit_char_whitelist=0123456789.,+-eE']
    processo = subprocess.run(
        args,
        input=png.tobytes(),
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if processo.returncode != 0:
        mensagem = processo.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Tesseract falhou: {mensagem or processo.returncode}")

    texto_tsv = processo.stdout.decode("utf-8", errors="replace")
    leitor = csv.DictReader(io.StringIO(texto_tsv), delimiter="\t")
    tokens = []
    for linha in leitor:
        texto = _limpar_texto(linha.get("text", ""))
        try:
            confianca = float(linha.get("conf", -1))
        except (TypeError, ValueError):
            confianca = -1
        if not texto or confianca < 0:
            continue

        x = float(linha["left"])
        y = float(linha["top"])
        w = float(linha["width"])
        h = float(linha["height"])
        bbox_rotacionada = {"x1": x, "y1": y, "x2": x + w, "y2": y + h}
        bbox_ampliada = _mapear_bbox_rotacionada(
            bbox_rotacionada,
            orientacao,
            largura_ampliada,
            altura_ampliada,
        )
        bbox = {chave: round(valor / escala, 2) for chave, valor in bbox_ampliada.items()}
        tokens.append(
            {
                "text": texto,
                "raw_text": str(linha.get("text", "")),
                "confidence": round(confianca / 100.0, 4),
                "bbox": bbox,
                "variant": variante,
                "orientation": orientacao,
                "engine": "tesseract",
            }
        )
    return tokens


def _carregar_easyocr():
    global _EASYOCR_READER, _EASYOCR_ERROR
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER
    if _EASYOCR_ERROR is not None:
        raise RuntimeError(_EASYOCR_ERROR)
    try:
        from backend.cpu_inference import configure_cpu_inference
        configure_cpu_inference()
        import easyocr

        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        return _EASYOCR_READER
    except Exception as erro:
        _EASYOCR_ERROR = str(erro)
        raise RuntimeError(_EASYOCR_ERROR) from erro


def _executar_easyocr(
    imagem: np.ndarray,
    escala: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    variante: str = "easyocr_clahe",
) -> list[dict]:
    leitor = _carregar_easyocr()
    resultados = leitor.readtext(
        imagem,
        detail=1,
        paragraph=False,
        min_size=7,
        text_threshold=0.30,
        low_text=0.15,
        link_threshold=0.20,
        # A page was previously enlarged to 1800 and then again by 1.5.
        # Keep one bounded global pass; small uncertain labels are reread
        # at their original resolution in the existing component crops.
        canvas_size=1600,
        mag_ratio=1.0,
    )
    tokens = []
    for pontos, texto_bruto, confianca in resultados:
        texto = _limpar_texto(str(texto_bruto))
        if not texto:
            continue
        xs = [float(ponto[0]) / escala + offset_x for ponto in pontos]
        ys = [float(ponto[1]) / escala + offset_y for ponto in pontos]
        tokens.append(
            {
                "text": texto,
                "raw_text": str(texto_bruto),
                "confidence": round(float(confianca), 4),
                "bbox": {
                    "x1": round(min(xs), 2),
                    "y1": round(min(ys), 2),
                    "x2": round(max(xs), 2),
                    "y2": round(max(ys), 2),
                },
                "variant": variante,
                "orientation": "normal",
                "engine": "easyocr",
            }
        )
    return tokens


def _tiles_ocr(imagem: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    altura, largura = imagem.shape[:2]
    cortes_x = ((0, int(largura * 0.58)), (int(largura * 0.42), largura))
    cortes_y = ((0, int(altura * 0.58)), (int(altura * 0.42), altura))
    return [
        (x1, y1, imagem[y1:y2, x1:x2])
        for y1, y2 in cortes_y
        for x1, x2 in cortes_x
        if x2 - x1 >= 40 and y2 - y1 >= 40
    ]


def preparar_mosaico_componentes(
    imagem: np.ndarray,
    deteccoes: list[dict],
    tamanho_celula: int = 280,
    colunas: int = 4,
) -> tuple[np.ndarray | None, list[dict]]:
    """Amplia a vizinhança de cada componente em um mosaico para o OCR."""
    if not deteccoes:
        return None, []
    altura_imagem, largura_imagem = imagem.shape[:2]
    linhas = math.ceil(len(deteccoes) / colunas)
    mosaico = np.full(
        (linhas * tamanho_celula, colunas * tamanho_celula, 3),
        255,
        dtype=np.uint8,
    )
    mapas = []

    for indice, deteccao in enumerate(deteccoes):
        bbox = deteccao["bbox"]
        largura = max(1.0, bbox["x2"] - bbox["x1"])
        altura = max(1.0, bbox["y2"] - bbox["y1"])
        margem = min(180.0, max(55.0, max(largura, altura) * 1.10))
        crop_x1 = max(0, int(round(bbox["x1"] - margem)))
        crop_y1 = max(0, int(round(bbox["y1"] - margem)))
        crop_x2 = min(largura_imagem, int(round(bbox["x2"] + margem)))
        crop_y2 = min(altura_imagem, int(round(bbox["y2"] + margem)))
        recorte = imagem[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        if recorte.size == 0:
            continue
        nucleo_x1 = int(round(bbox["x1"] - crop_x1 + largura * 0.22))
        nucleo_y1 = int(round(bbox["y1"] - crop_y1 + altura * 0.22))
        nucleo_x2 = int(round(bbox["x2"] - crop_x1 - largura * 0.22))
        nucleo_y2 = int(round(bbox["y2"] - crop_y1 - altura * 0.22))
        if nucleo_x2 > nucleo_x1 and nucleo_y2 > nucleo_y1:
            recorte[nucleo_y1:nucleo_y2, nucleo_x1:nucleo_x2] = 255

        limite = tamanho_celula - 20
        escala = min(
            limite / max(1, recorte.shape[1]),
            limite / max(1, recorte.shape[0]),
        )
        largura_render = max(1, int(round(recorte.shape[1] * escala)))
        altura_render = max(1, int(round(recorte.shape[0] * escala)))
        render = cv2.resize(
            recorte,
            (largura_render, altura_render),
            interpolation=cv2.INTER_CUBIC,
        )
        coluna = indice % colunas
        linha = indice // colunas
        paste_x = coluna * tamanho_celula + (tamanho_celula - largura_render) // 2
        paste_y = linha * tamanho_celula + (tamanho_celula - altura_render) // 2
        mosaico[
            paste_y:paste_y + altura_render,
            paste_x:paste_x + largura_render,
        ] = render
        mapas.append(
            {
                "detection_id": deteccao["detection_id"],
                "component_class": deteccao.get("class"),
                "paste_x": paste_x,
                "paste_y": paste_y,
                "render_width": largura_render,
                "render_height": altura_render,
                "crop_x1": crop_x1,
                "crop_y1": crop_y1,
                "scale": escala,
            }
        )
    return mosaico, mapas


def _executar_easyocr_mosaico(
    mosaico: np.ndarray,
    mapas: list[dict],
) -> list[dict]:
    """Lê cada célula separadamente para impedir caixas entre componentes.

    O método anterior enviava o mosaico inteiro ao detector de texto. Em
    diagramas manuscritos ele podia unir duas células vizinhas e remapear uma
    palavra para uma caixa enorme. ``readtext_batched`` mantém a velocidade do
    lote sem permitir que a detecção de texto atravesse a célula atual.
    """
    leitor = _carregar_easyocr()
    tamanho_lote = 320
    imagens = []
    transformacoes = []
    mapas_validos = []
    for mapa in mapas:
        x1 = int(mapa["paste_x"])
        y1 = int(mapa["paste_y"])
        largura = int(mapa["render_width"])
        altura = int(mapa["render_height"])
        recorte = mosaico[y1:y1 + altura, x1:x1 + largura]
        if recorte.size == 0:
            continue
        escala_lote = min(
            (tamanho_lote - 16) / max(1, largura),
            (tamanho_lote - 16) / max(1, altura),
        )
        largura_lote = max(1, int(round(largura * escala_lote)))
        altura_lote = max(1, int(round(altura * escala_lote)))
        redimensionado = cv2.resize(
            recorte,
            (largura_lote, altura_lote),
            interpolation=cv2.INTER_CUBIC,
        )
        tela = np.full((tamanho_lote, tamanho_lote, 3), 255, dtype=np.uint8)
        offset_x = (tamanho_lote - largura_lote) // 2
        offset_y = (tamanho_lote - altura_lote) // 2
        tela[offset_y:offset_y + altura_lote, offset_x:offset_x + largura_lote] = (
            redimensionado
        )
        imagens.append(tela)
        transformacoes.append(
            {
                "scale": escala_lote,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "width": largura,
                "height": altura,
            }
        )
        mapas_validos.append(mapa)

    if not imagens:
        return []

    resultados_lote = leitor.readtext_batched(
        imagens,
        n_width=tamanho_lote,
        n_height=tamanho_lote,
        detail=1,
        paragraph=False,
        decoder="greedy",
        min_size=7,
        text_threshold=0.25,
        low_text=0.12,
        link_threshold=0.18,
        canvas_size=tamanho_lote,
        mag_ratio=1.0,
        batch_size=min(4, len(imagens)),
        workers=0,
    )
    tokens = []
    for resultados, mapa, transformacao in zip(
        resultados_lote,
        mapas_validos,
        transformacoes,
    ):
        for pontos, texto_bruto, confianca in resultados:
            texto = _limpar_texto(str(texto_bruto))
            if not texto:
                continue
            xs_render = [
                min(
                    transformacao["width"],
                    max(
                        0.0,
                        (float(ponto[0]) - transformacao["offset_x"])
                        / transformacao["scale"],
                    ),
                )
                for ponto in pontos
            ]
            ys_render = [
                min(
                    transformacao["height"],
                    max(
                        0.0,
                        (float(ponto[1]) - transformacao["offset_y"])
                        / transformacao["scale"],
                    ),
                )
                for ponto in pontos
            ]
            if max(xs_render) - min(xs_render) < 2 or max(ys_render) - min(ys_render) < 2:
                continue
            xs = [
                mapa["crop_x1"] + valor / mapa["scale"]
                for valor in xs_render
            ]
            ys = [
                mapa["crop_y1"] + valor / mapa["scale"]
                for valor in ys_render
            ]
            tokens.append(
                {
                    "text": texto,
                "raw_text": str(texto_bruto),
                    "confidence": round(float(confianca), 4),
                    "bbox": {
                        "x1": round(min(xs), 2),
                        "y1": round(min(ys), 2),
                        "x2": round(max(xs), 2),
                        "y2": round(max(ys), 2),
                    },
                    "variant": "easyocr_component_batch",
                    "orientation": "normal",
                    "engine": "easyocr",
                    "source_detection_id": mapa["detection_id"],
                    "source_component_class": mapa.get("component_class"),
                }
            )
    return tokens


def _remapear_tokens_mosaico(
    tokens: list[dict],
    mapas: list[dict],
    variante: str,
) -> list[dict]:
    remapeados = []
    for token in tokens:
        bbox = token["bbox"]
        centro_x = (bbox["x1"] + bbox["x2"]) / 2.0
        centro_y = (bbox["y1"] + bbox["y2"]) / 2.0
        mapa = next(
            (
                item
                for item in mapas
                if item["paste_x"] <= centro_x <= item["paste_x"] + item["render_width"]
                and item["paste_y"] <= centro_y <= item["paste_y"] + item["render_height"]
            ),
            None,
        )
        if mapa is None:
            continue
        item = dict(token)
        item["bbox"] = {
            "x1": round(
                mapa["crop_x1"] + (bbox["x1"] - mapa["paste_x"]) / mapa["scale"],
                2,
            ),
            "y1": round(
                mapa["crop_y1"] + (bbox["y1"] - mapa["paste_y"]) / mapa["scale"],
                2,
            ),
            "x2": round(
                mapa["crop_x1"] + (bbox["x2"] - mapa["paste_x"]) / mapa["scale"],
                2,
            ),
            "y2": round(
                mapa["crop_y1"] + (bbox["y2"] - mapa["paste_y"]) / mapa["scale"],
                2,
            ),
        }
        item["variant"] = variante
        item["source_detection_id"] = mapa["detection_id"]
        remapeados.append(item)
    return remapeados


def _deslocar_tokens(
    tokens: list[dict],
    offset_x: float,
    offset_y: float,
    sufixo: str,
) -> list[dict]:
    saida = []
    for token in tokens:
        item = dict(token)
        item["bbox"] = {
            "x1": round(token["bbox"]["x1"] + offset_x, 2),
            "y1": round(token["bbox"]["y1"] + offset_y, 2),
            "x2": round(token["bbox"]["x2"] + offset_x, 2),
            "y2": round(token["bbox"]["y2"] + offset_y, 2),
        }
        item["variant"] = f"{token['variant']}_{sufixo}"
        saida.append(item)
    return saida


def _combinar_tokens_fragmentados(tokens: list[dict]) -> list[dict]:
    """Join up to six nearby fragments, including already readable numbers.

    Require the same OCR view, orientation and component crop. Keep longest
    valid spans so 1 + 0 + V becomes 10V rather than silently losing the 1.
    """
    proposals=[]
    for first in tokens:
        # Complete reference-comparison spans have already been segmented and
        # parsed. Joining them again mixes separate labels, especially after
        # rotating their boxes back into the original photograph.
        if first.get('reading_method') or first.get('source')=='handwriting_reference':continue
        from backend.reference_labels import reference_classes
        if first.get('engine')=='local_printed_word' and reference_classes(first['text']):continue
        chain=[first]
        while len(chain)<=6:
            if len(chain)>1:
                text=''.join(t['text'] for t in chain)
                measurement=parse_measurement(text)
                known=len(text)<=80 and model_name(text) in CATALOG
                unknown_model=(len(text)<=80 and bool(model_classes(text)) and
                               not model_classes(first['text']) and not parse_measurement(first['text']))
                parameter=re.fullmatch(r'(?:hfe|beta|β|B|vgs\(?th\)?|vth|gm|aol|avol|dcr|esr|isat)[:=](.+)',text,re.I)
                parameter=bool(parameter and (parse_measurement(parameter[1]) or parse_measurement(parameter[1],'resistor')))
                reference=len(text)<=16 and bool(_REFERENCE_GENERIC.fullmatch(_texto_chave(text))) and any(re.search(r'[A-Za-z]',t['text']) for t in chain)
                if measurement or known or unknown_model or reference or parameter:
                    candidate=dict(first,text=text,raw_text=' '.join(t.get('raw_text',t['text']) for t in chain),
                                   confidence=min(float(t.get('confidence',0)) for t in chain),
                                   bbox={key:func(t['bbox'][key] for t in chain) for key,func in [('x1',min),('y1',min),('x2',max),('y2',max)]},
                                   variant=first.get('variant','')+'_joined',
                                   joined_parts=[{'text':t['text'],'bbox':dict(t['bbox']),'engine':t.get('engine'),'variant':t.get('variant')} for t in chain])
                    if _token_relevante(candidate):proposals.append(candidate)
            if len(chain)==6:break
            last=chain[-1];a=last['bbox'];height=max(1.,a['y2']-a['y1']);nexts=[]
            for other in tokens:
                if any(other is t for t in chain):continue
                if any(other.get(k)!=first.get(k) for k in ('engine','variant','orientation','source_detection_id')):continue
                b=other['bbox'];h=max(1.,b['y2']-b['y1']);overlap=max(0.,min(a['y2'],b['y2'])-max(a['y1'],b['y1']))
                gap=b['x1']-a['x2']
                if b['x1']<=a['x1'] or gap<-.20*min(height,h) or gap>1.25*max(height,h) or overlap/min(height,h)<.55:continue
                # Separate multi-digit numbers are different labels, not digits
                # of one value. Tight one-digit splits can still be repaired.
                left_numeric=bool(re.fullmatch(r'[0-9OIl|]+',last['text']))
                right_numeric=bool(re.fullmatch(r'[0-9OIl|]+',other['text']))
                other_value=parse_measurement(other['text']) or parse_measurement(other['text'],'resistor')
                if left_numeric and other_value and (other_value['unit_explicit'] or other_value['prefix'] or len(other['text'])>1):continue
                if left_numeric and right_numeric and (len(last['text'])>1 or len(other['text'])>1 or gap>.45*max(height,h)):continue
                # Never append another number to a complete unit-labelled value.
                if parse_measurement(''.join(t['text'] for t in chain)):continue
                nexts.append((max(0.,gap),b['x1'],other))
            if not nexts:break
            chain.append(min(nexts,key=lambda item:item[:2])[2])
    accepted=[];occupied=set()
    for proposal in sorted(proposals,key=lambda p:(-len(p['joined_parts']),-p['confidence'],p['bbox']['x1'])):
        keys=[(p['text'],tuple(p['bbox'][k] for k in ('x1','y1','x2','y2')),p['engine'],p['variant']) for p in proposal['joined_parts']]
        if any(key in occupied for key in keys):continue
        accepted.append(proposal);occupied.update(keys)
    return accepted


def _without_joined_parts(tokens):
    joined=[t for t in tokens if t.get('joined_parts')]
    parts={(p['text'],tuple(p['bbox'][k] for k in ('x1','y1','x2','y2')),p.get('engine'),p.get('variant')) for t in joined for p in t['joined_parts']}
    return [t for t in tokens if t.get('joined_parts') or (t['text'],tuple(t['bbox'][k] for k in ('x1','y1','x2','y2')),t.get('engine'),t.get('variant')) not in parts]


def _reread_unit_numbers(image,tokens,command):
    """Small numeric-only rereads around unit anchors, with a bounded budget.

    Keep disagreements as alternatives for human review. Never whitelist
    digits on model codes or on the entire photo.
    """
    if not command:return [],[]
    deadline=time.monotonic()+20;seen=[];added=[];observations=[]
    for anchor in sorted(tokens,key=lambda t:-float(t.get('confidence',0))):
        parsed=parse_measurement(anchor['text']);dimension=unit_anchor(anchor['text'])
        if not dimension and not (parsed and parsed['corrections']):continue
        if float(anchor.get('confidence',0))<.30:continue
        b=anchor['bbox']
        if any(_bbox_iou(b,old)>.4 for old in seen):continue
        if len(seen)>=6 or time.monotonic()>=deadline:break
        seen.append(b);height=max(8,b['y2']-b['y1'])
        if parsed:
            dimension=parsed['unit']
            numeric_length=len(parsed['numeric_reading'])
            fraction=numeric_length/max(1,len(re.sub(r'\s+','',anchor['text'])))
            x1=max(0,int(b['x1']-2));x2=min(image.shape[1],int(b['x1']+(b['x2']-b['x1'])*fraction+2))
        else:
            x1=max(0,int(b['x1']-min(160,max(50,8*height))));x2=max(0,int(b['x1']-1))
        y1=max(0,int(b['y1']-height*.15));y2=min(image.shape[0],int(b['y2']+height*.15))
        if x2-x1<5 or y2-y1<5:continue
        # A nearby reference/model is not an unlabelled numeric value.
        if not parsed and any(t is not anchor and t['bbox']['x1']>=x1 and t['bbox']['x2']<=x2 and
                              t['bbox']['y1']<y2 and t['bbox']['y2']>y1 and
                              (_REFERENCE_GENERIC.fullmatch(_texto_chave(t['text'])) or model_ocr_reading(t['text'])['classes']) for t in tokens):continue
        crop=image[y1:y2,x1:x2];gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
        enlarged=cv2.resize(gray,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        try:
            results=_executar_tesseract_tsv(enlarged,command,'unit_numeric_focus','normal',3,enlarged.shape[1],enlarged.shape[0],digits_only=True,timeout=max(.1,min(6,deadline-time.monotonic())))
        except (RuntimeError,subprocess.SubprocessError):continue
        results=_deslocar_tokens(results,x1,y1,'unit_crop')
        good=[t for t in results if float(t.get('confidence',0))>=.65 and re.fullmatch(r'[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)',t['text'])]
        observations.extend(results)
        if len(good)!=1:continue
        number=good[0];prefix=parsed['prefix'] if parsed else unit_suffix(anchor['text'])[1]
        new_text=number['text']+prefix+SYMBOLS[dimension];reading=parse_measurement(new_text)
        if not reading:continue
        if parsed:
            anchor['numeric_focus_verified']=reading['normalized']==parsed['normalized']
            if reading['normalized']!=parsed['normalized']:
                anchor['numeric_alternatives']=[parsed['text'],reading['text']]
            continue
        candidate=dict(number,text=reading['text'],raw_text=number.get('raw_text',number['text'])+' '+anchor.get('raw_text',anchor['text']),
                       engine='unit_focus_tesseract',confidence=min(number['confidence'],anchor['confidence']),
                       bbox={key:func([number['bbox'][key],b[key]]) for key,func in [('x1',min),('y1',min),('x2',max),('y2',max)]},
                       source_detection_id=anchor.get('source_detection_id'),numeric_focus_verified=True,
                       joined_parts=[{'text':anchor['text'],'bbox':dict(b),'engine':anchor.get('engine'),'variant':anchor.get('variant')}])
        added.append(candidate)
    return added,observations


def _uncertain_ocr_detections(tokens,deteccoes):
    if not deteccoes:return []
    joined=_without_joined_parts(tokens+_combinar_tokens_fragmentados(tokens))
    preview=criar_componentes_logicos(joined,deteccoes)
    unresolved=set()
    for c in preview['components']:
        ref=c.get('reference',{});value=c.get('value',{});model=c.get('model',{})
        low_ref=ref.get('source')!='ocr' or float(ref.get('confidence') or 0)<.72
        low_value=c['type'] in VALUE_REQUIRED and (not value.get('text') or float(value.get('confidence') or 0)<.72 or value.get('needs_review'))
        low_model=c['type'] in {'transistor_npn','transistor_pnp','mosfet_n','mosfet_p','opamp','inductor','diode'} and (not model.get('text') or float(model.get('confidence') or 0)<.72 or model.get('needs_review'))
        if low_ref or low_value or low_model:unresolved.add(c['detection_id'])
    return [d for d in deteccoes if d['detection_id'] in unresolved]


def _reread_uncertain_regions(image,command,deteccoes,warnings):
    tokens=[];rects=[];height,width=image.shape[:2]
    for d in deteccoes:
        b=d['bbox'];margin=min(180,max(55,max(b['x2']-b['x1'],b['y2']-b['y1'])*1.1))
        rect=(max(0,int(b['x1']-margin)),max(0,int(b['y1']-margin)),min(width,int(b['x2']+margin)),min(height,int(b['y2']+margin)))
        # Overlapping doubts share one read, with exact coordinate restoration.
        for i,(x1,y1,x2,y2) in enumerate(rects):
            if min(x2,rect[2])>max(x1,rect[0]) and min(y2,rect[3])>max(y1,rect[1]):
                rects[i]=(min(x1,rect[0]),min(y1,rect[1]),max(x2,rect[2]),max(y2,rect[3]));break
        else:rects.append(rect)
    for x1,y1,x2,y2 in rects:
        crop=image[y1:y2,x1:x2]
        if not crop.size:continue
        variants,scale=_preparar_variantes(crop);base=variants[0][1]
        try:readings=_executar_tesseract_tsv(base,command,'targeted_context','normal',scale,base.shape[1],base.shape[0])
        except (RuntimeError,subprocess.TimeoutExpired) as error:
            warnings.append('Releitura local não concluída: '+str(error));continue
        for t in readings:
            b=t['bbox'];t['bbox']=dict(b,x1=b['x1']+x1,x2=b['x2']+x1,y1=b['y1']+y1,y2=b['y2']+y1)
        tokens.extend(readings)
    return tokens


def _line_fields(text):
    from backend.reference_labels import reference_classes
    match=re.fullmatch(r'\s*((?:LED|BAT|BT|CP|VS|IC|[RCLDQTMUV])\s*\d{1,5})(?:\s*[:=]\s*|\s+)(.+)',text,re.I)
    if match and reference_classes(match[1]) and (parse_measurement(match[2]) or model_ocr_reading(match[2])['classes']):
        return [re.sub(r'\s+','',match[1]),match[2]]
    return []


def _semantic_tokens(tokens):
    output=[]
    for token in tokens:
        # Lower-case n belongs to the photographed model inscription here;
        # 1n4007 is a diode code, never a fractional resistor value.
        if re.fullmatch(r'1[nN]\d{3,5}[A-Za-z]?',token.get('text','')):
            token=dict(token,text=token['text'].upper(),raw_text=token.get('raw_text',token['text']))
        fields=_line_fields(token.get('text',''))
        if not fields:output.append(token);continue
        for index,text in enumerate(fields):
            output.append(dict(token,text=text,token_id=str(token.get('token_id',id(token)))+':'+str(index),
                               parent_token_id=token.get('token_id') or str(id(token)),semantic_field=True))
    return output


def _consolidate_text_boxes(tokens):
    """One photographed text span, with competing readings kept as evidence."""
    from backend.word_extent import complete_span_priorities
    complete=complete_span_priorities(tokens)
    def rank(token):
        text=token['text'];measurement=parse_measurement(text)
        prefixed=parse_measurement(text,'resistor')
        try:reading=model_ocr_reading(text)
        except ValueError:reading=dict(classes=set(),needs_review=True)
        # A prefix such as k/M is observed electrical writing as well. A weak
        # omega hypothesis must not erase a stronger complete '100k' reading.
        # Complete models/phrases still outrank a cropped reference fragment.
        quality=4 if _line_fields(text) else 3 if (measurement and measurement['unit_explicit'] or prefixed and prefixed.get('prefix') or reading['classes'] and not reading['needs_review']) else 2 if _REFERENCE_GENERIC.fullmatch(_texto_chave(text)) else 1
        return quality,complete.get(id(token),0),float(token.get('confidence',0))
    result=[]
    for token in sorted(tokens,key=rank,reverse=True):
        b=token['bbox'];area=max(1,(b['x2']-b['x1'])*(b['y2']-b['y1']))
        for saved in result:
            a=saved['bbox'];other=max(1,(a['x2']-a['x1'])*(a['y2']-a['y1']))
            intersection=max(0,min(a['x2'],b['x2'])-max(a['x1'],b['x1']))*max(0,min(a['y2'],b['y2'])-max(a['y1'],b['y1']))
            coverage=intersection/min(area,other);iou=intersection/max(1,area+other-intersection)
            if iou<.45 and coverage<.88:continue
            # Separate lines, e.g. model and beta, may share broad engine
            # boxes. A large vertical center shift keeps them independent.
            if abs((a['y1']+a['y2'])-(b['y1']+b['y2']))/2>.45*min(a['y2']-a['y1'],b['y2']-b['y1']):continue
            saved.setdefault('spatial_alternatives',[]).append(dict(text=token['text'],raw_text=token.get('raw_text',token['text']),
                confidence=token.get('confidence',0),engine=token.get('engine'),bbox=dict(b)))
            saved['evidence_count']=int(saved.get('evidence_count',1))+int(token.get('evidence_count',1))
            # Keep one visual box without hiding conflicting electrical values.
            # The existing review screen consumes numeric_alternatives.
            saved_fields=_line_fields(saved['text']);other_fields=_line_fields(token['text'])
            primary=parse_measurement(saved_fields[-1] if saved_fields else saved['text'])
            alternative=parse_measurement(other_fields[-1] if other_fields else token['text'])
            if (primary and alternative and primary['unit_explicit'] and alternative['unit_explicit']
                    and (primary['normalized'],primary['unit'])!=(alternative['normalized'],alternative['unit'])):
                alternatives=saved.setdefault('numeric_alternatives',[])
                for value in (primary['text'],alternative['text']):
                    if value not in alternatives:alternatives.append(value)
            break
        else:result.append(copy.deepcopy(token))
    return result


def _executar_easyocr_linhas(page,regions,deteccoes=(),soft_page=None):
    from backend.ocr_regions import line_crop
    from backend.ocr_context import repair_gain_label,region_context,equals_number_box,retry_allowlist,gain_context
    reader=_carregar_easyocr();tokens=[];uncertain=[]
    def append(raw,confidence,box,index,angle,variant):
        text=_limpar_texto(str(raw))
        if not text:return None
        token=dict(text=text,raw_text=str(raw),confidence=round(float(confidence),4),bbox=dict(box),
                   engine='easyocr',variant=variant,orientation='normal',deskew_degrees=round(angle,2),line_region_id=index,
                   component_context=box.get('component_context') or region_context(box,deteccoes))
        token=repair_gain_label(token,deteccoes);tokens.append(token)
        return token
    for index,box in enumerate(regions):
        crop,angle=line_crop(soft_page if soft_page is not None else page,box)
        readings=reader.recognize(crop,decoder='greedy',detail=1,paragraph=False,batch_size=1,workers=0)
        good=False;gain_candidate=False
        for _,raw,confidence in readings:
            t=append(raw,confidence,box,index,angle,'ink_text_line')
            if t is None:continue
            parsed=parse_measurement(t['text'])
            good|=bool(_token_relevante(t) and confidence>=.60 and not (parsed and parsed['corrections']))
            gain_candidate|=bool(re.match(r'^\s*\(?[βBPpb]\s*[=:Ee]',str(raw)))
        gain_neighbor=gain_context(box,deteccoes) if gain_candidate else None
        if gain_neighbor and not gain_neighbor.get('ambiguous'):
            numeric=equals_number_box(page,box)
            if numeric:
                number_crop,_=line_crop(soft_page if soft_page is not None else page,numeric)
                for _,number,confidence in reader.recognize(number_crop,decoder='greedy',allowlist='0123456789.,',detail=1,paragraph=False,batch_size=1,workers=0):
                    if confidence<.50 or not re.fullmatch(r'\d+(?:[.,]\d+)?',str(number).strip()):continue
                    t=append('β='+str(number).strip(),confidence,box,index,angle,'gain_numeric_after_equal')
                    if t:
                        t.update(raw_text=str(readings[0][1]),raw_numeric_text=str(number),gain_equal_geometry_confirmed=True,
                                 contextual_corrections=[dict(from_text=str(readings[0][1]),to_text=t['text'],reason='bjt_label_and_visible_equal_numeric_crop')],needs_review=True)
                        good=True
        if not good:uncertain.append((index,box,angle))
    # Retry nearby text before remote fragments; use a different rendering and
    # the component's alphabet, rather than repeating identical beam readings.
    uncertain.sort(key=lambda item:((item[1].get('component_context') or {}).get('distance',99),item[0]))
    started=time.monotonic()
    for index,box,angle in uncertain[:6]:
        if time.monotonic()-started>8:break
        context=box.get('component_context') or {};allowlist=retry_allowlist(context.get('kind'))
        crop,_=line_crop(page,box)
        options=dict(decoder='beamsearch',beamWidth=5,detail=1,paragraph=False,batch_size=1,workers=0)
        if allowlist:options['allowlist']=allowlist
        for _,raw,confidence in reader.recognize(crop,**options):append(raw,confidence,box,index,angle,'ink_text_line_beam')
    return tokens


def extrair_tokens_impressos(imagem,comando,deteccoes):
    from backend.ocr_context import repair_gain_label
    from backend.terminal_identity import read_opamp_signs
    started=time.monotonic();tokens=[];raw=[];warnings=[]
    from backend.local_word_ocr import locate_words,read_words
    regions=locate_words(imagem);crops=[];boxes=[]
    for region in regions:
        box=region['bbox'];x1=max(0,int(box['x1'])-2);y1=max(0,int(box['y1'])-2)
        x2=min(imagem.shape[1],int(box['x2'])+3);y2=min(imagem.shape[0],int(box['y2'])+3)
        if x2<=x1 or y2<=y1:continue
        crops.append(imagem[y1:y2,x1:x2]);boxes.append(box)
    for row,box in zip(read_words(crops),boxes):
        if not row:continue
        text=_limpar_texto(row['text'])
        token=dict(text=text,raw_text=row['text'],confidence=row['confidence'],bbox=box,engine='local_printed_word',variant='printed_neural_word',orientation='normal')
        raw.append(copy.deepcopy(token))
        if row['confidence']<.85:continue
        from backend.printed_text import annotation_region
        if _token_relevante(token) and (text not in {'Q','T'} or row['confidence']>=.92 and annotation_region(text,box,deteccoes or [])):tokens.append(token)
    from backend.printed_text import read_annotations
    oriented=read_annotations(imagem,deteccoes or [])
    raw.extend(copy.deepcopy(oriented));tokens.extend(oriented)
    command=comando or localizar_tesseract()
    if command:
        page=cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY) if imagem.ndim==3 else imagem
        scale=min(4.,max(1.,2200/max(page.shape)))
        enlarged=cv2.resize(page,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        try:
            printed=_executar_tesseract_tsv(enlarged,command,'printed_explicit','normal',scale,enlarged.shape[1],enlarged.shape[0],timeout=10)
            raw.extend(copy.deepcopy(printed))
            tokens.extend(t for t in printed if t['confidence']>=.45 and _token_relevante(t))
        except (RuntimeError,OSError,subprocess.TimeoutExpired) as error:warnings.append('Leitura impressa não concluída: '+str(error))
    else:warnings.append('O leitor de texto impresso não está disponível nesta instalação.')
    signs=[t for d in deteccoes or [] for t in read_opamp_signs(imagem,d)]
    raw.extend(copy.deepcopy(signs));tokens.extend(signs)
    result=OCRTokenList(_consolidate_text_boxes([repair_gain_label(t,deteccoes or []) for t in tokens]))
    # Archive rejected/uncertain readings as evidence without assigning them.
    result.raw_tokens=raw;result.pending_readings=[]
    result.timings=dict(printed_seconds=round(time.monotonic()-started,4),handwritten_reference_seconds=0)
    return result,warnings


def extrair_tokens_ocr(
    imagem: np.ndarray,
    comando: str | None,
    deteccoes: list[dict] | None = None,
    detected_text_regions: list[dict] | None = None,
) -> tuple[list[dict], list[str]]:
    from backend.drawing_mode import digital
    if digital():return extrair_tokens_impressos(imagem,comando,deteccoes)
    from backend.handwriting_reference import recognize
    from backend.ocr_context import repair_gain_label
    started=time.monotonic()
    from backend.local_word_ocr import locate_words
    word_regions=locate_words(imagem)
    tokens,pending=recognize(imagem,deteccoes or [],detected_text_regions=list(detected_text_regions or [])+word_regions)
    # The reference reader scores glyphs in percent; association and other
    # OCR engines use probabilities. Convert once at the engine boundary.
    for token in tokens+pending:
        if token.get('source')=='handwriting_reference' and float(token.get('confidence',0))>1:
            token['confidence']=float(token['confidence'])/100.
    reference_seconds=time.monotonic()-started
    # Printed annotations need a print reader as well as handwriting samples.
    # Keep this optional and local: no model downloads or remote services.
    printed=[];warnings=[];command=comando or localizar_tesseract()
    if command and (len(deteccoes or [])>=3 or len(pending)>=2):
        page=cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY) if imagem.ndim==3 else imagem
        scale=min(3.,max(1.,1800/max(page.shape)))
        enlarged=cv2.resize(page,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        try:
            printed=_executar_tesseract_tsv(enlarged,command,'printed_annotation_fallback','normal',scale,enlarged.shape[1],enlarged.shape[0],timeout=5)
            printed=[t for t in printed if t['confidence']>=.60 and _token_relevante(t)]
        except (RuntimeError,OSError,subprocess.TimeoutExpired) as error:
            warnings.append('A leitura complementar de texto impresso não foi concluída: '+str(error))
    from backend.terminal_identity import read_opamp_signs
    tokens+= [t for d in deteccoes or [] for t in read_opamp_signs(imagem,d)]
    raw=copy.deepcopy(tokens+pending+printed)
    result=OCRTokenList(_consolidate_text_boxes([repair_gain_label(t,deteccoes or []) for t in tokens+printed]))
    result.raw_tokens=raw
    result.pending_readings=pending
    result.timings=dict(reference_seconds=round(reference_seconds,4),complement_seconds=round(time.monotonic()-started-reference_seconds,4))
    return result,warnings+([str(len(pending))+' escrita(s) precisam de revisão: caractere sem referência ou comparação ambígua.'] if pending else [])


_REFERENCE_CLASS_HINTS = (
    (re.compile(r'^R(?:L|S|B|E|C|IN|OUT)\d*$'),{'resistor'}),
    (re.compile(r'^C(?:IN|OUT)\d*$'),{'capacitor','capacitor_polarized'}),
    (re.compile(r'^[+-]?V(?:CC|EE|DD|SS)\d*$'),{'vcc'}),
    (re.compile(r"^R\d+$"), {"resistor"}),
    (re.compile(r"^(?:C|CP)\d+$"), {"capacitor", "capacitor_polarized"}),
    (re.compile(r"^LED\d*$"), {"led"}),
    (re.compile(r"^D\d+$"), {"diode", "led"}),
    (re.compile(r"^L\d+$"), {"inductor"}),
    (re.compile(r"^(?:V|VS)\d+$"), {"source_dc", "battery", "vcc"}),
    (re.compile(r"^(?:BAT|BT)\d*$"), {"battery"}),
    (re.compile(r"^GND\d*$"), {"gnd"}),
    (re.compile(r"^VCC\d*$"), {"vcc"}),
    (re.compile(r"^Q\d+$"), {"transistor_npn", "transistor_pnp", "mosfet_n", "mosfet_p"}),
    (re.compile(r"^M\d+$"), {"mosfet_n", "mosfet_p"}),
    (re.compile(r"^(?:U|IC)\d+$"), {"opamp"}),
)


def _classes_sugeridas_por_texto(texto: str) -> tuple[set[str], str | None]:
    chave = _texto_chave(texto)
    if model_name(texto) in CATALOG or re.fullmatch(r'1N\d{3,5}[A-Z]?',chave):return set(),None
    from backend.reference_labels import reference_classes
    classes = reference_classes(texto)
    if classes: return classes, "reference"

    from backend.drawing_mode import digital
    reading=parse_measurement(texto) if digital() else None
    if reading and reading['unit_explicit']:
        return {kind for kind in REFERENCE_PREFIXES if DIMENSIONS.get(kind)==reading['unit']},'value'

    if re.fullmatch(r"[A-Za-z]+",chave):return set(),None
    classes_valor = {
        classe
        for classe in REFERENCE_PREFIXES
        if _parece_valor(texto, classe)
    }
    return classes_valor, "value" if classes_valor else None


def _cobertura_token_na_deteccao(token_bbox: dict, det_bbox: dict) -> float:
    x1 = max(token_bbox["x1"], det_bbox["x1"])
    y1 = max(token_bbox["y1"], det_bbox["y1"])
    x2 = min(token_bbox["x2"], det_bbox["x2"])
    y2 = min(token_bbox["y2"], det_bbox["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(
        1.0,
        (token_bbox["x2"] - token_bbox["x1"])
        * (token_bbox["y2"] - token_bbox["y1"]),
    )
    return inter / area


def _cobertura_deteccao_no_token(token_bbox: dict, det_bbox: dict) -> float:
    x1 = max(token_bbox["x1"], det_bbox["x1"])
    y1 = max(token_bbox["y1"], det_bbox["y1"])
    x2 = min(token_bbox["x2"], det_bbox["x2"])
    y2 = min(token_bbox["y2"], det_bbox["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(
        1.0,
        (det_bbox["x2"] - det_bbox["x1"])
        * (det_bbox["y2"] - det_bbox["y1"]),
    )
    return inter / area


def _verified_complete_word(token):
    confidence=float(token.get('confidence',0))
    return bool(token.get('word_evidence')) and (
        token.get('independent_readers',0)>=2 and confidence>=.98 or
        confidence>=.99 and token.get('min_character_confidence',0)>=.99)


def _leads_beyond_word(image,detection,token):
    """Count only the suspected body's ink, excluding a nearby text row."""
    from backend.resistor_geometry import symbol_mask
    from backend.vcc_topology import symbol_ink_labels,external_leads
    mask=symbol_mask(image)
    _,labels=cv2.connectedComponents(mask,connectivity=8)
    owned=symbol_ink_labels(labels,[detection])
    own_mask=np.isin(labels,list(owned)).astype(np.uint8)*255
    return external_leads(image,token['bbox'],own_mask)


def remover_falsos_positivos_textuais(
    deteccoes: list[dict],
    tokens: list[dict],
    imagem: np.ndarray | None = None,
) -> tuple[list[dict], list[dict]]:
    """Remove caixas do YOLO que coincidem quase inteiramente com escrita."""
    mantidas = []
    removidas = []
    for deteccao in deteccoes:
        if deteccao.get("manual_reviewed"):
            mantidas.append(deteccao)
            continue
        falso_token = None
        motivo = None
        for token in tokens:
            chave = token.get("text", "").upper().replace(",", ".")
            minimo = 0.35 if token.get("engine") == "easyocr" else 0.55
            if float(token.get("confidence", 0.0)) < minimo:
                continue
            cobertura_deteccao = _cobertura_deteccao_no_token(
                token["bbox"],
                deteccao["bbox"],
            )
            cobertura_token = _cobertura_token_na_deteccao(
                token["bbox"],
                deteccao["bbox"],
            )
            eh_zero_de_tensao = (
                deteccao.get("class") == "vcc"
                and re.fullmatch(r"\d+(?:\.\d+)?V", chave)
                and cobertura_deteccao >= 0.62
            )
            texto_confirmado = (
                bool(re.search(r"[A-Z0-9]", chave))
                and len(_texto_chave(chave)) >= 1
                and cobertura_deteccao >= 0.65
                and (cobertura_token >= 0.45 or
                     cobertura_deteccao>=.90 and cobertura_token>=.15
                     and _verified_complete_word(token))
                and (
                    int(token.get("evidence_count", 1)) >= 2
                    or float(token.get("confidence", 0.0)) >= (
                        0.48 if token.get("engine") == "easyocr" else 0.65
                    )
                    or token.get("library_matched")
                )
            )
            if eh_zero_de_tensao or texto_confirmado:
                if imagem is not None:
                    from backend.recognition import _validar_classe_geometrica
                    from backend.vcc_topology import external_leads
                    validation = _validar_classe_geometrica(imagem, deteccao['bbox'], deteccao['class'])
                    minimum = 1 if deteccao['class'] in {'vcc','gnd'} else 2
                    # OCR can read a real symbol as A, O, 0 or H. Preserve a
                    # physically validated body with real connected leads.
                    if validation and validation[0]:
                        evidence = validation[2]
                        classes,tipo=_classes_sugeridas_por_texto(token['text'])
                        full_word=_verified_complete_word(token)
                        word_without_leads=(deteccao['class'] in {'capacitor','capacitor_polarized'}
                            and full_word and len(_texto_chave(token['text']))>=2
                            # Letter strokes can leave the small false frame.
                            # Real device leads must continue beyond the
                            # complete independently read word as well.
                            and _leads_beyond_word(imagem,deteccao,token)<minimum)
                        intrinsic_leads = bool(evidence.get('ring_with_two_connected_leads')
                                               or evidence.get('connected_curve_with_two_leads')
                                               or evidence.get('opposed_leads')
                                               or evidence.get('capacitor_body_with_opposed_leads'))
                        if not word_without_leads and (intrinsic_leads or external_leads(imagem,deteccao['bbox']) >= minimum):
                            continue
                falso_token = token
                motivo = (
                    "vcc_box_inside_voltage_text"
                    if eh_zero_de_tensao
                    else "component_box_matches_ocr_text"
                )
                break
        if falso_token is None:
            mantidas.append(deteccao)
            continue
        removida = dict(deteccao)
        removida["removed_as_text_false_positive"] = True
        removida["text_false_positive_reason"] = motivo
        removida["conflicting_text_token"] = {
            "token_id": falso_token.get("token_id"),
            "text": falso_token.get("text"),
            "confidence": falso_token.get("confidence"),
        }
        removidas.append(removida)
    return mantidas, removidas


def _completar_bbox(bbox: dict) -> dict:
    x1 = float(bbox["x1"])
    y1 = float(bbox["y1"])
    x2 = float(bbox["x2"])
    y2 = float(bbox["y2"])
    return {
        "x1": round(x1, 2),
        "y1": round(y1, 2),
        "x2": round(x2, 2),
        "y2": round(y2, 2),
        "width": round(max(0.0, x2 - x1), 2),
        "height": round(max(0.0, y2 - y1), 2),
        "center_x": round((x1 + x2) / 2.0, 2),
        "center_y": round((y1 + y2) / 2.0, 2),
    }


def ajustar_caixas_com_texto(
    deteccoes: list[dict],
    tokens: list[dict],
    imagem: np.ndarray | None = None,
) -> tuple[list[dict], list[dict]]:
    """Encolhe a caixa pelo lado ocupado por referência ou valor."""
    ajustadas = []
    registros = []

    for deteccao_original in deteccoes:
        deteccao = dict(deteccao_original)
        deteccao["bbox"] = _completar_bbox(deteccao_original["bbox"])
        from backend.drawing_mode import digital
        if imagem is not None and deteccao.get('class')=='resistor':
            from backend.resistor_contacts import body_for_box
            observed=body_for_box(imagem,deteccao['bbox'])
            if observed and observed['evidence'].get('opposed_leads'):
                # Whole-side clipping by a nearby word can erase the symbol's
                # short lead context even when no body corner is lost.
                ajustadas.append(deteccao)
                continue
        if imagem is not None and deteccao.get('class') in {'diode','led'}:
            from backend.diode_contacts import body_axes
            if body_axes(imagem,deteccao['bbox']):
                # A label beside a lead must not clip the lead above the
                # triangle. The observed triangle/bar already proves a part.
                ajustadas.append(deteccao)
                continue
        if imagem is not None and deteccao.get('class') in {'capacitor','capacitor_polarized'}:
            from backend.capacitor_geometry import capacitor_body_evidence
            body_ok,_=capacitor_body_evidence(imagem,deteccao['bbox'])
            if body_ok:
                # A nearby value can overlap the vertical span of the plates.
                # Cutting that span loses real symbol pixels and polarity.
                ajustadas.append(deteccao)
                continue
        if deteccao.get("manual_reviewed") or deteccao.get('class') in {'transistor_npn','transistor_pnp','mosfet_n','mosfet_p','opamp'}:
            # A label beside the base/gate can overlap the vertical range of
            # an emitter arrow or another lead. A rectangular text cut would
            # remove that valid symbol evidence far away from the label.
            ajustadas.append(deteccao)
            continue

        if imagem is not None and deteccao.get('class') in {'vcc', 'gnd'}:
            # A valid one-terminal symbol can sit beside a voltage or label.
            # Cutting its box to that label removes the circle/stem extent and
            # shifts the terminal. Keep the observed symbol's accepted extent.
            from backend.recognition import _validar_classe_geometrica
            validation = _validar_classe_geometrica(imagem, deteccao['bbox'], deteccao['class'])
            if validation and validation[0]:
                ajustadas.append(deteccao)
                continue
            if deteccao.get('class') == 'gnd':
                # Recheck the actual ink and its connection, rather than
                # trusting metadata from a previous recognition pass. Tiny
                # lower marks can fail the older three-bar validator while
                # OCR mistakes the same bar and stem for a reference.
                from backend.terminal_bar_ground import refine_learned_ground
                checked = refine_learned_ground(
                    imagem, {k: v for k, v in deteccao.items()
                             if k != 'terminal_bar_ground'}, anchors=deteccoes)
                if checked.get('terminal_bar_ground'):
                    ajustadas.append(deteccao)
                    continue

        bbox_inicial = dict(deteccao["bbox"])
        largura_minima = max(12.0, bbox_inicial["width"] * 0.25)
        altura_minima = max(12.0, bbox_inicial["height"] * 0.25)
        tokens_usados = []

        for token in sorted(
            tokens,
            key=lambda item: float(item.get("confidence", 0.0)),
            reverse=True,
        ):
            # A single undimensioned stroke is often a lead/body read as "1".
            # It can remain an OCR candidate, but cannot cut a component box.
            if re.fullmatch(r"[0-9+\-=−]", str(token.get("text", "")).strip()):
                continue
            minimo = 0.35 if token.get("engine") == "easyocr" else 0.55
            if float(token.get("confidence", 0.0)) < minimo:
                continue
            bbox = deteccao["bbox"]
            if _cobertura_token_na_deteccao(token["bbox"], bbox) < 0.15:
                continue

            tb = token["bbox"]
            centro_x = (bbox["x1"] + bbox["x2"]) / 2.0
            centro_y = (bbox["y1"] + bbox["y2"]) / 2.0
            token_x = (tb["x1"] + tb["x2"]) / 2.0
            token_y = (tb["y1"] + tb["y2"]) / 2.0
            largura = max(1.0, bbox["x2"] - bbox["x1"])
            altura = max(1.0, bbox["y2"] - bbox["y1"])
            deslocamento_x = (token_x - centro_x) / (largura / 2.0)
            deslocamento_y = (token_y - centro_y) / (altura / 2.0)
            if max(abs(deslocamento_x), abs(deslocamento_y)) < 0.25:
                continue

            novo = dict(bbox)
            lado = None
            folga = 2.0
            if abs(deslocamento_x) >= abs(deslocamento_y):
                if deslocamento_x < 0:
                    candidato = max(bbox["x1"], tb["x2"] + folga)
                    corte = candidato - bbox_inicial["x1"]
                    if (
                        bbox["x2"] - candidato >= largura_minima
                        and corte <= bbox_inicial["width"] * 0.60
                    ):
                        novo["x1"] = candidato
                        lado = "left"
                else:
                    candidato = min(bbox["x2"], tb["x1"] - folga)
                    corte = bbox_inicial["x2"] - candidato
                    if (
                        candidato - bbox["x1"] >= largura_minima
                        and corte <= bbox_inicial["width"] * 0.60
                    ):
                        novo["x2"] = candidato
                        lado = "right"
            else:
                if deslocamento_y < 0:
                    candidato = max(bbox["y1"], tb["y2"] + folga)
                    corte = candidato - bbox_inicial["y1"]
                    if (
                        bbox["y2"] - candidato >= altura_minima
                        and corte <= bbox_inicial["height"] * 0.60
                    ):
                        novo["y1"] = candidato
                        lado = "top"
                else:
                    candidato = min(bbox["y2"], tb["y1"] - folga)
                    corte = bbox_inicial["y2"] - candidato
                    if (
                        candidato - bbox["y1"] >= altura_minima
                        and corte <= bbox_inicial["height"] * 0.60
                    ):
                        novo["y2"] = candidato
                        lado = "bottom"

            if lado:
                deteccao["bbox"] = _completar_bbox(novo)
                tokens_usados.append(
                    {
                        "token_id": token.get("token_id"),
                        "text": token["text"],
                        "side": lado,
                    }
                )

        if tokens_usados:
            deteccao["bbox_before_text_adjustment"] = bbox_inicial
            deteccao["bbox_adjusted_for_text"] = True
            deteccao["bbox_adjustment_tokens"] = tokens_usados
            registros.append(
                {
                    "detection_id": deteccao["detection_id"],
                    "before": bbox_inicial,
                    "after": deteccao["bbox"],
                    "tokens": tokens_usados,
                }
            )
        ajustadas.append(deteccao)

    return ajustadas, registros


def filtrar_deteccoes_em_texto(
    deteccoes: list[dict],
    tokens: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Sinaliza sobreposição de texto sem apagar previsões do YOLO."""
    mantidas = []
    removidas = []
    for deteccao in deteccoes:
        if deteccao.get("manual_reviewed"):
            mantidas.append(deteccao)
            continue
        conflitos = []
        tokens_ajuste = {
            item.get("token_id")
            for item in deteccao.get("bbox_adjustment_tokens", [])
        }
        for token in tokens:
            bbox_conflito = deteccao.get(
                "bbox_before_text_adjustment",
                deteccao["bbox"],
            )
            if _cobertura_token_na_deteccao(token["bbox"], bbox_conflito) < 0.72:
                continue
            classes, tipo = _classes_sugeridas_por_texto(token["text"])
            if not classes or deteccao["class"] in classes:
                continue
            minimo = 0.12 if token.get("engine") == "easyocr" else 0.28
            if token["confidence"] < minimo:
                continue
            if tipo == "value" and len(classes) > 2:
                continue
            if token.get("token_id") in tokens_ajuste:
                continue
            conflitos.append(
                {
                    "token_id": token.get("token_id"),
                    "text": token["text"],
                    "confidence": token["confidence"],
                    "suggested_classes": sorted(classes),
                }
            )

        if conflitos:
            sinalizada = dict(deteccao)
            sinalizada["ocr_text_overlap_suspected"] = True
            sinalizada["conflicting_text_tokens"] = conflitos
            motivos = list(sinalizada.get("review_reasons", []))
            if "ocr_text_overlap" not in motivos:
                motivos.append("ocr_text_overlap")
            sinalizada["review_reasons"] = motivos
            sinalizada["review_required"] = True
            mantidas.append(sinalizada)
            removidas.append(sinalizada)
        else:
            mantidas.append(deteccao)
    return mantidas, removidas


def _distancia_entre_bboxes(a: dict, b: dict) -> float:
    dx = max(a["x1"] - b["x2"], b["x1"] - a["x2"], 0.0)
    dy = max(a["y1"] - b["y2"], b["y1"] - a["y2"], 0.0)
    return math.hypot(dx, dy)


def _distancia_token_componente(token: dict, deteccao: dict) -> float | None:
    from backend.reference_labels import local_distance
    return local_distance(token, deteccao)


def _detectar_tokens_por_amostras(
    imagem: np.ndarray,
    deteccoes: list[dict],
) -> list[dict]:
    """Localiza exemplos confirmados da escrita nas regiões dos componentes.

    A busca é restrita à vizinhança de componentes da classe correspondente e
    usa poucas escalas. Isso dá utilidade imediata às correções do usuário sem
    fazer uma varredura cara ou transformar um texto distante em componente.
    """
    indice = _carregar_indice_amostras()
    if not indice or not deteccoes:
        return []
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    altura_imagem, largura_imagem = cinza.shape[:2]
    encontrados = []

    for amostra in indice[-500:]:
        arquivo = str(amostra.get("template_file", ""))
        rotulo = _limpar_texto(str(amostra.get("label", "")))
        classe = str(amostra.get("component_type", ""))
        if not arquivo or not rotulo or not classe:
            continue
        caminho = OCR_SAMPLES_DIR / arquivo
        modelo_original = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
        if modelo_original is None or modelo_original.size == 0:
            continue

        melhor = None
        for deteccao in deteccoes:
            if deteccao.get("class") != classe:
                continue
            bbox = deteccao["bbox"]
            largura = max(1.0, bbox["x2"] - bbox["x1"])
            altura = max(1.0, bbox["y2"] - bbox["y1"])
            margem = min(230.0, max(85.0, max(largura, altura) * 2.0))
            roi_x1 = max(0, int(math.floor(bbox["x1"] - margem)))
            roi_y1 = max(0, int(math.floor(bbox["y1"] - margem)))
            roi_x2 = min(largura_imagem, int(math.ceil(bbox["x2"] + margem)))
            roi_y2 = min(altura_imagem, int(math.ceil(bbox["y2"] + margem)))
            roi = cinza[roi_y1:roi_y2, roi_x1:roi_x2]

            for escala in (0.90, 1.0, 1.10):
                modelo = cv2.resize(
                    modelo_original,
                    None,
                    fx=escala,
                    fy=escala,
                    interpolation=(
                        cv2.INTER_AREA if escala < 1.0 else cv2.INTER_CUBIC
                    ),
                )
                mh, mw = modelo.shape[:2]
                if mw < 5 or mh < 5 or mw > roi.shape[1] or mh > roi.shape[0]:
                    continue
                correlacao = cv2.matchTemplate(roi, modelo, cv2.TM_CCOEFF_NORMED)
                _, pontuacao, _, local = cv2.minMaxLoc(correlacao)
                candidato = (
                    float(pontuacao),
                    roi_x1 + int(local[0]),
                    roi_y1 + int(local[1]),
                    mw,
                    mh,
                    deteccao["detection_id"],
                )
                if melhor is None or candidato[0] > melhor[0]:
                    melhor = candidato

        if melhor is None or melhor[0] < 0.72:
            continue
        pontuacao, x, y, largura, altura, detection_id = melhor
        encontrados.append(
            {
                "text": rotulo,
                "confidence": round(min(0.99, pontuacao), 4),
                "bbox": {
                    "x1": round(float(x), 2),
                    "y1": round(float(y), 2),
                    "x2": round(float(x + largura), 2),
                    "y2": round(float(y + altura), 2),
                },
                "variant": "confirmed_handwriting_template",
                "orientation": "normal",
                "engine": "template",
                "source_detection_id": detection_id,
                "source_component_class": classe,
                "library_matched": True,
                "template_similarity": round(pontuacao, 4),
            }
        )
    return encontrados


def _aplicar_biblioteca_ocr(
    imagem: np.ndarray,
    tokens: list[dict],
    deteccoes: list[dict],
) -> list[dict]:
    """Corrige leituras usando amostras confirmadas da escrita do usuário."""
    indice = _carregar_indice_amostras()
    if not indice or not tokens:
        return tokens

    amostras = []
    for item in indice[-500:]:
        caminho = OCR_SAMPLES_DIR / str(item.get("file", ""))
        molde = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
        if molde is None or molde.shape != (OCR_SAMPLE_HEIGHT, OCR_SAMPLE_WIDTH):
            continue
        rotulo = _limpar_texto(str(item.get("label", "")))
        if not rotulo:
            continue
        amostras.append((item, molde, rotulo))
    if not amostras:
        return tokens

    saida = []
    for token_original in tokens:
        token = dict(token_original)
        recorte = _recortar_token(imagem, token.get("bbox", {}))
        normalizada = _normalizar_amostra_ocr(recorte)
        if normalizada is None:
            saida.append(token)
            continue

        _, tipo_atual = _classes_sugeridas_por_texto(token.get("text", ""))
        if model_classes(token.get('text','')):
            tipo_atual = 'model'
        melhor_por_rotulo: dict[str, tuple[float, dict]] = {}
        for item, molde, rotulo in amostras:
            tipo_amostra = item.get("kind")
            if tipo_atual and tipo_amostra != tipo_atual:
                continue
            classe = item.get("component_type")
            compativel = any(
                det.get("class") == classe
                and _distancia_token_componente(token, det) is not None
                for det in deteccoes
            )
            if deteccoes and not compativel:
                continue
            similaridade = float(
                cv2.matchTemplate(
                    normalizada,
                    molde,
                    cv2.TM_CCOEFF_NORMED,
                )[0, 0]
            )
            anterior = melhor_por_rotulo.get(rotulo)
            if anterior is None or similaridade > anterior[0]:
                melhor_por_rotulo[rotulo] = (similaridade, item)

        ordenadas = sorted(
            (
                (similaridade, rotulo, item)
                for rotulo, (similaridade, item) in melhor_por_rotulo.items()
            ),
            reverse=True,
        )
        if not ordenadas:
            saida.append(token)
            continue
        melhor, rotulo, item = ordenadas[0]
        segunda = ordenadas[1][0] if len(ordenadas) > 1 else -1.0
        chave_atual = _chave_leitura(token.get("text", ""))
        chave_nova = _chave_leitura(rotulo)
        limite = 0.74 if chave_atual == chave_nova else 0.83
        margem = 0.0 if chave_atual == chave_nova else 0.04
        if melhor >= limite and melhor - segunda >= margem:
            current=parse_measurement(token.get('text',''));candidate=parse_measurement(rotulo)
            if (current and candidate and current['unit_explicit'] and float(token.get('confidence',0))>=.60
                    and (current['normalized'],current['unit'])!=(candidate['normalized'],candidate['unit'])):
                token['numeric_alternatives']=list(dict.fromkeys(token.get('numeric_alternatives',[])+[current['text'],candidate['text']]))
                token['library_candidate']=dict(text=rotulo,similarity=round(melhor,4))
                saida.append(token)
                continue
            token["original_text"] = token.get("text", "")
            token["text"] = rotulo
            token["confidence"] = round(
                max(float(token.get("confidence", 0.0)), melhor),
                4,
            )
            token["library_matched"] = True
            token["template_similarity"] = round(melhor, 4)
            token["template_component_type"] = item.get("component_type")
        saida.append(token)
    return saida


def _chave_token(token: dict) -> str:
    return str(token.get("token_id") or id(token))


def _atribuir_tokens(tokens: list[dict], deteccoes: list[dict]) -> tuple[dict, list[dict]]:
    """Relaciona texto à caixa compatível mais próxima, sem misturar classes."""
    atribuicoes = {det["detection_id"]: [] for det in deteccoes}
    informacoes = []
    for token in tokens:
        classes, tipo = _classes_sugeridas_por_texto(token["text"])
        informacoes.append((token, classes, tipo))

    tokens_usados: set[str] = set()
    deteccoes_com_referencia: set[str] = set()
    referencia_por_deteccao: dict[str, dict] = {}
    pares_referencia = []

    # A second OCR view may read the numeric part of 100 mH as "L00".
    # Explicit units explain that same ink; retain the raw token for review,
    # but do not assign its alternate reading as a component name.
    measurement_regions=[]
    for other,_,kind in informacoes:
        if kind!='value' or float(other.get('confidence',0))<.55:continue
        parsed=parse_measurement(other['text'])
        if not parsed or not parsed['unit_explicit']:continue
        symbol=SYMBOLS[parsed['unit']]
        numeric=parse_measurement(parsed['numeric_reading']+symbol)
        if numeric:measurement_regions.append((other['bbox'],symbol,numeric['normalized']))

    for token, classes, tipo in informacoes:
        if tipo != "reference" or not classes or token.get('inferred_prefix'):
            continue
        from backend.drawing_mode import digital
        if digital() and float(token.get('confidence',0))<.60:continue
        explained_number=False
        for box,symbol,number in measurement_regions:
            if _cobertura_token_na_deteccao(token['bbox'],box)<.8:continue
            candidate=parse_measurement(token.get('raw_text',token['text'])+symbol)
            if candidate and candidate['normalized']==number:
                explained_number=True;break
        if explained_number:continue
        for deteccao in deteccoes:
            if deteccao["class"] not in classes:
                continue
            distancia = _distancia_token_componente(token, deteccao)
            if distancia is None:
                continue
            pares_referencia.append(
                (
                    distancia,
                    -float(token.get("confidence", 0.0)),
                    _chave_token(token),
                    deteccao["detection_id"],
                    token,
                )
            )

    from backend.reference_labels import annotation_group_owner
    from backend.text_assignment import globally_assign
    for distancia, _, token_id, detection_id, token in globally_assign(pares_referencia):
        if token_id in tokens_usados or detection_id in deteccoes_com_referencia:
            continue
        atribuido = dict(token)
        atribuido["distance_score"] = round(distancia, 4)
        atribuido["association_reason"] = "global_one_to_one_compatible_reference"
        atribuicoes[detection_id].append(atribuido)
        referencia_por_deteccao[detection_id] = atribuido
        tokens_usados.add(token_id)
        deteccoes_com_referencia.add(detection_id)

    deteccoes_com_valor: set[str] = set()
    pares_valor = []
    for token, classes, tipo in informacoes:
        token_id = _chave_token(token)
        if tipo != "value" or not classes or token_id in tokens_usados:
            continue
        from backend.drawing_mode import CURRENT
        compatible_detections=[d for d in deteccoes if d['class'] in classes]
        distances={d['detection_id']:_distancia_token_componente(token,d) for d in compatible_detections}
        nearest=min((v for v in distances.values() if v is not None),default=math.inf)
        def supported_owner(owner):
            # A neighbouring reference must not pull a word away from the
            # physical component immediately beside it, especially on a
            # rotated page with tightly spaced annotation groups.
            if owner and distances.get(owner) is not None and distances[owner]>nearest+max(12.,min(token['bbox']['x2']-token['bbox']['x1'],token['bbox']['y2']-token['bbox']['y1'])*.35):return None
            return owner
        written_owner=supported_owner(annotation_group_owner(token,referencia_por_deteccao)) if CURRENT.get()=='handwritten' and float(token.get('confidence',0))<.90 else None
        for deteccao in deteccoes:
            detection_id = deteccao["detection_id"]
            if written_owner and detection_id!=written_owner and float(referencia_por_deteccao[written_owner].get('confidence',0))>=.90:continue
            if deteccao["class"] in NO_VALUE_CLASSES:
                continue
            if deteccao["class"] not in classes:
                continue
            if not _valor_com_confianca(token,deteccao['class']):
                continue
            distancia = _distancia_token_componente(token, deteccao)
            if distancia is None:
                continue
            compatible_refs={cid:ref for cid,ref in referencia_por_deteccao.items()
                if next(d for d in deteccoes if d['detection_id']==cid)['class'] in classes}
            owner=supported_owner(annotation_group_owner(token,compatible_refs))
            if owner is not None and owner != detection_id:continue
            referencia = referencia_por_deteccao.get(detection_id)
            if referencia and distancia<=nearest+12:
                distancia_referencia = _distancia_entre_bboxes(
                    token["bbox"],
                    referencia["bbox"],
                )
                distancia = min(distancia, distancia_referencia * 0.55)
            parsed=parse_measurement(token['text'],deteccao['class'])
            priority=0 if parsed and parsed['unit_explicit'] else 1 if parsed and parsed['prefix'] else 2
            pares_valor.append(
                (
                    priority,
                    distancia,
                    -float(token.get("confidence", 0.0)),
                    token_id,
                    detection_id,
                    token,
                )
            )

    for priority, distancia, _, token_id, detection_id, token in globally_assign(pares_valor,value_pairs=True):
        if token_id in tokens_usados or detection_id in deteccoes_com_valor:
            continue
        atribuido = dict(token)
        atribuido["distance_score"] = round(distancia, 4)
        atribuido["association_reason"] = "nearest_compatible_component"
        atribuido['association_reason']='global_unit_first_one_to_one' if priority==0 else 'global_one_to_one_compatible_value'
        atribuicoes[detection_id].append(atribuido)
        tokens_usados.add(token_id)
        deteccoes_com_valor.add(detection_id)

    nao_atribuidos = [
        token for token in tokens if _chave_token(token) not in tokens_usados
    ]
    return atribuicoes, nao_atribuidos


def criar_componentes_logicos(tokens: list[dict], deteccoes: list[dict]) -> dict:
    from backend.annotation_semantics import recover_reference_readings
    tokens=recover_reference_readings(tokens,deteccoes)
    tokens=_semantic_tokens(tokens)
    tokens=_without_joined_parts(tokens+_combinar_tokens_fragmentados(tokens))
    atribuicoes, nao_atribuidos = _atribuir_tokens(tokens, deteccoes)
    referencias_reservadas = {
        _texto_chave(token["text"])
        for deteccao in deteccoes
        for token in atribuicoes.get(deteccao["detection_id"], [])
        if _referencia_compativel(token["text"], deteccao["class"])
    }
    ordenadas = sorted(
        deteccoes,
        key=lambda det: (
            det["bbox"].get("center_y", det["bbox"]["y1"]),
            det["bbox"].get("center_x", det["bbox"]["x1"]),
        ),
    )
    contadores: dict[str, int] = {}
    usados: set[str] = set()
    componentes = []

    for det in ordenadas:
        classe = det["class"]
        proximos = atribuicoes.get(det["detection_id"], [])
        referencias = [t for t in proximos if _referencia_compativel(t["text"], classe)]
        valores = (
            []
            if classe in NO_VALUE_CLASSES
            else [t for t in proximos if _valor_com_confianca(t, classe)]
        )
        referencias.sort(key=lambda t: (-t["confidence"], t["distance_score"]))
        valores.sort(key=lambda t: (-t["confidence"], t["distance_score"]))

        referencia_token = next(
            (t for t in referencias if _texto_chave(t["text"]) not in usados),
            None,
        )
        if referencia_token:
            referencia_texto = _texto_chave(referencia_token["text"])
            referencia_origem = "ocr"
            referencia_confianca = referencia_token["confidence"]
        else:
            prefixo = FALLBACK_PREFIXES.get(classe, "X")
            contadores[prefixo] = contadores.get(prefixo, 0) + 1
            if classe in {"gnd", "vcc"}:
                referencia_texto = f"{prefixo}{contadores[prefixo]}"
            else:
                referencia_texto = f"{prefixo}_AUTO{contadores[prefixo]}"
            while referencia_texto in usados or referencia_texto in referencias_reservadas:
                contadores[prefixo] += 1
                if classe in {"gnd", "vcc"}:
                    referencia_texto = f"{prefixo}{contadores[prefixo]}"
                else:
                    referencia_texto = f"{prefixo}_AUTO{contadores[prefixo]}"
            referencia_origem = "class" if classe in {"gnd", "vcc"} else "generated"
            referencia_confianca = (
                det.get("confidence") if referencia_origem == "class" else None
            )
        usados.add(referencia_texto)

        valor_token = valores[0] if valores else None
        if classe in NO_VALUE_CLASSES:
            valor = {
                "text": "",
                "normalized": None,
                "unit": None,
                "source": "not_applicable",
                "confidence": None,
            }
        else:
            valor = _normalizar_valor(valor_token["text"], classe) if valor_token else {
                "text": "",
                "normalized": None,
                "unit": None,
            }
            valor.update(
                {
                    "source": "ocr" if valor_token else "missing",
                    "confidence": valor_token["confidence"] if valor_token else None,
                }
            )
            if valor_token:
                valor['raw_text']=valor_token.get('raw_text',valor.get('raw_text',valor_token['text']))
                valor['numeric_focus_verified']=bool(valor_token.get('numeric_focus_verified'))
                valor['alternatives']=list(valor_token.get('numeric_alternatives',valor_token.get('alternatives',[])))
                if valor_token.get('needs_review'):valor['needs_review']=True
                for other in tokens:
                    if other is valor_token or not _valor_com_confianca(other,classe):continue
                    other_value=parse_measurement(other['text'],classe)
                    if (other_value and other_value['unit_explicit'] and
                        _bbox_iou(other['bbox'],valor_token['bbox'])>=.45 and
                        other_value['normalized']!=valor.get('normalized')):
                        for alternative in (valor['text'],other_value['text']):
                            if alternative not in valor['alternatives']:valor['alternatives'].append(alternative)
                if valor['alternatives']:valor['needs_review']=True
        exige_revisao = bool(
            det.get("review_required")
            or referencia_origem == "generated"
            or (classe in VALUE_REQUIRED and not valor["text"])
        )
        motivos = []
        if det.get("review_required"):
            motivos.append("classe_visual_duvidosa")
        if referencia_origem == "generated":
            motivos.append("referencia_nao_lida")
        if referencia_token and referencia_token.get('needs_review'):
            exige_revisao=True;motivos.append('referencia_ocr_ambigua')
        if classe in VALUE_REQUIRED and not valor["text"]:
            motivos.append("valor_nao_lido")

        if valor.get('needs_review') and valor.get('text'):
            exige_revisao=True
            motivos.append('numeric_ocr_conflict' if valor.get('alternatives') else 'numeric_ocr_corrected' if valor.get('corrections') else 'unit_inferred_from_component')
        componentes.append(
            {
                "component_id": referencia_texto,
                "detection_id": det["detection_id"],
                "type": classe,
                "class_confidence": det.get("confidence"),
                "bbox": det["bbox"],
                "reference": {
                    "text": referencia_texto,
                    "source": referencia_origem,
                    "confidence": referencia_confianca,
                },
                "value": valor,
                "ocr_tokens": proximos,
                "review_required": exige_revisao,
                "review_reasons": motivos,
            }
        )

    model_token_indices = enrich_models(componentes, tokens, deteccoes, _distancia_token_componente)
    nao_atribuidos = [t for t in nao_atribuidos if not any(t is tokens[i] for i in model_token_indices)]
    return {
        "components": componentes,
        "unassigned_tokens": nao_atribuidos,
        "reference_ocr_count": sum(
            1 for c in componentes if c["reference"]["source"] == "ocr"
        ),
        "value_ocr_count": sum(1 for c in componentes if c["value"]["source"] == "ocr"),
        "model_ocr_count": sum(1 for c in componentes if c["model"]["source"] == "ocr"),
        "review_count": sum(1 for c in componentes if c["review_required"]),
    }


def _desenhar_resultado(imagem: np.ndarray, resultado: dict, output_path: Path) -> None:
    canvas = imagem.copy()
    for token in resultado["tokens"]:
        b = token["bbox"]
        p1 = (int(round(b["x1"])), int(round(b["y1"])))
        p2 = (int(round(b["x2"])), int(round(b["y2"])))
        cv2.rectangle(canvas, p1, p2, (220, 120, 0), 1)
        cv2.putText(
            canvas,
            token["text"],
            (p1[0], max(14, p1[1] - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 120, 0),
            1,
            cv2.LINE_AA,
        )

    for componente in resultado["components"]:
        b = componente["bbox"]
        cor = (0, 140, 255) if componente["review_required"] else (30, 150, 60)
        p1 = (int(round(b["x1"])), int(round(b["y1"])))
        p2 = (int(round(b["x2"])), int(round(b["y2"])))
        cv2.rectangle(canvas, p1, p2, cor, 2)
        valor = componente["value"]["text"]
        rotulo = componente["component_id"] + (f" = {valor}" if valor else "")
        modelo = (componente.get('model') or {}).get('text','')
        if modelo and modelo != valor:
            rotulo += f' [{modelo}]'
        cv2.putText(
            canvas,
            rotulo,
            (p1[0], max(18, p1[1] - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            cor,
            2,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Não foi possível salvar {output_path.name}.")


def salvar_imagem_componentes(
    image_path: Path,
    resultado: dict,
    output_path: Path,
) -> None:
    imagem = cv2.imread(str(image_path))
    if imagem is None:
        raise FileNotFoundError(f"Imagem para OCR não encontrada: {image_path.name}")
    _desenhar_resultado(imagem, resultado, output_path)


@symbol_analysis_session
def executar_ocr_componentes(
    image_path: Path,
    detections_info: dict,
    output_json_path: Path,
    output_image_path: Path,
) -> dict:
    inicio_ocr = time.monotonic()
    comando = None
    imagem = cv2.imread(str(image_path))
    if imagem is None:
        raise FileNotFoundError(f"Imagem para OCR não encontrada: {image_path.name}")
    deteccoes = detections_info.get("detections") or []
    if not deteccoes:
        raise ValueError("Execute o reconhecimento YOLO antes do OCR.")

    # This belongs to the integrated photograph/OCR stage. Do not change the
    # isolated model's predictions, thresholds, classes or geometric filters.
    body_recovered_power=[]
    from backend.drawing_mode import CURRENT
    if CURRENT.get()=='handwritten':
        from backend.verified_power_contacts import recover_power
        body_recovered_power=recover_power(imagem,deteccoes,9)
        used_ids={d['detection_id'] for d in deteccoes}
        for item in body_recovered_power:
            index=1
            while f'OCR_POWER_{index:03d}' in used_ids:index+=1
            item['detection_id']=f'OCR_POWER_{index:03d}';used_ids.add(item['detection_id'])
        deteccoes=deteccoes+body_recovered_power

    tokens, avisos = extrair_tokens_ocr(
        imagem,
        comando,
        deteccoes=deteccoes,
        detected_text_regions=detections_info.get('detected_text_regions',[]),
    )
    from backend.voltage_text_recovery import recover_voltage_ports
    recovered_power=recover_voltage_ports(imagem,deteccoes,detections_info.get('geometry_rejections',[]),tokens)
    deteccoes=deteccoes+recovered_power
    from backend.printed_power import filter_ports
    deteccoes=filter_ports(imagem,deteccoes,tokens)
    from backend.printed_symbols import refine_models_from_text
    deteccoes=refine_models_from_text(imagem,deteccoes,tokens)
    from backend.typed_symbol_recovery import reconcile
    deteccoes=reconcile(imagem,deteccoes,detections_info.get('geometry_rejections',[]),tokens)
    from backend.annotation_symbol_context import reconcile_annotation_groups
    deteccoes=reconcile_annotation_groups(imagem,deteccoes,detections_info.get('geometry_rejections',[]),tokens)
    negativos_texto_adicionados = 0
    deteccoes_sem_texto, falsos_positivos = remover_falsos_positivos_textuais(
        deteccoes,
        tokens,
        imagem,
    )
    deteccoes_ajustadas, ajustes_caixa = ajustar_caixas_com_texto(
        deteccoes_sem_texto,
        tokens,
        imagem,
    )
    from backend.resistor_contacts import extend_boxes
    deteccoes_ajustadas=extend_boxes(imagem,deteccoes_ajustadas)
    from backend.drawing_mode import CURRENT
    if CURRENT.get()=='handwritten':
        from backend.symbol_semantics import resolve_marked_families
        ids={d['class']:d['class_id'] for d in deteccoes_ajustadas}
        ids.update(capacitor=1,capacitor_polarized=2,diode=3,led=4,mosfet_n=12,mosfet_p=13)
        # A letter/value removed from a broad model box must no longer supply
        # a polarity '+' or arrow for the actual, now separated symbol.
        deteccoes_ajustadas,_=resolve_marked_families(imagem,deteccoes_ajustadas,ids)
        from backend.symbol_bounds import refine_capacitor_after_text, refine_broad_ordinary_capacitors, refine_verified_polarized_frames
        deteccoes_ajustadas=refine_capacitor_after_text(imagem,deteccoes_ajustadas)
        deteccoes_ajustadas=refine_broad_ordinary_capacitors(imagem,deteccoes_ajustadas)
        deteccoes_ajustadas=refine_verified_polarized_frames(imagem,deteccoes_ajustadas)
    deteccoes_filtradas, deteccoes_texto = filtrar_deteccoes_em_texto(
        deteccoes_ajustadas,
        tokens,
    )
    # OCR box adjustments and observed-lead expansion must preserve the same
    # no-containment invariant as the initial detector.
    from backend.detection_boxes import remove_nested_boxes, merge_same_symbol_boxes
    deteccoes_filtradas, duplicate_rejections = merge_same_symbol_boxes(deteccoes_filtradas)
    deteccoes_filtradas,nested_rejections=remove_nested_boxes(deteccoes_filtradas,renumber=False)
    from backend.annotation_symbol_context import separate_annotation_frames
    deteccoes_filtradas=separate_annotation_frames(imagem,deteccoes_filtradas,tokens)
    logicos = criar_componentes_logicos(tokens, deteccoes_filtradas)
    resultado = {
        "schema_version": "stage6-1.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_image": image_path.name,
        "engine": ocr_status(),
        "elapsed_seconds": round(time.monotonic() - inicio_ocr, 2),
        "drawing_type": __import__("backend.drawing_mode",fromlist=["CURRENT"]).CURRENT.get(),
        "performance_policy": "explicit_domain_component_neighborhoods",
        "ocr_substage_seconds": getattr(tokens,'timings',{}),
        "text_line_regions": getattr(tokens,'text_regions',[]),
        "ignored_nontext_regions": getattr(tokens,'skipped_regions',[]),
        "text_hard_negative_samples_added": negativos_texto_adicionados,
        "ocr_verified_power_ports_added": len(recovered_power)+len(body_recovered_power),
        "count": len(logicos["components"]),
        "token_count": len(tokens),
        "tokens": tokens,
        "pending_readings": getattr(tokens,"pending_readings",[]),
        "raw_tokens": getattr(tokens,'raw_tokens',None) or list(tokens),
        "raw_token_count": len(getattr(tokens,'raw_tokens',None) or tokens),
        "ocr_warnings": avisos,
        "reading_policy": "units_first_then_numbers_references_and_models_with_raw_archive",
        "bbox_adjustment_count": len(ajustes_caixa),
        "bbox_adjustments": ajustes_caixa,
        "retained_detections": deteccoes_filtradas,
        "text_false_positive_count": len(falsos_positivos),
        "text_false_positives": falsos_positivos,
        "text_overlap_review_count": len(deteccoes_texto),
        "text_overlap_reviews": deteccoes_texto,
        "nested_box_rejections_after_ocr": nested_rejections,
        "same_symbol_duplicate_rejections_after_ocr": duplicate_rejections,
        **logicos,
    }
    output_json_path.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _desenhar_resultado(imagem, resultado, output_image_path)
    return resultado


def aplicar_correcoes_manuais(
    resultado: dict,
    edicoes: list[dict],
    detections_info: dict,
) -> dict:
    resultado = copy.deepcopy(resultado)
    por_id = {item.get("detection_id"): item for item in edicoes}
    deteccoes_por_id = {
        det["detection_id"]: det for det in detections_info.get("detections", [])
    }
    ids_esperados = set(deteccoes_por_id)
    if set(por_id) != ids_esperados:
        raise ValueError("Envie uma referência e um valor para cada componente detectado.")

    usados = set()
    for componente in resultado.get("components", []):
        edicao = por_id[componente["detection_id"]]
        referencia = _texto_chave(str(edicao.get("reference", "")))
        classe = componente["type"]
        valor_texto = (
            ""
            if classe in NO_VALUE_CLASSES
            else str(edicao.get("value", "")).strip()[:32]
        )
        if not referencia or len(referencia) > 16:
            raise ValueError("Toda referência deve ter entre 1 e 16 caracteres.")
        if referencia in usados:
            raise ValueError(f"A referência {referencia} está repetida.")
        usados.add(referencia)

        if any(k in edicao for k in ("model", "electrical_parameters", "parameter_policy")):
            componente.update(model_edit(componente, edicao))

        componente["component_id"] = referencia
        componente["reference"] = {
            "text": referencia,
            "source": "manual",
            "confidence": 1.0,
        }
        if classe in NO_VALUE_CLASSES:
            componente["value"] = {
                "text": "",
                "normalized": None,
                "unit": None,
                "source": "not_applicable",
                "confidence": None,
            }
        else:
            componente["value"] = {
                **_normalizar_valor(valor_texto, classe),
                "source": "manual" if valor_texto else "missing",
                "confidence": 1.0 if valor_texto else None,
            }
        motivos = []
        if deteccoes_por_id[componente["detection_id"]].get("review_required"):
            motivos.append("classe_visual_duvidosa")
        if componente["type"] in VALUE_REQUIRED and not valor_texto:
            motivos.append("valor_nao_informado")
        componente["review_reasons"] = motivos
        componente["review_required"] = bool(motivos)

    resultado["manual_edited_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    resultado["reference_ocr_count"] = sum(
        1 for c in resultado["components"] if c["reference"]["source"] == "ocr"
    )
    resultado["value_ocr_count"] = sum(
        1 for c in resultado["components"] if c["value"]["source"] == "ocr"
    )
    resultado["model_ocr_count"] = sum(1 for c in resultado["components"] if (c.get("model") or {}).get("source") == "ocr")
    resultado["review_count"] = sum(1 for c in resultado["components"] if c["review_required"])
    return resultado

# ---------------------------------------------------------------------------
# Compatibilidade com a API lógica das Etapas 5/6 antigas.
# Os testes externos e alguns scripts do projeto ainda importam estes nomes.
# A implementação delega ao pipeline atual e converte apenas o formato de saída.
# ---------------------------------------------------------------------------
def classificar_token(texto: str) -> str | None:
    _classes, tipo = _classes_sugeridas_por_texto(str(texto))
    return tipo if tipo in {"reference", "value"} else None


def montar_componentes_logicos(deteccoes: list[dict], tokens: list[dict]) -> list[dict]:
    """Compatibilidade com a API legada usada pelos scripts/testes antigos.

    O backend atual armazena a geometria de cada token em ``token["bbox"]``.
    Nas etapas antigas ela vinha achatada (x1/y1/x2/y2/center_x/center_y).
    Aceitamos os dois formatos aqui sem alterar o pipeline moderno.
    """
    tokens_atuais = []
    for token in tokens:
        novo = copy.deepcopy(token)
        if not isinstance(novo.get("bbox"), dict):
            x1 = float(novo.get("x1", 0.0))
            y1 = float(novo.get("y1", 0.0))
            x2 = float(novo.get("x2", x1))
            y2 = float(novo.get("y2", y1))
            novo["bbox"] = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": max(0.0, x2 - x1),
                "height": max(0.0, y2 - y1),
                "center_x": float(novo.get("center_x", (x1 + x2) / 2.0)),
                "center_y": float(novo.get("center_y", (y1 + y2) / 2.0)),
            }
        # OCR moderno trabalha com confiança 0..1; fixtures legadas usam 0..100.
        conf = float(novo.get("confidence", 0.0) or 0.0)
        if conf > 1.0:
            novo["confidence"] = conf / 100.0
        tokens_atuais.append(novo)

    atual = criar_componentes_logicos(tokens_atuais, deteccoes)
    saida = []
    contadores_legados: dict[str, int] = {}
    usados_legados: set[str] = set()
    for comp in atual.get("components", []):
        ref = comp.get("reference") or {}
        val = comp.get("value") or {}
        ref_source_original = str(ref.get("source") or "")
        ref_source = "provisional" if ref_source_original == "generated" else ref_source_original
        legacy_id = comp.get("component_id")
        if ref_source_original == "generated":
            prefixo = FALLBACK_PREFIXES.get(str(comp.get("type") or ""), "X")
            contadores_legados[prefixo] = contadores_legados.get(prefixo, 0) + 1
            legacy_id = f"{prefixo}{contadores_legados[prefixo]}"
            while legacy_id in usados_legados:
                contadores_legados[prefixo] += 1
                legacy_id = f"{prefixo}{contadores_legados[prefixo]}"
        usados_legados.add(str(legacy_id))
        saida.append({
            "component_uid": comp.get("detection_id"),
            "id": legacy_id,
            "type": comp.get("type"),
            "value": val.get("text") or None,
            "bbox": comp.get("bbox"),
            "reference_source": ref_source,
            "needs_review": bool(comp.get("review_required")),
            "confirmed_by_user": False,
        })
    return saida

def aplicar_edicoes_componentes(payload: dict, edicoes: list[dict]) -> dict:
    """Wrapper legado de revisão manual usado pelos testes/scripts antigos."""
    resultado = copy.deepcopy(payload)
    por_uid = {str(e.get("component_uid") or ""): e for e in edicoes}
    for comp in resultado.get("components", []):
        uid = str(comp.get("component_uid") or comp.get("detection_id") or "")
        edit = por_uid.get(uid)
        if not edit:
            continue
        if edit.get("id") is not None:
            comp["id"] = str(edit.get("id"))
        if edit.get("type") is not None:
            comp["type"] = str(edit.get("type"))
        comp["value"] = edit.get("value")
        comp["needs_review"] = False
        comp["confirmed_by_user"] = True
    resultado["review_count"] = sum(1 for c in resultado.get("components", []) if c.get("needs_review"))
    return resultado
