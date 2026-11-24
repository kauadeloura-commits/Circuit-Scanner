from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backend.ocr as ocr_module
from backend.ocr import (
    _aplicar_biblioteca_ocr,
    _limpar_texto,
    _token_relevante,
    ajustar_caixas_com_texto,
    aplicar_correcoes_manuais,
    criar_componentes_logicos,
    executar_ocr_componentes,
    filtrar_deteccoes_em_texto,
    ocr_status,
    preparar_imagem_texto_ao_redor,
    remover_falsos_positivos_textuais,
    salvar_amostras_ocr,
    salvar_negativos_texto_yolo,
)


def test_corrige_confusoes_comuns_da_escrita_manual():
    assert _limpar_texto("83") == "R3"
    assert _limpar_texto("lov") == "10V"
    token_ruim = _token("TXT_000", "teu4pA", 0, 0, 20, 10, confidence=0.37)
    token_ruim["engine"] = "easyocr"
    assert _token_relevante(token_ruim) is True


def test_texto_alfanumerico_generico_e_preservado_como_texto():
    token = _token("TXT_900", "LM358", 10, 10, 70, 35, confidence=0.42)
    token["engine"] = "easyocr"
    assert _token_relevante(token) is True

    resultado = criar_componentes_logicos(
        [token],
        [_deteccao("DET_001", "resistor", 120, 10, 190, 80)],
    )
    assert resultado["unassigned_tokens"][0]["text"] == "LM358"


def test_texto_reconhecido_vira_negativo_para_o_proximo_treino_yolo():
    imagem = np.full((100, 180, 3), 255, dtype=np.uint8)
    cv2.putText(
        imagem,
        "LED1",
        (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        3,
    )
    token = _token("TXT_901", "LED1", 10, 20, 105, 75, confidence=0.81)
    pasta_anterior = ocr_module.OCR_HARD_NEGATIVE_DIR
    with TemporaryDirectory() as pasta:
        ocr_module.OCR_HARD_NEGATIVE_DIR = Path(pasta)
        try:
            assert salvar_negativos_texto_yolo(imagem, [token]) == 1
            assert len(list(Path(pasta).glob("text_*.jpg"))) == 1
        finally:
            ocr_module.OCR_HARD_NEGATIVE_DIR = pasta_anterior


def test_texto_nunca_apaga_componente_detectado_pelo_yolo():
    deteccoes = [
        _deteccao("DET_001", "led", 10, 10, 90, 60, confidence=0.31),
        _deteccao("DET_002", "resistor", 100, 10, 180, 60, confidence=0.73),
    ]
    tokens = [
        _token("TXT_001", "R4", 20, 20, 55, 45, confidence=0.88),
        _token("TXT_002", "R3", 110, 20, 145, 45, confidence=0.90),
    ]
    for token in tokens:
        token["engine"] = "easyocr"

    mantidas, removidas = filtrar_deteccoes_em_texto(deteccoes, tokens)

    assert [item["detection_id"] for item in mantidas] == ["DET_001", "DET_002"]
    assert [item["detection_id"] for item in removidas] == ["DET_001"]
    assert mantidas[0]["ocr_text_overlap_suspected"] is True


def test_nao_remove_componente_confirmado_manualmente():
    deteccao = _deteccao("DET_001", "led", 10, 10, 90, 60, confidence=0.31)
    deteccao["manual_reviewed"] = True
    token = _token("TXT_001", "R4", 20, 20, 55, 45, confidence=0.88)
    token["engine"] = "easyocr"

    mantidas, removidas = filtrar_deteccoes_em_texto([deteccao], [token])

    assert mantidas == [deteccao]
    assert removidas == []


def test_le_texto_ao_redor_depois_de_localizar_componente():
    imagem = np.full((500, 500, 3), 120, dtype=np.uint8)
    deteccao = _deteccao("DET_001", "resistor", 200, 200, 300, 300)

    contexto = preparar_imagem_texto_ao_redor(imagem, [deteccao])

    assert np.all(contexto[0, 0] == 255)
    assert np.all(contexto[100, 100] == 120)
    assert np.all(contexto[250, 250] == 255)


def test_reduz_caixa_quando_texto_entra_no_lado_do_componente():
    deteccao = _deteccao("DET_001", "resistor", 100, 100, 200, 200)
    token = _token("TXT_001", "R1", 104, 130, 130, 160, confidence=0.90)
    token["engine"] = "easyocr"

    ajustadas, registros = ajustar_caixas_com_texto([deteccao], [token])

    assert ajustadas[0]["bbox"]["x1"] == 132.0
    assert ajustadas[0]["bbox"]["x2"] == 200.0
    assert ajustadas[0]["bbox_adjusted_for_text"] is True
    assert registros[0]["tokens"][0]["text"] == "R1"


def test_nao_reduz_caixa_com_ocr_de_baixa_confianca():
    deteccao = _deteccao("DET_001", "resistor", 100, 100, 200, 200)
    token = _token("TXT_001", "R1", 104, 130, 130, 160, confidence=0.20)
    token["engine"] = "easyocr"

    ajustadas, registros = ajustar_caixas_com_texto([deteccao], [token])

    assert ajustadas[0]["bbox"]["x1"] == 100.0
    assert registros == []


def test_reduz_mais_a_caixa_quando_texto_ocupa_metade_da_previsao():
    deteccao = _deteccao("DET_001", "resistor", 100, 100, 200, 200)
    token = _token("TXT_001", "R1", 104, 130, 150, 160, confidence=0.95)
    token["engine"] = "easyocr"

    ajustadas, registros = ajustar_caixas_com_texto([deteccao], [token])

    assert ajustadas[0]["bbox"]["x1"] == 152.0
    assert registros[0]["tokens"][0]["text"] == "R1"


def test_zero_de_10v_nao_permanece_como_componente_vcc():
    falso_vcc = _deteccao("DET_001", "vcc", 45, 18, 70, 52)
    vcc_real = _deteccao("DET_002", "vcc", 48, 70, 78, 110)
    token = _token("TXT_001", "10V", 20, 12, 88, 57, confidence=0.84)
    token["engine"] = "easyocr"

    mantidas, removidas = remover_falsos_positivos_textuais(
        [falso_vcc, vcc_real],
        [token],
    )

    assert [item["detection_id"] for item in mantidas] == ["DET_002"]
    assert removidas[0]["text_false_positive_reason"] == (
        "vcc_box_inside_voltage_text"
    )


def test_caixa_formada_pela_escrita_r4_e_removida_como_falso_resistor():
    falso_resistor = _deteccao("DET_001", "resistor", 20, 20, 78, 58)
    resistor_real = _deteccao("DET_002", "resistor", 105, 15, 180, 100)
    token = _token("TXT_001", "R4", 16, 16, 82, 62, confidence=0.61)
    token["engine"] = "easyocr"
    token["evidence_count"] = 2

    mantidas, removidas = remover_falsos_positivos_textuais(
        [falso_resistor, resistor_real],
        [token],
    )

    assert [item["detection_id"] for item in mantidas] == ["DET_002"]
    assert removidas[0]["text_false_positive_reason"] == (
        "component_box_matches_ocr_text"
    )


def test_caixa_grande_com_simbolo_e_texto_nao_e_apagada():
    caixa_completa = _deteccao("DET_001", "resistor", 10, 10, 180, 120)
    token = _token("TXT_001", "R4", 15, 20, 60, 50, confidence=0.72)
    token["engine"] = "easyocr"
    token["evidence_count"] = 2

    mantidas, removidas = remover_falsos_positivos_textuais(
        [caixa_completa],
        [token],
    )

    assert [item["detection_id"] for item in mantidas] == ["DET_001"]
    assert removidas == []


def _deteccao(detection_id, classe, x1, y1, x2, y2, confidence=0.90):
    return {
        "detection_id": detection_id,
        "class": classe,
        "confidence": confidence,
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


def _token(token_id, texto, x1, y1, x2, y2, confidence=0.95):
    return {
        "token_id": token_id,
        "text": texto,
        "confidence": confidence,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "variant": "test",
        "orientation": "normal",
    }


def test_associacao_texto_com_classe_visual():
    deteccoes = [
        _deteccao("DET_001", "resistor", 160, 170, 320, 250),
        _deteccao("DET_002", "capacitor", 600, 170, 760, 250),
    ]
    tokens = [
        _token("TXT_001", "R1", 175, 105, 225, 145),
        _token("TXT_002", "10k", 235, 105, 315, 145),
        _token("TXT_003", "C1", 615, 105, 665, 145),
        _token("TXT_004", "100uF", 675, 105, 790, 145),
    ]

    resultado = criar_componentes_logicos(tokens, deteccoes)
    componentes = {item["detection_id"]: item for item in resultado["components"]}

    assert componentes["DET_001"]["component_id"] == "R1"
    assert componentes["DET_001"]["value"]["text"] == "10k"
    assert componentes["DET_001"]["value"]["normalized"] == "10000"
    assert componentes["DET_002"]["component_id"] == "C1"
    assert componentes["DET_002"]["value"]["text"] == "100uF"
    assert resultado["reference_ocr_count"] == 2
    assert resultado["value_ocr_count"] == 2
    assert resultado["review_count"] == 0


def test_led_minusculo_e_referencia_valida():
    deteccao = _deteccao("DET_001", "led", 100, 100, 180, 180)
    token = _token("TXT_001", "led", 55, 110, 95, 140, confidence=0.62)
    token["engine"] = "easyocr"

    assert _token_relevante(token) is True
    resultado = criar_componentes_logicos([token], [deteccao])
    assert resultado["components"][0]["component_id"] == "LED"
    assert resultado["components"][0]["reference"]["source"] == "ocr"


def test_c1_e_c2_sao_associados_aos_capacitores_mais_proximos():
    deteccoes = [
        _deteccao("DET_001", "capacitor", 100, 120, 160, 190),
        _deteccao("DET_002", "capacitor", 330, 120, 390, 190),
    ]
    tokens = [
        _token("TXT_001", "c1", 65, 130, 95, 155, confidence=0.44),
        _token("TXT_002", "c2", 395, 130, 425, 155, confidence=0.41),
    ]
    for token in tokens:
        token["engine"] = "easyocr"

    resultado = criar_componentes_logicos(tokens, deteccoes)
    componentes = {
        item["detection_id"]: item["component_id"]
        for item in resultado["components"]
    }
    assert componentes == {"DET_001": "C1", "DET_002": "C2"}


def test_r4_vai_para_caixa_de_resistor_compativel_mais_proxima():
    deteccoes = [
        _deteccao("DET_001", "resistor", 100, 150, 170, 240),
        _deteccao("DET_002", "capacitor", 175, 120, 245, 210),
        _deteccao("DET_003", "resistor", 360, 150, 430, 240),
    ]
    tokens = [
        _token("TXT_001", "R4", 172, 170, 215, 200),
        _token("TXT_002", "5K", 172, 205, 215, 235),
    ]

    resultado = criar_componentes_logicos(tokens, deteccoes)
    componentes = {item["detection_id"]: item for item in resultado["components"]}

    assert componentes["DET_001"]["component_id"] == "R4"
    assert componentes["DET_001"]["value"]["text"] == "5K"
    assert componentes["DET_002"]["component_id"].startswith("C_AUTO")
    assert componentes["DET_003"]["component_id"].startswith("R_AUTO")


def test_correcao_r7_para_r4_alimenta_biblioteca_e_corrige_proxima_leitura():
    imagem = np.full((120, 220, 3), 255, dtype=np.uint8)
    cv2.putText(
        imagem,
        "R7",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 0),
        3,
    )
    token = _token("TXT_001", "R7", 15, 20, 85, 75, confidence=0.55)
    token["engine"] = "easyocr"
    deteccao = _deteccao("DET_001", "resistor", 105, 25, 180, 95)
    resultado = criar_componentes_logicos([token], [deteccao])

    with TemporaryDirectory() as pasta:
        pasta = Path(pasta)
        caminho_imagem = pasta / "entrada.png"
        assert cv2.imwrite(str(caminho_imagem), imagem)
        pasta_anterior = ocr_module.OCR_SAMPLES_DIR
        indice_anterior = ocr_module.OCR_SAMPLES_INDEX
        ocr_module.OCR_SAMPLES_DIR = pasta / "amostras"
        ocr_module.OCR_SAMPLES_INDEX = ocr_module.OCR_SAMPLES_DIR / "index.json"
        try:
            adicionadas = salvar_amostras_ocr(
                caminho_imagem,
                resultado,
                [{"detection_id": "DET_001", "reference": "R4", "value": ""}],
            )
            assert adicionadas == 1
            indice = json.loads(
                ocr_module.OCR_SAMPLES_INDEX.read_text(encoding="utf-8")
            )
            assert indice[0]["label"] == "R4"

            corrigidos = _aplicar_biblioteca_ocr(imagem, [token], [deteccao])
            assert corrigidos[0]["text"] == "R4"
            assert corrigidos[0]["library_matched"] is True
        finally:
            ocr_module.OCR_SAMPLES_DIR = pasta_anterior
            ocr_module.OCR_SAMPLES_INDEX = indice_anterior


def test_led_gnd_e_transistor_nao_possuem_valor():
    deteccoes = [
        _deteccao("DET_001", "led", 100, 100, 180, 180),
        _deteccao("DET_002", "gnd", 300, 100, 380, 180),
        _deteccao("DET_003", "transistor_npn", 500, 100, 600, 210),
    ]
    tokens = [
        _token("TXT_001", "LED1", 105, 50, 165, 85),
        _token("TXT_002", "3V", 110, 190, 155, 220),
        _token("TXT_003", "GND", 305, 50, 370, 85),
        _token("TXT_004", "Q1", 510, 50, 555, 85),
        _token("TXT_005", "BC547", 510, 220, 590, 250),
    ]

    resultado = criar_componentes_logicos(tokens, deteccoes)

    assert all(
        componente["value"]["source"] == "not_applicable"
        for componente in resultado["components"]
    )
    assert all(
        componente["value"]["text"] == ""
        for componente in resultado["components"]
    )


def test_sem_texto_gera_referencia_sem_inventar_valor():
    deteccoes = [
        _deteccao("DET_001", "resistor", 10, 10, 80, 50),
        _deteccao("DET_002", "resistor", 100, 10, 170, 50),
    ]
    resultado = criar_componentes_logicos([], deteccoes)

    assert [item["component_id"] for item in resultado["components"]] == [
        "R_AUTO1",
        "R_AUTO2",
    ]
    assert all(item["reference"]["source"] == "generated" for item in resultado["components"])
    assert all(item["value"]["text"] == "" for item in resultado["components"])
    assert resultado["review_count"] == 2


def test_correcao_manual_valida_referencias_unicas():
    deteccoes = [
        _deteccao("DET_001", "resistor", 10, 10, 80, 50),
        _deteccao("DET_002", "capacitor", 100, 10, 170, 50),
    ]
    resultado = criar_componentes_logicos([], deteccoes)
    corrigido = aplicar_correcoes_manuais(
        resultado,
        [
            {"detection_id": "DET_001", "reference": "R7", "value": "4k7"},
            {"detection_id": "DET_002", "reference": "C2", "value": "100uF"},
        ],
        {"detections": deteccoes},
    )

    assert corrigido["components"][0]["component_id"] == "R7"
    assert corrigido["components"][0]["value"]["normalized"] == "4700"
    assert corrigido["components"][1]["component_id"] == "C2"
    assert corrigido["review_count"] == 0


def test_tesseract_le_texto_impresso_e_salva_componentes():
    if not ocr_status()["ready"]:
        print("SKIP: Tesseract não instalado.")
        return

    imagem = np.full((500, 1000, 3), 255, dtype=np.uint8)
    cv2.putText(imagem, "R1", (150, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    cv2.putText(imagem, "10k", (245, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    cv2.rectangle(imagem, (170, 190), (330, 260), (0, 0, 0), 3)
    cv2.putText(imagem, "C1", (620, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    cv2.putText(imagem, "100uF", (715, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    cv2.rectangle(imagem, (640, 190), (800, 260), (0, 0, 0), 3)

    deteccoes = [
        _deteccao("DET_001", "resistor", 170, 190, 330, 260),
        _deteccao("DET_002", "capacitor", 640, 190, 800, 260),
    ]
    with TemporaryDirectory() as pasta:
        pasta = Path(pasta)
        entrada = pasta / "entrada.png"
        saida_json = pasta / "components.json"
        saida_imagem = pasta / "ocr.png"
        assert cv2.imwrite(str(entrada), imagem)

        resultado = executar_ocr_componentes(
            entrada,
            {"detections": deteccoes},
            saida_json,
            saida_imagem,
        )

        referencias = {item["component_id"] for item in resultado["components"]}
        valores = {item["value"]["text"].upper() for item in resultado["components"]}
        assert {"R1", "C1"}.issubset(referencias)
        assert "10K" in valores
        assert "100UF" in valores
        assert json.loads(saida_json.read_text(encoding="utf-8"))["count"] == 2
        assert saida_imagem.is_file() and saida_imagem.stat().st_size > 0


if __name__ == "__main__":
    test_associacao_texto_com_classe_visual()
    test_sem_texto_gera_referencia_sem_inventar_valor()
    test_correcao_manual_valida_referencias_unicas()
    test_tesseract_le_texto_impresso_e_salva_componentes()
    print("OK: testes da Etapa 6 passaram.")
