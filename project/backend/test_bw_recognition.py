import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import cv2
import numpy as np

from backend.recognition import (
    _tamanhos_piramide_monocromatica,
    _imagem_quase_monocromatica,
    _executar_predicao_em_blocos,
    _remover_pautas_longas,
    _restaurar_deteccoes_rotacionadas,
    criar_vista_binaria_normalizada,
    criar_vistas_monocromaticas_universais,
    consolidar_multiview,
    detectar_componentes,
)
from backend.real_dataset import _monochrome_training_variants


class TestBWRecognitionPreprocessing(unittest.TestCase):
    def test_remove_pauta_clara_mas_preserva_fio_escuro(self):
        img = np.full((260, 360, 3), 230, dtype=np.uint8)
        for x in range(30, 360, 55):
            cv2.line(img, (x, 0), (x, 259), (180, 180, 180), 2)
        cv2.line(img, (180, 20), (180, 240), (20, 20, 20), 4)

        limpa = _remover_pautas_longas(img)
        gray = cv2.cvtColor(limpa, cv2.COLOR_BGR2GRAY)

        self.assertGreater(int(gray[130, 30]), 210)
        self.assertLess(int(gray[130, 180]), 80)

    def test_caixa_rotacionada_volta_para_coordenadas_originais(self):
        detection = {
            "detection_id": "DET_001",
            "class_id": 8,
            "class": "gnd",
            "confidence": 0.80,
            "bbox": {
                "x1": 40.0, "y1": 25.0,
                "x2": 90.0, "y2": 75.0,
                "width": 50.0, "height": 50.0,
                "center_x": 65.0, "center_y": 50.0,
            },
        }
        restored = _restaurar_deteccoes_rotacionadas(
            [detection], (200, 300, 3), "cw"
        )[0]
        self.assertEqual(
            restored["bbox"],
            {
                "x1": 25.0, "y1": 110.0,
                "x2": 75.0, "y2": 160.0,
                "width": 50.0, "height": 50.0,
                "center_x": 50.0, "center_y": 135.0,
            },
        )

    def test_detecta_entrada_preto_branco(self):
        img = np.full((220, 320, 3), 255, dtype=np.uint8)
        cv2.line(img, (20, 110), (100, 110), (0, 0, 0), 4)
        cv2.rectangle(img, (100, 80), (220, 140), (0, 0, 0), 4)
        cv2.line(img, (220, 110), (300, 110), (0, 0, 0), 4)
        self.assertTrue(_imagem_quase_monocromatica(img))

    def test_nao_classifica_imagem_colorida_forte_como_pb(self):
        img = np.full((120, 160, 3), (30, 60, 210), dtype=np.uint8)
        self.assertFalse(_imagem_quase_monocromatica(img))

    def test_binaria_mantem_fundo_branco_e_traco_escuro(self):
        img = np.full((180, 260, 3), 245, dtype=np.uint8)
        cv2.line(img, (20, 90), (240, 90), (20, 20, 20), 3)
        out = criar_vista_binaria_normalizada(img)
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        self.assertGreater(float(np.mean(gray[:10, :])), 240.0)
        self.assertLess(float(np.mean(gray[87:94, 40:220])), 80.0)

    def test_binaria_corrige_polaridade_invertida(self):
        img = np.zeros((180, 260, 3), dtype=np.uint8)
        cv2.line(img, (20, 90), (240, 90), (255, 255, 255), 3)
        out = criar_vista_binaria_normalizada(img)
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        self.assertGreater(float(np.mean(gray[:10, :])), 240.0)
        self.assertLess(float(np.mean(gray[87:94, 40:220])), 80.0)

    def test_vistas_monocromaticas_universais(self):
        img = np.full((240, 360, 3), 248, dtype=np.uint8)
        cv2.rectangle(img, (80, 70), (280, 170), (15, 15, 15), 4)
        views = criar_vistas_monocromaticas_universais(img)
        self.assertEqual(set(views), {
            "monochrome_clahe", "monochrome_otsu",
            "monochrome_adaptive", "monochrome_stroke_normalized",
        })
        for view in views.values():
            self.assertEqual(view.shape, img.shape)
            self.assertGreater(float(np.mean(view[:8, :])), 220.0)

    def test_dataset_real_gera_tres_variantes_pb_por_imagem(self):
        img = np.full((180, 260, 3), (245, 240, 235), dtype=np.uint8)
        cv2.line(img, (20, 90), (240, 90), (10, 10, 10), 3)
        variants = _monochrome_training_variants(img)
        self.assertEqual(
            set(variants), {"mono_clahe", "mono_otsu", "mono_adaptive"}
        )
        self.assertTrue(all(view.shape == img.shape for view in variants.values()))

    def test_piramide_monocromatica_tem_multiplas_escalas(self):
        img = np.full((379, 333, 3), 255, dtype=np.uint8)
        sizes = _tamanhos_piramide_monocromatica(img)
        self.assertGreaterEqual(len(sizes), 3)
        self.assertEqual(sizes, sorted(set(sizes)))
        self.assertTrue(all(size < max(img.shape[:2]) for size in sizes))

    def test_inferencia_em_blocos_retorna_coordenadas_globais(self):
        img = np.full((1500, 1500, 3), 255, dtype=np.uint8)

        def fake_predict(model, recorte, confidence, imgsz, inference_source):
            # Uma caixa local fixa em cada bloco; basta validar que blocos
            # posteriores recebem deslocamento para coordenadas globais.
            return [{
                "detection_id": "DET_001",
                "class_id": 0,
                "class": "resistor",
                "confidence": 0.80,
                "bbox": {
                    "x1": 20.0, "y1": 30.0, "x2": 120.0, "y2": 90.0,
                    "width": 100.0, "height": 60.0,
                    "center_x": 70.0, "center_y": 60.0,
                },
                "inference_sources": [inference_source],
            }]

        with patch("backend.recognition._executar_predicao", side_effect=fake_predict):
            deteccoes = _executar_predicao_em_blocos(
                object(), img, 0.01, 960, "tile_test", tile_size=960, overlap=0.20
            )

        self.assertGreater(len(deteccoes), 1)
        self.assertTrue(any(det["bbox"]["x1"] > 20 for det in deteccoes))
        self.assertTrue(any(det["bbox"]["y1"] > 30 for det in deteccoes))

    def test_diagrama_pequeno_pode_ser_dividido_em_blocos_de_detalhe(self):
        img = np.full((379, 333, 3), 255, dtype=np.uint8)

        def fake_predict(model, recorte, confidence, imgsz, inference_source):
            return [{
                "detection_id": "DET_001",
                "class_id": 0,
                "class": "resistor",
                "confidence": 0.20,
                "bbox": {
                    "x1": 10.0, "y1": 10.0, "x2": 80.0, "y2": 55.0,
                    "width": 70.0, "height": 45.0,
                    "center_x": 45.0, "center_y": 32.5,
                },
                "inference_sources": [inference_source],
            }]

        with patch("backend.recognition._executar_predicao", side_effect=fake_predict):
            deteccoes = _executar_predicao_em_blocos(
                object(), img, 0.05, 640,
                "normalized_binary_bw_detail_tiles",
                tile_size=235, overlap=0.38,
            )

        self.assertGreater(len(deteccoes), 1)
        self.assertTrue(any(det["bbox"]["x1"] > 10 for det in deteccoes))
        self.assertTrue(any(det["bbox"]["y1"] > 10 for det in deteccoes))

    def test_blocos_reais_usam_inferencia_em_lote(self):
        img = np.full((892, 867, 3), 255, dtype=np.uint8)

        class Result:
            boxes = None
            names = {}

        class BatchModel:
            def __init__(self):
                self.calls = 0

            def predict(self, source, **kwargs):
                self.calls += 1
                return [Result() for _ in source]

        model = BatchModel()
        detections = _executar_predicao_em_blocos(
            model, img, 0.05, 640, "monochrome_batch_test",
            tile_size=339, overlap=0.42,
        )
        self.assertEqual(detections, [])
        self.assertLessEqual(model.calls, 2)

    def test_previsoes_fracas_repetidas_nao_sao_resgatadas(self):
        img = np.full((240, 320, 3), 255, dtype=np.uint8)
        cv2.circle(img, (160, 120), 38, (0, 0, 0), 3)

        def fake_prediction(model, image, confidence, imgsz, inference_source):
            if inference_source not in {
                "monochrome_adaptive", "monochrome_otsu",
            }:
                return []
            return [{
                "detection_id": "DET_001",
                "class_id": 6,
                "class": "source_dc",
                "confidence": 0.13,
                "bbox": {
                    "x1": 115.0, "y1": 75.0, "x2": 205.0, "y2": 165.0,
                    "width": 90.0, "height": 90.0,
                    "center_x": 160.0, "center_y": 120.0,
                },
                "inference_sources": [inference_source],
            }]

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.png"
            output = Path(temp_dir) / "output.png"
            result_json = Path(temp_dir) / "output.json"
            cv2.imwrite(str(source), img)
            with (
                patch("backend.recognition._carregar_modelo", return_value=object()),
                patch("backend.recognition._hash_modelo", return_value="test"),
                patch("backend.recognition._executar_predicao", side_effect=fake_prediction),
                patch("backend.recognition._deteccoes_triangulos_preenchidos", return_value=[]),
                patch("backend.recognition._deteccoes_gnd_geometrico", return_value=[]),
                patch("backend.recognition._deteccoes_resistores_geometricos", return_value=[]),
                patch("backend.recognition._deteccoes_transistores_geometricos", return_value=[]),
                patch("backend.recognition._deteccoes_opamp_geometricos", return_value=[]),
            ):
                payload = detectar_componentes(
                    source, output, result_json,
                    confidence=0.25, imgsz=640, strategy="auto",
                )

        self.assertEqual(payload["count"], 0)
        self.assertIn("mosfet_n", payload["dataset_classes_with_monochrome_runtime"])

    def test_consenso_nao_infla_confianca_da_rede(self):
        base = {
            "detection_id": "DET_001",
            "class_id": 0,
            "class": "resistor",
            "confidence": 0.70,
            "bbox": {
                "x1": 10.0, "y1": 10.0, "x2": 80.0, "y2": 50.0,
                "width": 70.0, "height": 40.0,
                "center_x": 45.0, "center_y": 30.0,
            },
        }
        detections = []
        for source in (
            "original_color", "normalized_grayscale",
            "monochrome_otsu", "monochrome_adaptive",
        ):
            detections.append({**base, "inference_sources": [source]})
        merged = consolidar_multiview(detections)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0]["confidence"], 0.70)
        self.assertGreater(
            merged[0]["derived_consensus_confidence"],
            merged[0]["confidence"],
        )

    def test_piramide_sozinha_nao_cria_falsos_diodos(self):
        img = np.full((360, 640, 3), 245, dtype=np.uint8)
        for y in range(30, 350, 32):
            cv2.line(img, (0, y), (639, y), (190, 190, 190), 2)

        def fake_prediction(model, image, confidence, imgsz, inference_source):
            if "_pyramid_" not in inference_source:
                return []
            return [{
                "detection_id": "DET_001", "class_id": 3,
                "class": "diode", "confidence": 0.91,
                "bbox": {
                    "x1": 20.0, "y1": 20.0, "x2": 70.0, "y2": 65.0,
                    "width": 50.0, "height": 45.0,
                    "center_x": 45.0, "center_y": 42.5,
                },
                "inference_sources": [inference_source],
            }]

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "ruled.png"
            output = Path(temp_dir) / "output.png"
            result_json = Path(temp_dir) / "output.json"
            cv2.imwrite(str(source), img)
            with (
                patch("backend.recognition._carregar_modelo", return_value=object()),
                patch("backend.recognition._hash_modelo", return_value="test"),
                patch("backend.recognition._executar_predicao", side_effect=fake_prediction),
                patch("backend.recognition._deteccoes_triangulos_preenchidos", return_value=[]),
                patch("backend.recognition._deteccoes_gnd_geometrico", return_value=[]),
                patch("backend.recognition._deteccoes_resistores_geometricos", return_value=[]),
                patch("backend.recognition._deteccoes_transistores_geometricos", return_value=[]),
                patch("backend.recognition._deteccoes_opamp_geometricos", return_value=[]),
            ):
                payload = detectar_componentes(
                    source, output, result_json,
                    confidence=0.25, imgsz=640, strategy="auto",
                )

        self.assertEqual(payload["count"], 0)


if __name__ == '__main__':
    unittest.main()
