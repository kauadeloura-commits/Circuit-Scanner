import random
import unittest

import cv2
import numpy as np

from backend.dataset_generator import _bbox_from_mask, _draw_symbol
from backend.recognition import (
    _deteccoes_capacitores_geometricos,
    _deteccoes_resistores_retangulares,
    _deteccoes_resistores_geometricos,
    _deteccoes_transistores_geometricos,
    _deteccoes_opamp_geometricos,
    _validar_geometria_led,
    _validar_geometria_diodo,
    _evidencia_setas_emissao_led,
    _validar_geometria_resistor,
    _validar_geometria_indutor,
    _validar_geometria_capacitor,
    _validar_geometria_gnd,
    _validar_geometria_transistor,
    _validar_geometria_vcc,
    filtrar_falsos_positivos_geometricos,
    corrigir_diodo_led_por_setas,
)


class FalsePositiveGeometryFilterTests(unittest.TestCase):
    def test_vcc_real_passa(self):
        img, mask = _draw_symbol("vcc", 256, random.Random(7))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x1, y1, x2, y2 = _bbox_from_mask(mask)
        ok, score, _ = _validar_geometria_vcc(
            img, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        )
        self.assertTrue(ok)
        self.assertGreaterEqual(score, 0.72)

    def test_gnd_lateral_girado_noventa_graus_passa(self):
        img = np.full((256, 256, 3), 255, dtype=np.uint8)
        # Haste vindo da direita; a barra mais longa fica junto da haste e as
        # duas seguintes diminuem para a esquerda.
        cv2.line(img, (120, 128), (230, 128), (0, 0, 0), 4)
        cv2.line(img, (120, 78), (120, 178), (0, 0, 0), 4)
        cv2.line(img, (92, 96), (92, 160), (0, 0, 0), 4)
        cv2.line(img, (67, 113), (67, 143), (0, 0, 0), 4)
        ok, score, evidence = _validar_geometria_gnd(
            img, {"x1": 55, "y1": 65, "x2": 235, "y2": 190}
        )
        self.assertTrue(ok, evidence)
        self.assertGreaterEqual(score, 0.78)
        self.assertIn(
            evidence["detected_orientation"], {"rotated_cw", "rotated_ccw"}
        )

    def test_ponto_de_conexao_nao_e_vcc(self):
        img = np.full((256, 256, 3), 255, dtype=np.uint8)
        cv2.line(img, (40, 128), (216, 128), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(img, (128, 128), 7, (0, 0, 0), -1, cv2.LINE_AA)
        ok, _, evidence = _validar_geometria_vcc(
            img, {"x1": 104, "y1": 104, "x2": 152, "y2": 152}
        )
        self.assertFalse(ok)
        self.assertEqual(evidence.get("validation_mode"), "connection_dot_topology")
        self.assertEqual(evidence.get("connection_branch_count"), 2)


    def test_bolinha_com_uma_ligacao_pode_ser_vcc(self):
        img = np.full((256, 256, 3), 255, dtype=np.uint8)
        cv2.line(img, (128, 128), (128, 220), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(img, (128, 128), 8, (0, 0, 0), -1, cv2.LINE_AA)
        ok, score, evidence = _validar_geometria_vcc(
            img, {"x1": 104, "y1": 104, "x2": 152, "y2": 152}
        )
        self.assertTrue(ok)
        self.assertGreaterEqual(score, 0.90)
        self.assertEqual(evidence.get("connection_branch_count"), 1)

    def test_bolinha_em_juncao_t_nao_e_vcc(self):
        img = np.full((256, 256, 3), 255, dtype=np.uint8)
        cv2.line(img, (40, 128), (216, 128), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(img, (128, 128), (128, 220), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(img, (128, 128), 8, (0, 0, 0), -1, cv2.LINE_AA)
        ok, _, evidence = _validar_geometria_vcc(
            img, {"x1": 104, "y1": 104, "x2": 152, "y2": 152}
        )
        self.assertFalse(ok)
        self.assertEqual(evidence.get("connection_branch_count"), 3)

    def test_bolinha_em_cruzamento_nao_e_vcc(self):
        img = np.full((256, 256, 3), 255, dtype=np.uint8)
        cv2.line(img, (40, 128), (216, 128), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(img, (128, 40), (128, 216), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(img, (128, 128), 8, (0, 0, 0), -1, cv2.LINE_AA)
        ok, _, evidence = _validar_geometria_vcc(
            img, {"x1": 104, "y1": 104, "x2": 152, "y2": 152}
        )
        self.assertFalse(ok)
        self.assertEqual(evidence.get("connection_branch_count"), 4)

    def _draw_transistor(self, pnp=False):
        img = np.full((256, 256, 3), 255, dtype=np.uint8)
        cv2.circle(img, (132, 128), 70, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, (30, 128), (102, 128), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, (104, 80), (104, 176), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, (104, 98), (166, 56), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, (104, 158), (166, 200), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, (166, 56), (210, 56), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, (166, 200), (210, 200), (0, 0, 0), 5, cv2.LINE_AA)
        if pnp:
            cv2.arrowedLine(img, (159, 196), (132, 177), (0, 0, 0), 3, cv2.LINE_AA, tipLength=0.35)
        else:
            cv2.arrowedLine(img, (132, 177), (159, 196), (0, 0, 0), 3, cv2.LINE_AA, tipLength=0.35)
        return img

    def test_transistor_com_forquilha_passa(self):
        img = self._draw_transistor(False)
        ok, score, evidence = _validar_geometria_transistor(
            img,
            {"x1": 20, "y1": 45, "x2": 220, "y2": 211},
            "transistor_npn",
        )
        self.assertTrue(ok)
        self.assertGreaterEqual(score, 0.72)
        self.assertTrue(evidence["fork_connected"])

    def test_transistor_espelhado_com_forquilha_passa(self):
        img = cv2.flip(self._draw_transistor(True), 1)
        ok, score, evidence = _validar_geometria_transistor(
            img,
            {"x1": 20, "y1": 45, "x2": 220, "y2": 211},
            "transistor_pnp",
        )
        self.assertTrue(ok)
        self.assertGreaterEqual(score, 0.72)
        self.assertEqual(evidence["symbol_orientation"], "left")

    def test_linha_pautada_nao_e_led(self):
        img = np.full((180, 260, 3), 245, dtype=np.uint8)
        cv2.line(img, (0, 90), (259, 90), (150, 150, 150), 2, cv2.LINE_AA)
        ok, _, evidence = _validar_geometria_led(
            img, {"x1": 55, "y1": 55, "x2": 205, "y2": 125}
        )
        self.assertFalse(ok)
        self.assertTrue(evidence["ruled_line_only"])

    def test_simbolo_led_real_passa(self):
        img, mask = _draw_symbol("led", 256, random.Random(19))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x1, y1, x2, y2 = _bbox_from_mask(mask)
        ok, _, evidence = _validar_geometria_led(
            img, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        )
        self.assertTrue(ok, evidence)

    def test_simbolo_diodo_real_passa(self):
        img, mask = _draw_symbol("diode", 256, random.Random(23))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x1, y1, x2, y2 = _bbox_from_mask(mask)
        ok, _, evidence = _validar_geometria_diodo(
            img, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        )
        self.assertTrue(ok, evidence)

    def test_diodo_com_duas_setas_e_corrigido_para_led(self):
        img = np.full((260, 260, 3), 255, dtype=np.uint8)
        cv2.line(img, (25, 135), (85, 135), (0, 0, 0), 4)
        cv2.polylines(
            img,
            [np.asarray([(85, 85), (85, 185), (150, 135)], np.int32)],
            True,
            (0, 0, 0),
            4,
        )
        cv2.line(img, (155, 82), (155, 188), (0, 0, 0), 4)
        cv2.line(img, (155, 135), (230, 135), (0, 0, 0), 4)
        cv2.arrowedLine(
            img, (172, 78), (202, 48), (0, 0, 0), 3, tipLength=0.25
        )
        cv2.arrowedLine(
            img, (194, 98), (224, 68), (0, 0, 0), 3, tipLength=0.25
        )
        bbox = {
            "x1": 20.0, "y1": 35.0, "x2": 240.0, "y2": 195.0,
            "width": 220.0, "height": 160.0,
            "center_x": 130.0, "center_y": 115.0,
        }
        rays, evidence = _evidencia_setas_emissao_led(img, bbox)
        self.assertTrue(rays, evidence)
        detections, count = corrigir_diodo_led_por_setas(img, [{
            "detection_id": "DET_001",
            "class_id": 3,
            "class": "diode",
            "confidence": 0.78,
            "bbox": bbox,
            "review_required": True,
            "review_reasons": ["visually_similar_classes"],
        }])
        self.assertEqual(count, 1)
        self.assertEqual(detections[0]["class"], "led")
        self.assertEqual(detections[0]["class_resolution"], "led_emission_arrows")

    def test_diodo_sem_setas_permanece_diodo(self):
        img = np.full((260, 260, 3), 255, dtype=np.uint8)
        cv2.line(img, (25, 135), (85, 135), (0, 0, 0), 4)
        cv2.polylines(
            img,
            [np.asarray([(85, 85), (85, 185), (150, 135)], np.int32)],
            True,
            (0, 0, 0),
            4,
        )
        cv2.line(img, (155, 82), (155, 188), (0, 0, 0), 4)
        cv2.line(img, (155, 135), (230, 135), (0, 0, 0), 4)
        bbox = {"x1": 20.0, "y1": 35.0, "x2": 240.0, "y2": 195.0}
        rays, evidence = _evidencia_setas_emissao_led(img, bbox)
        self.assertFalse(rays, evidence)

    def test_indutor_em_papel_cinza_pautado_nao_e_rejeitado(self):
        img = np.full((220, 360, 3), 220, dtype=np.uint8)
        for x in range(20, 360, 55):
            cv2.line(img, (x, 0), (x, 219), (185, 185, 185), 2)
        cv2.line(img, (35, 110), (75, 110), (25, 25, 25), 4)
        for center_x in (95, 135, 175, 215):
            cv2.ellipse(
                img, (center_x, 110), (20, 25), 0, 180, 360,
                (25, 25, 25), 4,
            )
        cv2.line(img, (235, 110), (325, 110), (25, 25, 25), 4)
        ok, _, evidence = _validar_geometria_indutor(
            img, {"x1": 30, "y1": 70, "x2": 330, "y2": 145}
        )
        self.assertTrue(ok, evidence)
        self.assertLess(evidence["ink_ratio"], 0.48)

    def test_capacitor_manual_recupera_placas_por_projecao(self):
        img = np.full((220, 300, 3), 220, dtype=np.uint8)
        for x in range(20, 300, 55):
            cv2.line(img, (x, 0), (x, 219), (185, 185, 185), 2)
        cv2.line(img, (25, 110), (120, 110), (20, 20, 20), 4)
        cv2.line(img, (120, 70), (123, 150), (20, 20, 20), 4)
        cv2.line(img, (155, 72), (153, 148), (20, 20, 20), 4)
        cv2.line(img, (154, 110), (275, 110), (20, 20, 20), 4)
        ok, _, evidence = _validar_geometria_capacitor(
            img, {"x1": 20, "y1": 55, "x2": 280, "y2": 165}
        )
        self.assertTrue(ok, evidence)
        self.assertEqual(
            evidence["projection_plate_evidence"]["vertical_groups"], 2
        )

    def test_pauta_nao_e_diodo(self):
        img = np.full((180, 260, 3), 245, dtype=np.uint8)
        cv2.line(img, (0, 70), (259, 70), (160, 160, 160), 2)
        cv2.line(img, (0, 110), (259, 110), (160, 160, 160), 2)
        ok, _, evidence = _validar_geometria_diodo(
            img, {"x1": 70, "y1": 45, "x2": 135, "y2": 125}
        )
        self.assertFalse(ok, evidence)

    def test_letras_ed_nao_sao_resistor(self):
        img = np.full((150, 180, 3), 255, dtype=np.uint8)
        cv2.putText(
            img, "ED", (28, 112), cv2.FONT_HERSHEY_SIMPLEX,
            2.7, (0, 0, 0), 5, cv2.LINE_AA,
        )
        ok, _, evidence = _validar_geometria_resistor(
            img, {"x1": 18, "y1": 22, "x2": 165, "y2": 130}
        )
        self.assertFalse(ok, evidence)

    def test_geometria_propoe_resistor_sem_yolo(self):
        img = np.full((320, 420, 3), 255, dtype=np.uint8)
        points = np.asarray([
            (45, 160), (90, 160), (105, 135), (125, 185),
            (145, 135), (165, 185), (185, 135), (205, 160), (260, 160),
        ], dtype=np.int32)
        cv2.polylines(img, [points], False, (0, 0, 0), 4, cv2.LINE_AA)
        detections = _deteccoes_resistores_geometricos(img)
        self.assertGreaterEqual(len(detections), 1)
        self.assertEqual(detections[0]["class"], "resistor")

    def test_geometria_propoe_resistor_iec_vertical(self):
        img = np.full((360, 260, 3), 255, dtype=np.uint8)
        cv2.line(img, (130, 15), (130, 100), (0, 0, 0), 5)
        cv2.rectangle(img, (108, 100), (152, 255), (0, 0, 0), 6)
        cv2.line(img, (130, 255), (130, 345), (0, 0, 0), 5)
        detections = _deteccoes_resistores_retangulares(img)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class"], "resistor")

    def test_resistor_iec_vertical_sobre_pauta_clara(self):
        img = np.full((360, 260, 3), 230, dtype=np.uint8)
        for x in range(20, 260, 55):
            cv2.line(img, (x, 0), (x, 359), (180, 180, 180), 2)
        # A pauta x=130 atravessa o vazio do corpo antes de o resistor ser
        # desenhado, reproduzindo a foto real do caderno.
        cv2.line(img, (130, 15), (130, 100), (20, 20, 20), 5)
        cv2.rectangle(img, (108, 100), (152, 255), (20, 20, 20), 6)
        cv2.line(img, (130, 255), (130, 345), (20, 20, 20), 5)

        detections = _deteccoes_resistores_retangulares(img)
        self.assertEqual(len(detections), 1)
        bbox = detections[0]["bbox"]
        self.assertLess(bbox["x1"], 110)
        self.assertGreater(bbox["x2"], 150)

    def test_pautas_e_outros_simbolos_nao_propoem_resistor_iec(self):
        img = np.full((420, 520, 3), 230, dtype=np.uint8)
        for x in range(25, 520, 55):
            cv2.line(img, (x, 0), (x, 419), (178, 178, 178), 2)
        # Fonte circular, triangulo e barra/terminal de diodo: formas que
        # geravam os falsos resistores mostrados na imagem real.
        cv2.circle(img, (105, 90), 32, (25, 25, 25), 4)
        cv2.line(img, (105, 20), (105, 58), (25, 25, 25), 4)
        cv2.line(img, (105, 122), (105, 175), (25, 25, 25), 4)
        cv2.polylines(
            img,
            [np.asarray([(210, 300), (285, 220), (285, 380)], np.int32)],
            True,
            (25, 25, 25),
            4,
        )
        cv2.line(img, (390, 220), (390, 380), (25, 25, 25), 5)
        cv2.line(img, (390, 300), (470, 300), (25, 25, 25), 5)

        self.assertEqual(_deteccoes_resistores_retangulares(img), [])

    def test_geometria_propoe_capacitor_por_placas(self):
        img = np.full((360, 420, 3), 255, dtype=np.uint8)
        cv2.line(img, (175, 90), (175, 170), (0, 0, 0), 5)
        cv2.line(img, (210, 90), (210, 170), (0, 0, 0), 5)
        cv2.line(img, (35, 130), (175, 130), (0, 0, 0), 5)
        cv2.line(img, (210, 130), (385, 130), (0, 0, 0), 5)
        detections = _deteccoes_capacitores_geometricos(img)
        self.assertGreaterEqual(len(detections), 1)
        self.assertEqual(detections[0]["class"], "capacitor")

    def test_capacitor_cortado_na_borda_ainda_e_proposto(self):
        img = np.full((360, 420, 3), 255, dtype=np.uint8)
        cv2.line(img, (2, 90), (2, 170), (0, 0, 0), 5)
        cv2.line(img, (30, 90), (30, 170), (0, 0, 0), 5)
        cv2.line(img, (30, 130), (300, 130), (0, 0, 0), 5)
        detections = _deteccoes_capacitores_geometricos(img)
        self.assertGreaterEqual(len(detections), 1)

    def test_corpo_circular_de_led_nao_vira_resistor(self):
        img, _ = _draw_symbol("led", 256, random.Random(31))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        detections = _deteccoes_resistores_geometricos(img)
        self.assertEqual(detections, [])

    def test_geometria_propoe_opamp_sem_yolo(self):
        img, _ = _draw_symbol("opamp", 256, random.Random(41))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        detections = _deteccoes_opamp_geometricos(img)
        self.assertGreaterEqual(len(detections), 1)
        self.assertEqual(detections[0]["class"], "opamp")

    def test_linhas_de_caderno_nao_propoem_opamp(self):
        img = np.full((320, 520, 3), 245, dtype=np.uint8)
        for y in range(25, 320, 32):
            cv2.line(img, (0, y), (519, y), (180, 180, 180), 2)
        self.assertEqual(_deteccoes_opamp_geometricos(img), [])

    def test_geometria_propoe_bjt_sem_yolo(self):
        img = self._draw_transistor(False)
        detections = _deteccoes_transistores_geometricos(img, [])
        self.assertGreaterEqual(len(detections), 1)
        self.assertEqual(detections[0]["class"], "transistor_npn")

    def test_geometria_nao_sobrescreve_indicio_pnp_do_yolo(self):
        img = self._draw_transistor(True)
        hint = {
            "class_id": 11,
            "class": "transistor_pnp",
            "confidence": 0.14,
            "bbox": {
                "x1": 55.0, "y1": 55.0, "x2": 210.0, "y2": 210.0,
                "width": 155.0, "height": 155.0,
                "center_x": 132.5, "center_y": 132.5,
            },
        }
        detections = _deteccoes_transistores_geometricos(img, [hint])
        self.assertGreaterEqual(len(detections), 1)
        self.assertEqual(detections[0]["class"], "transistor_pnp")
        self.assertEqual(
            detections[0]["bjt_subtype_source"],
            "overlapping_yolo_candidate",
        )

    def test_linha_diagonal_nao_e_transistor(self):
        img = np.full((256, 256, 3), 255, dtype=np.uint8)
        cv2.line(img, (35, 220), (220, 35), (0, 0, 0), 5, cv2.LINE_AA)
        ok, _, evidence = _validar_geometria_transistor(
            img,
            {"x1": 30, "y1": 30, "x2": 225, "y2": 225},
            "transistor_pnp",
        )
        self.assertFalse(ok)
        self.assertFalse(evidence["fork_connected"])

    def test_pautas_nao_passam_com_confianca_um(self):
        img = np.full((600, 520, 3), 245, dtype=np.uint8)
        for x in range(45, 520, 70):
            cv2.line(img, (x, 0), (x, 599), (175, 175, 175), 2)
        for y in range(60, 600, 70):
            cv2.line(img, (0, y), (519, y), (185, 185, 185), 2)
        classes = [
            "capacitor", "capacitor_polarized", "inductor", "battery",
            "gnd", "resistor", "mosfet_n", "mosfet_p",
        ]
        detections = []
        for i, classe in enumerate(classes):
            x1 = 15 + i * 60
            detections.append({
                "detection_id": f"DET_{i+1:03d}",
                "class_id": i,
                "class": classe,
                "confidence": 1.0,
                "best_view_confidence": 1.0,
                "bbox": {
                    "x1": float(x1), "y1": 45.0,
                    "x2": float(x1 + 35), "y2": 245.0,
                    "width": 35.0, "height": 200.0,
                    "center_x": float(x1 + 17.5), "center_y": 145.0,
                },
            })
        kept, rejected = filtrar_falsos_positivos_geometricos(img, detections)
        self.assertEqual(kept, [])
        self.assertEqual({d["class"] for d in rejected}, set(classes))

    def test_simbolos_estruturais_reais_passam_filtro(self):
        classes = [
            "resistor", "capacitor", "capacitor_polarized", "inductor",
            "battery", "gnd", "mosfet_n", "mosfet_p",
        ]
        for index, classe in enumerate(classes):
            with self.subTest(classe=classe):
                img, mask = _draw_symbol(classe, 256, random.Random(index + 30))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                x1, y1, x2, y2 = _bbox_from_mask(mask)
                detection = {
                    "detection_id": "DET_001",
                    "class_id": index,
                    "class": classe,
                    "confidence": 0.95,
                    "best_view_confidence": 0.95,
                    "bbox": {
                        "x1": float(x1), "y1": float(y1),
                        "x2": float(x2), "y2": float(y2),
                        "width": float(x2 - x1), "height": float(y2 - y1),
                        "center_x": float((x1 + x2) / 2),
                        "center_y": float((y1 + y2) / 2),
                    },
                }
                kept, rejected = filtrar_falsos_positivos_geometricos(
                    img, [detection]
                )
                self.assertEqual(len(kept), 1, rejected)

    def test_filtro_remove_duas_falsas_deteccoes(self):
        img = np.full((300, 500, 3), 255, dtype=np.uint8)
        cv2.circle(img, (100, 150), 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.line(img, (260, 230), (430, 60), (0, 0, 0), 4, cv2.LINE_AA)
        detections = [
            {
                "detection_id": "DET_001", "class_id": 9, "class": "vcc", "confidence": 0.8,
                "bbox": {"x1": 75.0, "y1": 125.0, "x2": 125.0, "y2": 175.0,
                         "width": 50.0, "height": 50.0, "center_x": 100.0, "center_y": 150.0},
            },
            {
                "detection_id": "DET_002", "class_id": 10, "class": "transistor_npn", "confidence": 0.75,
                "bbox": {"x1": 250.0, "y1": 50.0, "x2": 440.0, "y2": 240.0,
                         "width": 190.0, "height": 190.0, "center_x": 345.0, "center_y": 145.0},
            },
        ]
        kept, rejected = filtrar_falsos_positivos_geometricos(img, detections)
        self.assertEqual(len(kept), 0)
        self.assertEqual(len(rejected), 2)


if __name__ == "__main__":
    unittest.main()
