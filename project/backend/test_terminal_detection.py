import unittest
from unittest.mock import patch

import cv2
import numpy as np

from backend.terminal_detection import detectar_terminais_componentes


class TerminalDetectionTests(unittest.TestCase):
    def _input(self, component_type, bbox, component_id="X1"):
        return {
            "schema_version": "stage6-1.0",
            "components": [
                {
                    "component_id": component_id,
                    "detection_id": "DET_001",
                    "type": component_type,
                    "bbox": bbox,
                    "review_required": False,
                    "review_reasons": [],
                }
            ],
        }

    def test_resistor_horizontal(self):
        data = self._input(
            "resistor",
            {"x1": 100, "y1": 80, "x2": 300, "y2": 120, "width": 200, "height": 40},
            "R1",
        )
        result = detectar_terminais_componentes(data, 500, 300)
        comp = result["components"][0]
        self.assertEqual(comp["orientation"]["axis"], "horizontal")
        self.assertEqual([p["pin"] for p in comp["terminals"]], ["1", "2"])
        self.assertEqual(comp["terminals"][0]["position"], {"x": 101.0, "y": 100.0})
        self.assertEqual(comp["terminals"][1]["position"], {"x": 299.0, "y": 100.0})

    def test_resistor_vertical(self):
        data = self._input(
            "resistor",
            {"x1": 80, "y1": 50, "x2": 120, "y2": 250, "width": 40, "height": 200},
            "R2",
        )
        result = detectar_terminais_componentes(data, 400, 400)
        comp = result["components"][0]
        self.assertEqual(comp["orientation"]["axis"], "vertical")
        self.assertEqual(comp["terminals"][0]["position"], {"x": 100.0, "y": 51.0})
        self.assertEqual(comp["terminals"][1]["position"], {"x": 100.0, "y": 249.0})

    def test_rotated_two_terminal_uses_local_angle(self):
        image = np.full((220, 360, 3), 255, dtype=np.uint8)
        cv2.line(image, (80, 130), (280, 90), (0, 0, 0), 4, cv2.LINE_AA)
        data = self._input(
            "resistor",
            {"x1": 75, "y1": 80, "x2": 285, "y2": 140, "width": 210, "height": 60},
            "R3",
        )
        result = detectar_terminais_componentes(data, 360, 220, image=image)
        comp = result["components"][0]
        self.assertEqual(comp["orientation"]["axis"], "horizontal")
        self.assertLess(comp["orientation"]["angle_degrees"], -5.0)
        p1, p2 = comp["terminals"]
        self.assertGreater(p1["position"]["y"], p2["position"]["y"])

    def test_houghlinesp_accepts_2d_n_by_4_output(self):
        """Regressão: alguns builds do OpenCV retornam HoughLinesP como (N,4)."""
        image = np.full((180, 320, 3), 255, dtype=np.uint8)
        data = self._input(
            "resistor",
            {"x1": 50, "y1": 60, "x2": 270, "y2": 120, "width": 220, "height": 60},
            "R4",
        )
        fake_lines = np.array([
            [5, 35, 210, 35],
            [8, 34, 205, 34],
        ], dtype=np.int32)
        with patch("backend.terminal_detection.cv2.HoughLinesP", return_value=fake_lines):
            result = detectar_terminais_componentes(data, 320, 180, image=image)
        comp = result["components"][0]
        self.assertEqual(comp["orientation"]["axis"], "horizontal")
        self.assertEqual(len(comp["terminals"]), 2)
        self.assertEqual(comp["orientation"]["angle_source"], "local_hough")

    def test_ground_has_one_terminal_on_top(self):
        data = self._input(
            "gnd",
            {"x1": 50, "y1": 50, "x2": 110, "y2": 130, "width": 60, "height": 80},
            "GND1",
        )
        result = detectar_terminais_componentes(data, 300, 300)
        pins = result["components"][0]["terminals"]
        self.assertEqual(len(pins), 1)
        self.assertEqual(pins[0]["pin"], "GND")
        self.assertEqual(pins[0]["position"], {"x": 80.0, "y": 51.0})

    def test_source_dc_is_two_sided_not_top_bottom(self):
        data = self._input(
            "source_dc",
            {"x1": 40, "y1": 70, "x2": 240, "y2": 170, "width": 200, "height": 100},
            "V1",
        )
        result = detectar_terminais_componentes(data, 300, 240)
        comp = result["components"][0]
        self.assertEqual(comp["orientation"]["axis"], "horizontal")
        self.assertEqual(comp["terminals"][0]["position"], {"x": 41.0, "y": 120.0})
        self.assertEqual(comp["terminals"][1]["position"], {"x": 239.0, "y": 120.0})

    def test_opamp_has_only_three_visible_pins_and_correct_sign_order(self):
        data = self._input(
            "opamp",
            {"x1": 20, "y1": 20, "x2": 220, "y2": 180, "width": 200, "height": 160},
            "U1",
        )
        result = detectar_terminais_componentes(data, 300, 240)
        pins = result["components"][0]["terminals"]
        self.assertEqual([p["pin"] for p in pins], ["+", "-", "OUT"])
        self.assertLess(pins[0]["position"]["y"], pins[1]["position"]["y"])

    def test_transistor_without_image_or_model_is_not_invented(self):
        data = self._input(
            "transistor_npn",
            {"x1": 10, "y1": 10, "x2": 110, "y2": 80, "width": 100, "height": 70},
            "Q1",
        )
        result = detectar_terminais_componentes(data, 200, 200)
        comp = result["components"][0]
        self.assertEqual(comp["terminals"], [])
        self.assertTrue(comp["terminal_detection"]["review_required"])
        self.assertTrue(comp["transistor_geometry"].get("training_required"))


class PriorityComponentConnectionTests(unittest.TestCase):
    """Valida os símbolos que estão sendo usados nos testes atuais do TC."""

    def _detect_drawn_symbol(self, component_type, component_id):
        import random
        from backend.dataset_generator import _draw_symbol, _bbox_from_mask

        gray, symbol_mask = _draw_symbol(component_type, 256, random.Random(4411))
        bbox_raw = _bbox_from_mask(symbol_mask)
        self.assertIsNotNone(bbox_raw)
        x1, y1, x2, y2 = bbox_raw
        bbox = {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": x2 - x1, "height": y2 - y1,
        }
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        data = {
            "schema_version": "stage6-1.0",
            "components": [{
                "component_id": component_id,
                "detection_id": "DET_001",
                "type": component_type,
                "bbox": bbox,
                "review_required": False,
                "review_reasons": [],
            }],
        }
        result = detectar_terminais_componentes(data, 256, 256, image=image)
        return result["components"][0]

    def test_priority_resistor_connections(self):
        comp = self._detect_drawn_symbol("resistor", "R1")
        pins = comp["terminals"]
        self.assertEqual([p["pin"] for p in pins], ["1", "2"])
        self.assertLess(pins[0]["position"]["x"], pins[1]["position"]["x"])
        self.assertEqual([p["electrical_role"] for p in pins], ["terminal_1", "terminal_2"])

    def test_priority_capacitor_connections(self):
        comp = self._detect_drawn_symbol("capacitor", "C1")
        pins = comp["terminals"]
        self.assertEqual([p["pin"] for p in pins], ["1", "2"])
        self.assertLess(pins[0]["position"]["x"], pins[1]["position"]["x"])

    def test_priority_diode_anode_cathode(self):
        comp = self._detect_drawn_symbol("diode", "D1")
        pins = comp["terminals"]
        self.assertEqual([p["pin"] for p in pins], ["A", "K"])
        self.assertEqual([p["electrical_role"] for p in pins], ["anode", "cathode"])
        self.assertLess(pins[0]["position"]["x"], pins[1]["position"]["x"])

    def test_priority_led_anode_cathode(self):
        comp = self._detect_drawn_symbol("led", "LED1")
        pins = comp["terminals"]
        self.assertEqual([p["pin"] for p in pins], ["A", "K"])
        self.assertLess(pins[0]["position"]["x"], pins[1]["position"]["x"])

    def test_priority_gnd_connection_is_top(self):
        comp = self._detect_drawn_symbol("gnd", "GND1")
        pin = comp["terminals"][0]
        self.assertEqual(pin["pin"], "GND")
        self.assertEqual(pin["electrical_role"], "ground")
        self.assertEqual(pin["terminal_side"], "top")

    def test_priority_vcc_connection_is_bottom(self):
        comp = self._detect_drawn_symbol("vcc", "VCC1")
        pin = comp["terminals"][0]
        self.assertEqual(pin["pin"], "VCC")
        self.assertEqual(pin["electrical_role"], "power")
        self.assertEqual(pin["terminal_side"], "bottom")

    def _assert_transistor_pins(self, component_type):
        comp = self._detect_drawn_symbol(component_type, "Q1")
        pins = {p["pin"]: p for p in comp["terminals"]}
        self.assertEqual(set(pins), {"B", "C", "E"})
        self.assertEqual(pins["B"]["electrical_role"], "base")
        self.assertEqual(pins["C"]["electrical_role"], "collector")
        self.assertEqual(pins["E"]["electrical_role"], "emitter")
        # Nova regra: B/C/E devem cair SOBRE a forquilha do transistor,
        # e não nas bordas da bounding box do YOLO.
        self.assertLess(pins["B"]["position"]["x"], pins["C"]["position"]["x"])
        self.assertLess(pins["C"]["position"]["y"], pins["E"]["position"]["y"])
        self.assertLess(pins["C"]["position"]["x"], 200)
        self.assertLess(pins["E"]["position"]["x"], 200)
        self.assertGreater(pins["C"]["position"]["x"], 120)
        self.assertGreater(pins["E"]["position"]["x"], 120)
        self.assertEqual(pins["B"]["terminal_side"], "fork_base")
        self.assertEqual(pins["C"]["terminal_side"], "fork_collector")
        self.assertEqual(pins["E"]["terminal_side"], "fork_emitter")
        self.assertTrue(pins["B"]["source"].startswith("transistor_fork"))
        self.assertTrue(comp["transistor_geometry"]["source"].startswith("transistor_fork"))

    def test_priority_npn_bce(self):
        comp = self._detect_drawn_symbol("transistor_npn", "Q1")
        self._assert_transistor_pins("transistor_npn")
        self.assertEqual(comp["transistor_geometry"]["transistor_type"], "NPN")

    def test_priority_pnp_bce_or_safe_review(self):
        comp = self._detect_drawn_symbol("transistor_pnp", "Q2")
        self.assertEqual(comp["transistor_geometry"]["transistor_type"], "PNP")
        pins = {p["pin"]: p for p in comp["terminals"]}
        if pins:
            self.assertEqual(set(pins), {"B", "C", "E"})
            self.assertNotEqual(pins["C"]["position"], pins["E"]["position"])
        else:
            self.assertTrue(comp["terminal_detection"]["review_required"])
            self.assertTrue(comp["transistor_geometry"].get("training_required"))


class TransistorForkRealGeometryTests(unittest.TestCase):
    def _draw_real_style_bjt(self, fork_side="right"):
        img = np.full((260, 260, 3), 255, dtype=np.uint8)
        # Caixa propositalmente alta: reproduz o erro antigo em que o aspect
        # ratio fazia o transistor ser tratado como "vertical".
        bbox = {"x1": 70, "y1": 35, "x2": 180, "y2": 225, "width": 110, "height": 190}
        bx = 112 if fork_side == "right" else 138
        # barra da base espessa
        cv2.rectangle(img, (bx-5, 86), (bx+5, 174), (20,20,20), -1)
        # terminal da base no lado oposto à forquilha
        if fork_side == "right":
            cv2.line(img, (76,130), (bx-5,130), (20,20,20), 4, cv2.LINE_AA)
            cv2.line(img, (bx+5,108), (166,70), (20,20,20), 4, cv2.LINE_AA)
            cv2.line(img, (bx+5,152), (166,190), (20,20,20), 4, cv2.LINE_AA)
        else:
            cv2.line(img, (bx+5,130), (176,130), (20,20,20), 4, cv2.LINE_AA)
            cv2.line(img, (bx-5,108), (84,70), (20,20,20), 4, cv2.LINE_AA)
            cv2.line(img, (bx-5,152), (84,190), (20,20,20), 4, cv2.LINE_AA)
        return img, bbox

    def _run(self, component_type, fork_side):
        image, bbox = self._draw_real_style_bjt(fork_side)
        data = {
            "schema_version": "stage6-1.0",
            "components": [{
                "component_id": "Q1",
                "detection_id": "DET_001",
                "type": component_type,
                "bbox": bbox,
                "review_required": False,
                "review_reasons": [],
            }],
        }
        return detectar_terminais_componentes(data, 260, 260, image=image)["components"][0]

    def test_tall_bbox_does_not_rotate_real_bjt_right_fork(self):
        comp = self._run("transistor_npn", "right")
        self.assertEqual(comp["orientation"]["axis"], "horizontal")
        self.assertEqual(comp["transistor_geometry"].get("fork_side"), "right")
        self.assertTrue(comp["transistor_geometry"].get("branch_distinct"))
        pins = {p["pin"]: p for p in comp["terminals"]}
        if pins:
            self.assertAlmostEqual(pins["B"]["position"]["y"], 130, delta=12)
            self.assertNotEqual(pins["C"]["position"], pins["E"]["position"])
        else:
            # Este desenho não contém seta: a política atual exige treino/revisão
            # em vez de inventar qual ramo é emissor.
            self.assertTrue(comp["terminal_detection"]["review_required"])
            self.assertTrue(comp["transistor_geometry"].get("training_required"))

    def test_mirrored_pnp_fork_left_is_detected(self):
        comp = self._run("transistor_pnp", "left")
        pins = {p["pin"]: p for p in comp["terminals"]}
        self.assertEqual(comp["orientation"]["axis"], "horizontal")
        self.assertEqual(comp["transistor_geometry"].get("fork_side"), "left")
        self.assertLess(pins["C"]["position"]["x"], pins["B"]["position"]["x"])
        self.assertLess(pins["E"]["position"]["x"], pins["B"]["position"]["x"])
        self.assertLess(pins["C"]["position"]["y"], pins["E"]["position"]["y"])




class TightHandDrawnRegressionTests(unittest.TestCase):
    def test_led_tight_bbox_with_page_line_stays_vertical(self):
        img = np.full((180, 180, 3), 225, np.uint8)
        # pauta horizontal clara
        cv2.line(img, (0, 90), (179, 90), (175, 175, 175), 1)
        # LED vertical: terminais, triângulo/diagonais e barra horizontal
        cv2.line(img, (82, 35), (82, 70), (25, 25, 25), 3)
        cv2.line(img, (68, 70), (82, 88), (25, 25, 25), 3)
        cv2.line(img, (96, 70), (82, 88), (25, 25, 25), 3)
        cv2.line(img, (66, 91), (98, 91), (25, 25, 25), 3)
        cv2.line(img, (82, 91), (82, 132), (25, 25, 25), 3)
        # setas de emissão à direita
        cv2.line(img, (100, 68), (116, 56), (25, 25, 25), 2)
        cv2.line(img, (104, 80), (122, 68), (25, 25, 25), 2)
        info = {
            "schema_version": "test",
            "components": [{
                "component_id": "LED_TIGHT", "type": "led",
                # bbox deliberadamente apertada e mais larga por causa das setas
                "bbox": {"x1": 62, "y1": 66, "x2": 126, "y2": 108, "width": 64, "height": 42},
            }],
        }
        out = detectar_terminais_componentes(info, 180, 180, img)
        c = out["components"][0]
        self.assertEqual(c["orientation"]["axis"], "vertical")
        pins = {t["pin"]: t["position"] for t in c["terminals"]}
        self.assertLess(pins["A"]["y"], pins["K"]["y"])
        self.assertLess(abs(pins["A"]["x"] - pins["K"]["x"]), 18)

    def test_pnp_tight_bbox_never_publishes_collapsed_ce(self):
        img = np.full((200, 200, 3), 240, np.uint8)
        cv2.line(img, (90, 65), (90, 135), (20, 20, 20), 4)
        cv2.line(img, (45, 100), (90, 100), (20, 20, 20), 3)
        cv2.line(img, (90, 85), (135, 48), (20, 20, 20), 3)
        cv2.line(img, (90, 115), (135, 152), (20, 20, 20), 3)
        cv2.line(img, (118, 137), (108, 128), (20, 20, 20), 2)
        cv2.line(img, (108, 128), (110, 140), (20, 20, 20), 2)
        info = {
            "schema_version": "test",
            "components": [{
                "component_id": "Q_TIGHT", "type": "transistor_pnp",
                "bbox": {"x1": 78, "y1": 78, "x2": 112, "y2": 126, "width": 34, "height": 48},
            }],
        }
        out = detectar_terminais_componentes(info, 200, 200, img)
        c = out["components"][0]
        pins = {t["pin"]: t["position"] for t in c["terminals"]}
        if pins:
            self.assertNotEqual(pins["C"], pins["E"])
            self.assertGreater(np.hypot(pins["C"]["x"]-pins["E"]["x"], pins["C"]["y"]-pins["E"]["y"]), 8)
        else:
            self.assertTrue(c["terminal_detection"]["review_required"])
            self.assertTrue(c["transistor_geometry"].get("training_required"))


if __name__ == "__main__":
    unittest.main()

class LedElectricalAxisRegressionTests(unittest.TestCase):
    def test_vertical_led_ignores_emission_arrows_that_widen_bbox(self):
        image = np.full((240, 260, 3), 235, dtype=np.uint8)
        bbox = {"x1": 35, "y1": 30, "x2": 225, "y2": 210, "width": 190, "height": 180}
        x = 105
        cv2.line(image, (x, 30), (x, 82), (25, 25, 25), 4, cv2.LINE_AA)
        cv2.line(image, (x, 142), (x, 210), (25, 25, 25), 4, cv2.LINE_AA)
        pts = np.array([[78, 82], [132, 82], [105, 130]], dtype=np.int32)
        cv2.polylines(image, [pts.reshape((-1,1,2))], True, (25,25,25), 4, cv2.LINE_AA)
        cv2.line(image, (76, 136), (134, 136), (25,25,25), 4, cv2.LINE_AA)
        cv2.arrowedLine(image, (142, 105), (190, 70), (25,25,25), 3, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(image, (150, 125), (205, 88), (25,25,25), 3, cv2.LINE_AA, tipLength=0.25)
        data = {"schema_version":"stage6-1.0","components":[{"component_id":"LED1","detection_id":"DET_LED","type":"led","bbox":bbox,"review_required":False,"review_reasons":[]}]}
        result = detectar_terminais_componentes(data, 260, 240, image=image)
        comp = result["components"][0]
        pins = {p["pin"]: p for p in comp["terminals"]}
        self.assertEqual(comp["orientation"]["axis"], "vertical")
        self.assertIn(comp["orientation"]["source"], {"opposite_border_electrical_axis", "opposite_border_alignment", "opposite_collinear_wire_pair", "capacitor_parallel_plates"})
        self.assertLess(pins["A"]["position"]["y"], pins["K"]["position"]["y"])
        self.assertAlmostEqual(pins["A"]["position"]["x"], x, delta=12)
        self.assertAlmostEqual(pins["K"]["position"]["x"], x, delta=12)


class LedPhotoContaminationRegressionTests(unittest.TestCase):
    def test_vertical_led_body_wins_over_false_horizontal_border_noise(self):
        image = np.full((260, 300, 3), 235, dtype=np.uint8)
        bbox = {"x1": 30, "y1": 25, "x2": 260, "y2": 235, "width": 230, "height": 210}
        x = 118
        cv2.line(image, (x, 52), (x, 92), (25,25,25), 4, cv2.LINE_AA)
        cv2.line(image, (x, 155), (x, 205), (25,25,25), 4, cv2.LINE_AA)
        pts = np.array([[88,94],[148,94],[118,140]], dtype=np.int32)
        cv2.polylines(image, [pts.reshape((-1,1,2))], True, (25,25,25), 4, cv2.LINE_AA)
        cv2.line(image, (85,148), (151,148), (25,25,25), 4, cv2.LINE_AA)
        cv2.arrowedLine(image, (155,118), (205,80), (25,25,25), 3, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(image, (162,140), (220,98), (25,25,25), 3, cv2.LINE_AA, tipLength=0.25)
        # Ruído/texto encostando nas duas laterais da bbox: reproduz o falso
        # eixo horizontal observado em fotos.
        cv2.line(image, (30,182), (70,182), (25,25,25), 5, cv2.LINE_AA)
        cv2.line(image, (220,182), (260,182), (25,25,25), 5, cv2.LINE_AA)
        data = {"schema_version":"stage6-1.0","components":[{
            "component_id":"LED1","detection_id":"DET_LED","type":"led",
            "bbox":bbox,"review_required":False,"review_reasons":[]
        }]}
        comp = detectar_terminais_componentes(data, 300, 260, image=image)["components"][0]
        pins = {p["pin"]: p for p in comp["terminals"]}
        self.assertEqual(comp["orientation"]["axis"], "vertical")
        self.assertIn(comp["orientation"]["source"], {"led_body_rectangle", "opposite_collinear_wire_pair"})
        self.assertLess(pins["A"]["position"]["y"], pins["K"]["position"]["y"])
        self.assertAlmostEqual(pins["A"]["position"]["x"], x, delta=14)
        self.assertAlmostEqual(pins["K"]["position"]["x"], x, delta=14)


class PnpBranchSeparationRegressionTests(unittest.TestCase):
    def test_pnp_collector_and_emitter_stay_on_opposite_fork_branches(self):
        image = np.full((280, 280, 3), 255, dtype=np.uint8)
        bbox = {"x1": 65, "y1": 30, "x2": 205, "y2": 245, "width": 140, "height": 215}
        bx = 118
        cv2.rectangle(image, (bx-5, 82), (bx+5, 182), (20,20,20), -1)
        cv2.line(image, (70,132), (bx-5,132), (20,20,20), 4, cv2.LINE_AA)
        # Ramo de coletor (superior) e emissor (inferior).
        cv2.line(image, (bx+5,108), (184,62), (20,20,20), 4, cv2.LINE_AA)
        cv2.line(image, (bx+5,158), (184,210), (20,20,20), 4, cv2.LINE_AA)
        # Seta PNP no emissor, mais tinta próxima ao ramo inferior.
        cv2.arrowedLine(image, (170,198), (145,174), (20,20,20), 4, cv2.LINE_AA, tipLength=0.35)
        # Distrator próximo ao emissor que não pode puxar o coletor para a mesma trilha.
        cv2.line(image, (165,190), (198,216), (20,20,20), 3, cv2.LINE_AA)
        data = {"schema_version":"stage6-1.0","components":[{
            "component_id":"Q1","detection_id":"DET_Q","type":"transistor_pnp",
            "bbox":bbox,"review_required":False,"review_reasons":[]
        }]}
        comp = detectar_terminais_componentes(data, 280, 280, image=image)["components"][0]
        pins = {p["pin"]: p for p in comp["terminals"]}
        by = pins["B"]["position"]["y"]
        cy = pins["C"]["position"]["y"]
        ey = pins["E"]["position"]["y"]
        self.assertLess(cy, by)
        self.assertGreater(ey, by)
        self.assertGreater(abs(ey-cy), 35)



class CapacitorElectricalAxisRegressionTests(unittest.TestCase):
    def test_vertical_capacitor_uses_wire_axis_not_wide_bbox(self):
        # Reproduz o caso real: duas placas horizontais deixam a bbox larga,
        # mas os terminais elétricos entram por cima/baixo.
        image = np.full((240, 280, 3), 235, dtype=np.uint8)
        bbox = {"x1": 35, "y1": 25, "x2": 235, "y2": 215, "width": 200, "height": 190}
        x = 128
        # fios verticais
        cv2.line(image, (x, 25), (x, 92), (25,25,25), 4, cv2.LINE_AA)
        cv2.line(image, (x, 148), (x, 215), (25,25,25), 4, cv2.LINE_AA)
        # placas horizontais do capacitor
        cv2.line(image, (82, 104), (174, 104), (25,25,25), 5, cv2.LINE_AA)
        cv2.line(image, (82, 136), (174, 136), (25,25,25), 5, cv2.LINE_AA)
        # ruído/desenho lateral que alarga visualmente o símbolo
        cv2.line(image, (185, 118), (220, 118), (80,80,80), 2, cv2.LINE_AA)
        data = {"schema_version":"stage6-1.0","components":[{
            "component_id":"C1","detection_id":"DET_C","type":"capacitor",
            "bbox":bbox,"review_required":False,"review_reasons":[]
        }]}
        result = detectar_terminais_componentes(data, 280, 240, image=image)
        comp = result["components"][0]
        pins = comp["terminals"]
        self.assertEqual(comp["orientation"]["axis"], "vertical")
        self.assertIn(comp["orientation"]["source"], {"opposite_border_electrical_axis", "opposite_border_alignment", "opposite_collinear_wire_pair", "capacitor_parallel_plates"})
        self.assertLess(pins[0]["position"]["y"], pins[1]["position"]["y"])
        self.assertAlmostEqual(pins[0]["position"]["x"], x, delta=12)
        self.assertAlmostEqual(pins[1]["position"]["x"], x, delta=12)
        # Não pode voltar ao erro antigo: pino esquerdo/direito por causa da bbox larga.
        self.assertLess(abs(pins[0]["position"]["x"] - pins[1]["position"]["x"]), 18)

    def test_handdrawn_vertical_capacitor_wide_tight_bbox_uses_parallel_plates(self):
        image = np.full((190, 190, 3), 225, dtype=np.uint8)
        # Linhas pautadas claras atravessando a região.
        cv2.line(image, (0, 74), (189, 74), (175,175,175), 1)
        cv2.line(image, (0, 112), (189, 112), (175,175,175), 1)
        x = 96
        # Fio superior/inferior e duas placas horizontais curtas desenhadas à mão.
        cv2.line(image, (x, 42), (x, 82), (35,35,35), 3, cv2.LINE_AA)
        cv2.line(image, (78, 87), (112, 85), (35,35,35), 3, cv2.LINE_AA)
        cv2.line(image, (79, 96), (113, 94), (35,35,35), 3, cv2.LINE_AA)
        cv2.line(image, (x+1, 96), (x+2, 142), (35,35,35), 3, cv2.LINE_AA)
        # Bbox propositalmente mais larga que alta: reproduz o erro real.
        bbox = {"x1": 70, "y1": 76, "x2": 122, "y2": 105, "width": 52, "height": 29}
        data = {"schema_version":"stage6-1.0","components":[{
            "component_id":"C_HAND","detection_id":"DET_CH","type":"capacitor",
            "bbox":bbox,"review_required":False,"review_reasons":[]
        }]}
        result = detectar_terminais_componentes(data, 190, 190, image=image)
        comp = result["components"][0]
        pins = comp["terminals"]
        self.assertEqual(comp["orientation"]["axis"], "vertical")
        self.assertEqual(comp["orientation"]["source"], "capacitor_parallel_plates")
        self.assertLess(pins[0]["position"]["y"], pins[1]["position"]["y"])
        self.assertLess(abs(pins[0]["position"]["x"] - pins[1]["position"]["x"]), 14)

    def test_horizontal_capacitor_still_uses_left_right_wires(self):
        image = np.full((260, 260, 3), 240, dtype=np.uint8)
        bbox = {"x1": 25, "y1": 35, "x2": 235, "y2": 225, "width": 210, "height": 190}
        y = 126
        cv2.line(image, (25, y), (92, y), (20,20,20), 4, cv2.LINE_AA)
        cv2.line(image, (150, y), (235, y), (20,20,20), 4, cv2.LINE_AA)
        cv2.line(image, (104, 82), (104, 170), (20,20,20), 5, cv2.LINE_AA)
        cv2.line(image, (138, 82), (138, 170), (20,20,20), 5, cv2.LINE_AA)
        data = {"schema_version":"stage6-1.0","components":[{
            "component_id":"C2","detection_id":"DET_C2","type":"capacitor",
            "bbox":bbox,"review_required":False,"review_reasons":[]
        }]}
        result = detectar_terminais_componentes(data, 260, 260, image=image)
        comp = result["components"][0]
        pins = comp["terminals"]
        self.assertEqual(comp["orientation"]["axis"], "horizontal")
        self.assertLess(pins[0]["position"]["x"], pins[1]["position"]["x"])
        self.assertAlmostEqual(pins[0]["position"]["y"], y, delta=12)
        self.assertAlmostEqual(pins[1]["position"]["y"], y, delta=12)


class ResistorBodyAndSnapRegressionTests(unittest.TestCase):
    def _run(self, image, bbox, components=None):
        if components is None:
            components = [{
                "component_id": "R1",
                "detection_id": "DET_001",
                "type": "resistor",
                "bbox": bbox,
                "review_required": False,
                "review_reasons": [],
            }]
        data = {"schema_version": "stage6-1.0", "components": components}
        return detectar_terminais_componentes(data, image.shape[1], image.shape[0], image=image)

    def test_vertical_resistor_body_overrides_wide_yolo_bbox(self):
        image = np.full((230, 230, 3), 255, dtype=np.uint8)
        # Corpo real vertical, mas bbox propositalmente mais larga que alta.
        cv2.rectangle(image, (102, 82), (126, 148), (25, 25, 25), 3)
        cv2.line(image, (114, 52), (114, 82), (25, 25, 25), 3)
        cv2.line(image, (114, 148), (114, 178), (25, 25, 25), 3)
        bbox = {"x1": 38, "y1": 48, "x2": 190, "y2": 182, "width": 152, "height": 134}
        result = self._run(image, bbox)
        comp = result["components"][0]
        self.assertEqual(comp["orientation"]["axis"], "vertical")
        self.assertEqual(comp["orientation"]["source"], "resistor_body_rectangle")
        p1, p2 = comp["terminals"]
        self.assertAlmostEqual(p1["position"]["x"], 114, delta=5)
        self.assertAlmostEqual(p2["position"]["x"], 114, delta=5)
        self.assertLess(p1["position"]["y"], p2["position"]["y"])

    def test_resistor_terminal_centers_on_nearby_connection_dot(self):
        image = np.full((230, 230, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (102, 84), (126, 148), (25, 25, 25), 3)
        cv2.line(image, (114, 55), (114, 84), (25, 25, 25), 3)
        cv2.circle(image, (114, 50), 7, (25, 25, 25), -1)
        cv2.line(image, (114, 148), (114, 180), (25, 25, 25), 3)
        bbox = {"x1": 70, "y1": 43, "x2": 160, "y2": 182, "width": 90, "height": 139}
        result = self._run(image, bbox)
        comp = result["components"][0]
        top = comp["terminals"][0]
        self.assertAlmostEqual(top["position"]["x"], 114, delta=3)
        self.assertAlmostEqual(top["position"]["y"], 50, delta=4)
        self.assertTrue(top.get("connection_snap", {}).get("applied"))
        self.assertIn(top.get("connection_snap", {}).get("mode"), {"junction_center", "wire_centerline"})

    def test_close_terminals_remain_inside_their_own_separate_boxes(self):
        image = np.full((180, 260, 3), 255, dtype=np.uint8)
        # Dois resistores horizontais com um trecho curto comum entre as bboxes.
        cv2.rectangle(image, (45, 78), (95, 102), (25, 25, 25), 3)
        cv2.line(image, (20, 90), (45, 90), (25, 25, 25), 3)
        cv2.line(image, (95, 90), (108, 90), (25, 25, 25), 3)
        cv2.rectangle(image, (112, 78), (162, 102), (25, 25, 25), 3)
        cv2.line(image, (108, 90), (112, 90), (25, 25, 25), 3)
        cv2.line(image, (162, 90), (190, 90), (25, 25, 25), 3)
        components = [
            {"component_id":"R1","detection_id":"D1","type":"resistor","bbox":{"x1":20,"y1":65,"x2":106,"y2":115,"width":86,"height":50},"review_required":False,"review_reasons":[]},
            {"component_id":"R2","detection_id":"D2","type":"resistor","bbox":{"x1":110,"y1":65,"x2":190,"y2":115,"width":80,"height":50},"review_required":False,"review_reasons":[]},
        ]
        result = self._run(image, components[0]["bbox"], components=components)
        r1 = result["components"][0]["terminals"][1]["position"]
        r2 = result["components"][1]["terminals"][0]["position"]
        self.assertLess(r1["x"],106)
        self.assertGreater(r2["x"],110)
        self.assertAlmostEqual(r1["y"], r2["y"], delta=0.01)
        self.assertEqual(result["nearby_terminal_snap_count"], 0)

class RotationInvariantStage7RegressionTests(unittest.TestCase):
    def test_emitter_anchor_recovers_base_and_collector_at_many_angles(self):
        from backend.terminal_detection import _refine_bc_from_emitter_anchor
        for angle in (0, 30, 60, 90, 120, 150, 180, 225, 270, 315):
            with self.subTest(angle=angle):
                img = np.full((260, 260, 3), 245, np.uint8)
                b = np.array([130.0, 130.0]); c = np.array([195.0, 75.0]); e = np.array([195.0, 185.0])
                cv2.line(img, tuple(b.astype(int)), tuple(c.astype(int)), (10,10,10), 4)
                cv2.line(img, tuple(b.astype(int)), tuple(e.astype(int)), (10,10,10), 4)
                cv2.line(img, (70,130), tuple(b.astype(int)), (10,10,10), 4)
                M = cv2.getRotationMatrix2D((130,130), angle, 1.0)
                img = cv2.warpAffine(img, M, (260,260), borderValue=(245,245,245))
                def tr(p):
                    return np.array([M[0,0]*p[0]+M[0,1]*p[1]+M[0,2], M[1,0]*p[0]+M[1,1]*p[1]+M[1,2]])
                bb, cc, ee = tr(b), tr(c), tr(e)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); ys, xs = np.nonzero(gray < 180)
                bbox = {"x1":float(xs.min()-4),"y1":float(ys.min()-4),"x2":float(xs.max()+4),"y2":float(ys.max()+4),"width":float(xs.max()-xs.min()+8),"height":float(ys.max()-ys.min()+8)}
                out, meta = _refine_bc_from_emitter_anchor(img, bbox, tuple(ee), 260, 260)
                self.assertTrue(meta.get("success"))
                self.assertLess(float(np.linalg.norm(np.array(out["B"]) - bb)), 9.0)
                self.assertLess(float(np.linalg.norm(np.array(out["C"]) - cc)), 7.0)

    def test_rotated_resistor_uses_real_long_axis(self):
        img = np.full((240,240,3), 245, np.uint8)
        box = cv2.boxPoints(((120,120),(32,92),27)).astype(int)
        cv2.drawContours(img,[box],0,(20,20,20),3)
        # leads along long-axis direction
        theta = np.deg2rad(27+90)
        u = np.array([np.cos(theta),np.sin(theta)])
        p1=np.array([120.,120.])-u*75; p2=np.array([120.,120.])+u*75
        cv2.line(img,tuple(np.round(np.array([120.,120.])-u*46).astype(int)),tuple(np.round(p1).astype(int)),(20,20,20),3)
        cv2.line(img,tuple(np.round(np.array([120.,120.])+u*46).astype(int)),tuple(np.round(p2).astype(int)),(20,20,20),3)
        bbox={"x1":45,"y1":40,"x2":195,"y2":200,"width":150,"height":160}
        info={"schema_version":"test","components":[{"component_id":"RROT","type":"resistor","bbox":bbox}]}
        out=detectar_terminais_componentes(info,240,240,img)
        pins=out["components"][0]["terminals"]
        self.assertGreater(np.linalg.norm(np.array([pins[1]["position"]["x"]-pins[0]["position"]["x"], pins[1]["position"]["y"]-pins[0]["position"]["y"]])), 80)

    def test_rotated_capacitor_uses_parallel_plates(self):
        img=np.full((240,240,3),245,np.uint8)
        center=np.array([120.,120.]); plate_angle=np.deg2rad(28); u=np.array([np.cos(plate_angle),np.sin(plate_angle)]); n=np.array([-u[1],u[0]])
        for off in (-10,10):
            a=center+n*off-u*28; b=center+n*off+u*28
            cv2.line(img,tuple(np.round(a).astype(int)),tuple(np.round(b).astype(int)),(20,20,20),3)
        cv2.line(img,tuple(np.round(center-n*10).astype(int)),tuple(np.round(center-n*65).astype(int)),(20,20,20),3)
        cv2.line(img,tuple(np.round(center+n*10).astype(int)),tuple(np.round(center+n*65).astype(int)),(20,20,20),3)
        bbox={"x1":65,"y1":60,"x2":175,"y2":180,"width":110,"height":120}
        info={"schema_version":"test","components":[{"component_id":"CROT","type":"capacitor","bbox":bbox}]}
        out=detectar_terminais_componentes(info,240,240,img)
        comp=out["components"][0]; pins=comp["terminals"]
        self.assertIn(comp["orientation"]["source"], {"capacitor_parallel_plates_any_angle","capacitor_parallel_plates"})
        self.assertGreater(np.linalg.norm(np.array([pins[1]["position"]["x"]-pins[0]["position"]["x"], pins[1]["position"]["y"]-pins[0]["position"]["y"]])), 70)

class AnyAngleBaseBarRegressionTests(unittest.TestCase):
    def test_base_collector_from_emitter_anchor_at_many_angles(self):
        from backend.terminal_detection import _refine_bc_from_emitter_basebar_any_angle
        for angle in (0,30,60,90,120,150,180,225,270,315):
            with self.subTest(angle=angle):
                img=np.full((280,280,3),245,np.uint8)
                B=np.array([130.,135.]); C=np.array([190.,75.]); E=np.array([190.,195.])
                btop=np.array([130.,95.]); bbot=np.array([130.,175.]); lead=np.array([65.,135.])
                cv2.line(img,tuple(btop.astype(int)),tuple(bbot.astype(int)),(15,15,15),4)
                cv2.line(img,tuple(lead.astype(int)),tuple(B.astype(int)),(15,15,15),4)
                cv2.line(img,(130,105),tuple(C.astype(int)),(15,15,15),4)
                cv2.line(img,(130,165),tuple(E.astype(int)),(15,15,15),4)
                M=cv2.getRotationMatrix2D((130,135),angle,1.0)
                img=cv2.warpAffine(img,M,(280,280),borderValue=(245,245,245))
                def tr(p):
                    return np.array([M[0,0]*p[0]+M[0,1]*p[1]+M[0,2],M[1,0]*p[0]+M[1,1]*p[1]+M[1,2]])
                bb,cc,ee=tr(B),tr(C),tr(E)
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);ys,xs=np.nonzero(gray<180)
                bbox={"x1":float(xs.min()-10),"y1":float(ys.min()-10),"x2":float(xs.max()+10),"y2":float(ys.max()+10)}
                bbox["width"]=bbox["x2"]-bbox["x1"];bbox["height"]=bbox["y2"]-bbox["y1"]
                out,meta=_refine_bc_from_emitter_basebar_any_angle(img,bbox,tuple(ee),280,280)
                self.assertTrue(meta.get("success"))
                self.assertLess(float(np.linalg.norm(np.array(out["B"])-bb)),6.0)
                self.assertLess(float(np.linalg.norm(np.array(out["C"])-cc)),6.0)
