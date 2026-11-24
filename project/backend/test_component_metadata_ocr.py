import unittest
from pathlib import Path

from backend.component_metadata import (
    DEFAULT_COMPONENT_PROFILES,
    build_component_metadata,
    model_classes_for_text,
    parse_hfe_text,
)
from backend.ocr import (
    _combinar_tokens_eletricos,
    _limpar_texto,
    aplicar_correcoes_manuais,
    classificar_token,
    criar_componentes_logicos,
)


def _bbox(x, y, width=24, height=12):
    return {
        "x1": float(x),
        "y1": float(y),
        "x2": float(x + width),
        "y2": float(y + height),
        "width": float(width),
        "height": float(height),
        "center_x": float(x + width / 2),
        "center_y": float(y + height / 2),
    }


def _token(token_id, text, x, y, confidence=0.92):
    return {
        "token_id": token_id,
        "text": text,
        "confidence": confidence,
        "bbox": _bbox(x, y),
        "engine": "easyocr",
        "variant": "test_handwriting",
    }


class ComponentMetadataAndOcrTests(unittest.TestCase):
    def test_hfe_accepts_case_beta_word_and_greek_beta(self):
        self.assertEqual(parse_hfe_text("hFE=220"), 220.0)
        self.assertEqual(parse_hfe_text("HfE : 135"), 135.0)
        self.assertEqual(parse_hfe_text("beta=98"), 98.0)
        self.assertEqual(parse_hfe_text("β = 310"), 310.0)
        self.assertEqual(_limpar_texto("β = 310"), "β=310")
        self.assertEqual(parse_hfe_text("hFE=2OO"), 200.0)
        self.assertEqual(parse_hfe_text("beta=I8O"), 180.0)
        self.assertEqual(parse_hfe_text("B=150"), 150.0)
        self.assertIsNone(parse_hfe_text("B150"))

    def test_plain_number_never_becomes_transistor_gain(self):
        self.assertIsNone(parse_hfe_text("220"))
        self.assertIsNone(parse_hfe_text("R2 220"))
        self.assertNotEqual(classificar_token("220"), "hfe")

    def test_model_names_are_case_insensitive(self):
        self.assertIn("transistor_npn", model_classes_for_text("bc547"))
        self.assertIn("mosfet_n", model_classes_for_text("irf540n"))
        self.assertIn("opamp", model_classes_for_text("lm358"))
        self.assertIn("diode", model_classes_for_text("1n4148"))
        self.assertIn("transistor_npn", model_classes_for_text("bcS47"))
        self.assertIn("mosfet_n", model_classes_for_text("irf54On"))
        self.assertEqual(model_classes_for_text("BC557"), {"transistor_pnp"})
        self.assertEqual(model_classes_for_text("2N7000"), {"mosfet_n"})
        self.assertEqual(classificar_token("Lm358"), "model")

    def test_requested_classes_always_receive_common_generic_model(self):
        for component_type in (
            "diode",
            "transistor_npn",
            "transistor_pnp",
            "mosfet_n",
            "mosfet_p",
            "opamp",
        ):
            with self.subTest(component_type=component_type):
                metadata = build_component_metadata(component_type)
                self.assertEqual(
                    metadata["model"]["text"],
                    DEFAULT_COMPONENT_PROFILES[component_type]["model"],
                )
                self.assertEqual(metadata["model"]["source"], "generic_default")
                self.assertTrue(metadata["model"]["assumed"])

    def test_ocr_assigns_model_and_explicit_hfe_to_nearest_transistor(self):
        detection = {
            "detection_id": "DET_Q1",
            "class": "transistor_npn",
            "confidence": 0.94,
            "bbox": _bbox(100, 100, 60, 60),
            "review_required": False,
        }
        tokens = [
            _token("T1", "q1", 75, 82),
            _token("T2", "Bc547", 166, 104),
            _token("T3", "β=225", 166, 125),
        ]
        component = criar_componentes_logicos(tokens, [detection])["components"][0]
        self.assertEqual(component["component_id"], "Q1")
        self.assertEqual(component["model"]["text"], "BC547")
        self.assertEqual(component["model"]["source"], "ocr")
        self.assertEqual(component["parameters"]["hfe"]["value"], 225.0)
        self.assertEqual(component["parameters"]["hfe"]["source"], "ocr")

    def test_split_bc547_and_hfe_are_rejoined_only_when_valid(self):
        tokens = [
            _token("A", "BC", 10, 10),
            _token("B", "547", 36, 10),
            _token("C", "hFE=", 10, 40),
            _token("D", "180", 48, 40),
        ]
        combined = _combinar_tokens_eletricos(tokens)
        texts = {item["text"] for item in combined}
        self.assertIn("BC547", texts)
        self.assertIn("hFE=180", texts)

    def test_manual_blank_model_reapplies_generic_and_hfe_is_validated(self):
        detection = {
            "detection_id": "DET_Q1",
            "class": "transistor_npn",
            "confidence": 0.9,
            "bbox": _bbox(20, 20, 50, 50),
        }
        logical = criar_componentes_logicos([], [detection])
        result = {"components": logical["components"]}
        edited = aplicar_correcoes_manuais(
            result,
            [{
                "detection_id": "DET_Q1",
                "reference": "Q1",
                "value": "",
                "model": "",
                "parameters": {"hfe": "275"},
            }],
            {"detections": [detection]},
        )
        component = edited["components"][0]
        self.assertEqual(component["model"]["text"], "BC547")
        self.assertEqual(component["model"]["source"], "generic_default")
        self.assertEqual(component["parameters"]["hfe"]["value"], 275.0)
        self.assertEqual(component["parameters"]["hfe"]["source"], "manual")

        with self.assertRaisesRegex(ValueError, "hFE/β"):
            aplicar_correcoes_manuais(
                {"components": logical["components"]},
                [{
                    "detection_id": "DET_Q1",
                    "reference": "Q1",
                    "value": "",
                    "model": "BC547",
                    "parameters": {"hfe": "5000"},
                }],
                {"detections": [detection]},
            )

    def test_editor_is_available_without_swagger(self):
        source = (Path(__file__).parent / "main.py").read_text(encoding="utf-8")
        self.assertIn("/component-properties", source)
        self.assertIn("Editar modelos e valores", source)
        self.assertIn("hFE=", source)
        self.assertIn("β=", source)


if __name__ == "__main__":
    unittest.main()
