import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import main


class Stage7AssistedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_projects = main.PROJECTS_DIR
        main.PROJECTS_DIR = Path(self.tmp.name) / "projects"
        main.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        main.PROJECTS_DIR = self.old_projects
        self.tmp.cleanup()

    def test_persisted_resistor_manual_points_override_geometry(self):
        pid = "PRJ_TEST"
        pdir = main.PROJECTS_DIR / pid
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "terminal_corrections.json").write_text(json.dumps({
            "schema_version": "terminal-corrections-1.0",
            "components": [{
                "component_id": "DET_001",
                "type": "resistor",
                "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 70},
                "points": {"1": {"x": 20, "y": 8}, "2": {"x": 20, "y": 72}},
            }],
        }), encoding="utf-8")
        result = {"components": [{
            "component_id": "DET_001",
            "type": "resistor",
            "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 70},
            "terminals": [],
            "terminal_detection": {"review_required": True},
        }]}
        applied = main._apply_saved_terminal_corrections(pid, result)
        self.assertEqual(applied, 1)
        comp = result["components"][0]
        self.assertEqual([t["pin"] for t in comp["terminals"]], ["1", "2"])
        self.assertEqual(comp["terminals"][0]["position"], {"x": 20.0, "y": 8.0})
        self.assertFalse(comp["terminal_detection"]["review_required"])
        self.assertEqual(comp["terminals"][0]["source"], "manual_ground_truth_persisted")

    def test_persisted_diode_keeps_anode_cathode_names(self):
        pid = "PRJ_DIODE"
        pdir = main.PROJECTS_DIR / pid
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "terminal_corrections.json").write_text(json.dumps({
            "components": [{
                "component_id": "D1", "type": "diode",
                "bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 50},
                "points": {"A": {"x": 25, "y": 2}, "K": {"x": 25, "y": 48}},
            }]
        }), encoding="utf-8")
        result = {"components": [{
            "component_id": "D1", "type": "diode",
            "bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 50},
            "terminals": [], "terminal_detection": {"review_required": True},
        }]}
        self.assertEqual(main._apply_saved_terminal_corrections(pid, result), 1)
        pins = {t["pin"]: t["electrical_role"] for t in result["components"][0]["terminals"]}
        self.assertEqual(pins, {"A": "anode", "K": "cathode"})

    @patch("backend.main._terminal_result_for_manual_edit")
    @patch("backend.main.buscar_projeto")
    def test_deferred_transistor_does_not_block_stage8(self, buscar, terminal_result):
        buscar.return_value = {"project_id": "PRJ_Q1"}
        terminal_result.return_value = {"components": [{
            "component_id": "Q1",
            "type": "transistor_npn",
            "bbox": {"x1": 10, "y1": 10, "x2": 80, "y2": 80},
            "terminals": [],
            "review_required": True,
            "terminal_detection": {
                "status": "awaiting_stage8_wires",
                "method": "transistor_points_after_stage8_wires",
                "review_required": True,
                "review_reasons": ["aguardando_etapa8_linhas_para_BCE"],
            },
            "transistor_geometry": {"source": "awaiting_stage8_wires"},
        }]}

        readiness = main.verificar_prontidao_etapa7("PRJ_Q1")

        self.assertTrue(readiness["ready_for_stage8"])
        self.assertEqual(readiness["unresolved_count"], 0)
        self.assertEqual(readiness["deferred_transistor_count"], 1)
        self.assertEqual(readiness["deferred_to_stage8"][0]["component_id"], "Q1")

    @patch("backend.main._terminal_result_for_manual_edit")
    @patch("backend.main.buscar_projeto")
    def test_unresolved_non_transistor_does_not_block_wire_first_pipeline(self, buscar, terminal_result):
        buscar.return_value = {"project_id": "PRJ_R1"}
        terminal_result.return_value = {"components": [{
            "component_id": "R1",
            "type": "resistor",
            "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 80},
            "terminals": [],
            "terminal_detection": {"status": "review", "review_required": True},
        }]}

        readiness = main.verificar_prontidao_etapa7("PRJ_R1")

        self.assertTrue(readiness["ready_for_stage8"])
        self.assertFalse(readiness["terminals_ready_for_connectivity"])
        self.assertEqual(readiness["unresolved_count"], 1)
        self.assertEqual(readiness["deferred_transistor_count"], 0)


if __name__ == "__main__":
    unittest.main()
