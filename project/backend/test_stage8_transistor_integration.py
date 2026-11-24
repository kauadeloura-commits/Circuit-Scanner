import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend import main


class Stage8TransistorIntegrationTests(unittest.TestCase):
    def test_wires_then_refinement_rewrites_terminals_and_keeps_wires(self):
        with tempfile.TemporaryDirectory() as td:
            old_projects = main.PROJECTS_DIR
            try:
                main.PROJECTS_DIR = Path(td)
                pid = "PRJ_TEST8"
                pdir = Path(td) / pid
                pdir.mkdir(parents=True)
                from backend.test_npn_wire_contacts import fixture
                img,sk,component,expected=fixture(notebook=False)
                cv2.imwrite(str(pdir / "original.png"), img)
                component['terminals']=[{'pin':pin,'position':dict(x=270,y=270),
                    'source':'trained_terminal_keypoint_model'} for pin in ('B','C','E')]
                terminals={'components':[component]}
                project = {
                    "project_id": pid, "stored_filename":"original.png",
                    "processed_filename":None, "terminals_info":json.dumps(terminals),
                    "ocr_info":None, "detections_info":None,
                }
                fake_result={"segment_count":3,"endpoint_candidate_count":6,"junction_candidate_count":0,"nodes":[],"segments":[]}
                fake_mask=sk.copy()
                with patch.object(main, "detectar_fios", return_value=(fake_result,fake_mask,sk,np.zeros_like(sk))), \
                     patch.object(main, "atualizar_terminais"), patch.object(main, "atualizar_fios"):
                    result=main._executar_etapa8_fios(pid,project)
                self.assertTrue((pdir/"wires.json").exists())
                self.assertTrue((pdir/"terminals.json").exists())
                self.assertTrue((pdir/"transistor_wire_refinement.png").exists())
                self.assertEqual(result["transistor_wire_refinement"]["refined_count"],1)
                saved=json.loads((pdir/"terminals.json").read_text(encoding="utf-8"))
                pins={t["pin"]:t["position"] for t in saved["components"][0]["terminals"]}
                box=component['bbox']
                for pin,p in pins.items():
                    self.assertLess(np.linalg.norm(np.array([p['x'],p['y']])-expected[pin]),5)
                    self.assertTrue(box['x1']<p['x']<box['x2'] and box['y1']<p['y']<box['y2'])
            finally:
                main.PROJECTS_DIR = old_projects

if __name__ == '__main__':
    unittest.main()
