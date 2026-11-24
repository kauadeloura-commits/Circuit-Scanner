import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import backend.database as database
import backend.main as main


class StorageRecoveryTests(unittest.TestCase):
    def test_rebuilds_sqlite_index_from_existing_project_folder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            projects = root / "projects"
            project = projects / "PRJ_0042"
            project.mkdir(parents=True)
            cv2.imwrite(str(project / "original.png"), np.full((30, 40, 3), 255, np.uint8))
            (project / "detections.json").write_text(json.dumps({"count": 1, "detections": []}), encoding="utf-8")
            cv2.imwrite(str(project / "detections.png"), np.full((30, 40, 3), 255, np.uint8))
            db_path = root / "database" / "circuit_scanner.db"
            db_path.parent.mkdir(parents=True)

            with patch.object(database, "DATABASE_PATH", db_path), \
                 patch.object(main, "DATABASE_PATH", db_path), \
                 patch.object(main, "PROJECTS_DIR", projects):
                database.inicializar_banco()
                self.assertEqual(database.contar_projetos(), 0)
                info = main._recuperar_indice_projetos_do_disco()
                self.assertEqual(info["recovered"], 1)
                self.assertEqual(database.contar_projetos(), 1)
                restored = database.buscar_projeto("PRJ_0042")
                self.assertIsNotNone(restored)
                self.assertEqual(restored["stored_filename"], "original.png")
                self.assertEqual(restored["detection_filename"], "detections.png")
                self.assertEqual(restored["status"], "componentes_detectados")


    def test_removes_legacy_recuperado_suffix_from_database(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path = root / "database" / "circuit_scanner.db"
            db_path.parent.mkdir(parents=True)
            with patch.object(database, "DATABASE_PATH", db_path):
                database.inicializar_banco()
                internal = database.inserir_projeto_temporario(
                    "TMP_1", "PRJ_0099 (recuperado)", "2026-09-14T12:00:00-03:00",
                    "criado", "original.png", "original.png"
                )
                database.finalizar_projeto(internal, "PRJ_0099", "projects/PRJ_0099")
                self.assertEqual(database.limpar_nomes_recuperados(), 1)
                restored = database.buscar_projeto("PRJ_0099")
                self.assertEqual(restored["name"], "PRJ_0099")



if __name__ == "__main__":
    unittest.main()
