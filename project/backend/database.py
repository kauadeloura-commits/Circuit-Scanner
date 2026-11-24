from pathlib import Path
from contextlib import contextmanager
import os
import sqlite3

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "database"
DATABASE_DIR.mkdir(exist_ok=True)
DATABASE_PATH = Path(os.environ.get("CIRCUIT_SCANNER_DB", str(DATABASE_DIR / "circuit_scanner.db"))).resolve()
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def conectar():
    conexao = sqlite3.connect(DATABASE_PATH)
    conexao.row_factory = sqlite3.Row
    try:
        yield conexao
    finally:
        conexao.close()


def _colunas_da_tabela(conexao):
    linhas = conexao.execute("PRAGMA table_info(projects)").fetchall()
    return {linha[1] for linha in linhas}


def inicializar_banco():
    with conectar() as conexao:
        conexao.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                path TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                processed_filename TEXT,
                processing_info TEXT,
                detection_filename TEXT,
                detections_info TEXT,
                ocr_filename TEXT,
                ocr_info TEXT,
                terminals_filename TEXT,
                terminals_info TEXT,
                wires_filename TEXT,
                wires_info TEXT
            )
            """
        )

        # Migra bancos das Etapas 2, 3 ou 4.
        colunas = _colunas_da_tabela(conexao)

        if "processed_filename" not in colunas:
            conexao.execute(
                "ALTER TABLE projects ADD COLUMN processed_filename TEXT"
            )

        if "processing_info" not in colunas:
            conexao.execute(
                "ALTER TABLE projects ADD COLUMN processing_info TEXT"
            )

        if "detection_filename" not in colunas:
            conexao.execute(
                "ALTER TABLE projects ADD COLUMN detection_filename TEXT"
            )

        if "detections_info" not in colunas:
            conexao.execute(
                "ALTER TABLE projects ADD COLUMN detections_info TEXT"
            )

        if "ocr_filename" not in colunas:
            conexao.execute(
                "ALTER TABLE projects ADD COLUMN ocr_filename TEXT"
            )

        if "ocr_info" not in colunas:
            conexao.execute(
                "ALTER TABLE projects ADD COLUMN ocr_info TEXT"
            )

        if "terminals_filename" not in colunas:
            conexao.execute(
                "ALTER TABLE projects ADD COLUMN terminals_filename TEXT"
            )

        if "terminals_info" not in colunas:
            conexao.execute(
                "ALTER TABLE projects ADD COLUMN terminals_info TEXT"
            )

        if "wires_filename" not in colunas:
            conexao.execute(
                "ALTER TABLE projects ADD COLUMN wires_filename TEXT"
            )

        if "wires_info" not in colunas:
            conexao.execute(
                "ALTER TABLE projects ADD COLUMN wires_info TEXT"
            )

        conexao.commit()


def inserir_projeto_temporario(
    project_id_temporario,
    name,
    created_at,
    status,
    original_filename,
    stored_filename,
):
    with conectar() as conexao:
        cursor = conexao.execute(
            """
            INSERT INTO projects (
                project_id,
                name,
                created_at,
                status,
                path,
                original_filename,
                stored_filename,
                processed_filename,
                processing_info,
                detection_filename,
                detections_info,
                ocr_filename,
                ocr_info,
                terminals_filename,
                terminals_info,
                wires_filename,
                wires_info
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)
            """,
            (
                project_id_temporario,
                name,
                created_at,
                status,
                "",
                original_filename,
                stored_filename,
            ),
        )
        conexao.commit()
        return cursor.lastrowid


def finalizar_projeto(id_interno, project_id, path):
    with conectar() as conexao:
        conexao.execute(
            "UPDATE projects SET project_id = ?, path = ? WHERE id = ?",
            (project_id, path, id_interno),
        )
        conexao.commit()


def excluir_projeto_por_id_interno(id_interno):
    with conectar() as conexao:
        conexao.execute("DELETE FROM projects WHERE id = ?", (id_interno,))
        conexao.commit()


def atualizar_processamento(
    project_id,
    status,
    processed_filename,
    processing_info,
):
    with conectar() as conexao:
        # Reprocessar invalida qualquer detecção anterior.
        conexao.execute(
            """
            UPDATE projects
            SET status = ?,
                processed_filename = ?,
                processing_info = ?,
                detection_filename = NULL,
                detections_info = NULL,
                ocr_filename = NULL,
                ocr_info = NULL,
                terminals_filename = NULL,
                terminals_info = NULL,
                wires_filename = NULL,
                wires_info = NULL
            WHERE project_id = ?
            """,
            (
                status,
                processed_filename,
                processing_info,
                project_id,
            ),
        )
        conexao.commit()


def atualizar_deteccoes(
    project_id,
    status,
    detection_filename,
    detections_info,
):
    with conectar() as conexao:
        conexao.execute(
            """
            UPDATE projects
            SET status = ?,
                detection_filename = ?,
                detections_info = ?,
                ocr_filename = NULL,
                ocr_info = NULL,
                terminals_filename = NULL,
                terminals_info = NULL,
                wires_filename = NULL,
                wires_info = NULL
            WHERE project_id = ?
            """,
            (
                status,
                detection_filename,
                detections_info,
                project_id,
            ),
        )
        conexao.commit()


def atualizar_ocr(
    project_id,
    status,
    ocr_filename,
    ocr_info,
):
    with conectar() as conexao:
        conexao.execute(
            """
            UPDATE projects
            SET status = ?,
                ocr_filename = ?,
                ocr_info = ?,
                terminals_filename = NULL,
                terminals_info = NULL,
                wires_filename = NULL,
                wires_info = NULL
            WHERE project_id = ?
            """,
            (
                status,
                ocr_filename,
                ocr_info,
                project_id,
            ),
        )
        conexao.commit()


def atualizar_terminais(
    project_id,
    status,
    terminals_filename,
    terminals_info,
):
    with conectar() as conexao:
        conexao.execute(
            """
            UPDATE projects
            SET status = ?,
                terminals_filename = ?,
                terminals_info = ?,
                wires_filename = NULL,
                wires_info = NULL
            WHERE project_id = ?
            """,
            (status, terminals_filename, terminals_info, project_id),
        )
        conexao.commit()


def atualizar_terminais_pos_fios(
    project_id,
    status,
    terminals_filename,
    terminals_info,
):
    """Atualiza terminais refinados pela Etapa 8 sem apagar wires.* já gerados."""
    with conectar() as conexao:
        conexao.execute(
            """
            UPDATE projects
            SET status = ?,
                terminals_filename = ?,
                terminals_info = ?
            WHERE project_id = ?
            """,
            (status, terminals_filename, terminals_info, project_id),
        )
        conexao.commit()


def atualizar_fios(
    project_id,
    status,
    wires_filename,
    wires_info,
):
    with conectar() as conexao:
        conexao.execute(
            """
            UPDATE projects
            SET status = ?,
                wires_filename = ?,
                wires_info = ?
            WHERE project_id = ?
            """,
            (status, wires_filename, wires_info, project_id),
        )
        conexao.commit()


def listar_resumos_projetos():
    """Small navigation list: avoid loading vision JSON and electrical artifacts."""
    with conectar() as conexao:
        linhas=conexao.execute('SELECT project_id,name,created_at,status FROM projects ORDER BY id DESC').fetchall()
        return [dict(linha) for linha in linhas]


def listar_projetos():
    with conectar() as conexao:
        linhas = conexao.execute(
            """
            SELECT
                id,
                project_id,
                name,
                created_at,
                status,
                path,
                original_filename,
                stored_filename,
                processed_filename,
                processing_info,
                detection_filename,
                detections_info,
                ocr_filename,
                ocr_info,
                terminals_filename,
                terminals_info,
                wires_filename,
                wires_info
            FROM projects
            ORDER BY id DESC
            """
        ).fetchall()

        return [dict(linha) for linha in linhas]


def buscar_projeto(project_id):
    with conectar() as conexao:
        linha = conexao.execute(
            """
            SELECT
                id,
                project_id,
                name,
                created_at,
                status,
                path,
                original_filename,
                stored_filename,
                processed_filename,
                processing_info,
                detection_filename,
                detections_info,
                ocr_filename,
                ocr_info,
                terminals_filename,
                terminals_info,
                wires_filename,
                wires_info
            FROM projects
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()

        return None if linha is None else dict(linha)


def registrar_projeto_recuperado(
    *,
    project_id: str,
    name: str,
    created_at: str,
    status: str,
    path: str,
    original_filename: str,
    stored_filename: str,
    processed_filename: str | None = None,
    processing_info: str | None = None,
    detection_filename: str | None = None,
    detections_info: str | None = None,
    ocr_filename: str | None = None,
    ocr_info: str | None = None,
    terminals_filename: str | None = None,
    terminals_info: str | None = None,
    wires_filename: str | None = None,
    wires_info: str | None = None,
):
    """Reinsere no SQLite um projeto que continua existindo em projects/.

    Isso protege o sistema quando o arquivo de banco é removido/trocado durante
    uma atualização do backend. Nunca sobrescreve um registro já existente.
    """
    with conectar() as conexao:
        conexao.execute(
            """
            INSERT OR IGNORE INTO projects (
                project_id, name, created_at, status, path, original_filename,
                stored_filename, processed_filename, processing_info,
                detection_filename, detections_info, ocr_filename, ocr_info,
                terminals_filename, terminals_info, wires_filename, wires_info
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id, name, created_at, status, path, original_filename,
                stored_filename, processed_filename, processing_info,
                detection_filename, detections_info, ocr_filename, ocr_info,
                terminals_filename, terminals_info, wires_filename, wires_info,
            ),
        )
        conexao.commit()



def limpar_nomes_recuperados() -> int:
    """Remove o sufixo legado ' (recuperado)' dos nomes salvos no SQLite.

    Versões antigas usavam esse texto ao reconstruir o índice a partir da pasta
    projects/. O usuário não deve vê-lo no front-end nem mantê-lo no banco.
    """
    with conectar() as conexao:
        linhas = conexao.execute(
            "SELECT project_id, name FROM projects WHERE name LIKE ?",
            ("% (recuperado)",),
        ).fetchall()
        alterados = 0
        for linha in linhas:
            nome = str(linha["name"] or "")
            sufixo = " (recuperado)"
            if nome.endswith(sufixo):
                novo = nome[:-len(sufixo)].strip() or str(linha["project_id"])
                conexao.execute(
                    "UPDATE projects SET name = ? WHERE project_id = ?",
                    (novo, linha["project_id"]),
                )
                alterados += 1
        conexao.commit()
        return alterados


def contar_projetos() -> int:
    with conectar() as conexao:
        linha = conexao.execute("SELECT COUNT(*) AS total FROM projects").fetchone()
        return int(linha["total"] if linha is not None else 0)
