from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.database import conectar, inicializar_banco


def main():
    obrigatorios = [
        "backend/ocr.py",
        "scripts/stage6_acceptance.py",
        "tests/test_stage6_ocr.py",
        "frontend/project.html",
        "frontend/js/project.js",
    ]
    faltando = [caminho for caminho in obrigatorios if not (ROOT / caminho).is_file()]
    assert not faltando, f"Arquivos faltando: {faltando}"

    inicializar_banco()
    with conectar() as conexao:
        colunas = {linha[1] for linha in conexao.execute("PRAGMA table_info(projects)")}
    assert {"ocr_filename", "ocr_info"}.issubset(colunas)

    html = (ROOT / "frontend" / "project.html").read_text(encoding="utf-8")
    js = (ROOT / "frontend" / "js" / "project.js").read_text(encoding="utf-8")
    assert "botao-ocr" in html
    assert "/ocr/components" in js
    print("TESTE OK — estrutura da Etapa 6")


if __name__ == "__main__":
    main()
