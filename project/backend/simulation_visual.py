from pathlib import Path
import json

HTML=(Path(__file__).resolve().parent.parent/"frontend/simulation.html").read_text(encoding="utf8")

def simulation_page(project_id=None,base=None):
    page=HTML
    if base and base.startswith('/share/'):
        page=page.replace('Suas alterações são salvas automaticamente no projeto e nos esquemáticos baixados.',
                          'Neste link compartilhado, você pode experimentar alterações nesta sessão do navegador.')
    return page.replace("__PAYLOAD__",json.dumps(dict(base=base or "/projects/"+project_id)).replace("<","\\u003c"))
