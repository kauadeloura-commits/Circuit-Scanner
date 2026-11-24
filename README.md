# Circuit Scanner

Reconhecimento de circuitos eletrônicos a partir de fotografias, com identificação de componentes, valores, terminais e conexões, revisão manual, simulação e exportação para KiCad.

O programa está na pasta `project/`. Esta cópia inclui os modelos de IA, conjuntos de dados e projetos salvos presentes na instalação original. Fotografias e resultados de análises estão em `project/projects/`.

## Executar no Windows

Abra um terminal na pasta `project` e execute:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe iniciar_scanner.py
```

Alguns scripts e registros da instalação original usam caminhos em `C:\tc2\Circuit_Scanner_Etapa5_Corrigido\project`. Confira esses caminhos se executar a cópia em outro local.

O ambiente Python instalado (`venv`), caches, cópias antigas do backend e ZIPs que duplicam pastas não são versionados. Os arquivos originais da instalação permanecem preservados.

A identificação automática pode exigir correção manual de componentes, textos e conexões antes da simulação ou exportação.
