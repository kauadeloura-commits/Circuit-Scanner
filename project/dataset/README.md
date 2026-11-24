# Dataset — Etapa 4

Esta pasta usa o formato de detecção do YOLO:

- `images/train`, `images/val`, `images/test`
- `labels/train`, `labels/val`, `labels/test`
- um arquivo `.txt` para cada imagem
- cada linha do rótulo: `class_id center_x center_y width height`, normalizados entre 0 e 1.

`dataset.yaml` já contém as 15 classes.

Para gerar o conjunto pelo terminal:

    python scripts/generate_dataset.py 60

Ou use a página `http://127.0.0.1:5500/dataset.html` com backend e frontend ligados.

Para treinamento real na Etapa 5, recomenda-se complementar os dados sintéticos com desenhos, scans e fotos reais anotados/revisados.
