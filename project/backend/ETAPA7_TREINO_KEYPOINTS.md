# Etapa 7 — treinamento de B/C/E dos transistores

## Onde ficam os arquivos

A pasta `backend/` não contém o modelo treinado. O projeto usa:

- `project/models/terminal_keypoints.pt`
- `project/models/terminal_keypoints_summary.json`
- `project/terminal_keypoints/raw/...`

Ao atualizar o backend, preserve `models/` e `terminal_keypoints/`.

## Corrigir um transistor

Abra:

`http://127.0.0.1:8010/projects/PRJ_XXXX/terminal-trainer`

Marque B, C e E e use **Salvar correção + amostra**.

A correção:
1. atualiza imediatamente `terminals.png`;
2. fica persistida em `projects/PRJ_XXXX/terminal_corrections.json`;
3. cria/substitui a amostra de treino.

## Conferir status

`GET /terminal-learning/status`

Os campos mais importantes são:
- `sample_count`
- `samples_by_class`
- `model_exists`
- `model_loadable`
- `model_load_error`
- `dataset_quality`
- `new_samples_since_training`

`dataset_quality.edge_clipped_keypoint_count > 0` indica que existem labels antigos possivelmente cortados nas bordas do crop.

## Diagnosticar um projeto que errou

`GET /projects/PRJ_XXXX/terminal-learning/diagnostics`

Essa rota informa para cada transistor se o modelo realmente rodou, as confianças B/C/E e o motivo de uma previsão ter sido rejeitada.

## Treinar

```bash
python -m backend.terminal_learning status
python -m backend.terminal_learning train --epochs 80 --imgsz 320 --batch 8
```

O dataset agora separa train/val por projeto para impedir que dois transistores da mesma foto contaminem os dois conjuntos.

## Política de inferência

- correção manual persistida: prioridade máxima;
- modelo treinado carregável: prioridade automática;
- modelo rejeitou a previsão: revisão manual, sem fallback geométrico;
- sem modelo carregável: heurística geométrica.

## Reparar amostras antigas sem remarcar

Se `/terminal-learning/status` mostrar `dataset_quality.edge_clipped_keypoint_count > 0`, rode:

```bash
python -m backend.terminal_learning repair
```

ou no Swagger:

`POST /terminal-learning/repair-samples`

O backend usa os pontos B/C/E já guardados nos arquivos `meta` para reconstruir os crops/labels. Isso **não** altera o `.pt` já treinado; se houver reparos, faça um novo treino depois.
