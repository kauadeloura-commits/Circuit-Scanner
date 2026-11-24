# Etapa 7 v3.9 — memória visual local de transistores

## Problema corrigido
A v3.8 só reutilizava ground truth conhecido quando a imagem completa tinha similaridade >= 98,5%. Recompressão, redimensionamento, recorte ou reupload podiam impedir a correspondência e fazer o sistema exibir `TREINAR B/C/E` para um transistor já ensinado.

## Nova política
1. Correção manual do projeto atual.
2. Ground truth de amostra conhecida pela memória visual local do transistor.
3. Modelo `terminal_keypoints.pt`.
4. Revisão manual.
5. Heurística apenas quando não há modelo carregável.

A memória local compara contexto apertado e contexto amplo da bbox com os crops salvos em `terminal_keypoints/raw/images`, somente dentro da mesma classe NPN/PNP. A imagem completa é apenas evidência adicional.

Quando a imagem-fonte completa é reconhecida, os keypoints manuais são reaplicados pela escala global da imagem. Em reuploads/crops, são mapeados relativamente à bbox atual.

## Diagnóstico
`GET /projects/{project_id}/terminal-learning/diagnostics`

Para amostra conhecida, observe:
- `effective_source = known_training_ground_truth`
- `known_training_match.match_method = local_visual_memory`
- `local_similarity`
- `wide_similarity`
- `point_mapping`

## Testes
67 testes automatizados passaram nesta versão.
