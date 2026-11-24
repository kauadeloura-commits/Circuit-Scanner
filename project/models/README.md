# Modelos — Etapa 5

O modelo ativo revisado está em:

`models/circuit_scanner_yolo.pt`

Para conferir o modelo e o dataset:

    python scripts\audit_stage5.py

Se futuramente for necessário treinar novamente, faça primeiro um teste:

    python scripts\train_yolo.py --epochs 10

Treinamento mais sério:

    python scripts\train_yolo.py --epochs 50

O script também salva:

- `training_summary.json`
- `dataset_runtime.yaml`
- `training/circuit_scanner/` com os resultados do treino

O projeto corrigido já contém um modelo treinado. O arquivo
`active_model_summary.json` registra qual peso foi ativado, as métricas reais e
o caminho do modelo de rollback.
