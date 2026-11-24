# Circuit Scanner — extensão de dataset real

Esta extensão corrige o problema de generalização observado na Etapa 5: o YOLO reconhecia o dataset sintético, mas não símbolos desenhados à mão.

Principais adições:

- `frontend/annotate.html`: ferramenta de anotação no navegador;
- `backend/real_dataset.py`: armazenamento das fotos e bounding boxes;
- `real_data/`: fontes reais originais e anotações;
- `dataset_mixed/`: sintético + real para treinamento;
- `dataset_real_eval/`: apenas originais reais para avaliação honesta;
- `scripts/train_yolo_mixed.py`: novo treinamento;
- `scripts/validate_yolo_mixed.py`: validação geral;
- `scripts/validate_yolo_real.py`: validação somente em fotos reais.

Correções da revisão da Etapa 5:

- o detector não escolhe mais `processed.png` automaticamente;
- a imagem original colorida é combinada com uma versão em escala de cinza;
- caixas repetidas e conflitos de nome na mesma posição são consolidados;
- classes incertas são marcadas para revisão;
- scripts de treino/validação geram caminhos válidos para a pasta atual;
- `scripts/audit_stage5.py` informa classes reais ausentes ou desequilibradas.

Leia também `README.md` antes de usar.
