# Treinamento refinado de componentes — v5.38.0

Treinamento real executado em CPU: 11 épocas de ajuste da localização e das classes e 12 épocas da rede inteira. As fases pararam por estagnação na validação. O modelo é o YOLO26n de 15 classes já usado no programa, sem aumento de arquitetura ou necessidade de GPU.

A primeira versão escolhida regrediu no reconhecimento completo e foi rejeitada. A versão entregue preserva 75% dos parâmetros originais e incorpora 25% do checkpoint treinado da oitava época da segunda fase. Essa proporção foi comparada com outras na validação, mantendo as classes e os formatos existentes.

## Dados e conferência

700 imagens de treino: 75 páginas com muitos componentes, 420 recortes de contexto com cotas por classe, 90 exemplos sintéticos para preservar formas conhecidas, 70 fundos reais e 45 negativos de texto. Validação: 167 imagens. Teste por grupos de origem: 62 imagens. As fontes originais foram agrupadas antes dos recortes e transformações, com união de anotações de duplicatas. Fotos CAD com GNDs visíveis sem anotação foram excluídas. Anotações vazias não foram aceitas automaticamente como negativos.

Parte dessas fotos pode ter sido usada no treinamento anterior do modelo original; as métricas por grupos não são prova de generalização totalmente independente. Três fotos recentes foram reservadas e passaram a servir também à análise de erros depois da rejeição do primeiro candidato. Duas fotos adicionais foram verificadas visualmente e testadas depois, sem treinamento nelas. PRJ_0429 é uma cópia exata de PRJ_0428 e foi contado somente uma vez.

## Validação do refinamento

| Métrica | Original | Entregue |
|---|---:|---:|
| Precisão | 94.30% | 94.05% |
| Recuperação | 85.01% | 86.67% |
| mAP50 | 89.33% | 90.02% |
| mAP50–95 | 55.23% | 55.69% |

Zero detecções falsas em 35 recortes de texto/fundo na validação e zero em 9 recortes manuscritos das fotos de avaliação, no limite de confiança 0,25. Esses números se referem a esses controles, não a qualquer escrita possível.

## Reconhecimento completo do programa

Comparação com os mesmos filtros e a mesma configuração do programa. Correspondência exige classe correta e sobreposição IoU ≥ 0,50; uma caixa demasiadamente grande também conta como erro de localização.

| Foto | Corretos antes | Corretos depois | Falsas detecções antes → depois |
|---|---:|---:|---:|
| PRJ_0413 | 12 | 14 | 3 → 2 |
| PRJ_0428 | 22 | 21 | 1 → 2 |
| PRJ_0431 | 8 | 8 | 0 → 0 |
| PRJ_0409 | 14 | 15 | 2 → 1 |
| PRJ_0411 | 22 | 21 | 4 → 5 |

Total nas cinco fotos, 95 componentes anotados: 78 → 79 correspondências corretas; 10 → 10 falsas detecções; 17 → 16 faltantes/erros de classe ou localização. O ganho agregado é pequeno e não há teste de significância estatística.

No PRJ_0413, o GND omitido e o MOSFET N passaram a ser identificados. No PRJ_0409, um capacitor polarizado passou a ser classificado corretamente. Persistem erros de subtipo em transistores/MOSFETs, componentes omitidos e caixas de resistores muito grandes em algumas fotos. A melhora em recuperação não equivale à correção completa de todos esses problemas.

## Preservação e limites

O OCR, seus 56 recortes de referência, os terminais e o código do programa não foram treinados ou alterados por este pacote. Não há promessa de processamento em 30 segundos ou reconhecimento perfeito. Apenas os pesos do detector e seu resumo de treinamento são atualizados. Fotos, banco de projetos e correções manuais são preservados.

Modelo anterior SHA256: `d0a3efdac624fe62980e3b7bef76d45054df63c2a1289e3bd7c2efd9e7299f4f`. Modelo entregue SHA256: `4d98de590a6eb19b59350d9046345cad4b6a8590dc76432fa63a45ea1b7314a3`.

Método e métricas seguem as APIs documentadas do [treinamento Ultralytics](https://docs.ultralytics.com/modes/train/), [validação](https://docs.ultralytics.com/modes/val/) e [métricas de detecção](https://docs.ultralytics.com/guides/yolo-performance-metrics/). Os valores acima são medidos nesta sessão.
