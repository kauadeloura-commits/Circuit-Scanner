# Revisão completa do detector — v5.13.0

## Diagnóstico técnico

A revisão do pipeline encontrou causas concretas que explicam os falsos
positivos, falsos negativos, caixas aninhadas e o treinamento excessivamente
demorado. Não foi feita troca de arquitetura nem alteração das etapas de
terminais, fios e junções.

1. O limiar interno chegou a ser cinco vezes menor que o solicitado pela tela.
2. Muitas transformações da mesma imagem eram tratadas como evidências extras e
   podiam elevar artificialmente a confiança até 1.00.
3. A regra de agrupamento aceitava cobertura da caixa menor como duplicidade.
   Assim, uma caixa grande envolvendo vários símbolos engolia caixas pequenas.
4. O agrupamento comparava uma nova caixa com qualquer membro do grupo. Isso
   permitia agrupamento em cadeia entre componentes vizinhos.
5. O treinamento continuava a partir do último modelo, mesmo se ele estivesse
   degradado.
6. Páginas inteiras reduziam componentes pequenos durante o treino. A quantidade
   de imagens existia, mas cada símbolo podia ocupar poucos pixels após resize.
7. Negativos artificiais de texto não representavam suficientemente fios,
   pautas e regiões reais entre os componentes.
8. Não havia uma auditoria objetiva das caixas nem comparação por cor de caneta.
9. O treinamento de 80 épocas não tinha parada solicitável com validação do
   melhor checkpoint já produzido.

## Alterações implementadas

### ARQUIVO: `backend/recognition.py`

FUNÇÕES: `detectar_componentes`, `consolidar_multiview`,
`remover_duplicadas`, `_mesma_regiao_de_componente`,
`remover_caixas_englobantes_multicomponente`, `_executar_predicao` e
`_executar_predicao_em_blocos`.

PROBLEMA:

- confiança inflada por vistas correlacionadas;
- agrupamento em cadeia;
- caixas grandes absorvendo componentes vizinhos;
- NMS sem configuração explícita;
- resgate de previsões fracas;
- geometria criando objetos sem proposta confiável da rede.

ALTERAÇÃO:

- a confiança exibida é a melhor confiança real da rede;
- o consenso é mantido apenas como pontuação auxiliar para desempate;
- agrupamento usa uma âncora, compatibilidade de centro e razão de tamanho;
- uma caixa contendo duas caixas independentes é rejeitada como englobante;
- YOLO usa explicitamente `iou=0.55`, `agnostic_nms=False` e `max_det=300`;
- limiar interno nunca fica abaixo de 0.12;
- removido o resgate global em 0.10/0.12;
- geometria somente valida/rejeita propostas da IA;
- tiling permanece para páginas grandes, com coordenadas globais e deduplicação.

MOTIVO:

Evitar que repetir a mesma imagem fabrique evidência e separar corretamente
componentes próximos.

EFEITO ESPERADO:

Menos falsos positivos, menos caixas aninhadas, preservação de componentes
vizinhos e confiança interpretável.

### ARQUIVO: `backend/real_dataset.py`

FUNÇÕES: `_real_page_tiles`, `_labels_for_crop`, `_write_real_tiles` e
`build_mixed_dataset`.

PROBLEMA:

Os componentes anotados ficavam pequenos quando a página inteira era reduzida
para o `imgsz` do YOLO. Regiões reais com fios e pautas não eram exploradas como
negativos difíceis.

ALTERAÇÃO:

- cada caixa real gera recorte focado, preservando componentes vizinhos válidos;
- caixas parcialmente cortadas não entram como exemplos completos;
- regiões da mesma página sem componentes viram negativos difíceis;
- divisão treino/validação/teste continua agrupada pela imagem/projeto de origem;
- variantes monocromáticas continuam limitadas ao split correto.

MOTIVO:

Mais imagens não ajudam quando o objeto chega pequeno ao detector. A escala útil
é tão importante quanto a quantidade.

EFEITO ESPERADO:

Maior recall em resistores, capacitores, diodos, transistores e demais símbolos
pequenos, com redução de falsos positivos em fios e pautas.

### ARQUIVO: `backend/component_learning.py`

FUNÇÕES: `_stable_foundation_model`, `_candidate_quality_gate`,
`_negative_false_positive_summary`, `_train_worker`,
`request_component_training_stop` e `_stop_after_current_epoch`.

PROBLEMA:

- degradação acumulativa ao treinar sempre sobre o último modelo;
- ativação de candidato com média aceitável, mas classe rara colapsada;
- ausência de parada segura antes de 80 épocas.

ALTERAÇÃO:

- treinos partem de base estável e imutável;
- o modelo anterior continua preservado;
- qualidade mínima: precisão 0.60, recall 0.35 e mAP50 0.45;
- regressões por classe também bloqueiam ativação;
- negativos reais são usados na auditoria de falsos positivos;
- novo comando finaliza após a época atual e valida `best.pt` normalmente;
- augmentations moderadas substituem mosaic agressivo.

MOTIVO:

Mais épocas sobre um peso degradado apenas consolidam erros. O peso precisa ser
promovido por métricas reais e por classe.

EFEITO ESPERADO:

Treinamento recuperável, menor risco de overfitting e nenhum modelo novo ativo
sem aprovação objetiva.

### ARQUIVO: `backend/component_diagnostics.py`

FUNÇÕES: `audit_real_annotations`, `evaluate_pen_colors`,
`start_component_diagnostic` e funções auxiliares de matching.

PROBLEMA:

Não havia evidência sobre qualidade das labels, tamanho relativo das caixas,
cores de caneta, TP, FP, FN ou desempenho por classe.

ALTERAÇÃO:

- relatório de caixas grandes, pequenas, duplicadas e concorrentes;
- detecção de margem vazia e sugestão de caixa mais justa, sem alterar a label;
- estatísticas de área e pixels por classe;
- razão de desequilíbrio maior/menor;
- avaliação do holdout real em preto, azul, vermelho e verde;
- TP, FP, FN, precisão, recall e F1 gerais e por classe;
- comparação automática do relatório atual com o anterior.

MOTIVO:

Separar problemas de label, cor, modelo e pós-processamento antes de modificar
parâmetros.

EFEITO ESPERADO:

Decisões baseadas em métricas, identificação direta das imagens que precisam de
correção e comparação antes/depois reproduzível.

### ARQUIVOS: `backend/component_trainer_page.py` e `backend/main.py`

FUNÇÕES/ROTAS:

- `/component-learning/dataset-audit`;
- `/component-learning/diagnostic`;
- `/component-learning/diagnostic/status`;
- `/component-learning/stop-and-validate`;
- interface de treinamento.

PROBLEMA:

Os diagnósticos exigiam acesso manual à documentação da API e o treinamento não
podia ser encerrado de forma segura pela interface.

ALTERAÇÃO:

- botões `Auditar anotações` e `Testar modelo e cores`;
- resultados mostrados na própria tela;
- botão `Finalizar após esta época e validar`;
- versão do sistema atualizada.

EFEITO ESPERADO:

Fluxo completo pela interface, sem depender de `/docs`.

## Como executar a revisão no dataset real

1. Abra `http://127.0.0.1:8010/component-trainer`.
2. Clique em `Auditar anotações`.
3. Se houver problemas, corrija primeiro caixas grandes, duplicadas ou com
   classes concorrentes.
4. Clique em `Testar modelo e cores`.
5. Compare precision, recall, F1, FP e FN por cor e por classe.
6. Corrija as imagens indicadas e monte novamente o dataset misto.
7. Treine e execute novamente o diagnóstico. A tela e o JSON mostrarão a
   comparação antes/depois.

## Interpretação

- Muitos `box_has_excessive_empty_margin`: labels precisam ser apertadas.
- Muitos `possible_duplicate_annotation`: existem caixas repetidas.
- Muitos `different_classes_compete_in_annotation`: duas classes ocupam a
  mesma região no ground truth.
- Recall baixo com precisão boa: objetos pequenos/escala ou classe pouco
  aprendida.
- Precisão baixa com recall alto: modelo inventando componentes ou labels
  incompletas.
- F1 de uma cor mais de 0.08 abaixo das outras: viés de cor ainda presente.
- FP alto em todas as cores: negativos reais insuficientes ou labels ausentes.
- FN alto em todas as cores: escala, caixas erradas ou classe colapsada.

## Limitação necessária

O código foi revisado e testado neste pacote, mas os números verdadeiros só
podem ser calculados no computador que contém o dataset completo e o peso
treinado. Por isso a auditoria foi incorporada ao programa, em vez de inferir
qualidade a partir apenas da contagem de imagens.
