# Treinamento de OCR e proteção contra componentes sobre texto — versão 5.46.0

Treinamento real executado em CPU em 14/10/2026. Os arquivos de pesos são usados pelo programa; não se trata apenas de preparar exemplos.

## Dados e separação

- Foto fornecida: `1cb354be8d8a518911e8fed96a5c43e224b4059b79cc41dcdd4bea603235bf3e` (SHA-256).
- 21 caracteres recortados e conferidos, além de oito exemplos completos de palavras/valores. A última letra/dígito ambíguo da referência inferior não recebeu um rótulo de OCR.
- 1.518 glifos originais renderizados de fontes do Windows. Não são distribuídos arquivos TTF.
- Treino: 14 faces de sete famílias (Arial, Calibri, Times New Roman, Consolas, Comic Sans, Ink Free e Segoe Print). Validação: Tahoma/Courier New; teste: Verdana/Georgia. As famílias foram separadas antes das ampliações.
- OCR: 69 classes; 14,386 exemplos de treino após variações de inclinação, espessura, escala e resolução. 25 épocas; MLP sobre características de traços/HOG e uma cabeça que distingue tipografia de manuscrito. O classificador antigo continua disponível.
- Detector: 357 imagens de treino, 207 de validação e 110 de teste. O treino combina exemplos positivos dos 15 tipos de componentes, a fotografia rotulada e páginas/recortes contendo apenas texto. Referências como R1…R10, C1…C10 e L1…L9 também entram nos exemplos de contexto.
- O detector principal foi refinado por sete épocas (parada antecipada). O melhor checkpoint foi da época 2. A validação escolheu uma interpolação conservadora: 75% dos parâmetros anteriores e 25% dos refinados. Essa escolha precedeu o teste final.
- Dois filtros supervisionados complementam o YOLO: 57 épocas para glifos/palavras isoladas e 90 épocas para escrita com pautas, sombras, desfoque e caixas parciais. As falsas propostas verificadas na foto de treino entraram como negativos difíceis. Nenhuma foto de teste foi adicionada ao treino para corrigir seu resultado.

## Lógica em execução

O OCR consulta os exemplos de traços e o especialista de fontes. Referências são lidas como uma letra de componente seguida de números; valores são interpretados com unidade e contexto do componente. I/l/1 e O/o/0 são reunidos apenas em posições numéricas. Um exemplo completo com correspondência forte tem prioridade sobre uma hipótese formada por letras isoladas.

O detector usa os negativos de texto aprendidos e verifica o corpo e o entorno da caixa. O filtro isolado protege componentes com fios externos. O filtro de contexto usa um limiar de 99,99%; uma bobina cuja curva e dois fios foram traçados no mesmo componente de tinta é preservada pela prova física. Caixas corrigidas manualmente são preservadas antes desses filtros.

## Resultados observados

| Verificação | Anterior | Versão 5.46.0 |
|---|---:|---:|
| Rótulos completos em fontes de teste (56 casos) | 29/56 | 44/56 |
| Glifos individuais em fontes reservadas | — | 91.67% em 276 casos |
| YOLO mAP50 no teste (15 classes) | 96.900% | 96.945% |
| YOLO recall no teste | 94.005% | 93.757% |
| YOLO mAP50–95 no teste | 62.522% | 62.147% |
| Caixas brutas sobre texto em 83 páginas de teste | 3 | 1 |
| Caixas sobre texto após os filtros nessas páginas | 0 | 0 |

O ganho do YOLO em mAP50 foi pequeno; o recall e o mAP50–95 caíram ligeiramente. A principal melhora medida foi no OCR e na separação texto/componente, não um aumento amplo comprovado da precisão de todos os símbolos.

Filtro isolado: 351/356 textos rejeitados; 391/391 componentes preservados. Filtro contextual: 447/476 textos rejeitados; 391/391 componentes preservados. São testes de recortes, separados das medidas do circuito completo. Nos testes de regressão, três bobinas verdadeiras inicialmente foram rejeitadas pelo filtro contextual; a proteção pela curva com dois fios corrigiu os três casos antes da instalação.

Passaram 166 testes de regressão, incluindo os valores e modelos das fotos PRJ_0425, PRJ_0428 e PRJ_0435, circuitos com muitos componentes, terminais, junções e gravação de arquivos. Uma nova inferência do PRJ_0443 preservou os sete símbolos corretos e os valores de 5 V, 4,7 kΩ, 3,3 kΩ e 22 kΩ.

## Limites conhecidos

Na nova foto usada para treino, as falsas caixas sobre as escritas foram retiradas. O circuito completo ainda apresentou limitações de reconhecimento: seis caixas físicas, com VCC ausente e confusões de classe no resistor em zigue-zague e em um transistor, além de associações de valores afetadas por essas classes. Isso não foi contabilizado como uma foto independente acertada. Os recortes exatos de treino também não são uma medida independente de precisão.

Os resultados não garantem ausência de falsos positivos em toda escrita futura, nem dispensam a revisão do circuito. Não foi comprovado um tempo completo de processamento de 30 segundos nesta atualização. Os modelos adicionais executam localmente em CPU, sem serviço pago, GPU ou novo treinamento ao abrir o programa.

## Arquivos e rastreabilidade

- Build: `2026-10-14-supervised-fonts-text-component-guard-v5.46.0`.
- Pesos YOLO ativos: `dc4d523528894e0cfd246611ac2c45953504a5439eeeef5374030752ebf16ebd`.
- Pesos OCR anteriores preservados: `be5e7d98957c830b2662339e24109875a85db28ca49e43925e247ad5e9a6b328`.
- `glyph_specialist.npz`: `499b3314cbfdd671fe059e35b533a8e78485c164413b0eedd5978ade60fc3d4a`.
- `text_component_classifier.npz`: `bf7d79e32fd60085bf97a0229c70a3bdd4c6d9449be0c302e0ab838c668227df`.
- `text_context_classifier.npz`: `af0a4e5cad1e486a035259778c07a64870d989d9efa14804b8ccce2e46a609a8`.

Os JSONs anexos incluem métricas por classe, separação das fontes, hiperparâmetros, histórico e anotações dos recortes. Os dados dos projetos e as correções manuais não entram no pacote de atualização.
