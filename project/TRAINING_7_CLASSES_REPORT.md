# Treino de terminais — v5.7.1

## Correção NPN v5.7.1

- NPN voltado para a direita: o ramo inferior direito com seta é `E`.
- NPN espelhado para a esquerda: o ramo inferior esquerdo com seta é `E`.
- A seta é sempre o emissor; o outro ramo é o coletor.
- A trava só atua quando a forquilha horizontal está clara. Em símbolos
  rotacionados, a seta e o modelo treinado continuam definindo B/C/E.
- 55 amostras NPN horizontais corrigidas confirmaram a regra: 52 à direita,
  3 à esquerda e nenhuma violação.
- O modelo dedicado NPN/PNP foi retreinado diretamente com os pontos corrigidos.

Resultado do teste BJT separado (24 transistores): pose mAP50 de 99,5% e
mAP50–95 de 92,9%. Nos 14 NPN de teste, o modelo aceitou os 14 e não houve
nenhuma inversão C/E antes ou depois da trava de segurança.

## Escopo deste modelo

Foram usadas somente as classes:

- `vcc`
- `transistor_npn`
- `transistor_pnp`
- `diode`
- `led`
- `resistor`
- `capacitor`

MOSFET, amplificador operacional, genérico, bateria, fonte DC e as demais
classes foram preservadas no dataset, mas não entraram neste treino.

## Dataset

- 589 amostras corrigidas nas sete classes.
- Divisão por `project_id`: nenhum circuito aparece em mais de um subconjunto.
- 429 amostras originais de treino, 59 de validação e 101 de teste.
- O treino foi balanceado para 480 imagens com aumentação somente fotométrica;
  validação e teste não foram aumentados.
- Ordem fixa dos keypoints: B/C/E para BJT; A/K para diodo e LED; 1/2 para
  resistor e capacitor; VCC para alimentação. Slots ausentes ficam invisíveis.

## Resultado do checkpoint escolhido

| Conjunto | Precisão pose | Recall pose | mAP50 pose | mAP50–95 pose |
|---|---:|---:|---:|---:|
| Validação (59) | 94,2% | 94,2% | 98,1% | 92,3% |
| Teste separado (101) | 96,5% | 95,1% | 98,7% | 91,0% |

Na simulação da inferência usada pelo backend, 98 de 101 amostras de teste
tiveram previsão aceita. A amostra de LED ainda é pequena (19 no total), então
novos LEDs corrigidos continuam importantes apesar do resultado atual.

## Política de uso no backend

- Evidência física forte tem prioridade: seta e barra/base do transistor,
  placas do capacitor, barra catódica e eixos claros de fios/corpo.
- O modelo de sete classes assume os pontos quando a geometria é ambígua e a
  previsão da classe esperada supera os limites de confiança.
- Se a previsão for rejeitada, o fluxo geométrico anterior continua ativo.
- O modelo antigo de BJT não foi sobrescrito.

## Como treinar novamente depois de adicionar correções

Execute na raiz do projeto:

```bash
python backend/all_component_terminal_learning.py train --epochs 60 --imgsz 416 --batch 16 --patience 15
```

Para atualizar também o modelo dedicado de NPN/PNP com as correções:

```bash
python backend/all_component_terminal_learning.py train-bjt --epochs 40 --imgsz 416 --batch 16 --patience 12
```

O melhor checkpoint será gravado em:

```text
models/all_component_terminal_keypoints.pt
```

O relatório completo será atualizado em:

```text
models/all_component_terminal_keypoints_summary.json
```

O comando requer Python com `ultralytics`, `torch`, `torchvision`, `opencv-python`
e `numpy`. Em máquina sem GPU, o treino usa CPU automaticamente conforme a
configuração atual do script.

## Verificação

- `97 passed, 20 subtests passed` no conjunto completo de regressão do backend.
- Endpoint de status: `/terminal-learning/all-components/status`.
- Teste de carregamento do modelo: `/terminal-learning/all-components/model-runtime`.
