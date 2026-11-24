# Correcao e dataset de terminais — v5.6

## Objetivo

Permitir que todos os terminais das classes existentes no dataset sejam
corrigidos visualmente e armazenar essas correcoes como ground truth para um
futuro modelo de keypoints por classe.

Classes atuais:

- resistor;
- capacitor e capacitor polarizado;
- diodo e LED;
- indutor;
- fonte DC e bateria;
- GND e VCC;
- transistor NPN e PNP;
- MOSFET N e P;
- amplificador operacional.

Classes futuras tambem podem ser corrigidas. Quando uma classe ainda nao tiver
um esquema conhecido, o editor permite adicionar e remover nomes de pinos.

## Como corrigir

1. Abra `http://127.0.0.1:8010/workflow`.
2. Entre no projeto.
3. Clique em `Corrigir terminais de todos os componentes`.
4. Selecione o componente e o pino.
5. Clique exatamente no ponto de conexao eletrica.
6. Salve o componente atual ou todos os componentes alterados.

Ao salvar:

- `terminal_corrections.json` preserva a correcao no projeto;
- `terminals.json` e `terminals.png` sao regenerados;
- a Etapa 8 de linhas e executada novamente;
- uma amostra e gravada em `terminal_training_all_components/raw/<classe>/`.

## Formato das amostras

Cada componente corrigido gera:

- um PNG recortado contendo o simbolo e todos os pontos marcados;
- um JSON com classe, bbox, esquema de pinos, papel eletrico, coordenadas
  globais e coordenadas normalizadas no recorte e na bbox.

O ID da amostra e estavel por projeto/componente. Uma nova correcao da mesma
peca substitui a amostra anterior, evitando duplicacao e vies no treinamento.

O arquivo `terminal_training_all_components/classes.json` registra as classes,
quantidade de amostras e variacoes de esquema de pinos. Se uma classe possuir
mais de um esquema, o status sinaliza o conflito antes do futuro treinamento.

## Status

`GET /terminal-learning/all-components/status`

O retorno apresenta total de amostras, amostras por classe, esquemas encontrados
e se o conjunto esta estruturalmente pronto para a futura etapa de treinamento.

## Pasta que deve ser preservada

Ao atualizar apenas o backend, nao apague:

- `terminal_training_all_components/`;
- `terminal_keypoints/`;
- `projects/`;
- `models/`;
- `database/`.

## Build

`2026-09-22-workflow-loading-fix-v5.6.1`
