# Treinamento visual do reconhecimento de componentes — v5.8

Abra `http://127.0.0.1:8010/component-trainer`. Não é necessário usar
`/docs`.

## Fluxo recomendado

1. Em **Imagens reais**, adicione uma foto ou captura completa do circuito.
2. Escolha a classe e desenhe uma caixa justa ao redor de cada símbolo.
3. Use **Selecionar/mover** para arrastar a caixa ou seus quatro cantos.
4. Exclua caixas incorretas e pressione **Salvar anotações**.
5. Repita com circuitos variados e acompanhe a contagem por classe.
6. Gere a base sintética, monte o dataset misto e inicie o treinamento.
7. Acompanhe as épocas e as métricas na própria página.

As imagens e correções ficam em `real_data/`. O dataset preparado para YOLO
fica em uma pasta versionada dentro de `dataset_builds/`; a avaliação somente
com fotos reais fica no mesmo build. `dataset_builds/active.json` informa qual
versão será usada no próximo treino. Isso evita erros de acesso negado no
Windows quando uma pasta de dataset anterior estiver aberta ou bloqueada.

## O que cada anotação ensina

Esta página treina a primeira etapa: **onde o componente está e qual é sua
classe**. Ela não marca base/coletor/emissor, anodo/catodo ou outros pinos.
Esses terminais continuam no editor de terminais de cada projeto, depois do
reconhecimento do componente.

O registro atual possui IDs estáveis para resistor, capacitor não polarizado,
capacitor polarizado, diodo, LED, indutor, fonte DC, bateria, GND, VCC,
transistor NPN, transistor PNP, MOSFET N, MOSFET P e amplificador operacional.

## Segurança do modelo

- O treino parte de `models/circuit_scanner_yolo.pt`.
- Antes do treino, o peso ativo é copiado para `models/backups/`.
- O modelo ativo só é substituído depois que `best.pt` existe e o mapa das 15
  classes foi validado.
- Se o servidor reiniciar durante o treino, o modelo anterior permanece ativo.
- O relatório final fica em `models/training_summary_mixed.json`.

## Quantidade e qualidade dos exemplos

Procure obter ao menos 30 símbolos reais bem revisados de cada classe antes de
usar a métrica como indicação séria. Para classes difíceis, 100 ou mais caixas
variadas costumam ser mais úteis. Misture rotações, tamanhos, desenhos,
espelhamentos, fundos, iluminações e fios próximos. Separe circuitos novos para
validação; o construtor já evita colocar variações da mesma imagem real em
treino e validação.

Para os transistores, inclua NPN e PNP em orientações para a direita e para a
esquerda. A seta e os terminais são corrigidos na etapa de terminais; aqui a
caixa deve conter o símbolo inteiro.
