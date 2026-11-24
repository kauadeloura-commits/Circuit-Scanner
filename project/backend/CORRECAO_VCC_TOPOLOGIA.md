# Correção de reconhecimento: VCC por topologia do nó

## Problema
O YOLO podia classificar uma bolinha de junção como VCC. Validar somente o formato visual do símbolo não era suficiente em esquemas onde VCC é representado como um nó/terminal isolado.

## Regra implementada
Quando uma detecção do YOLO é `vcc`, o backend procura uma bolinha preenchida perto da caixa detectada e conta quantos ramos elétricos saem dela:

- 1 ramo: candidato VCC aceito;
- 0 ramos: rejeitado;
- 2 ou mais ramos: rejeitado como junção/ponto de conexão.

O número de ramos é obtido por máscara de primeiro plano + transformada de distância + análise de um anel ao redor do centro da bolinha. A cor do fio não é usada como regra fixa.

Se não houver uma bolinha preenchida, o sistema mantém compatibilidade com o símbolo VCC clássico do dataset (chevron/seta com uma haste).

## Componentes básicos
Resistor, capacitor, capacitor polarizado, diodo, LED e GND ganharam um resgate conservador de baixa confiança. Quando existe evidência visual suficiente, a detecção é mantida com `review_required=true` em vez de simplesmente desaparecer. VCC e BJT não usam esse resgate.

## Campos úteis em detections.json
Uma detecção VCC validada por nó contém em `geometry_validation.evidence`:

- `validation_mode: connection_dot_topology`
- `connection_branch_count`
- `node_center`
- `node_radius`
- `junction_rejected`

O JSON também informa `basic_component_rescue_count`.
