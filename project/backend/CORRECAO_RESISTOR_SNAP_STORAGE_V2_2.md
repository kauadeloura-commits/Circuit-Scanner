# Correção v2.2 — resistor, snap local e recuperação do banco

## 1. Resistor
A orientação do resistor retangular não depende mais da proporção da bounding box do YOLO. O módulo procura o retângulo real do corpo, mede o seu eixo longo e usa esse eixo para decidir horizontal/vertical. Se o corpo não for encontrado (por exemplo resistor zig-zag), o sistema usa os fios opostos/bbox como fallback.

## 2. Snap geométrico dos terminais
Depois de localizar um terminal, a Etapa 7 faz um ajuste curto e local:
- prioriza o centro de uma bolinha/junção próxima;
- senão centraliza o ponto sobre o fio local;
- terminais de componentes diferentes que ficarem a até 8 px e possuírem tinta ligando os dois são alinhados para a mesma coordenada.

Esse snap NÃO cria nets e NÃO substitui as Etapas 8–10. Ele serve para melhorar a geometria visual e fornecer pontos iniciais mais coerentes.

## 3. Banco/projetos no front-end
O arquivo SQLite é somente o índice dos projetos. A partir da v2.2, se o banco estiver vazio/trocado mas `projects/PRJ_*` ainda existir, o backend reinsere automaticamente os projetos no SQLite e a rota `GET /projects` volta a listá-los.

Rotas de diagnóstico:
- `GET /system/storage`
- `POST /system/storage/reindex`

IMPORTANTE: o ZIP v2.2 NÃO contém `database/circuit_scanner.db`, para não sobrescrever o banco existente ao atualizar o backend.
