# Etapa 8 + refinamento de transistores v5.1

Build: `2026-09-21-stage8-transistor-wire-refinement-v5.1`

## Mudança principal

Depois da detecção dos fios, NPN/PNP são refinados usando os contatos reais do skeleton no perímetro da bounding box.

- B e E da Etapa 7/modelo são usados como âncoras semânticas.
- E continua sendo o ramo definido pela seta na Etapa 7.
- C NÃO reutiliza a previsão antiga: é reconstruído como um terceiro contato de fio distinto e no lado oposto da forquilha em relação a E.
- B/C/E devem corresponder a três contatos distintos.
- Correções manuais nunca são sobrescritas.
- Terminais provisórios de transistor não são usados como sementes para filtrar fios da Etapa 8, evitando dependência circular.

## Fluxo

`POST /projects/{project_id}/wires`

Gera/atualiza:

- `wires.json`
- `wires.png`
- `wires_mask.png`
- `wires_skeleton.png`
- `terminals.json` (quando algum transistor é refinado)
- `terminals.png` (quando algum transistor é refinado)
- `transistor_wire_refinement.png` (diagnóstico dos B/C/E finais)

## Testes

Suíte completa: 80 testes, 0 falhas.
