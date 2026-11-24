# Etapa 8 — Fios + refinamento de terminais de transistor

Build: `2026-09-21-stage8-wire-refined-transistors-v5.1`

## Fluxo

1. A Etapa 7 produz terminais aproximados.
2. `POST /projects/{project_id}/wires` executa a Etapa 8.
3. A Etapa 8 mascara corpos dos componentes e OCR, remove artefatos de pauta quando possível, cria `wires_mask.png`, afina para `wires_skeleton.png` e extrai segmentos para `wires.json`.
4. Para NPN/PNP, pontos antigos B/C/E **não são usados como sementes de fio**. O perímetro da bounding box é usado para preservar qualquer um dos três condutores reais.
5. A Etapa 7B procura três caminhos distintos chegando ao transistor:
   - dois caminhos aproximadamente paralelos = par C/E;
   - caminho com orientação diferente = B;
   - evidência anterior da seta/emissor decide qual do par é E;
   - o outro caminho é C.
6. Se não houver três caminhos coerentes, ou C/E não puder ser identificado com segurança, o componente é marcado para revisão em vez de receber uma certeza falsa.
7. Correções manuais e `known_training_ground_truth` nunca são sobrescritos.

## Arquivos gerados

- `wires.json`
- `wires.png`
- `wires_mask.png`
- `wires_skeleton.png`
- `wires_erased_mask.png`
- `transistor_wire_refinement.png` (quando existe contexto de terminais/transistores)
- `terminals.json` e `terminals.png` são atualizados quando o refinamento por fios consegue melhorar transistores automáticos.

## Teste

1. Inicie o backend:
   `uvicorn backend.main:app --reload --port 8010`
2. Execute no Swagger:
   `POST /projects/{project_id}/wires`
3. Abra:
   `http://127.0.0.1:8010/projects-files/PRJ_XXXX/wires.png`
4. Confira o skeleton:
   `http://127.0.0.1:8010/projects-files/PRJ_XXXX/wires_skeleton.png`
5. Confira o refinamento dos transistores:
   `http://127.0.0.1:8010/projects-files/PRJ_XXXX/transistor_wire_refinement.png`
6. Confira o resultado final:
   `http://127.0.0.1:8010/projects-files/PRJ_XXXX/terminals.png`
7. Consulte:
   `GET /projects/{project_id}/stage8-status`

## Limite de escopo

A Etapa 8 representa fios e caminhos. `junction_candidate` ainda não significa conexão elétrica confirmada. Distinguir T, junção com ponto e cruzamento sem conexão pertence à Etapa 9.
