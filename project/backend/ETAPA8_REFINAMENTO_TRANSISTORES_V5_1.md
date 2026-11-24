# Circuit Scanner — Etapa 8 + Refinamento de Transistores v5.1

Build: `2026-09-21-stage8-wire-refined-transistors-v5.1`

## Fluxo

1. Etapa 7 gera terminais provisórios.
2. `POST /projects/{project_id}/wires` executa a Etapa 8.
3. São gerados `wires.json`, `wires.png`, `wires_mask.png`, `wires_skeleton.png`.
4. Após os fios, a Etapa 7B refina automaticamente somente NPN/PNP não corrigidos manualmente.
5. O refinamento exige três caminhos distintos chegando à bbox do transistor.
6. A geometria dos fios define Base vs par C/E. A evidência da seta/modelo anterior é usada somente para identificar qual ramo do par é o Emissor.
7. Correções manuais nunca são sobrescritas.

## Arquivos visuais

- `wires.png`: fios detectados sobre a imagem.
- `wires_mask.png`: máscara binária de condutores.
- `wires_skeleton.png`: centerline de 1 px dos fios.
- `transistor_wire_refinement.png`: diagnóstico dos fios e B/C/E refinados.
- `terminals.png`: preview final dos terminais após o refinamento.

## Rotas

- `POST /projects/{project_id}/wires`
- `GET /projects/{project_id}/stage8-status`
- `POST /projects/{project_id}/refine-transistors-from-wires`

A última rota reaplica somente o refinamento B/C/E usando um `wires.json` já existente.

## Segurança contra falso positivo

Se menos de três caminhos distintos forem encontrados, o sistema não inventa B/C/E: marca o transistor para revisão. Se a geometria dos três fios for clara mas a identidade C/E for ambígua, B é mantida e C/E são marcados como incertos.

## Banco e arquivos persistentes

Ao instalar esta versão, preserve fora de `backend/`:

- `database/`
- `projects/`
- `models/`
- `terminal_keypoints/`

## Testes

A suíte interna desta entrega possui 81 testes e passa integralmente.
