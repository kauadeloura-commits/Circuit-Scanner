# Etapa 8 - Fios + refinamento dos transistores

Build: `2026-09-21-stage8-wires-transistor-refine-v5.1`

## Fluxo
1. Execute normalmente detecção/OCR/terminais.
2. Rode `POST /projects/{project_id}/wires`.
3. O backend gera `wires.json`, `wires.png`, `wires_mask.png`, `wires_skeleton.png`.
4. Na mesma execução, NPN/PNP sem correção manual são refinados usando três caminhos distintos do skeleton.
5. Os terminais atualizados são gravados novamente em `terminals.json` e `terminals.png`.
6. `transistor_wire_refinement.png` mostra skeleton + B/C/E finais.

## Regras importantes
- Correções manuais nunca são sobrescritas.
- B/C/E finais ficam sobre o centerline dos fios detectados.
- Os três pinos precisam vir de três caminhos diferentes.
- O par de caminhos externos mais paralelos forma C/E; o terceiro caminho é B.
- O E anterior (seta/modelo) é usado somente para escolher qual dos dois caminhos da forquilha é o emissor. Ele não define a posição final.
- Se não houver três caminhos coerentes, o transistor continua marcado para revisão.
- Junções e cruzamentos ainda não são decididos aqui; isso pertence à Etapa 9.

## Arquivos para conferir
- `wires.png`: fios detectados sobre a imagem.
- `wires_skeleton.png`: centerlines de 1 pixel.
- `terminals.png`: terminais após refinamento.
- `transistor_wire_refinement.png`: diagnóstico específico dos transistores.
- `wires.json`: representação da Etapa 8 + relatório de refinamento.

## Rotas
- `POST /projects/{project_id}/wires`
- `GET /projects/{project_id}/stage8-status`
- `GET /system/version`
