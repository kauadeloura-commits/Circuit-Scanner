# Etapa 8 + Etapa 7B — fios e refinamento dos transistores

Build: `2026-09-21-stage8-transistor-wire-refinement-v5.1`

## Fluxo

1. Etapa 7 gera terminais provisórios.
2. `POST /projects/{project_id}/wires` detecta fios e skeleton.
3. Para NPN/PNP automáticos, a Etapa 7B procura três fios externos distintos na bbox.
4. O fio da base é escolhido pela região da base, mas a coordenada final é o contato real do fio.
5. Entre os outros dois fios, a seta/evidência de seta define `E`; o outro vira `C`.
6. `terminals.json` e `terminals.png` são regenerados automaticamente.
7. Correções manuais salvas nunca são sobrescritas.
8. Se não houver três fios coerentes, o transistor fica em revisão e não é inventado um terceiro terminal.

## Arquivos gerados

- `wires.json`
- `wires.png`
- `wires_mask.png`
- `wires_skeleton.png`
- `wires_erased_mask.png`
- `terminals.json` (atualizado)
- `terminals.png` (atualizado)
- `transistor_wire_refinement.png` (diagnóstico)

## Teste

Execute:

```text
POST /projects/PRJ_XXXX/wires
```

Depois compare:

```text
projects/PRJ_XXXX/wires.png
projects/PRJ_XXXX/terminals.png
projects/PRJ_XXXX/transistor_wire_refinement.png
```

No `terminals.json`, um transistor refinado deve ter:

```json
"terminal_detection": {
  "status": "wire_refined",
  "method": "stage8_three_distinct_wire_contacts"
}
```

Cada terminal B/C/E terá também `wire_id`, permitindo verificar que os três pertencem a fios distintos.
