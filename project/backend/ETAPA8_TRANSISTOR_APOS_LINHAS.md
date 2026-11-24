# Etapa 8 + Ordem Física do Transistor (v5.6)

## Regra arquitetural
- Classes diferentes de transistor continuam usando seus terminais da Etapa 7.
- Para `transistor_npn` e `transistor_pnp`, terminais automáticos B/C/E ficam adiados até a Etapa 8.
- O gate `GET /projects/{id}/stage7-readiness` permite iniciar a Etapa 8 quando
  os únicos pontos pendentes forem B/C/E automáticos de transistores; outros
  componentes incompletos continuam bloqueando o avanço.
- Correções manuais B/C/E já salvas são preservadas.
- `POST /projects/{project_id}/wires` detecta fios/skeleton e, em seguida, refina SOMENTE transistores.

## Como B/C/E são definidos
1. O detector principal fornece a classe (`transistor_npn` ou `transistor_pnp`) e bbox.
2. A Etapa 8 preserva qualquer condutor que alcance o perímetro da bbox do transistor; B/C/E antigos não são usados como sementes.
3. O skeleton é analisado ao redor da bbox.
4. São exigidos 3 caminhos físicos distintos.
5. Os três caminhos de fio são selecionados sem receber nomes.
6. A classe NPN/PNP já definida pelo detector principal determina o sentido esperado da seta.
7. A seta física escolhe e bloqueia o terminal E.
   - A presença da seta define o ramo emissor. Se o sentido estimado conflitar
     com a classe NPN/PNP, o ramo não é trocado silenciosamente: fica sinalizado para revisão.
8. A barra reta escolhe B somente entre os dois contatos restantes.
9. C é obrigatoriamente o único contato que sobrou; o modelo nunca escolhe C diretamente.
10. Ground truth e pose treinada são usados apenas quando a seta ou a barra não puderem ser lidas.
11. As posições finais permanecem na centerline dos fios da Etapa 8.
12. Evidência ausente ou ambígua gera revisão em vez de permitir troca silenciosa de B/C/E.

## Reanalise visual mantida na v5.6
- O recorte do transistor e analisado em quatro versoes: original, contraste
  local, nitidez e binarizacao com fechamento leve.
- A escolha usa consenso espacial entre as tentativas para evitar que uma
  unica binarizacao fraca troque o ramo da seta.
- Discordancia real continua visivel como revisao; resultados fortes e
  concordantes seguem automaticamente.

## Correcao manual e continuidade
- Acesse `http://127.0.0.1:8010/workflow`; nao e necessario usar `/docs`.
- Em cada projeto, `Corrigir pontos B/C/E` abre o editor visual.
- A correcao e persistida em `terminal_corrections.json`, atualiza
  `terminals.json`/`terminals.png` e alimenta o treino de transistores.
- Ao salvar, a Etapa 8 e executada novamente e as linhas sao regeneradas.
- Reconhecimento, OCR e recalculo de terminais tambem continuam
  automaticamente ate as linhas.

## Treinamento de transistor
- O dataset continua usando a ordem fixa `B, C, E`.
- A auditoria informa amostras com keypoints colapsados e possíveis trocas semânticas.
- O treino dá mais peso à localização dos keypoints e reduz augmentations que podem apagar a seta.
- NPN e PNP permanecem classes separadas; previsões de classe divergente não são aceitas.

## Arquivos gerados
- `wires.json`
- `wires.png`
- `wires_mask.png`
- `wires_skeleton.png`
- `wires_erased_mask.png`
- `transistor_wire_refinement.png`
- `terminals.json` (atualizado após fios)
- `terminals.png` (atualizado após fios)

## Teste
1. Rode o backend.
2. Faça detecção/OCR normalmente.
3. Abra `/workflow` e escolha o projeto.
4. Clique em `Reconhecer e gerar linhas`.
5. Se necessario, use `Corrigir pontos B/C/E`; ao salvar, as linhas sao refeitas.
6. Confira `transistor_wire_refinement.png` e `terminals.png` na propria tela.

## Build
`2026-09-22-workflow-loading-fix-v5.6.1`
