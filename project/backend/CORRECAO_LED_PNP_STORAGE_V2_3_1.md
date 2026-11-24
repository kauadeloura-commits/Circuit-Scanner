# Correção Etapa 7 v2.3.1

## LED em foto
- A orientação do LED não depende apenas das bordas da bbox.
- Quando texto/setas contaminam as laterais, o eixo do corpo principal pode substituir o falso eixo de borda.
- A/K são posicionados ao longo do eixo central do corpo, ignorando setas de emissão fora desse corredor.

## PNP
- C e E usam busca restrita a lados opostos da base e a metades verticais diferentes da forquilha.
- O coletor não pode fazer snap no mesmo ramo do emissor.
- NPN mantém a lógica já validada.

## Banco/front-end
- Novos projetos recuperados usam somente o ID/nome normal, sem '(recuperado)'.
- Na inicialização, registros antigos com esse sufixo são corrigidos no próprio SQLite.
- O ZIP de atualização não inclui circuit_scanner.db.

## Testes
- 44 testes automáticos aprovados.
