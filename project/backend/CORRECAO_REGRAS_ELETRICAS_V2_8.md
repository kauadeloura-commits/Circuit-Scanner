# Etapa 7 v2.8 — regras elétricas do símbolo

Correções:

- Diodo/LED: a barra do cátodo define `K`; o terminal no lado oposto é `A`.
- A orientação elétrica é perpendicular à barra do cátodo.
- A barra é distinguida de linhas do papel e da outra lateral do símbolo usando o ápice formado pelas diagonais.
- BJT NPN/PNP: o ramo que contém a seta é sempre `E` (emissor).
- O outro ramo da forquilha é `C` (coletor).
- `B` permanece na linha reta da base.
- Não se assume mais que C é sempre o ramo superior ou E sempre o inferior.

Testes de regressão específicos foram incluídos em `test_symbol_electrical_rules.py`.
