# Correção do eixo elétrico do capacitor — v2.1

Problema observado em desenho manual: a bounding box do capacitor pode ficar mais larga por causa das placas, texto ou ruído. O algoritmo anterior podia classificá-lo como horizontal mesmo quando as placas estavam horizontais e os terminais elétricos entravam por cima e por baixo.

Correção: `capacitor` e `capacitor_polarized` agora usam a mesma inferência de eixo elétrico por bordas opostas já aplicada a LED/diodo. O sistema procura os fios que entram em lados opostos da bbox e usa esses traços para escolher horizontal/vertical e para posicionar os terminais.

- fios em cima + embaixo -> eixo elétrico vertical;
- fios à esquerda + à direita -> eixo elétrico horizontal;
- bbox/aspect ratio fica apenas como fallback.

Foram adicionados testes de regressão para capacitor vertical com bbox larga e capacitor horizontal.
