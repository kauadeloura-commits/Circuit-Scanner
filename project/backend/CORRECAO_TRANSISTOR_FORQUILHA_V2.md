# Correção dos terminais de transistor — v1.9

Problema observado em imagens reais: a bounding box do transistor ficava alta por incluir trechos de coletor/emissor. A rotina anterior usava a proporção da bounding box para decidir a orientação e, por isso, alguns BJTs canônicos eram interpretados como verticais. B, C e E acabavam deslocados ou caíam em posições de fallback.

A partir da v1.9, NPN e PNP usam primeiro a geometria interna do símbolo:

1. Localiza a barra vertical da base por morfologia.
2. Localiza duas diagonais que realmente nascem perto dessa barra.
3. Descobre se a forquilha abre para a esquerda ou para a direita.
4. Coloca B na interseção do terminal da base com a barra.
5. Coloca C na ponta real do ramo diagonal superior.
6. Coloca E na ponta real do ramo diagonal inferior.
7. A bounding box serve apenas para limitar a região de busca, não para definir B/C/E.

Também foram adicionados testes para:
- transistor com bounding box alta sem rotacionar a geometria;
- NPN com forquilha à direita;
- PNP espelhado com forquilha à esquerda;
- preservação dos testes anteriores de resistor, capacitor, diodo, LED, VCC, GND, OCR e regeneração automática.

Para conferir a versão:

GET /system/version

Build esperado:

2026-09-14-stage7-transistor-fork-v1.9
