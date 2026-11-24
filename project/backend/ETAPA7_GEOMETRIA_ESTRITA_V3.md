# Etapa 7 — Geometria estrita v3.0

## Regra principal

A classe reconhecida vem primeiro. Depois cada classe usa uma rotina geométrica própria.

### Transistor NPN/PNP
- A classe `transistor_npn` ou `transistor_pnp` é preservada antes de calcular os pinos.
- A barra reta define B.
- São exigidos dois ramos distintos da forquilha.
- O ramo que contém a cabeça da seta é E. O outro é C.
- NPN espera seta para fora; PNP espera seta para dentro. Divergência é marcada para revisão.
- Se a seta não puder ser confirmada, C/E aparecem com `?` no preview e o componente é marcado para revisão, em vez de fingir alta confiança.

### Resistor
- Retângulo real do corpo tem prioridade sobre a bounding box.
- Em resistor desenhado/zig-zag, PCA da tinta central é usada como fallback.
- O terminal é rastreado a partir do corpo ao longo do eixo do componente até o traço elétrico real.

### Capacitor
- As duas placas paralelas têm prioridade sobre bbox e texto.
- O eixo elétrico é perpendicular às placas.
- Os terminais são rastreados pelo eixo que passa no centro comum das placas.

### Segurança
Resultados geometricamente ambíguos são marcados para revisão. O sistema não deve apresentar uma identificação C/E incerta como se fosse confirmada.
