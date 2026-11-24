# Correção da Etapa 7 — LED/diodo por eixo elétrico

Problema observado em desenho manual: as setas de emissão do LED alargavam a bounding box e faziam um LED vertical ser interpretado como horizontal. Com isso A/K eram colocados à esquerda/direita.

Correção: para LED e diodo, a orientação passa a priorizar o par de bordas opostas onde os fios elétricos realmente entram no símbolo. Se houver traços alinhados no topo e na base, o eixo é vertical; se houver traços alinhados à esquerda e à direita, o eixo é horizontal. As setas de emissão não participam desse cálculo.

Além disso, quando o eixo é obtido dessa forma, os pontos dos terminais são retirados diretamente das interseções dos fios com as bordas da detecção. O Hough não pode mais inclinar A/K usando as setas do LED.

Build: 2026-09-14-stage7-led-axis-v2.0
