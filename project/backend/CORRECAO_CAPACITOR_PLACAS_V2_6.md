# Correção do capacitor v2.6

A Etapa 7 agora detecta explicitamente o par de placas paralelas do capacitor.

- placas horizontais -> eixo elétrico vertical;
- placas verticais -> eixo elétrico horizontal;
- linhas pautadas longas recebem penalização;
- duas bordas do mesmo fio grosso são rejeitadas pela quantidade de tinta entre elas;
- a regra das placas tem prioridade quando o par é estruturalmente confiável;
- fios/borda e bbox continuam como fallback.

Isso corrige capacitores desenhados à mão cuja bounding box fica mais larga do que alta.
