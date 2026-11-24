# Correção de falsos positivos — VCC e transistores

## Problema
O detector aceitava pontos/junções como `vcc` e linhas diagonais como `transistor_npn` ou `transistor_pnp`, principalmente em imagens P&B analisadas com limiar de candidato baixo.

## Correção imediata (sem retreinar)
Após o consenso do YOLO, as classes mais problemáticas passam por validação geométrica:

- `vcc`: exige haste vertical + dois ramos diagonais formando o chevron superior.
- `transistor_npn` / `transistor_pnp`: exige haste da base + chegada horizontal da base + ramo diagonal superior + ramo diagonal inferior, conectados como uma forquilha BJT.

Detecções rejeitadas aparecem em `detections.json` nos campos:

- `geometry_rejected_count`
- `geometry_rejections`
- `rejection_reason`

Razões atuais:

- `vcc_shape_missing`
- `bjt_fork_missing`

## Correção para o próximo treinamento
`dataset_generator.py` agora também cria hard negatives com labels YOLO vazios:

- pontos isolados;
- ponto de junção com fios;
- T;
- X;
- linha diagonal;
- diagonais paralelas;
- V sem símbolo;
- canto de fio.

Esses negativos só afetam um novo modelo depois de regenerar o dataset e retreinar.

## Testes
Execute:

```bash
python -m unittest backend.test_false_positive_filters backend.test_bw_recognition backend.test_terminal_detection -v
```

Resultado esperado nesta versão: 26 testes OK.
