# Etapa 7 — Detecção de terminais dos componentes

Esta versão prioriza os componentes atualmente usados nos testes do TC:

- VCC
- GND
- resistor
- capacitor não polarizado
- transistor NPN
- transistor PNP
- diodo
- LED

## Objetivo

A Etapa 7 não constrói ainda as redes (`NET_1`, `NET_2`, ...). Ela identifica onde cada pino elétrico do componente está localizado para que a Etapa 8 detecte fios e a Etapa 10 possa relacionar fio + terminal corretamente.

## Regras elétricas priorizadas

| Classe | Terminais | Regra atual do símbolo treinado |
|---|---|---|
| `vcc` | `VCC` | conexão na haste inferior |
| `gnd` | `GND` | conexão na haste superior |
| `resistor` | `1`, `2` | esquerda/direita; topo/base se vertical |
| `capacitor` | `1`, `2` | esquerda/direita; topo/base se vertical |
| `diode` | `A`, `K` | anodo à esquerda e catodo à direita no símbolo canônico treinado |
| `led` | `A`, `K` | anodo à esquerda e catodo à direita no símbolo canônico treinado |
| `transistor_npn` | `B`, `C`, `E` | base à esquerda; coletor superior direito; emissor inferior direito |
| `transistor_pnp` | `B`, `C`, `E` | base à esquerda; coletor superior direito; emissor inferior direito |

Os símbolos do dataset recebem pequenas rotações. O algoritmo usa a orientação da bounding box e a imagem local para ajustar os pontos.

## Correção importante da versão 1.2

A versão anterior procurava o pixel escuro mais próximo da posição teórica. Em transistores, isso podia fazer `C` e `E` grudarem na circunferência do símbolo.

A versão 1.2 usa a **extremidade real do traço na borda elétrica esperada**. Cada terminal também recebe:

- `electrical_role`: papel elétrico (`base`, `collector`, `emitter`, `anode`, `cathode`, `power`, `ground`, ...)
- `terminal_side`: lado geométrico (`left`, `right`, `top`, `bottom`)
- `position`: coordenada usada posteriormente para associar fios
- `search_radius_px`: região de tolerância para a Etapa 8

Exemplo de transistor NPN:

```json
{
  "pin": "C",
  "electrical_role": "collector",
  "terminal_side": "right",
  "position": {"x": 212.0, "y": 57.0}
}
```

## Endpoint

Depois da Etapa 6:

```text
POST /projects/{project_id}/terminals
```

Arquivos gerados:

- `terminals.json`
- `terminals.png`

## Testes

Execute, na pasta que contém `backend`:

```bash
python -m unittest backend.test_terminal_detection -v
```

A suíte possui testes específicos para os oito tipos priorizados e testes de regressão da Etapa 7.

## Limite atual importante

Diodo/LED e transistor usam a orientação canônica presente no dataset atual para nomear os pinos polarizados. A primeira versão do dataset aplica pequenas rotações, mas não treina todas as possíveis inversões de 90°/180°. Se futuramente o detector aceitar símbolos completamente invertidos, será necessário também inferir visualmente a barra do catodo e a seta do emissor antes de considerar `A/K` e `C/E` automaticamente confirmados.

## Ajuste específico — transistores BJT NPN/PNP

Na versão stage7-1.3, os terminais de `transistor_npn` e `transistor_pnp` não são mais posicionados na borda da bounding box do YOLO.

O fluxo agora é:

1. usar a classe reconhecida para saber se o componente é NPN ou PNP;
2. localizar, dentro da bbox, segmentos de linha com HoughLinesP;
3. normalizar pequenas rotações do símbolo;
4. identificar a linha da base, a haste central e os dois ramos diagonais da forquilha;
5. estimar as junções B/C/E diretamente sobre o desenho da forquilha;
6. refinar C/E usando cantos locais da imagem;
7. usar uma posição interna sobre a forquilha como fallback se as linhas não estiverem legíveis.

O `terminals.json` passa a incluir `transistor_geometry`, com `transistor_type`, grupos de linhas encontrados e a origem da estimativa. Isso ajuda a depurar casos reais sem voltar a depender das extremidades da bbox.
