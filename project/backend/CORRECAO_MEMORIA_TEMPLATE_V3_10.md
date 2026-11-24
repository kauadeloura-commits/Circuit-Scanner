# Etapa 7 v3.10 — Memória visual por template

Problema: a v3.9 ainda podia pedir TREINAR B/C/E para um transistor já ensinado quando a bounding box atual mudava muito em relação à bbox usada no treino.

Correção:
- o crop salvo no treino é procurado diretamente na imagem atual em múltiplas escalas;
- a bbox atual limita apenas a região de busca, não define mais a correspondência;
- quando o template conhecido é encontrado, B/C/E são transportados pela mesma transformação de escala/translação;
- a memória conhecida continua tendo prioridade sobre o modelo Pose;
- modelo Pose continua sendo usado para imagens novas.

Diagnóstico disponível em `/projects/{project_id}/terminal-learning/diagnostics`:
- `known_training_match.template_similarity`
- `known_training_match.template_scale`
- `known_training_match.point_mapping`

Se `point_mapping` for `multiscale_template_transform`, a amostra antiga foi reconhecida diretamente pelo crop visual treinado.
