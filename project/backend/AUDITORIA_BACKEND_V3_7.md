# Auditoria do backend — Etapa 7 v3.7

## Achados principais

1. **O modelo treinado não fica dentro de `backend/`.**
   O código usa `BASE_DIR = Path(__file__).resolve().parent.parent`, portanto:
   - backend: `project/backend/`
   - modelo: `project/models/terminal_keypoints.pt`
   - amostras: `project/terminal_keypoints/`

2. **`model_exists: true` era insuficiente.**
   Antes o status só confirmava que o arquivo existia. Agora informa também `model_loadable`, `model_task`, classes, tamanho, data e erro de carregamento.

3. **O modelo treinado podia ser descartado cedo demais.**
   Havia limiar de 0.50 para caixa e média dos keypoints. Agora a inferência é diagnosticada em níveis: aceita, aceita com revisão ou rejeita.

4. **O Pose-model estava reclassificando NPN/PNP.**
   Isso contrariava o pipeline. Agora a classe do detector principal é autoridade e o modelo de pose só localiza B/C/E.

5. **Heurística podia sobrescrever um modelo rejeitado.**
   Agora, se existe `terminal_keypoints.pt` carregável, a heurística de transistor NÃO é usada. Se a previsão do modelo for rejeitada, o transistor é enviado para revisão/treino.

6. **Snap global podia deslocar B/C/E aprendidos.**
   O snap entre terminais próximos não atua mais em transistores nem em ground-truth/modelo treinado.

7. **Amostras de treino podiam clipar keypoints.**
   O recorte antigo partia principalmente da bbox e podia cortar B/C/E fora dela. Novas correções criam o crop contendo bbox + todos os keypoints com margem.

8. **Validação podia ter vazamento entre circuitos.**
   Agora train/val é separado por `project_id`: transistores da mesma imagem não ficam em train e val ao mesmo tempo.

9. **Correções manuais podiam se perder quando DET_xxx mudava.**
   Além do ID, o backend agora tenta reaplicar ground-truth por tipo + IoU da bbox.

10. **Conexões SQLite não eram fechadas explicitamente.**
    O context manager agora fecha a conexão, eliminando warnings de recursos.

## Rotas para diagnóstico

### Status completo

`GET /terminal-learning/status`

Verifique principalmente:
- `model_exists`
- `model_loadable`
- `model_load_error`
- `dataset_quality`
- `new_samples_since_training`

### Runtime do arquivo .pt

`GET /terminal-learning/model-runtime`

Confirma se o arquivo existe **e abre como modelo Pose**.

### Diagnóstico por transistor de um projeto

`GET /projects/{project_id}/terminal-learning/diagnostics`

Para cada NPN/PNP mostra:
- confiança da box;
- confiança média e individual de B/C/E;
- coordenadas previstas;
- classe prevista pelo pose-model (apenas diagnóstico);
- se foi aceita/rejeitada;
- motivo da rejeição.

## Política atual

1. Correção manual persistida — prioridade máxima.
2. Modelo `terminal_keypoints.pt` carregável — prioridade automática.
3. Modelo rejeitou a previsão — marcar para revisão, sem heurística.
4. Modelo não existe/não carrega — heurística geométrica como fallback.

## Pastas que NÃO devem ser substituídas ao atualizar o backend

- `database/`
- `projects/`
- `models/`
- `terminal_keypoints/`

Substitua apenas `backend/`.
