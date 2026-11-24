# Etapa 7 automática — pontos de terminais após OCR

Fluxo atual:

1. `POST /projects/{project_id}/detect` reconhece os componentes.
2. `POST /projects/{project_id}/ocr` lê referências/valores e, no mesmo request, executa automaticamente a Etapa 7.
3. A resposta do OCR inclui `terminals_generated_automatically`, `terminal_preview_url`, `terminals_url` e `terminals_json_url`.
4. `terminals.png` é uma imagem de depuração com bounding box, identificação do componente e pontos dos pinos.
5. `terminals.json` contém as coordenadas e a função elétrica de cada pino.

A rota `POST /projects/{project_id}/terminals` continua existindo para forçar o recálculo manual.

Ao editar componentes em `PUT /projects/{project_id}/ocr/components`, os terminais também são recalculados automaticamente.

Se somente a detecção de terminais falhar, o OCR permanece salvo e a resposta inclui `terminal_detection_warning` para facilitar a depuração.
