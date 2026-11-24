# Etapa 7 — regeneração automática dos terminais

Versão: `2026-09-14-stage7-live-regeneration-v1.8`

## Regra do fluxo

Os terminais são sempre derivados da lista de componentes mais recente.

1. `POST /projects/{id}/detect`
   - salva as detecções automáticas;
   - invalida OCR/terminais antigos;
   - gera `terminals.json` e `terminals.png` provisórios usando classe + bbox.

2. `PUT /projects/{id}/detections`
   - salva inclusões, remoções, correções de classe e bbox feitas pelo usuário;
   - apaga `ocr.png`, `components.json`, `terminals.png` e `terminals.json` antigos;
   - recria imediatamente `terminals.json` e `terminals.png` usando a lista corrigida;
   - o OCR fica invalidado e deve ser executado novamente para atualizar referência/valor.

3. `POST /projects/{id}/ocr`
   - reconstrói `components.json` a partir das detecções atuais;
   - recria os terminais novamente, agora usando os componentes lógicos do OCR.

4. `PUT /projects/{id}/ocr/components`
   - salva correções manuais de referência/valor;
   - recria `terminals.json` e `terminals.png` novamente.

5. `POST /projects/{id}/terminals`
   - pode ser usado como recálculo manual;
   - prefere `ocr_info` quando existe;
   - caso o OCR esteja invalidado, usa as detecções atuais.

## Garantia contra arquivo obsoleto

Antes de cada nova geração, `terminals.png` e `terminals.json` antigos são removidos. Se a nova geração falhar, não permanece um preview velho na pasta do projeto.

## Como validar

- Rode o reconhecimento automático e confira `terminals.png`.
- Adicione manualmente um componente que o detector deixou de fora e salve a correção.
- Sem chamar `/terminals`, abra novamente `terminals.png`.
- O novo componente deve aparecer e o `terminals.json` deve ter a nova contagem.
- Depois rode OCR; os terminais devem ser gerados novamente usando as referências/valores atualizados.
