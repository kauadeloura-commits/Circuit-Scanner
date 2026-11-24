# v3.8 — memória de amostras já treinadas

Problema corrigido: uma foto já corrigida e usada no treino podia voltar a exibir `TREINAR B/C/E` quando o modelo Pose tivesse baixa confiança.

Nova ordem de prioridade para transistores:
1. correção manual persistida no próprio projeto;
2. ground truth de uma amostra conhecida da mesma foto, localizada por similaridade da imagem completa + posição/bbox do transistor;
3. modelo `models/terminal_keypoints.pt`;
4. revisão manual quando o modelo treinado existe mas não produz uma previsão aceitável;
5. heurística apenas quando não há modelo treinado carregável.

A memória de amostra conhecida reutiliza B/C/E originalmente marcados pelo usuário. Ela também suporta a mesma foto reaberta em outro projeto e pequenas diferenças na bbox do detector.

O endpoint `/projects/{project_id}/terminal-learning/diagnostics` agora informa `known_training_match` e `effective_source`.
