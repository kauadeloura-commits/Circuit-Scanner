# Correção NPN v3.11

Diagnóstico real observado no projeto PRJ_0242:
- modelo carregável: true
- box_confidence NPN ~0.462
- mean keypoint confidence ~0.719
- B/C/E preditos colapsados
- geometry_reasons: B_C_colapsados, C_E_colapsados
- known_training_match: null

A correção não reduz thresholds. Em vez disso, adiciona memória visual por features locais (SIFT; ORB como fallback) + RANSAC para reencontrar uma amostra manual já ensinada mesmo com mudança de escala, rotação, compressão e bbox. Os B/C/E manuais são transportados pela transformação afim estimada.

Novos diagnósticos em known_training_match:
- feature_similarity
- feature_inliers
- feature_good_matches
- feature_inlier_ratio
- feature_bbox_iou
- feature_detector
- point_mapping=feature_affine_transform quando utilizado

O modelo treinado continua sendo rejeitado quando B/C/E colapsam; não aceitamos uma pose geometricamente inválida apenas por ter confiança alta.
