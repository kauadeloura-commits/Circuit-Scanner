from pathlib import Path
import json,csv,hashlib,collections,statistics
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'evaluation_20261019';OUT=ROOT.parent.parent/'outputs/correcao_e_reavaliacao_20261019';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def csvfile(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter=';');w.writeheader()
  for row in rows:w.writerow({k:(format(v,'.10g').replace('.',',') if isinstance(v,float) else v) for k,v in row.items()})
before=read(BASE/'analysis/summary.json');after=read(ROOT/'analysis/summary.json')
assert after['final'] and after['completed']==323
assert sha(ROOT/'cases_frozen.json')==sha(BASE/'cases_frozen.json')
assert sha(ROOT/'topology_frozen.json')==sha(BASE/'topology_frozen.json')
sealed=read(ROOT/'revision_asset_seal.json');changes=[]
for row in sealed:
 p=ROOT/'runtime'/row['path']
 if not p.is_file() or sha(p)!=row['sha256']:changes.append(row['path'])
assert not changes,changes
cases=read(ROOT/'cases_frozen.json')
assert all(sha(Path(c['input']))==c['input_sha256'] for c in cases)
old={r['case_id']:r for r in read(BASE/'analysis/cases.json')};new=read(ROOT/'analysis/cases.json')
assert all(r['manual_edits']==0 for r in new)
counts=collections.Counter(c['class_name'] for c in cases if c['cohort']=='isolated')
assert len(counts)==15 and set(counts.values())=={20}
metrics=[]
def metric(label,a,b,n):
 assert n>0
 metrics.append(dict(criterio=label,acertos_antes=a,acertos_depois=b,total=n,porcentagem_antes=round(100*a/n,4),porcentagem_depois=round(100*b/n,4),diferenca_pontos_percentuais=round(100*(b-a)/n,4)))
for g,label in [('isolated','Componentes isolados'),('whole','Componentes nos circuitos completos')]:
 a=next(x for x in before['class_metrics'] if x['cohort']==g and x['class_name']=='ALL');b=next(x for x in after['class_metrics'] if x['cohort']==g and x['class_name']=='ALL');assert a['gt']==b['gt'];metric(label,a['tp'],b['tp'],b['gt'])
for k,label in [('reference','Nomes lidos e associados corretamente'),('value','Valores lidos e associados corretamente'),('model','Modelos lidos e associados corretamente')]:
 a,b=before['ocr'][k],after['ocr'][k];assert a['eligible']==b['eligible'];metric(label,a['correct'],b['correct'],b['eligible'])
metric('Circuitos com deteccao integral',before['whole_integral_detection'],after['whole_integral_detection'],23)
assert before['graph']['circuits']==after['graph']['circuits']==11
metric('Circuitos com todas as ligacoes corretas',before['graph']['exact'],after['graph']['exact'],11)
classrows=[]
for b in after['class_metrics']:
 a=next(x for x in before['class_metrics'] if x['cohort']==b['cohort'] and x['class_name']==b['class_name'])
 classrows.append(dict(grupo=b['cohort'],componente=b['class_name'],esperados=b['gt'],corretos_antes=a['tp'],corretos_depois=b['tp'],fn_antes=a['fn'],fn_depois=b['fn'],fp_antes=a['fp'],fp_depois=b['fp'],revocacao_antes=a['recall'],revocacao_depois=b['recall'],precisao_antes=a['precision'],precisao_depois=b['precision']))
caserows=[]
for b in new:
 a=old[b['case_id']]
 caserows.append(dict(teste=b['case_id'],grupo=b['cohort'],origem=b['source_id'],esperados=b['expected_components'],corretos_antes=a['correct_components'],corretos_depois=b['correct_components'],fn_antes=a['false_negatives'],fn_depois=b['false_negatives'],fp_antes=a['false_positives'],fp_depois=b['false_positives'],nomes_antes=a['reference_correct'],nomes_depois=b['reference_correct'],valores_antes=a['value_correct'],valores_depois=b['value_correct'],modelos_antes=a['model_correct'],modelos_depois=b['model_correct'],ligacoes_exatas_antes=a['graph_exact'],ligacoes_exatas_depois=b['graph_exact'],analise_segundos_antes=a['analysis_seconds'],analise_segundos_depois=b['analysis_seconds'],omitidos_depois=json.dumps(b['missed_names'],ensure_ascii=False),classes_erradas_depois=json.dumps(b['misclassified_names'],ensure_ascii=False),erros_depois=json.dumps(b['handler_errors'],ensure_ascii=False)))
cache_reuse=[];internal_errors=[];nested=[]
for c in cases:
 folder=ROOT/'results'/c['id'];run=read(folder/'result.json')
 assert run['manual_edits']==run['prediction_overrides']==0
 for s in run['stages']:
  if not s['ok'] and s.get('http_status',0)==500:internal_errors.append(dict(case=c['id'],stage=s['stage'],error=s.get('error')))
 for stage in ['detect','ocr_and_automatic_geometry']:
  p=folder/(stage+'_response.json')
  if p.exists() and read(p).get('processing_reused'):cache_reuse.append([c['id'],stage])
 p=folder/'detections.json'
 if p.exists():
  ds=read(p).get('detections',[])
  for i,a in enumerate(ds):
   for b in ds[i+1:]:
    aa,bb=a['bbox'],b['bbox']
    contains=lambda x,y:all(x[k]<=y[k] for k in ['x1','y1']) and all(x[k]>=y[k] for k in ['x2','y2'])
    if contains(aa,bb) or contains(bb,aa):nested.append([c['id'],a['detection_id'],b['detection_id']])
assert not cache_reuse,'Final rerun unexpectedly reused previous analysis'
summary=dict(before_version='5.48.0',after_version='5.49.0',cases=323,metrics=metrics,latency_before=before['latency'],latency_after=after['latency'],graph_before=before['graph'],graph_after=after['graph'],stage_errors_before=before['stage_errors'],stage_errors_after=after['stage_errors'],regression_tests=read(ROOT/'regression_tests.json'),internal_errors=internal_errors,nested_boxes=nested,inputs_unchanged=True,ground_truth_unchanged=True,revision_assets_unchanged_during_run=True,manual_prediction_corrections=0,cached_stage_reuses=cache_reuse,interpretation='Same historical cases used to investigate and fix bugs; regression comparison, not independent generalization accuracy. Ground truths visually assisted by AI without independent human adjudication. No validated electrical simulation.')
csvfile('Porcentagens_antes_e_depois.csv',metrics);csvfile('Classes_antes_e_depois.csv',classrows);csvfile('Testes_antes_e_depois.csv',caserows)
save(OUT/'Comparacao_completa.json',summary);save(ROOT/'comparison_complete.json',summary)
print(json.dumps(dict(metrics=metrics,latency_after=after['latency'],internal_errors=internal_errors,nested_boxes=nested),ensure_ascii=True,indent=2))
