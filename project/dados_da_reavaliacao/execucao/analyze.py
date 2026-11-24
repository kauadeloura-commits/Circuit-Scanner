from pathlib import Path
import json, math, collections, itertools, re, csv, statistics, hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image,ImageDraw,ImageFont
from prepare import ROOT,CLASSES,dump,sha
OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
PT=['Resistor','Capacitor','Capacitor polarizado','Diodo','LED','Indutor','Fonte DC','Bateria','GND','VCC','Transistor NPN','Transistor PNP','MOSFET N','MOSFET P','Amplificador operacional']
def read(p,default=None):
    return json.loads(p.read_text(encoding='utf8')) if p.exists() else ({} if default is None else default)
def iou(a,b):
    x=max(0,min(a['x2'],b['x2'])-max(a['x1'],b['x1']));y=max(0,min(a['y2'],b['y2'])-max(a['y1'],b['y1']))
    inter=x*y;union=(a['x2']-a['x1'])*(a['y2']-a['y1'])+(b['x2']-b['x1'])*(b['y2']-b['y1'])-inter
    return inter/union if union>0 else 0
def matches(gt,pred,threshold=.5):
    if not gt or not pred:return []
    mat=np.array([[iou(g,p.get('bbox',p)) for p in pred] for g in gt]);eligible=np.where(mat>=threshold,mat,0)
    a,b=linear_sum_assignment(eligible,maximize=True)
    return [(int(x),int(y),float(mat[x,y])) for x,y in zip(a,b) if mat[x,y]>=threshold]
def pred_class(p):return p.get('class',p.get('type','unknown'))
def detection_scores(gt,pred,threshold=.5):
    matched=matches(gt,pred,threshold);gseen={x for x,_,_ in matched};pseen={y for _,y,_ in matched}
    rows=[]
    for x,y,ov in matched:
        good=CLASSES[int(gt[x]['class_id'])]==pred_class(pred[y]);rows.append(dict(gt_index=x,pred_index=y,iou=ov,status='correct' if good else 'wrong_class'))
    rows.extend(dict(gt_index=x,pred_index=None,iou=None,status='missed') for x in range(len(gt)) if x not in gseen)
    rows.extend(dict(gt_index=None,pred_index=y,iou=None,status='extra') for y in range(len(pred)) if y not in pseen)
    counts=collections.Counter(r['status'] for r in rows);tp=counts['correct'];wrong=counts['wrong_class'];miss=counts['missed'];extra=counts['extra']
    return dict(gt=len(gt),pred=len(pred),tp=tp,wrong_class=wrong,missed_localization=miss,extra=extra,fn=wrong+miss,fp=wrong+extra,precision=tp/len(pred) if pred else None,recall=tp/len(gt) if gt else None,rows=rows)
def normtext(s):return re.sub(r'\s+','',str(s or '')).upper().replace('Ω','Ω')
def numeric(s):
    if s is None:return None
    s=str(s).strip().replace(' ','').replace(',','.').replace('µ','u').replace('μ','u').replace('Ω','Ω')
    # Embedded engineering notation e.g. 4k7, 3k3, 1k5.
    s=re.sub(r'(\d+)([kKMGmunp])(\d+)',r'\1.\3\2',s)
    m=re.fullmatch(r'([+-]?\d+(?:\.\d+)?)([kKMGmunp]?)(Ω|[FHVWA]|ohm|OHM|Ohm)?',s)
    if not m:return None
    v=float(m[1])*{'':1,'k':1e3,'K':1e3,'M':1e6,'G':1e9,'m':1e-3,'u':1e-6,'n':1e-9,'p':1e-12}[m[2]]
    unit=(m[3] or '').replace('OHM','Ω').replace('Ohm','Ω').replace('ohm','Ω')
    return v,unit
def samevalue(g,v):
    expected=numeric(g);actual=numeric(v.get('text'))
    if actual is None and isinstance(v.get('normalized'),(int,float)):
        unit=str(v.get('unit') or '').replace('ohm','Ω');actual=(float(v['normalized']),unit)
    if expected and actual:
        sameunit=expected[1]==actual[1] or not actual[1]
        return sameunit and math.isclose(expected[0],actual[0],rel_tol=1e-6,abs_tol=1e-12)
    return normtext(g)==normtext(v.get('text'))
def describe(g):
    s=PT[int(g['class_id'])]+' '+(g.get('reference') or g['gt_id'])
    return s
def pairs(nets):return {tuple(sorted(p)) for net in nets for p in itertools.combinations(set(net),2)}
def graph_score(case,folder,gt_graph):
    if not gt_graph:return None
    circuit=read(folder/'circuit.json');term=read(folder/'terminals.json');comps=term.get('components',[]);gt=case['ground_truth'];gtby={g['gt_id']:g for g in gt}
    cmatch=matches(gt,comps,.5);maps={}
    for x,y,_ in cmatch:
        if CLASSES[int(gt[x]['class_id'])]==comps[y].get('type'):maps[str(comps[y]['component_id'])]=gt[x]['gt_id']
    logical={c['id']:str(c.get('metadata',{}).get('source_component_id')) for c in circuit.get('components',[])}
    passive={'resistor','capacitor','inductor'};pinmaps={}
    for c in comps:
        sid=str(c.get('component_id'));gid=maps.get(sid)
        if not gid:continue
        if c.get('type') in passive and len(c.get('terminals',[]))==2:
            g=gtby[gid];t1,t2=c['terminals'];p1=t1.get('position',{});p2=t2.get('position',{})
            axis='x' if abs(float(p1.get('x',0))-float(p2.get('x',0)))>abs(float(p1.get('y',0))-float(p2.get('y',0))) else 'y'
            ordered=sorted(c['terminals'],key=lambda t:float(t.get('position',{}).get(axis,0)))
            for n,t in enumerate(ordered):pinmaps[(sid,str(t['pin']))]=str(n+1)
    transformed=[]
    for net in circuit.get('nets',[]):
        members=[]
        for member in net.get('members',[]):
            cid,pin=member.rsplit('.',1);sid=logical.get(cid,cid);gid=maps.get(sid)
            members.append((gid+'.'+pinmaps.get((sid,pin),pin)) if gid else 'UNMATCHED:'+member)
        transformed.append(members)
    expected=pairs(gt_graph['nets']);actual=pairs(transformed)
    tp=len(expected&actual);fp=len(actual-expected);fn=len(expected-actual)
    return dict(expected_pairs=len(expected),predicted_pairs=len(actual),tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,exact=expected==actual,expected_nets=gt_graph['nets'],predicted_mapped_nets=transformed,missing_pairs=[list(p) for p in sorted(expected-actual)],extra_pairs=[list(p) for p in sorted(actual-expected)],matched_components=len(maps),generated=bool(circuit))
def csvfile(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields,delimiter=';');w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def analyze(final=False):
    cases=read(ROOT/'cases_frozen.json',[]);topology=read(ROOT/'topology_frozen.json');case_rows=[];object_rows=[];field_rows=[];confusion=collections.Counter();full=[]
    for case in cases:
        cid=case['id'];folder=ROOT/'results'/cid;run=read(folder/'result.json')
        if not run:continue
        gt=case['ground_truth'];before=read(folder/'detections_before_ocr.json').get('detections',[]);pred=read(folder/'detections.json').get('detections',before)
        score=detection_scores(gt,pred);raw=detection_scores(gt,before);loose=detection_scores(gt,pred,.25)
        ocr=read(folder/'components.json');comps=ocr.get('components',[]);terms=read(folder/'terminals.json');wires=read(folder/'wires.json');circuit=read(folder/'circuit.json')
        c_matches=matches(gt,comps,.5);bygt={x:comps[y] for x,y,_ in c_matches}
        refmultiplicity=collections.Counter(normtext(g.get('reference')) for g in gt if g.get('reference'))
        fields=[]
        for gi,g in enumerate(gt):
            c=bygt.get(gi);correcttype=bool(c and c.get('type')==CLASSES[int(g['class_id'])])
            for kind in ['reference','value','model']:
                if not g.get(kind):continue
                if kind=='reference' and refmultiplicity[normtext(g[kind])]>1:eligible=False
                else:eligible=True
                observed=(c or {}).get(kind) or {};source=observed.get('source','missing');read_by_ocr=source=='ocr'
                textok=samevalue(g[kind],observed) if kind=='value' else normtext(g[kind])==normtext(observed.get('text'))
                correct=bool(eligible and correcttype and read_by_ocr and textok)
                f=dict(case_id=cid,source_id=case['source_id'],gt_id=g['gt_id'],class_name=CLASSES[int(g['class_id'])],field=kind,expected=g[kind],observed=observed.get('text',''),source=source,component_detected_correctly=correcttype,eligible=eligible,read_by_ocr=read_by_ocr,text_matches=textok,correctly_assigned=correct,reason='non_unique_reference' if not eligible else 'component_not_correctly_matched' if not correcttype else 'not_read_by_ocr' if not read_by_ocr else 'different_text_or_value' if not textok else 'correct')
                fields.append(f);field_rows.append(f)
        missed=[];wrong=[];detected=[];extra=[]
        for r in score['rows']:
            g=gt[r['gt_index']] if r['gt_index'] is not None else None;p=pred[r['pred_index']] if r['pred_index'] is not None else None
            expected=CLASSES[int(g['class_id'])] if g else '(background)';observed=pred_class(p) if p else '(missed)'
            confusion[(case['cohort'],expected,observed)]+=1
            row=dict(case_id=cid,cohort=case['cohort'],source_id=case['source_id'],gt_id=g.get('gt_id') if g else None,expected_class=expected,observed_class=observed,status=r['status'],iou=r['iou'],expected_reference=g.get('reference') if g else None,expected_value=g.get('value') if g else None,expected_model=g.get('model') if g else None,prediction_id=p.get('detection_id') if p else None,prediction_confidence=p.get('confidence') if p else None,gt_bbox=g if g else None,pred_bbox=p.get('bbox') if p else None)
            oc=next((oc for oc in comps if p and oc.get('detection_id')==p.get('detection_id')),None)
            for field in ['reference','value','model']:
                observed_field=(oc or {}).get(field) or {};row['observed_'+field]=observed_field.get('text');row[field+'_source']=observed_field.get('source')
            object_rows.append(row)
            if r['status']=='correct':detected.append(describe(g))
            elif r['status']=='missed':missed.append(describe(g))
            elif r['status']=='wrong_class':wrong.append(describe(g)+' → '+(PT[CLASSES.index(observed)] if observed in CLASSES else observed))
            else:extra.append((PT[CLASSES.index(observed)] if observed in CLASSES else observed)+' '+str(p.get('detection_id','')))
        stages=run.get('stages') or read(folder/'progress.json').get('stages',[]);times={s['stage']:s['seconds'] for s in stages};issues=read(folder/'review_issues_readonly_response.json',[])
        graph=graph_score(case,folder,topology.get(cid))
        main_seconds=sum(times.get(k,0) for k in ['process','detect','ocr_and_automatic_geometry'])
        row=dict(case_id=cid,cohort=case['cohort'],source_id=case['source_id'],source_type=case['source_type'],width=case['width'],height=case['height'],expected_components=len(gt),detected_components=len(pred),correct_components=score['tp'],missed_localization=score['missed_localization'],wrong_class=score['wrong_class'],extra_boxes=score['extra'],false_negatives=score['fn'],false_positives=score['fp'],precision=score['precision'],recall=score['recall'],raw_correct=raw['tp'],correct_iou25=loose['tp'],integral_detection=score['tp']==len(gt) and score['fp']==0,reference_eligible=sum(f['field']=='reference' and f['eligible'] for f in fields),reference_correct=sum(f['field']=='reference' and f['correctly_assigned'] for f in fields),value_eligible=sum(f['field']=='value' and f['eligible'] for f in fields),value_correct=sum(f['field']=='value' and f['correctly_assigned'] for f in fields),model_eligible=sum(f['field']=='model' and f['eligible'] for f in fields),model_correct=sum(f['field']=='model' and f['correctly_assigned'] for f in fields),ocr_token_count=ocr.get('token_count',0),references_reported_by_app=ocr.get('reference_ocr_count',0),values_reported_by_app=ocr.get('value_ocr_count',0),models_reported_by_app=ocr.get('model_ocr_count',0),terminals_generated=terms.get('terminal_count',0),junctions_review=(wires.get('stage9') or {}).get('review_count'),graph_generated=bool(circuit),graph_exact=graph.get('exact') if graph else None,simulation_gate_allowed=any(s['stage']=='simulation_gate' and s['ok'] for s in stages),process_seconds=times.get('process'),detection_seconds=times.get('detect'),ocr_geometry_seconds=times.get('ocr_and_automatic_geometry'),analysis_seconds=main_seconds,all_handler_seconds=sum(times.values()),case_seconds=run.get('total_seconds'),execution_status=run['status'],handler_errors=[dict(stage=s['stage'],error=s.get('error'),http_status=s.get('http_status')) for s in stages if not s['ok']],detected_names=detected,missed_names=missed,misclassified_names=wrong,extra_names=extra,graph_issues=issues,manual_edits=run.get('manual_edits',0))
        row.update(junction_seconds=times.get('junctions'),connectivity_seconds=times.get('connectivity_diagnostic'),circuit_seconds=times.get('circuit_diagnostic'))
        case_rows.append(row);full.append(dict(case=case,summary=row,score=score,raw_score=raw,loose_score=loose,fields=fields,graph=graph,components=comps))
    summaries=[]
    for cohort in ['isolated','whole']:
        rr=[r for r in case_rows if r['cohort']==cohort]
        for kind in CLASSES+['ALL']:
            oo=[r for r in object_rows if r['cohort']==cohort and (kind=='ALL' or r['expected_class']==kind or r['observed_class']==kind)]
            gt_n=sum(r['expected_class']!='(background)' and (kind=='ALL' or r['expected_class']==kind) for r in oo);pred_n=sum(r['observed_class']!='(missed)' and (kind=='ALL' or r['observed_class']==kind) for r in oo)
            tp=sum(r['status']=='correct' and (kind=='ALL' or r['expected_class']==kind) for r in oo);fp=pred_n-tp;fn=gt_n-tp
            pre=tp/pred_n if pred_n else None;rec=tp/gt_n if gt_n else None;f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
            summaries.append(dict(cohort=cohort,class_name=kind,gt=gt_n,pred=pred_n,tp=tp,fp=fp,fn=fn,precision=pre,recall=rec,f1=f1))
    def distribution(vals):
        vals=[v for v in vals if v is not None]
        return dict(n=len(vals),mean=float(np.mean(vals)) if vals else None,median=float(np.median(vals)) if vals else None,p95=float(np.percentile(vals,95)) if vals else None,min=min(vals) if vals else None,max=max(vals) if vals else None,up_to_30_seconds=sum(x<=30 for x in vals),up_to_60_seconds=sum(x<=60 for x in vals))
    ocr_summary={}
    for field in ['reference','value','model']:
        rr=[f for f in field_rows if f['field']==field and f['eligible']];conditional=[f for f in rr if f['component_detected_correctly']]
        correct=sum(f['correctly_assigned'] for f in rr)
        ocr_summary[field]=dict(eligible=len(rr),correct=correct,end_to_end=correct/len(rr) if rr else None,conditional_eligible=len(conditional),conditional_correct=sum(f['correctly_assigned'] for f in conditional),conditional_accuracy=sum(f['correctly_assigned'] for f in conditional)/len(conditional) if conditional else None,read_by_ocr=sum(f['read_by_ocr'] and f['component_detected_correctly'] for f in rr),non_unique_excluded=sum(f['field']==field and not f['eligible'] for f in field_rows))
    graph_rows=[dict(case_id=f['case']['id'],**f['graph']) for f in full if f['graph']]
    graph_summary=dict(circuits=len(graph_rows),generated=sum(g['generated'] for g in graph_rows),exact=sum(g['exact'] for g in graph_rows),tp=sum(g['tp'] for g in graph_rows),fp=sum(g['fp'] for g in graph_rows),fn=sum(g['fn'] for g in graph_rows))
    graph_summary['precision']=graph_summary['tp']/(graph_summary['tp']+graph_summary['fp']) if graph_summary['tp']+graph_summary['fp'] else None
    graph_summary['recall']=graph_summary['tp']/(graph_summary['tp']+graph_summary['fn']) if graph_summary['tp']+graph_summary['fn'] else None
    summary=dict(completed=len(case_rows),required=len(cases),final=final,class_metrics=summaries,ocr=ocr_summary,graph=graph_summary,latency={cohort:distribution([r['analysis_seconds'] for r in case_rows if r['cohort']==cohort]) for cohort in ['isolated','whole']},whole_integral_detection=sum(r['integral_detection'] for r in case_rows if r['cohort']=='whole'),whole_simulation_gate_allowed=sum(r['simulation_gate_allowed'] for r in case_rows if r['cohort']=='whole'),case_statuses=dict(collections.Counter(r['execution_status'] for r in case_rows)),stage_errors=dict(collections.Counter(e['stage']+': '+str(e['error']) for r in case_rows for e in r['handler_errors'])),exposure=dict(unique_source_images=len({c['source_sha256'] for c in cases}),cases_in_historical_train=sum(any('train' in s for s in c['historical_partition_locations']) for c in cases),all_historical=True))
    if final:assert len(case_rows)==len(cases)
    dump(OUT/'summary.json',summary);dump(OUT/'cases.json',case_rows);dump(OUT/'case_details.json',full);dump(OUT/'objects.json',object_rows);dump(OUT/'fields.json',field_rows);dump(OUT/'graphs.json',graph_rows)
    csvfile('casos.csv',case_rows);csvfile('componentes.csv',object_rows);csvfile('ocr.csv',field_rows);csvfile('classes.csv',summaries);csvfile('ligacoes.csv',graph_rows)
    csvfile('confusao.csv',[dict(cohort=k[0],expected=k[1],observed=k[2],count=v) for k,v in sorted(confusion.items())])
    print(json.dumps({k:v for k,v in summary.items() if k not in ['class_metrics','stage_errors']},ensure_ascii=False,indent=2))
if __name__=='__main__':
    import sys
    analyze('--final' in sys.argv)
