from pathlib import Path
import copy,math
from backend.performance import file_hash,read_json,write_json,digest

FILENAME='terminal_review_draft.json'
def shape(component):
    return digest({'type':component.get('type'),'bbox':{k:float(component['bbox'][k]) for k in ('x1','y1','x2','y2')}})
def load(folder,image_path):
    data=read_json(Path(folder)/FILENAME)
    if data.get('image_sha256')!=file_hash(image_path) or not isinstance(data.get('components'),list):return {'components':[]}
    data['components']=[r for r in data['components'] if isinstance(r,dict) and isinstance(r.get('points'),dict)
        and isinstance(r.get('pin_schema'),list) and all(isinstance(p,dict) and isinstance(p.get('pin'),str) for p in r['pin_schema'])
        and all(isinstance(q,dict) and all(isinstance(q.get(k),(int,float)) and math.isfinite(q[k]) for k in ('x','y')) for q in r['points'].values())]
    return data

def save(folder,image_path,components,edits,find_component,schema_for,clean_schema,fixed_schemas):
    if not isinstance(edits,list) or not edits:raise ValueError('Informe as marcações dos componentes alterados.')
    records=copy.deepcopy(load(folder,image_path)['components']);seen=set()
    for edit in edits:
        if not isinstance(edit,dict):raise ValueError('Marcação inválida.')
        cid=str(edit.get('component_id') or '');component=find_component(components,cid)
        if not component:raise ValueError('Componente não encontrado: '+cid)
        if cid in seen:raise ValueError('Componente repetido: '+cid)
        seen.add(cid);points=edit.get('points')
        if not isinstance(points,dict):raise ValueError(cid+': as marcações precisam ser um objeto.')
        if component.get('type') not in fixed_schemas and 'pin_schema' in edit:
            schema=clean_schema(edit['pin_schema'])
        else:schema=schema_for(component,requested_schema=edit.get('pin_schema'),points=points)
        names={pin for pin,_ in schema};clean={};b=component['bbox']
        if set(points)-names:raise ValueError(cid+': nome de pino inválido.')
        for pin,q in points.items():
            try:x,y=float(q['x']),float(q['y'])
            except (KeyError,TypeError,ValueError):raise ValueError(cid+'.'+pin+': posição inválida.')
            if not math.isfinite(x) or not math.isfinite(y) or not b['x1']<x<b['x2'] or not b['y1']<y<b['y2']:
                raise ValueError(cid+'.'+pin+': marque o ponto dentro da caixa do componente.')
            clean[pin]={'x':round(x,2),'y':round(y,2)}
        record={'component_id':component.get('component_id') or cid,'detection_id':component.get('detection_id'),
                'shape':shape(component),'points':clean,'pin_schema':[{'pin':p,'role':r} for p,r in schema]}
        records=[r for r in records if r.get('component_id')!=record['component_id'] and not (record['detection_id'] and r.get('detection_id')==record['detection_id'])];records.append(record)
    write_json(Path(folder)/FILENAME,{'schema_version':'terminal-review-draft-1.0','image_sha256':file_hash(image_path),'components':records})
    return len(edits)

def restore(folder,image_path,editable,components):
    records=load(folder,image_path)['components']
    for c,source in zip(editable,components):
        for record in records:
            same=record.get('component_id')==c['component_id'] or bool(source.get('detection_id') and record.get('detection_id')==source['detection_id'])
            if same and record.get('shape')==shape(source):
                c['points']=copy.deepcopy(record['points']);c['pin_schema']=copy.deepcopy(record['pin_schema']);c['pins']=[p['pin'] for p in c['pin_schema']]
                c['draft_restored']=True;break

def discard(folder,ids):
    path=Path(folder)/FILENAME;data=read_json(path)
    if isinstance(data.get('components'),list):
        data['components']=[c for c in data['components'] if isinstance(c,dict) and c.get('component_id') not in ids and c.get('detection_id') not in ids]
        write_json(path,data)
