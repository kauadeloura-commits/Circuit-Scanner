"""Human review of the logical circuit, without altering image geometry."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import re
from backend.circuit_document import validar_circuito, VALUE_REQUIRED
from backend.circuit_types import PIN_SCHEMAS
from backend.component_models import automatic_model_fields, model_edit

NET_NAME=re.compile(r'^[A-Za-z0-9_+\-]{1,64}$')


def review_component_models(circuit):
    result=deepcopy(circuit)
    for c in result['components']:
        metadata=c.setdefault('metadata',{})
        filled=automatic_model_fields(dict(type=c['type'],**{k:metadata[k] for k in ('model','electrical_parameters','parameter_policy') if k in metadata}))
        for key in ('model','electrical_parameters','parameter_policy','parameter_pending'):
            if key in filled:metadata[key]=filled[key]
    return result

def revision(circuit):
    return hashlib.sha256(json.dumps(circuit,sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf8')).hexdigest()

def review_issues(circuit):
    issues=[];dest={member:net['name'] for net in circuit['nets'] for member in net['members']}
    for c in circuit['components']:
        cid=c['id'];expected=PIN_SCHEMAS.get(c['type'])
        if expected is None:issues.append(dict(code='unknown_type',component_id=cid,message='Escolha um tipo de componente conhecido.',blocking=True))
        elif set(c['pins'])!=set(expected):issues.append(dict(code='pin_schema',component_id=cid,message='Confira os pinos exigidos para este tipo: '+', '.join(expected),blocking=True))
        if c['type'] in VALUE_REQUIRED and not (c.get('value') or '').strip():
            issues.append(dict(code='value_missing',component_id=cid,message='Informe o valor do componente.',blocking=True))
        if len(c['pins'])>1 and len({dest[cid+'.'+pin] for pin in c['pins']})==1:
            issues.append(dict(code='possible_short',component_id=cid,message='Todos os pinos estao na mesma rede; confira o possivel curto.',blocking=False))
    for net in circuit['nets']:
        if len(net['members'])==1:issues.append(dict(code='single_pin_net',net=net['name'],message='Rede com um unico pino; confira se esta aberta intencionalmente.',blocking=False))
    if any(c['type'] in {'source_dc','battery'} for c in circuit['components']) and not any(c['type']=='gnd' for c in circuit['components']) and not any(n['name'] in {'GND','0'} for n in circuit['nets']):
        issues.append(dict(code='missing_ground',message='Fonte sem referencia de terra identificada.',blocking=False))
    return issues

def apply_review(current,payload):
    if not isinstance(payload,dict):raise ValueError('Revisao deve ser um objeto.')
    components=payload.get('components');nets=payload.get('nets')
    if not isinstance(components,list) or not components or len(components)>1000:raise ValueError('Informe de 1 a 1000 componentes.')
    if not isinstance(nets,list) or len(nets)>5000:raise ValueError('Lista de redes invalida.')
    original={c['id']:c for c in current['components']};used_origins=set();clean=[]
    for c in components:
        if not isinstance(c,dict):raise ValueError('Componente invalido.')
        cid=c.get('id');kind=c.get('type');pins=c.get('pins')
        if not isinstance(cid,str) or len(cid)>64:raise ValueError('Identificador invalido.')
        if not isinstance(kind,str) or kind not in PIN_SCHEMAS:raise ValueError('Tipo de componente desconhecido.')
        if not isinstance(pins,list) or any(not isinstance(p,str) for p in pins) or set(pins)!=set(PIN_SCHEMAS[kind]) or len(pins)!=len(PIN_SCHEMAS[kind]):raise ValueError('Pinos nao correspondem ao tipo de '+cid+'.')
        value=c.get('value');unit=c.get('unit')
        if value is not None and (not isinstance(value,str) or len(value)>100 or any(ord(ch)<32 for ch in value)):raise ValueError('Valor invalido em '+cid+'.')
        if unit is not None and (not isinstance(unit,str) or len(unit)>24):raise ValueError('Unidade invalida.')
        origin=c.get('original_id')
        if origin is not None and (not isinstance(origin,str) or origin not in original or origin in used_origins):raise ValueError('Origem de componente invalida ou repetida.')
        if origin:used_origins.add(origin)
        metadata=deepcopy(original[origin].get('metadata',{})) if origin else {'source_component_id':None}
        if origin and original[origin]['type']!=kind:
            for field in ('model','electrical_parameters','parameter_policy','parameter_pending'):metadata.pop(field,None)
        context=dict(type=kind,**{k:metadata[k] for k in ('model','electrical_parameters','parameter_policy') if k in metadata})
        if any(k in c for k in ('model','electrical_parameters','parameter_policy')):
            context.update(model_edit(context,c))
        context=automatic_model_fields(context)
        for field in ('model','electrical_parameters','parameter_policy','parameter_pending'):
            if field in context:metadata[field]=context[field]
        metadata.update(reference_source='manual_review',value_source='manual_review',review_required=False)
        clean.append(dict(id=cid,type=kind,pins=list(PIN_SCHEMAS[kind]),value=value.strip() or None if value is not None else None,
                          value_text=value.strip() or None if value is not None else None,unit=unit,metadata=metadata))
    clean_nets=[]
    for net in nets:
        if not isinstance(net,dict) or not isinstance(net.get('name'),str) or not NET_NAME.fullmatch(net['name']):raise ValueError('Nome de rede invalido. Use letras, numeros, _, + ou -.')
        members=net.get('members')
        if not isinstance(members,list) or not members:raise ValueError('Uma rede deve conter pelo menos um pino.')
        clean_nets.append(dict(name=net['name'],members=members))
    result=deepcopy(current);result['components']=clean;result['nets']=clean_nets
    if 'name' in payload:
        if not isinstance(payload['name'],str) or not payload['name'].strip() or len(payload['name'])>120:raise ValueError('Nome de circuito invalido.')
        result['name']=payload['name'].strip()
    validation=validar_circuito(result)
    if not validation['valid']:raise ValueError('; '.join(validation['errors']))
    history=deepcopy(current.get('review',{}))
    history.setdefault('automatic_warnings',deepcopy(current.get('warnings',[])))
    history.update(edited=True,saved_at=datetime.now().astimezone().isoformat(timespec='seconds'),confirmed_at=None,warnings_acknowledged=False)
    result.update(review=history,status='draft',confirmed_by_user=False,warnings=review_issues(result))
    result['validation']=dict(structurally_valid=True,review_required=True,component_count=len(clean),net_count=len(clean_nets),pin_count=sum(len(c['pins']) for c in clean))
    return result

def confirm_review(current,acknowledge_warnings):
    validation=validar_circuito(current)
    if not validation['valid']:raise ValueError('; '.join(validation['errors']))
    issues=review_issues(current)
    blockers=[issue['message'] for issue in issues if issue.get('blocking')]
    if blockers:raise ValueError('Corrija antes de confirmar: '+'; '.join(blockers))
    if (issues or current.get('warnings')) and acknowledge_warnings is not True:
        raise ValueError('Confira os avisos e marque a confirmacao da revisao.')
    result=deepcopy(current);review=result.setdefault('review',{})
    review.setdefault('automatic_warnings',deepcopy(current.get('warnings',[])))
    review.update(confirmed_at=datetime.now().astimezone().isoformat(timespec='seconds'),warnings_acknowledged=acknowledge_warnings is True)
    result.update(status='confirmed',confirmed_by_user=True,warnings=issues)
    result['validation']['review_required']=False
    return result
