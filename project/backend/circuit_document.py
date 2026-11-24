"""Stage 11: an image-independent, validated representation of the circuit."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime
import math
import re
from backend.circuit_types import PIN_SCHEMAS
from backend.component_models import automatic_model_fields

PREFIXES={'resistor':'R','capacitor':'C','capacitor_polarized':'C','diode':'D','led':'LED',
          'inductor':'L','source_dc':'V','battery':'BAT','gnd':'GND','vcc':'VCC',
          'transistor_npn':'Q','transistor_pnp':'Q','mosfet_n':'M','mosfet_p':'M','opamp':'U'}
VALUE_REQUIRED={'resistor','capacitor','capacitor_polarized','inductor','source_dc','battery'}
ID_PATTERN=re.compile(r'^[A-Za-z][A-Za-z0-9_-]*$')


def validar_circuito(circuit: dict) -> dict:
    errors=[]
    def error(message):errors.append(message)
    if not isinstance(circuit,dict):return {'valid':False,'errors':['Circuito deve ser um objeto JSON.']}
    if circuit.get('schema_version')!='circuit-1.0':error('Versao do formato de circuito invalida.')
    components=circuit.get('components');nets=circuit.get('nets')
    if not isinstance(components,list) or not isinstance(nets,list):
        return {'valid':False,'errors':errors+['components e nets devem ser listas.']}
    ids=set();known_pins=set()
    for c in components:
        if not isinstance(c,dict):error('Componente invalido.');continue
        cid=c.get('id')
        if not isinstance(cid,str) or not ID_PATTERN.fullmatch(cid):error('Identificador de componente invalido.');continue
        if cid in ids:error('Componente repetido: '+cid)
        ids.add(cid)
        if not isinstance(c.get('type'),str) or not c['type']:error('Componente sem classe: '+cid)
        if c.get('value') is not None and not isinstance(c.get('value'),str):error('Valor deve ser texto ou null: '+cid)
        pins=c.get('pins')
        if not isinstance(pins,list) or any(not isinstance(pin,str) or not pin or ':' in pin or '.' in pin for pin in pins):
            error('Lista de pinos invalida: '+cid);continue
        if len(set(pins))!=len(pins):error('Pinos repetidos: '+cid)
        known_pins.update(cid+'.'+pin for pin in pins)
    net_names=set();assigned=set()
    for net in nets:
        if not isinstance(net,dict):error('Rede invalida.');continue
        name=net.get('name')
        if not isinstance(name,str) or not name:
            error('Nome de rede ausente ou invalido.');continue
        if name in net_names:error('Nome de rede repetido.')
        net_names.add(name)
        members=net.get('members')
        if not isinstance(members,list) or any(not isinstance(m,str) for m in members):error('Membros da rede invalidos.');continue
        for member in members:
            if member not in known_pins:error('Pino referenciado nao existe: '+member)
            if member in assigned:error('Pino pertence a mais de uma rede: '+member)
            assigned.add(member)
    for missing in sorted(known_pins-assigned):error('Pino sem rede declarada: '+missing)
    return {'valid':not errors,'errors':errors}


def gerar_circuito(connectivity: dict, *, project_id=None, name=None) -> dict:
    if connectivity.get('schema_version')!='stage10-connectivity-1.0':raise ValueError('Execute a etapa 10 antes de gerar circuit.json.')
    source_components=connectivity.get('components') or []
    if not source_components:raise ValueError('Nenhum componente disponivel para gerar o circuito.')
    warnings=deepcopy(connectivity.get('warnings') or [])
    references={};reserved=set();counters={};components=[]
    for c in source_components:
        cid=str(c['component_id']);reference=c.get('reference')
        text=reference.get('text') if isinstance(reference,dict) else reference
        source=reference.get('source') if isinstance(reference,dict) else 'provided'
        if text and source not in {'provisional_detection_id','generated','pending_ocr'} and ID_PATTERN.fullmatch(str(text)):
            references[cid]=str(text);reserved.add(str(text))
        elif ID_PATTERN.fullmatch(cid) and not cid.startswith('DET_'):
            references[cid]=cid;reserved.add(cid)
    used=set();source_to_id={}
    for c in source_components:
        cid=str(c['component_id']);kind=str(c.get('type') or 'unknown');ref=references.get(cid)
        c=automatic_model_fields(c)
        generated=not ref or ref in used
        if generated:
            prefix=PREFIXES.get(kind,'X');number=counters.get(prefix,0)+1
            while prefix+str(number) in reserved|used:number+=1
            counters[prefix]=number;ref=prefix+str(number)
            warnings.append(dict(code='reference_generated',component_id=ref,message='Referencia atribuida automaticamente; confirmar identificacao.'))
        used.add(ref);source_to_id[cid]=ref
        pins=[str(t['pin']) for t in c.get('terminals',[])]
        expected=PIN_SCHEMAS.get(kind) or []
        order={pin:i for i,pin in enumerate(expected)}
        pins=sorted(pins,key=lambda pin:(order.get(pin,len(order)),pin))
        raw=c.get('value');text=raw.get('text') if isinstance(raw,dict) else raw
        value=raw.get('normalized') if isinstance(raw,dict) else raw
        if value is None:value=text or None
        if isinstance(value,(int,float)):
            if not math.isfinite(value):raise ValueError('Valor nao finito no componente '+ref)
            value=format(value,'.15g')
        elif value is not None:value=str(value)
        if kind in VALUE_REQUIRED and not value:
            warnings.append(dict(code='value_missing',component_id=ref,message='Valor do componente nao informado.'))
        metadata=dict(source_component_id=cid,reference_source='generated' if generated else
                      (c.get('reference',{}).get('source','provided') if isinstance(c.get('reference'),dict) else 'provided'),
                      value_source=raw.get('source','provided') if isinstance(raw,dict) else 'provided',
                      review_required=bool(c.get('review_required') or (c.get('terminal_detection') or {}).get('review_required')))
        for field in ('model','electrical_parameters','parameter_policy','parameter_pending'):
            if field in c:metadata[field]=deepcopy(c[field])
        components.append(dict(id=ref,type=kind,value=value,value_text=str(text) if text else None,
                               unit=raw.get('unit') if isinstance(raw,dict) else None,pins=pins,metadata=metadata))
    nets=[];by_terminal={r['terminal_ref']:r for r in connectivity.get('terminal_connections',[])}
    for net in connectivity.get('nets',[]):
        # Unattached image fragments have no logical component member.
        if not net.get('terminal_refs'):continue
        members=[]
        for terminal_ref in net['terminal_refs']:
            terminal=by_terminal.get(terminal_ref)
            if not terminal or terminal['component_id'] not in source_to_id:raise ValueError('Rede referencia terminal desconhecido.')
            members.append(source_to_id[terminal['component_id']]+'.'+terminal['pin'])
        nets.append(dict(name=net['name'],members=sorted(members)))
    # Warnings use logical identifiers too; they carry no image coordinates.
    for warning in warnings:
        source_id=warning.get('component_id')
        if source_id in source_to_id:warning['component_id']=source_to_id[source_id]
        terminal=by_terminal.get(warning.get('terminal_ref'))
        if terminal:warning['terminal_ref']=source_to_id[terminal['component_id']]+'.'+terminal['pin']
    result=dict(schema_version='circuit-1.0',project_id=project_id or connectivity.get('project_id'),
                name=name or 'Circuito reconhecido',created_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                components=components,nets=nets,warnings=warnings,
                confirmed_by_user=False,status='draft',image_independent=True,
                source_connectivity_signature=connectivity.get('source_signature'))
    validation=validar_circuito(result)
    if not validation['valid']:raise ValueError('Circuito invalido: '+'; '.join(validation['errors']))
    result['validation']=dict(structurally_valid=True,review_required=bool(warnings),
                              component_count=len(components),net_count=len(nets),pin_count=sum(len(c['pins']) for c in components))
    return result
