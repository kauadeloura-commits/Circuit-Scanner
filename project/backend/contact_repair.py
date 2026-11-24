"""Recover only an isolated net's observed external contact in saved projects."""
from copy import deepcopy
import math
from backend.circuit_connectivity import construir_conectividade
from backend.circuit_document import validar_circuito

def repair_isolated_contacts(circuit,connectivity,wires,image):
    if circuit.get('review',{}).get('visual_connections_edited'):return circuit,[]
    source={s['component_id']:s for s in connectivity.get('components',[])}
    aliases={}
    pins=[]
    for c in circuit['components']:
        origin=c.get('metadata',{}).get('source_component_id') or c['id'];s=source.get(origin)
        if not s or s['type']!=c['type']:continue
        for t in s.get('terminals',[]):
            if t['pin'] not in c['pins']:continue
            logical=c['id']+'.'+t['pin'];aliases[origin+':'+t['pin']]=logical
            p=t.get('position') or {}
            if 'x' in p and 'y' in p:pins.append((logical,(float(p['x']),float(p['y']))))
    dest={m:n for n in circuit['nets'] for m in n['members']}
    possible=any(math.dist(p,q)<=.75 and dest[a]['name']!=dest[b]['name']
                 and min(len(dest[a]['members']),len(dest[b]['members']))==1
                 for i,(a,p) in enumerate(pins) for b,q in pins[i+1:])
    if not possible or image is None:return circuit,[]
    if callable(image):image=image()
    if image is None:return circuit,[]
    fresh=construir_conectividade(image,wires,{'components':list(source.values())})
    result=deepcopy(circuit);changes=[]
    for contact in fresh.get('recovered_coincident_contacts',[]):
        a,b=(aliases.get(ref) for ref in contact['terminal_refs'])
        if not a or not b:continue
        dest={m:n for n in result['nets'] for m in n['members']};na,nb=dest[a],dest[b]
        if na is nb or min(len(na['members']),len(nb['members']))!=1:continue
        if len(na['members'])>1 or na['name'].upper() in {'GND','0'}:target,isolated=na,nb
        else:target,isolated=nb,na
        target['members']=sorted(set(target['members']+isolated['members']))
        result['nets']=[n for n in result['nets'] if n is not isolated]
        changes.append(dict(terminal_refs=[a,b],from_net=isolated['name'],to_net=target['name'],evidence=contact['evidence']))
    if not changes:return circuit,[]
    result.setdefault('automatic_contact_repairs',[]).extend(changes)
    result.setdefault('warnings',[]).extend(dict(code='observed_contact_recovered',message='Contato do fio recuperado entre '+', '.join(c['terminal_refs'])+'.',**c) for c in changes)
    if 'validation' in result:result['validation']['net_count']=len(result['nets'])
    assert validar_circuito(result)['valid']
    return result,changes
