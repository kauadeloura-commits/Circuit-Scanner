"""Explicit valued VCC symbols supply a rail; unvalued symbols remain labels."""
from backend.electrical_values import value_number

def rail_sources(circuit):
    members={m:n['name'] for n in circuit['nets'] for m in n['members']}
    grounds={n['name'] for n in circuit['nets'] if n['name'].upper() in {'GND','0'}}
    for c in circuit['components']:
        if c['type']=='gnd':grounds.add(members[c['id']+'.GND'])
    declared={}
    for c in circuit['components']:
        if c['type'] not in {'source_dc','battery'}:continue
        if members[c['id']+'.-'] in grounds:declared.setdefault(members[c['id']+'.+'],[]).append(value_number(c['value'],c['type'],c.get('unit')))
        elif members[c['id']+'.+'] in grounds:declared.setdefault(members[c['id']+'.-'],[]).append(-value_number(c['value'],c['type'],c.get('unit')))
    rails={};result=[]
    for c in circuit['components']:
        if c['type']!='vcc' or not c.get('value'):continue
        value=value_number(c['value'],'source_dc',c.get('unit'));net=members[c['id']+'.VCC']
        if net in grounds and value!=0:raise ValueError(c['id']+': VCC com tensão diferente de zero está ligado ao terra.')
        if net in rails and abs(rails[net]-value)>1e-9:raise ValueError('Tensões VCC diferentes na mesma rede '+net+'.')
        if net in declared and any(abs(v-value)>1e-9 for v in declared[net]):raise ValueError(c['id']+': tensão VCC conflita com a fonte já ligada à rede.')
        if net not in rails and net not in declared and net not in grounds:result.append(dict(component_id=c['id'],net=net,value=value))
        rails[net]=value
    return result
