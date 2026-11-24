"""Native CircuitJS serialization and reviewed-pin geometry, not outro simulador import.

Element post orders follow the official CircuitJS Java classes. Each disconnected
wire group gets a net label preserving the confirmed connectivity without loops.
"""
import math
from backend.kicad_export import confirmed, model_text
from backend.circuit_types import PIN_SCHEMAS
from backend.circuit_review import revision
from backend.semiconductor_models import parameters, falstad_fit
from backend.electrical_values import value_number
from backend.power_rails import rail_sources

def escape(text):
    # CustomLogicModel.escape: backslash, plus, equals, spaces, newlines.
    return str(text).replace('\\','\\\\').replace('+','\\p').replace('=','\\q').replace(' ','\\s').replace('\n','\\n').replace('#','\\h').replace('&','\\a').replace('\r','\\r') or '\\0'

def export_circuitjs(circuit,diagram):
    confirmed(circuit);dest={m:n['name'] for n in circuit['nets'] for m in n['members']}
    placement={c['id']:c for c in diagram['components']};lines=['$ 3 0.000005 10.20027730826997 50 5 50 5e-11']
    warnings=[];bindings=[];posts={};reserved={};index=0;wire_requests=[];label_requests=[];bridges=[]
    def pos(p):return 8*round(p[0]/2),8*round(p[1]/2)
    def emit(code,a,b,flags,suffix=''):
        nonlocal index
        if code=='w':wire_requests.append((a,b));return None
        element=index;index+=1;lines.append(f'{code} {a[0]} {a[1]} {b[0]} {b[1]} {flags}'+(' '+suffix if suffix else ''));return element
    def label(net,p):
        # labeledelement (207) internally unites equal names. Ground gets an actual g element.
        label_requests.append((net,p))
    def reserve(pin,point,net):
        previous=reserved.get(point)
        if previous is not None and previous!=net:raise ValueError('Terminais de redes diferentes coincidem no desenho: '+pin+'. Revise as posições.')
        reserved[point]=net;posts[pin]=point
    rails={r['component_id']:r for r in rail_sources(circuit)};grounded=set()
    ground_nets={n['name'] for n in circuit['nets'] if n['name'].upper() in {'GND','0'}}
    ground_nets.update(dest[c['id']+'.GND'] for c in circuit['components'] if c['type']=='gnd')
    for n in circuit['nets']:
        if len(n['members'])==1:warnings.append('Terminal sem ligação: '+n['members'][0]+'. Confira o fio no desenho e conecte antes de simular.')
    for c in circuit['components']:
        kind=c['type'];p={pin:pos(v) for pin,v in placement[c['id']]['pins'].items()};schema=PIN_SCHEMAS[kind];code=None;suffix='';flags=0
        nets={pin:dest[c['id']+'.'+pin] for pin in schema}
        for pin,pt in p.items():reserve(c['id']+'.'+pin,pt,nets[pin])
        a=p[schema[0]];b=p[schema[-1]];actual=dict(p);physical=[]
        if kind in {'resistor','capacitor','capacitor_polarized','inductor','source_dc','battery'}:
            value=c.get('value')
            if kind=='inductor' and not value:
                _,resolved=parameters(c);value=str(resolved['parameters'].get('inductance',{}).get('value',''))
            value=value_number(value,kind,c.get('unit'))
            code={'resistor':'r','capacitor':'c','capacitor_polarized':'209','inductor':'l','source_dc':'v','battery':'v'}[kind]
            suffix=f'{value:.15g}'
            if kind in {'capacitor','capacitor_polarized'}:suffix+=' 0 0'+(' 1' if kind=='capacitor_polarized' else '')
            if kind=='inductor':
                _,resolved=parameters(c);suffix+=' '+str(resolved['parameters'].get('initial_current',{}).get('value',0))
            if kind in {'source_dc','battery'}:
                # VoltageElm post 1 is positive, the opposite of outro simulador's node order.
                a,b=b,a;suffix=f'0 40 {value:.15g} 0 0 0.5'
        elif kind in {'diode','led'}:
            code='d' if kind=='diode' else '162';flags=2;suffix='default' if kind=='diode' else 'default-led 1 0 0 0.01'
            warnings.append(c['id']+': Falstad utiliza o modelo genérico de diodo/LED; o nome comercial aparece como identificação.')
        elif kind.startswith('transistor_') or kind.startswith('mosfet_') or kind=='opamp':
            fit=falstad_fit(c)
            if kind.startswith('transistor_') or kind.startswith('mosfet_'):
                control=schema[0];upper=schema[1];lower=schema[2]
                # CircuitJS sizes BJT/MOSFET posts at +/-16 px. Route reviewed pins to those posts.
                center=(8*round((p[upper][0]+p[lower][0])/16),8*round((p[upper][1]+p[lower][1])/16))
                dx,dy=center[0]-a[0],center[1]-a[1]
                if abs(dx)>=abs(dy):b=(center[0],a[1]);nx,ny=0,-1
                else:b=(a[0],center[1]);nx,ny=1,0
                if a==b:b=(a[0]+64,a[1]);nx,ny=0,-1
                # interpPoint2 multiplies its perpendicular by dsign; horizontal direction cancels.
                first=(b[0]+16*nx,b[1]+16*ny);second=(b[0]-16*nx,b[1]-16*ny)
                if kind.startswith('transistor_'):
                    code='t';polarity=1 if kind=='transistor_npn' else -1
                    if polarity<0:first,second=second,first
                    suffix=f'{polarity} 0 0 {fit["BF"]:.15g} default'
                    warnings.append(c['id']+': transistor aproximado com o hFE informado ou estimado; o comportamento varia com corrente e temperatura.')
                else:
                    code='f';flags=32+(1 if kind=='mosfet_p' else 0)
                    # N uses posts G,S,D; the P symbol labels its posts G,D,S.
                    if kind=='mosfet_p':first,second=second,first
                    suffix=f'{abs(fit["VTO"]):.15g} {fit["KP"]:.15g}'
                    warnings.append(c['id']+': MOSFET aproximado com VGS(th) e KP; capacitâncias e efeitos térmicos não são representados pelo modelo genérico.')
                if math.dist(first,p[upper])+math.dist(second,p[lower])>math.dist(second,p[upper])+math.dist(first,p[lower]):
                    flags|=1 if code=='t' else 8;first,second=second,first
                actual={control:a,upper:first,lower:second}
            else:
                code='a';flags=8
                left=(8*round((p['+'][0]+p['-'][0])/16),8*round((p['+'][1]+p['-'][1])/16));a=left;b=p['OUT']
                if abs(b[0]-a[0])>=abs(b[1]-a[1]):b=(b[0],a[1]);nx,ny=0,-1
                else:b=(a[0],b[1]);nx,ny=1,0
                if a==b:b=(a[0]+80,a[1]);nx,ny=0,-1
                actual={'-':(a[0]+16*nx,a[1]+16*ny),'+':(a[0]-16*nx,a[1]-16*ny),'OUT':b}
                if math.dist(actual['+'],p['+'])+math.dist(actual['-'],p['-'])>math.dist(actual['-'],p['+'])+math.dist(actual['+'],p['-']):
                    flags|=1;actual['+'],actual['-']=actual['-'],actual['+']
                suffix=f'15 -15 1000000 0 0 {fit["open_loop_gain"]:.15g}'
                warnings.append(c['id']+': amplificador aproximado com ganho informado ou estimado e alimentação interna de ±15 V.')
        elif kind=='gnd':
            code='g';grounded.add(nets['GND']);b=(a[0],a[1]+24)
        elif kind=='vcc':
            if c['id'] in rails:code='R';b=(a[0],a[1]-32);suffix=f'0 40 {rails[c["id"]]["value"]:.15g} 0 0 0.5'
        else:raise ValueError('Componente sem conversão Falstad: '+c['id'])
        element=emit(code,a,b,flags,suffix) if code else None
        for pin,pt in actual.items():
            reserve(c['id']+'.'+pin,pt,nets[pin]);label(nets[pin],pt)
            if pt!=p[pin]:bridges.append((pt,p[pin],nets[pin]))
        bindings.append(dict(component_id=c['id'],type=kind,element_index=element,pins={pin:dict(position=pt,net=nets[pin],post_index=(['-','+','OUT'].index(pin) if kind=='opamp' else (1 if pin=='+' else 0) if kind in {'source_dc','battery'} else ['G','S','D'].index(pin) if kind=='mosfet_n' else schema.index(pin))) for pin,pt in actual.items()}))
        text=' | '.join(filter(None,[c['id'],model_text(c)]))
        cx,cy=pos(placement[c['id']]['center']);emit('x',(cx+20,cy-28),(cx+20,cy-28),4,'14 '+escape(text))
    for net in sorted(ground_nets-grounded):
        p=next(p for name,p in label_requests if name==net)
        emit('g',p,(p[0],p[1]+24),0)
    # Native three-pin symbols have a fixed lead spacing. Connect their posts
    # to photographed anchors with straight orthogonal segments on the grid.
    from backend.schematic_routing import orthogonal
    for a,b,net in bridges:
        foreign=[(p[0]/4,p[1]/4) for p,n in reserved.items() if n!=net]
        path=orthogonal((a[0]/4,a[1]/4),(b[0]/4,b[1]/4),[],foreign)
        path=[pos(p) for p in path]
        for p,q in zip(path,path[1:]):
            if p!=q:reserve('bridge',p,net);reserve('bridge',q,net);emit('w',p,q,0)
    # Geometry cannot join two different nets through a coincident endpoint after rounding.
    def safe(pt,net):return pt not in reserved or reserved[pt]==net
    for route in diagram['routes']:
        for path in route['paths']:
            for a,b in zip(path,path[1:]):
                a,b=pos(a),pos(b)
                if a!=b and safe(a,route['net']) and safe(b,route['net']):
                    reserve('wire',a,route['net']);reserve('wire',b,route['net']);emit('w',a,b,0)
                elif a!=b:warnings.append(route['net']+': trecho visual próximo de outra rede omitido; conexão preservada pelas etiquetas.')
    # Ideal wires/labels must not form voltage-source loops in CircuitJS.
    # Keep a spanning forest of the wires, then one label per connected group.
    parent={}
    def find(p):
        parent.setdefault(p,p)
        while parent[p]!=p:parent[p]=parent[parent[p]];p=parent[p]
        return p
    for a,b in wire_requests:
        ra,rb=find(a),find(b)
        if ra==rb:continue
        parent[ra]=rb;lines.append(f'w {a[0]} {a[1]} {b[0]} {b[1]} 0');index+=1
    labelled=set()
    for net,p in label_requests:
        group=(net,find(p))
        if group in labelled:continue
        labelled.add(group);emit(207,p,(p[0]+24,p[1]),4,escape(net))
    data=dict(engine='CircuitJS/Falstad',source_revision=revision(circuit),layout_version=diagram['version'],
                circuit='\n'.join(lines)+'\n',components=bindings,warnings=list(dict.fromkeys(warnings)),
                model_mode='reviewed_parameters_with_documented_approximations')
    from backend.falstad_presentation import clean_presentation
    from backend.falstad_compact import compact_native
    from backend.falstad_edit import check_native_connections
    return check_native_connections(circuit,compact_native(circuit,clean_presentation(circuit,data)))

def preview_circuitjs(circuit,diagram):
    import copy
    preview=copy.deepcopy(circuit);preview.update(status='confirmed',confirmed_by_user=True)
    data=export_circuitjs(preview,diagram);data['source_revision']=revision(circuit)
    return data
