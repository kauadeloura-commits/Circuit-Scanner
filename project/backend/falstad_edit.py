"""Recover supported CircuitJS elements and physical connections from its public dump."""
import copy,math,re

def xml_entries(text):
    import xml.etree.ElementTree as ET
    if '<!DOCTYPE' in text.upper() or '<!ENTITY' in text.upper():raise ValueError('Formato XML inválido.')
    try:root=ET.fromstring(text)
    except ET.ParseError:raise ValueError('Circuito XML incompleto.') from None
    if root.tag!='cir':raise ValueError('Formato de circuito não reconhecido.')
    models={e.get('nm'):e.attrib for e in root if e.tag=='mm'};lines=[]
    aliases={'PolarCapacitor':'209','pc':'209','LED':'162','LabeledNode':'207','ln':'207'}
    from backend.circuitjs_export import escape
    for e in root:
        if e.tag in {'mm','dm','tm','o','h','adjustable'}:continue
        code=aliases.get(e.tag,e.tag)
        if code not in {'r','c','209','l','v','g','R','d','162','t','f','a','w','207','x'}:raise ValueError('Este componente Falstad ainda não pode ser salvo no projeto: '+e.tag)
        positions=e.get('x','').split()
        if len(positions)!=4:raise ValueError('Posição de componente inválida.')
        flags=e.get('f','0');extra=[]
        if code=='r':extra=[e.get('r','1000')]
        elif code in {'c','209'}:extra=[e.get('c','1e-5'),e.get('vd','0'),e.get('iv','0')]+([e.get('mv','1')] if code=='209' else [])
        elif code=='l':extra=[e.get('l','1'),e.get('ic','0')]
        elif code in {'v','R'}:extra=[e.get('wf','0'),e.get('fr','40'),e.get('maxv','5'),e.get('bias','0'),e.get('phaseShift','0'),e.get('dutyCycle','.5')]
        elif code=='t':extra=[e.get('pn','1'),e.get('vbe','0'),e.get('vbc','0'),e.get('be','100'),escape(e.get('mo','default'))]
        elif code=='f':
            model=models.get(e.get('mo'),{})
            if model.get('bt')=='1':raise ValueError('O MOSFET de quatro terminais ainda não pode ser salvo no projeto.')
            extra=[model.get('vt',e.get('vt','1.5')),model.get('be',e.get('be','.02'))]
        elif code=='a':extra=[e.get('ma','15'),e.get('mi','-15'),'1000000','0','0',e.get('ga','100000')]
        elif code in {'d','162'}:extra=[escape(e.get('mo','default'))]
        elif code in {'207','x'}:extra=[escape(e.get('te',''))]
        lines.append(' '.join([code]+positions+[flags]+extra))
    return '\n'.join(lines)
from backend.circuit_types import PIN_SCHEMAS
from backend.circuit_review import review_issues

def unescape(s):
    return re.sub(r'\\(.)',lambda m:{'s':' ','p':'+','q':'=','n':'\n','r':'\r','h':'#','a':'&','0':''}.get(m[1],m[1]),s)

def parse_deck(text):
    if not isinstance(text,str) or len(text)>1_000_000:raise ValueError('Circuito Falstad muito grande ou inválido.')
    if text.lstrip().startswith('<'):text=xml_entries(text)
    entries=[]
    for line in text.splitlines():
        fields=line.split()
        if not fields or fields[0] in {'$','o','h','%','?','!','.'}:continue
        if fields[0] not in {'r','c','209','l','v','g','R','d','162','t','f','a','w','207','x'}:raise ValueError('Este componente Falstad ainda não pode ser salvo no projeto: '+fields[0])
        if len(fields)<6:raise ValueError('Elemento Falstad incompleto.')
        try:a=tuple(int(x) for x in fields[1:3]);b=tuple(int(x) for x in fields[3:5]);flags=int(fields[5])
        except ValueError:raise ValueError('Posição de componente inválida.') from None
        if max(abs(x) for x in a+b)>100_000:raise ValueError('Posição fora da área de desenho.')
        entries.append(dict(code=fields[0],a=a,b=b,flags=flags,extra=fields[6:]))
    if len(entries)>5000:raise ValueError('Há elementos demais no circuito.')
    return entries

def number(value):
    n=float(value)
    if not math.isfinite(n) or abs(n)>1e15:raise ValueError('Valor do componente inválido.')
    return n

def describe(e):
    code=e['code'];a,b=e['a'],e['b'];flags=e['flags'];s=e['extra']
    kind={'r':'resistor','c':'capacitor','209':'capacitor_polarized','l':'inductor','v':'source_dc','g':'gnd','R':'vcc','d':'diode','162':'led'}.get(code)
    value=None;params={}
    if code in {'r','c','209','l'}:value=str(number(s[0]))
    if code in {'v','R'}:
        if int(s[0])!=0:raise ValueError('Salve uma fonte DC; a forma de onda pode ser exportada diretamente pelo Falstad.')
        value=str(number(s[2]))
    if code in {'t','f','a'}:
        dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
        if length<1:raise ValueError('O componente precisa de espaço entre seus terminais.')
        sign=1 if (dx if dy==0 else dy)>0 else -1
        offset=16*sign*(-1 if flags&(8 if code=='f' else 1) else 1)
        if code=='t':
            polarity=int(s[0]);kind='transistor_npn' if polarity>0 else 'transistor_pnp';offset*=polarity
            params['hfe']=dict(value=number(s[3]),unit='A/A',source='manual_simulator',estimated=False)
        elif code=='f':
            kind='mosfet_p' if flags&1 else 'mosfet_n';offset=-offset
            params.update(vgs_threshold=dict(value=abs(number(s[0]))*(-1 if kind=='mosfet_p' else 1),unit='V',source='manual_simulator',estimated=False),mos_kp=dict(value=number(s[1]),unit='A/V²',source='manual_simulator',estimated=False))
        else:kind='opamp';offset=16*sign*(-1 if flags&1 else 1);params['open_loop_gain']=dict(value=number(s[5]),unit='V/V',source='manual_simulator',estimated=False)
        center=b if code!='a' else a
        first=(math.floor(center[0]+offset*dy/length+.48),math.floor(center[1]-offset*dx/length+.48))
        second=(math.floor(center[0]-offset*dy/length+.48),math.floor(center[1]+offset*dx/length+.48))
        posts={'t':{'B':a,'C':first,'E':second},'f':{'G':a,'S':first,'D':second},'a':{'-':first,'+':second,'OUT':b}}[code]
        if kind=='mosfet_p':posts={'G':a,'D':first,'S':second}
    else:
        posts={'g':{'GND':a},'R':{'VCC':a},'v':{'-':a,'+':b}}.get(code)
        if posts is None:posts=dict(zip(PIN_SCHEMAS[kind],(a,b)))
    return kind,value,posts,params

def check_native_connections(circuit,data):
    """Check native posts, not drawing metadata, before publishing a layout."""
    entries=parse_deck(data['circuit'])
    for binding in data['components']:
        index=binding.get('element_index')
        if index is None:continue
        actual=describe(entries[index])[2]
        recorded={pin:tuple(p['position']) for pin,p in binding['pins'].items()}
        if actual!=recorded:raise ValueError('Os fios não coincidem com os terminais do simulador: '+binding['component_id'])
    imported,_,_=import_edits(circuit,data,data['circuit'])
    expected={frozenset(n['members']) for n in circuit['nets']}
    if {frozenset(n['members']) for n in imported['nets']}!=expected:
        raise ValueError('A organização do desenho alterou uma ligação. Reabra o circuito para reconstruir os fios.')
    return data


def import_edits(current,baseline,text):
    entries=parse_deck(text);old_entries=parse_deck(baseline['circuit'])
    bound={b.get('element_index') for b in baseline['components']}
    implicit_grounds=[e for i,e in enumerate(old_entries) if e['code']=='g' and i not in bound]
    old_order=[(i,e) for i,e in enumerate(old_entries) if e['code'] not in {'w','207','x'}]
    new_order=[(i,e) for i,e in enumerate(entries) if e['code'] not in {'w','207','x'}]
    # Moving/compressing the drawing retains native element order. Identifying
    # equal resistors by their old absolute coordinates would swap their names
    # after such a move. Wire/text insertion does not affect this physical order.
    same_order=[e['code'] for _,e in old_order]==[e['code'] for _,e in new_order]
    ordinal={new[0]:old[0] for old,new in zip(old_order,new_order)} if same_order else {}
    ignored={i for i,e in new_order if e['code']=='g' and
             (ordinal[i] not in bound if same_order else e in implicit_grounds)}
    parts=[(i,e,*describe(e)) for i,e in new_order if i not in ignored]
    if not parts:raise ValueError('O desenho não tem componentes para salvar.')
    old={c['id']:c for c in current['components']};assigned={};used=set()
    if same_order:
        by_index={b['element_index']:b['component_id'] for b in baseline['components'] if b.get('element_index') is not None}
        for i,e,kind,value,posts,params in parts:
            cid=by_index.get(ordinal[i])
            if cid:assigned[i]=cid;used.add(cid)
    candidates=[]
    for binding in baseline['components']:
        index=binding.get('element_index')
        if index is None or index>=len(old_entries):continue
        original=old_entries[index]
        for i,e,kind,value,posts,params in parts:
            if e['code']!=original['code']:continue
            distance=math.dist(e['a'],original['a'])+math.dist(e['b'],original['b'])
            candidates.append((distance,binding['component_id'],i))
    for distance,cid,i in sorted(candidates):
        if cid in used or i in assigned:continue
        assigned[i]=cid;used.add(cid)
    components=[];geometry=[];posts_by_member={};counts={};bindings=[]
    for i,e,kind,value,posts,params in parts:
        cid=assigned.get(i)
        if cid and old[cid]['type'] in {'battery','source_dc'} and kind=='source_dc':kind=old[cid]['type']
        if not cid:
            prefix={'resistor':'R','capacitor':'C','capacitor_polarized':'C','inductor':'L','source_dc':'V','vcc':'VCC','gnd':'GND','diode':'D','led':'LED','opamp':'U'}.get(kind,'Q' if kind.startswith('transistor') else 'M')
            counts[prefix]=counts.get(prefix,0)+1;cid=prefix+str(counts[prefix])
            while cid in old or cid in used:counts[prefix]+=1;cid=prefix+str(counts[prefix])
            used.add(cid)
        original=old.get(cid);c=copy.deepcopy(original) if original and original['type']==kind else dict(id=cid,type=kind,metadata={})
        c.update(id=cid,type=kind,pins=list(PIN_SCHEMAS[kind]))
        if value is not None:
            c.update(value=value,value_text=value,unit={'resistor':'ohm','capacitor':'farad','capacitor_polarized':'farad','inductor':'henry','source_dc':'volt','battery':'volt','vcc':'volt'}.get(kind))
        c.setdefault('metadata',{}).setdefault('electrical_parameters',{}).update(params)
        components.append(c);posts_by_member.update({cid+'.'+pin:p for pin,p in posts.items()})
        geometry.append(dict(component_id=cid,type=kind,terminals=[dict(pin=pin,position=dict(x=p[0]*.78125,y=p[1]*.78125)) for pin,p in posts.items()]))
        bindings.append(dict(component_id=cid,type=kind,element_index=i,pins={pin:dict(position=p) for pin,p in posts.items()}))
    # Ground/rail markers represented only by a net label remain logical parts.
    for binding in baseline['components']:
        cid=binding['component_id']
        if binding.get('element_index') is not None or cid in used:continue
        c=copy.deepcopy(old[cid]);components.append(c)
        posts={pin:tuple(v['position']) for pin,v in binding['pins'].items()}
        posts_by_member.update({cid+'.'+pin:p for pin,p in posts.items()})
        geometry.append(dict(component_id=cid,type=c['type'],terminals=[dict(pin=pin,position=dict(x=p[0]*.78125,y=p[1]*.78125)) for pin,p in posts.items()]))
        bindings.append(dict(component_id=cid,type=c['type'],element_index=None,pins={pin:dict(position=p) for pin,p in posts.items()}))
    parent={}
    def find(p):
        parent.setdefault(p,p)
        while parent[p]!=p:parent[p]=parent[parent[p]];p=parent[p]
        return p
    def join(a,b):parent[find(a)]=find(b)
    points=set(posts_by_member.values());wires=[];labels={};grounds=[]
    for e in entries:
        if e['code']=='w':wires.append((e['a'],e['b']));points.update((e['a'],e['b']))
        if e['code']=='g':grounds.append(e['a']);points.add(e['a'])
        if e['code']=='207':labels.setdefault(unescape(e['extra'][0]),[]).append(e['a']);points.add(e['a'])
    from backend.digital_twin import on_segment
    for a,b in wires:
        join(a,b)
        for p in points:
            if on_segment(p,a,b):join(a,p)
    for group in list(labels.values())+[grounds]:
        for p in group[1:]:join(group[0],p)
    groups={}
    for m,p in posts_by_member.items():groups.setdefault(find(p),[]).append(m)
    nets=[];names=set();ground=find(grounds[0]) if grounds else None
    for index,(root,members) in enumerate(groups.items(),1):
        name='GND' if root==ground else next((label for label,ps in labels.items() if find(ps[0])==root and re.fullmatch(r'[A-Za-z0-9_+\-]{1,64}',label)),f'NET_{index}')
        while name in names:name+='_'+str(index)
        names.add(name);nets.append(dict(name=name,members=sorted(members)))
    result=copy.deepcopy(current);result.update(components=components,nets=nets,status='confirmed',confirmed_by_user=True)
    result.setdefault('review',{}).update(visual_connections_edited=True,edit_source='falstad')
    result['warnings']=review_issues(result);result['validation']=dict(structurally_valid=True,component_count=len(components),net_count=len(nets),pin_count=len(posts_by_member),review_required=bool(result['warnings']))
    dest={m:n['name'] for n in nets for m in n['members']}
    for binding in bindings:
        for pin,reading in binding['pins'].items():reading['net']=dest[binding['component_id']+'.'+pin]
    return result,dict(components=geometry,nets=[],native_bindings=bindings),text
