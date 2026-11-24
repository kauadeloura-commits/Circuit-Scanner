"""Put legacy native drawings on CircuitJS's 8 pixel editing grid safely."""
import copy

def fit_grid(circuit,data):
    from backend.falstad_edit import parse_deck,describe,import_edits
    result=copy.deepcopy(data);old=parse_deck(data['circuit']);entries=copy.deepcopy(old)
    def snap(p):return tuple(8*round(v/8) for v in p)
    anchors={};bindings=result['components']
    for binding in bindings:
        index=binding.get('element_index')
        if index is None:continue
        e=entries[index];a,b=snap(e['a']),snap(e['b'])
        if e['code'] in {'t','f','a'}:
            if abs(b[0]-a[0])>=abs(b[1]-a[1]):b=(b[0],a[1])
            else:b=(a[0],b[1])
        elif e['code'] not in {'g','R'}:
            if abs(b[0]-a[0])>=abs(b[1]-a[1]):a=(a[0],8*round((a[1]+b[1])/16));b=(b[0],a[1])
            else:a=(8*round((a[0]+b[0])/16),a[1]);b=(a[0],b[1])
        if a==b and e['code'] not in {'g','R'}:b=(a[0]+32,a[1])
        e.update(a=a,b=b)
        previous=describe(old[index])[2];posts=describe(e)[2]
        for pin,p in posts.items():anchors.setdefault(previous[pin],[]).append(p)
        for pin,p in binding['pins'].items():p['position']=list(posts[pin])
    shared={p:(values[0] if len(set(values))==1 else snap(p)) for p,values in anchors.items()}
    extra=[]
    for e in entries:
        if e['code'] in {'w','g','207'}:e.update(a=shared.get(e['a'],snap(e['a'])),b=shared.get(e['b'],snap(e['b'])))
        elif e['code']=='x':e.update(a=snap(e['a']),b=snap(e['b']))
        if e['code']=='w' and e['a'][0]!=e['b'][0] and e['a'][1]!=e['b'][1]:
            corner=(e['b'][0],e['a'][1]);extra.append(dict(code='w',a=corner,b=e['b'],flags=0,extra=[]));e['b']=corner
    for p,values in anchors.items():
        for q in set(values):
            if q!=shared[p]:extra.append(dict(code='w',a=shared[p],b=q,flags=0,extra=[]))
    def serialize(entries,extras):
        text=data['circuit']
        if text.lstrip().startswith('<'):
            import xml.etree.ElementTree as ET
            tree=ET.fromstring(text);tree.set('f',str(int(tree.get('f','0'))|2));i=0
            for node in tree:
                if node.tag in {'mm','dm','tm','o','h','adjustable'}:continue
                e=entries[i];i+=1;node.set('x',' '.join(map(str,e['a']+e['b'])))
            for e in extras:ET.SubElement(tree,'w',x=' '.join(map(str,e['a']+e['b'])),f='0')
            return ET.tostring(tree,encoding='unicode')+'\n'
        lines=[];i=0
        for line in text.splitlines():
            f=line.split()
            if not f:continue
            if f[0]=='$':f[1]=str(int(f[1])|2)
            elif f[0] not in {'o','h','%','?','!','.'}:
                e=entries[i];i+=1;f[1:5]=map(str,e['a']+e['b'])
            lines.append(' '.join(f))
        lines += [' '.join([e['code'],*map(str,e['a']+e['b']),str(e['flags']),*e['extra']]) for e in extras]
        return '\n'.join(lines)+'\n'
    result['circuit']=serialize(entries,extra)
    from backend.falstad_presentation import clean_presentation
    result.pop('presentation_version',None)
    result=clean_presentation(circuit,result)
    expected={frozenset(n['members']) for n in circuit['nets']}
    try:
        changed,_,_=import_edits(circuit,result,result['circuit'])
        if {frozenset(n['members']) for n in changed['nets']}==expected:return result
    except (ValueError,KeyError,IndexError):pass
    # When snapping would join distinct nets, rebuild only drawing geometry.
    # Keep every existing native component's model and electrical attributes.
    from backend.digital_twin import make_diagram
    from backend.circuitjs_export import export_circuitjs
    geometry=dict(components=[dict(component_id=b['component_id'],type=b['type'],terminals=[dict(pin=pin,position=dict(x=p['position'][0]*.78125,y=p['position'][1]*.78125)) for pin,p in b['pins'].items()]) for b in data['components']],nets=[])
    fresh=export_circuitjs(circuit,make_diagram(circuit,geometry));fresh_entries=parse_deck(fresh['circuit'])
    original_by_id={b['component_id']:old[b['element_index']] for b in data['components'] if b.get('element_index') is not None}
    replacements={b['element_index']:original_by_id[b['component_id']] for b in fresh['components'] if b['component_id'] in original_by_id and b.get('element_index') is not None}
    if data['circuit'].lstrip().startswith('<'):
        import xml.etree.ElementTree as ET
        original_tree=ET.fromstring(data['circuit']);tree=ET.Element('cir',dict(original_tree.attrib));tree.set('f',str(int(tree.get('f','0'))|2))
        nodes=[n for n in original_tree if n.tag not in {'mm','dm','tm','o','h','adjustable'}]
        original_nodes={b['component_id']:nodes[b['element_index']] for b in data['components'] if b.get('element_index') is not None}
        by_index={b['element_index']:b['component_id'] for b in fresh['components'] if b.get('element_index') is not None}
        for node in original_tree:
            if node.tag in {'mm','dm','tm'}:tree.append(copy.deepcopy(node))
        for i,e in enumerate(fresh_entries):
            if i in replacements:
                node=copy.deepcopy(original_nodes[by_index[i]]);node.set('x',' '.join(map(str,e['a']+e['b'])));node.set('f',str(e['flags']))
            else:
                node=ET.Element(e['code'],x=' '.join(map(str,e['a']+e['b'])),f=str(e['flags']))
                if e['code']=='x':
                    from backend.falstad_edit import unescape
                    node.set('te',unescape(' '.join(e['extra'][1:])));node.set('sz',e['extra'][0])
            tree.append(node)
        fresh['circuit']=ET.tostring(tree,encoding='unicode')+'\n'
    else:
        lines=[];i=0
        for line in fresh['circuit'].splitlines():
            f=line.split()
            if f and f[0] not in {'$','o','h','%','?','!','.'}:
                if i in replacements:f=f[:6]+replacements[i]['extra']
                i+=1
            lines.append(' '.join(f))
        fresh['circuit']='\n'.join(lines)+'\n'
    for key in ('source_revision','edited'): 
        if key in data:fresh[key]=data[key]
    changed,_,_=import_edits(circuit,fresh,fresh['circuit'])
    if {frozenset(n['members']) for n in changed['nets']}!=expected:raise ValueError('Não foi possível encaixar o desenho na grade preservando as ligações.')
    return fresh
