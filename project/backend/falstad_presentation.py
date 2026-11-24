"""Label-free CircuitJS drawings with physical wires and local grounds."""
import copy,math
VERSION='physical-wires-local-grounds-1'

def clean_presentation(circuit,data):
    if data.get('presentation_version')==VERSION:return data
    from backend.falstad_edit import parse_deck,unescape
    from backend.circuitjs_export import escape
    from backend.digital_twin import on_segment
    from backend.kicad_export import model_text
    from backend.schematic_routing import orthogonal
    result=copy.deepcopy(data);text=data['circuit'];entries=parse_deck(text)
    dest={m:n['name'] for n in circuit['nets'] for m in n['members']}
    positions={b['component_id']+'.'+pin:tuple(p['position']) for b in data['components'] for pin,p in b['pins'].items()}
    ground_nets={n['name'] for n in circuit['nets'] if n['name'].upper() in {'GND','0'}}
    ground_nets.update(dest[c['id']+'.GND'] for c in circuit['components'] if c['type']=='gnd')
    grounds={e['a'] for e in entries if e['code']=='g'}
    virtual={tuple(b['pins']['GND']['position']) for b in data['components'] if b['type']=='gnd' and b.get('element_index') is None}
    grounds.update(virtual)
    points=set(positions.values())|grounds
    wires=[(e['a'],e['b']) for e in entries if e['code']=='w']
    points.update(p for line in wires for p in line)
    points.update(e['a'] for e in entries if e['code']=='207')
    parent={};grounded={};forest=[]
    def find(p):
        parent.setdefault(p,p);grounded.setdefault(p,p in grounds)
        while parent[p]!=p:parent[p]=parent[parent[p]];p=parent[p]
        return p
    def join(a,b):
        a,b=find(a),find(b)
        if a==b:return False
        parent[a]=b;grounded[b]=grounded[a] or grounded[b];return True
    edges=set()
    for a,b in wires:
        if a==b:continue
        chain=sorted((p for p in points if on_segment(p,a,b)),key=lambda p:math.dist(a,p))
        edges.update(tuple(sorted((p,q))) for p,q in zip(chain,chain[1:]) if p!=q)
    # Retain local branches but never join two trees that each have a ground.
    for a,b in sorted(edges,key=lambda e:(math.dist(*e),e)):
        ra,rb=find(a),find(b)
        if ra==rb or grounded[ra] and grounded[rb]:continue
        join(a,b);forest.append((a,b))
    geometry=[dict(id=b['component_id'],pins={pin:(p['position'][0]/4,p['position'][1]/4) for pin,p in b['pins'].items()}) for b in data['components'] if b['type']!='gnd']
    # Replace old label-only joins with visible wires avoiding foreign pins/bodies.
    for net in circuit['nets']:
        if net['name'] in ground_nets:continue
        members=[positions[m] for m in net['members']]
        while len({find(p) for p in members})>1:
            _,a,b=min((math.dist(a,b),a,b) for i,a in enumerate(members) for b in members[i+1:] if find(a)!=find(b))
            roots={find(p) for m,p in positions.items() if dest[m]!=net['name']}
            foreign=[(p[0]/4,p[1]/4) for p in points if find(p) in roots]
            path=orthogonal((a[0]/4,a[1]/4),(b[0]/4,b[1]/4),geometry,foreign)
            path=[(int(round(p[0]*4)),int(round(p[1]*4))) for p in path]
            for p,q in zip(path,path[1:]):
                if p!=q:join(p,q);forest.append((p,q));points.update((p,q))
    # Cutting a redundant ground-to-ground path must not leave long wire stubs.
    protected=set(positions.values())|grounds
    while True:
        degree={}
        for a,b in forest:
            for p in (a,b):degree[p]=degree.get(p,0)+1
        leaves={p for p,count in degree.items() if count==1 and p not in protected and grounded[find(p)]}
        if not leaves:break
        forest=[(a,b) for a,b in forest if a not in leaves and b not in leaves]
    added_grounds=set(virtual)
    for m,p in positions.items():
        if dest[m] in ground_nets and not grounded[find(p)]:added_grounds.add(p);grounded[find(p)]=True
    added_grounds.difference_update(e['a'] for e in entries if e['code']=='g')
    captions={c['id']:' | '.join(filter(None,[c['id'],model_text(c)])) for c in circuit['components']}
    def caption(text):return captions.get(text.split(' | ',1)[0],text)
    remap={};kept=[]
    for i,e in enumerate(entries):
        if e['code'] in {'w','207'}:continue
        remap[i]=len(kept);kept.append((i,e))
    if text.lstrip().startswith('<'):
        import xml.etree.ElementTree as ET
        tree=ET.fromstring(text);i=0
        for node in list(tree):
            if node.tag in {'mm','dm','tm','o','h','adjustable'}:continue
            e=entries[i];i+=1
            if e['code'] in {'w','207'}:tree.remove(node)
            elif e['code']=='x':node.set('te',caption(node.get('te','')))
        for p in sorted(added_grounds):ET.SubElement(tree,'g',x=f'{p[0]} {p[1]} {p[0]} {p[1]+24}',f='0')
        for a,b in forest:ET.SubElement(tree,'w',x=f'{a[0]} {a[1]} {b[0]} {b[1]}',f='0')
        result['circuit']=ET.tostring(tree,encoding='unicode')+'\n'
    else:
        lines=[];i=0
        for line in text.splitlines():
            fields=line.split()
            if not fields or fields[0] in {'$','o','h','%','?','!','.'}:lines.append(line);continue
            e=entries[i];i+=1
            if e['code'] in {'w','207'}:continue
            if e['code']=='x':line=' '.join(fields[:7]+[escape(caption(unescape(' '.join(fields[7:]))))])
            lines.append(line)
        lines.extend(f'g {p[0]} {p[1]} {p[0]} {p[1]+24} 0' for p in sorted(added_grounds))
        lines.extend(f'w {a[0]} {a[1]} {b[0]} {b[1]} 0' for a,b in forest)
        result['circuit']='\n'.join(lines)+'\n'
    for binding in result['components']:
        if binding.get('element_index') is not None:binding['element_index']=remap[binding['element_index']]
        elif binding['type']=='gnd':
            p=tuple(binding['pins']['GND']['position'])
            if p in added_grounds:binding['element_index']=len(kept)+sorted(added_grounds).index(p)
    result['warnings']=[w for w in result.get('warnings',[]) if 'etiquetas' not in w]
    result['presentation_version']=VERSION
    return result
