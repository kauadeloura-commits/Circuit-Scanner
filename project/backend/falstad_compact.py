"""Remove empty grid strips, preserving native posts and the electrical graph.

Coordinates are compressed monotonically on each axis. Equal coordinates remain
equal and separate tracks keep at least one grid cell. Fixed-size CircuitJS
three-pin posts and symbol bodies reserve their own uncompressed space.
"""
import bisect
import copy
import math

VERSION='native-empty-space-compaction-1'
GRID=8
NON_ELEMENTS={'$','o','h','%','?','!','.'}


def _body(e):
    a,b=e['a'],e['b'];code=e['code']
    if code in {'g','R'}:
        return (min(a[0],b[0])-16,min(a[1],b[1]),max(a[0],b[0])+16,max(a[1],b[1]))
    if code in {'t','f'}:
        if a[1]==b[1]:
            left,right=(40,16) if a[0]<b[0] else (16,40)
            return b[0]-left,b[1]-24,b[0]+right,b[1]+24
        above,below=(40,16) if a[1]<b[1] else (16,40)
        return b[0]-24,b[1]-above,b[0]+24,b[1]+below
    if code=='a':
        # Input spacing is fixed at +/-16. Reserve a normal-size triangle,
        # rather than retaining the photographed distance to the output.
        if a[1]==b[1]:
            end=a[0]+math.copysign(min(48,abs(b[0]-a[0])),b[0]-a[0])
            return min(a[0],end)-8,a[1]-24,max(a[0],end)+8,a[1]+24
        end=a[1]+math.copysign(min(48,abs(b[1]-a[1])),b[1]-a[1])
        return a[0]-24,min(a[1],end)-8,a[0]+24,max(a[1],end)+8
    x,y=(a[0]+b[0])/2,(a[1]+b[1])/2
    return x-24,y-24,x+24,y+24


def _axis(values,intervals,gap):
    values=set(values);bands=[]
    for lo,hi in sorted((GRID*math.floor(lo/GRID),GRID*math.ceil(hi/GRID)) for lo,hi in intervals):
        if bands and lo<=bands[-1][1]:bands[-1][1]=max(hi,bands[-1][1])
        else:bands.append([lo,hi])
        values.update((lo,hi))
    old=sorted(values);new=[32]
    for a,b in zip(old,old[1:]):
        protected=any(lo<=a and b<=hi for lo,hi in bands)
        new.append(new[-1]+(b-a if protected else min(b-a,gap)))
    def mapped(value):
        i=bisect.bisect_right(old,value)-1
        if i<0:return new[0]+value-old[0]
        if i==len(old)-1:return new[-1]+value-old[-1]
        return new[i]+(value-old[i])*(new[i+1]-new[i])/(old[i+1]-old[i])
    return mapped


def _graph(entries):
    """Native terminal partitions, including implicit grounds and label nodes."""
    from backend.digital_twin import on_segment
    from backend.falstad_edit import describe,unescape
    parent={};tags={};points=set();grounds=[];labels={};wires=[]
    def find(p):
        parent.setdefault(p,p)
        while parent[p]!=p:parent[p]=parent[parent[p]];p=parent[p]
        return p
    def join(a,b):parent[find(a)]=find(b)
    for i,e in enumerate(entries):
        if e['code']=='w':wires.append((e['a'],e['b']));points.update((e['a'],e['b']))
        elif e['code']=='207':labels.setdefault(unescape(' '.join(e['extra'])),[]).append(e['a']);points.add(e['a'])
        elif e['code']!='x':
            for pin,p in describe(e)[2].items():points.add(p);tags[i,pin]=p
            if e['code']=='g':grounds.append(e['a'])
    rows={};columns={}
    for p in points:rows.setdefault(p[1],[]).append(p);columns.setdefault(p[0],[]).append(p)
    for a,b in wires:
        join(a,b)
        nearby=rows[a[1]] if a[1]==b[1] else columns[a[0]] if a[0]==b[0] else points
        for p in nearby:
            if on_segment(p,a,b):join(a,p)
    for group in [grounds,*labels.values()]:
        for p in group[1:]:join(group[0],p)
    groups={}
    for tag,p in tags.items():groups.setdefault(find(p),[]).append(tag)
    return {tuple(sorted(g)) for g in groups.values()}


def _write(text,entries):
    if text.lstrip().startswith('<'):
        import xml.etree.ElementTree as ET
        tree=ET.fromstring(text);i=0
        for node in tree:
            if node.tag in {'mm','dm','tm','o','h','adjustable'}:continue
            e=entries[i];i+=1;node.set('x',' '.join(map(str,e['a']+e['b'])))
        return ET.tostring(tree,encoding='unicode')+'\n'
    lines=[];i=0
    for line in text.splitlines():
        f=line.split()
        if f and f[0] not in NON_ELEMENTS:
            e=entries[i];i+=1;f[1:5]=map(str,e['a']+e['b']);line=' '.join(f)
        lines.append(line)
    return '\n'.join(lines)+'\n'


def _captions(entries,bindings):
    """Move captions near their symbol, without changing text or font size."""
    from backend.falstad_edit import unescape
    bodies={b['component_id']:_body(entries[b['element_index']]) for b in bindings if b.get('element_index') is not None}
    placed=[]
    wire_boxes=[(min(e['a'][0],e['b'][0])-2,min(e['a'][1],e['b'][1])-2,
                 max(e['a'][0],e['b'][0])+2,max(e['a'][1],e['b'][1])+2)
                for e in entries if e['code']=='w']
    def intersects(a,b):return max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])
    for e in entries:
        if e['code']!='x':continue
        text=unescape(' '.join(e['extra'][1:] if e['extra'] and e['extra'][0].isdigit() else e['extra']))
        cid=text.split(' | ',1)[0]
        if cid not in bodies:continue
        x1,y1,x2,y2=bodies[cid];cx,cy=(x1+x2)/2,(y1+y2)/2
        width=max(16,len(text)*7.5);candidates=[]
        for offset in (0,16,32,48,64,80,96,128):
            candidates.extend([(x2+12+offset,cy-8),(cx-width/2,y1-12-offset),
                               (cx-width/2,y2+24+offset),(x1-width-12-offset,cy-8)])
        for x,y in candidates:
            x,y=GRID*round(x/GRID),GRID*round(y/GRID);box=(x-4,y-16,x+width+4,y+4)
            if x<8 or y<16 or any(intersects(box,b) for b in [*bodies.values(),*placed,*wire_boxes]):continue
            e['a']=e['b']=(int(x),int(y));placed.append(box);break


def _size(entries):
    points=[p for e in entries if e['code']!='x' for p in (e['a'],e['b'])]
    return (max(p[0] for p in points)-min(p[0] for p in points),
            max(p[1] for p in points)-min(p[1] for p in points)) if points else (0,0)


def _overlaps(entries):
    boxes={i:_body(e) for i,e in enumerate(entries) if e['code'] not in {'w','207','x'}}
    return {(i,j) for i,a in boxes.items() for j,b in boxes.items() if i<j
            and max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])}


def compact_native(circuit,data):
    if data.get('compact_version')==VERSION:return data
    from backend.falstad_edit import parse_deck,describe
    original=parse_deck(data['circuit'])
    if not original:return data
    expected=_graph(original)
    for gap in (GRID,16,24,32):
        entries=copy.deepcopy(original)
        virtual={(i,pin):tuple(p['position']) for i,binding in enumerate(data['components'])
                 if binding.get('element_index') is None for pin,p in binding['pins'].items()}
        # Recompute symbol centers after a pass; stop when no empty strips remain.
        for _ in range(6):
            physical=[e for e in entries if e['code'] not in {'w','207','x'}]
            boxes=[_body(e) for e in physical]
            # A freehand diagonal is rigid: anisotropic compression must not
            # remove an existing junction in the middle of a slanted segment.
            boxes += [(min(e['a'][0],e['b'][0]),min(e['a'][1],e['b'][1]),max(e['a'][0],e['b'][0]),max(e['a'][1],e['b'][1]))
                      for e in entries if e['code']=='w' and e['a'][0]!=e['b'][0] and e['a'][1]!=e['b'][1]]
            posts=[p for e in physical for p in describe(e)[2].values()]
            points=[p for e in entries if e['code']!='x' for p in (e['a'],e['b'])]+posts+list(virtual.values())
            fx=_axis([p[0] for p in points],[(b[0],b[2]) for b in boxes],gap)
            fy=_axis([p[1] for p in points],[(b[1],b[3]) for b in boxes],gap)
            mapped=lambda p:(int(round(fx(p[0]))),int(round(fy(p[1]))))
            changed=copy.deepcopy(entries)
            for e in changed:e['a'],e['b']=mapped(e['a']),mapped(e['b'])
            if any({pin:mapped(p) for pin,p in describe(a)[2].items()}!=describe(b)[2]
                   for a,b in zip(entries,changed) if a['code'] not in {'w','207','x'}):break
            if _graph(changed)!=expected:break
            if _overlaps(changed)-_overlaps(entries):break
            done=all(a['a']==b['a'] and a['b']==b['b'] for a,b in zip(entries,changed))
            entries=changed
            virtual={key:mapped(p) for key,p in virtual.items()}
            if done:break
        if _graph(entries)!=expected:continue
        # Component indices and every non-geometric parameter are unchanged.
        result=copy.deepcopy(data)
        for binding_index,binding in enumerate(result['components']):
            index=binding.get('element_index')
            if index is not None:
                posts=describe(entries[index])[2]
                for pin,p in binding['pins'].items():p['position']=list(posts[pin])
            else:
                # Virtual markers must follow the same map as visible wires.
                for pin,p in binding['pins'].items():p['position']=list(virtual[binding_index,pin])
        _captions(entries,result['components'])
        result['circuit']=_write(data['circuit'],entries)
        result['compact_version']=VERSION
        before,after=_size(original),_size(entries)
        result['compaction']=dict(before=list(before),after=list(after),
            wire_length_before=round(sum(math.dist(e['a'],e['b']) for e in original if e['code']=='w'),2),
            wire_length_after=round(sum(math.dist(e['a'],e['b']) for e in entries if e['code']=='w'),2),
            electrical_connections_preserved=True)
        return result
    return data
