"""Spatial schematic derived from reviewed pins/nets, independent of recognition.

Coordinates are a presentation sidecar: the confirmed circuit remains authoritative.
No electrical connection is inferred merely because two drawn lines cross.
"""
import html
import math
from collections import Counter
from pathlib import Path
import json
from backend.circuit_types import PIN_SCHEMAS

VERSION = 'photo-routes-observed-leads-8'

def read_geometry(folder):
    def read(name):
        try:return json.loads((Path(folder)/name).read_text(encoding='utf8'))
        except (OSError, ValueError):return {}
    override=read('visual_geometry.json');circuit=read('circuit.json')
    if override and circuit:
        from backend.circuit_review import revision
        if override.get('source_revision')==revision(circuit):return override['connectivity'],{}
    return read('connectivity.json'), read('wires.json')

def point(p):
    x,y=float(p['x']),float(p['y'])
    if not all(math.isfinite(v) and abs(v)<1e6 for v in (x,y)):raise ValueError('Coordenada inválida.')
    return x,y

def on_segment(p,a,b):
    length=math.dist(a,b)
    if length<1e-9:return math.dist(p,a)<.002
    cross=abs((p[0]-a[0])*(b[1]-a[1])-(p[1]-a[1])*(b[0]-a[0]))/length
    return cross<.002 and min(a[0],b[0])-.001<=p[0]<=max(a[0],b[0])+.001 and min(a[1],b[1])-.001<=p[1]<=max(a[1],b[1])+.001

def protect_routes(routes,positions,member_net,warnings):
    """KiCad connects a foreign pin on a wire even without a junction dot.

    Keep clear captured segments. Detour only those hitting another net's
    endpoints, preserving all reviewed pins and electrical memberships.
    """
    for route in routes:
        foreign={p for m,p in positions.items() if member_net[m]!=route['net']}
        foreign.update(p for r in routes if r['net']!=route['net'] for path in r['paths'] for p in path)
        other_edges=[(a,b) for r in routes if r['net']!=route['net'] for path in r['paths'] for a,b in zip(path,path[1:])]
        new=[];changed=False
        def clear(path):
            return (all(not any(on_segment(p,a,b) for p in foreign) for a,b in zip(path,path[1:]))
                    and all(not any(on_segment(p,a,b) for a,b in other_edges) for p in path[1:-1]))
        for path in route['paths']:
            if not clear(path):
                # Reroute the whole branch, not a segment whose provisional
                # bend itself coincides with a foreign wire endpoint.
                from backend.schematic_routing import orthogonal
                candidate=orthogonal(path[0],path[-1],[],foreign)
                if clear(candidate):new.append(candidate);changed=True;continue
            rebuilt=[path[0]]
            for a,b in zip(path,path[1:]):
                if not any(on_segment(p,a,b) for p in foreign):rebuilt.append(b);continue
                replacement=None
                for distance in range(4,205,4):
                    for offset in (distance,-distance):
                        for candidate in ([a,(a[0],a[1]+offset),(b[0],a[1]+offset),b],
                                          [a,(a[0]+offset,a[1]),(a[0]+offset,b[1]),b]):
                            candidate=[p for i,p in enumerate(candidate) if not i or p!=candidate[i-1]]
                            if all(min(p)>=2 for p in candidate) and clear(candidate):replacement=candidate;break
                        if replacement:break
                    if replacement:break
                if replacement is None:
                    from backend.schematic_routing import RoutingConflict
                    raise RoutingConflict('Não foi possível desenhar '+route['net']+' sem tocar outra rede. Revise os terminais coincidentes.')
                rebuilt.extend(replacement[1:]);changed=True
            new.append(rebuilt)
        route['paths']=new
        if changed:warnings.append(route['net']+': traçado ajustado para não encostar em terminais de outra rede.')

def shapes(kind):
    """Shared SVG/KiCad primitives in a canonical frame. Pin leads are explicit."""
    lines=[];circles=[];texts=[]
    def line(*pts):lines.append(list(pts))
    if kind in {'gnd','vcc'}:
        line((0,0),(0,16))
        if kind=='gnd':
            for y,w in [(16,12),(21,8),(26,3)]:line((-w,y),(w,y))
        else:circles.append((0,21,5))
        return lines,circles,texts
    if kind=='opamp':
        line((-30,-16),(-15,-16));line((-30,16),(-15,16));line((15,0),(30,0))
        line((-15,-25),(15,0),(-15,25),(-15,-25));texts=[(-10,-12,'+'),(-10,20,'−')]
    elif kind.startswith('transistor_') or kind.startswith('mosfet_'):
        line((-30,0),(-6,0));line((-6,-18),(-6,18))
        if kind.startswith('transistor_'):
            line((-6,-8),(15,-24),(30,-24));line((-6,8),(15,24),(30,24))
            if kind=='transistor_npn':line((7,12),(15,24),(2,19))
            else:line((8,21),(-1,12),(12,14))
        else:
            line((0,-18),(0,18));line((0,-15),(15,-24),(30,-24));line((0,15),(15,24),(30,24))
            line((0,0),(13,0));line((8,-4),(13,0),(8,4)) if kind=='mosfet_n' else line((5,-4),(0,0),(5,4))
    else:
        edge=12
        if kind in {'capacitor','capacitor_polarized','battery'}:edge=3
        line((-30,0),(-edge,0));line((edge,0),(30,0))
        if kind=='resistor':line((-12,-5),(12,-5),(12,5),(-12,5),(-12,-5))
        elif kind in {'capacitor','capacitor_polarized','battery'}:
            line((-3,-12),(-3,12));w=6 if kind=='battery' else 12;line((3,-w),(3,w))
            if kind in {'capacitor_polarized','battery'}:texts=[(-10,-14,'+')]
        elif kind=='inductor':
            # Four digital coil lobes, without a spurious conductor through them.
            pts=[(-12,0)]
            for i in range(4):
                for j in range(1,9):
                    t=math.pi*j/8;pts.append((-12+6*i+3*(1-math.cos(t)),-5*math.sin(t)))
            line(*pts)
        elif kind in {'diode','led'}:
            line((-12,-10),(12,0),(-12,10),(-12,-10));line((12,-12),(12,12))
            if kind=='led':
                for dx in (0,10):line((dx,-15),(dx+9,-24));line((dx+3,-24),(dx+9,-24),(dx+9,-18))
        elif kind=='source_dc':
            circles.append((0,0,12));texts=[(-5,3,'+'),(5,3,'−')]
    return lines,circles,texts

def frame(kind,pins):
    """Affine frame preserves pin identities, including mirrored C/E and +/- ."""
    schema=PIN_SCHEMAS[kind]
    if len(schema)==1:
        ox,oy=pins[schema[0]];direction=1 if kind=='gnd' else -1
        return lambda x,y:(ox+x*.22,oy+y*.22*direction)
    if len(schema)==2:
        a,b=(pins[p] for p in schema);ux=(b[0]-a[0])/60;uy=(b[1]-a[1])/60
        norm=math.hypot(ux,uy);factor=min(1, .32/max(norm,1e-9))
        return lambda x,y:((a[0]+b[0])/2+ux*x-uy*y*factor,(a[1]+b[1])/2+uy*x+ux*y*factor)
    if kind=='opamp':
        plus,minus,out=(pins[p] for p in schema);left=((plus[0]+minus[0])/2,(plus[1]+minus[1])/2)
        ux,uy=(out[0]-left[0])/60,(out[1]-left[1])/60
        vx,vy=(minus[0]-plus[0])/32,(minus[1]-plus[1])/32
        ox,oy=(left[0]+out[0])/2,(left[1]+out[1])/2
    else:
        base,collector,emitter=(pins[p] for p in schema);right=((collector[0]+emitter[0])/2,(collector[1]+emitter[1])/2)
        ux,uy=(right[0]-base[0])/60,(right[1]-base[1])/60
        vx,vy=(emitter[0]-collector[0])/48,(emitter[1]-collector[1])/48
        ox,oy=(base[0]+right[0])/2,(base[1]+right[1])/2
    return lambda x,y:(ox+x*ux+y*vx,oy+x*uy+y*vy)

def make_diagram(circuit,connectivity=None,wires=None):
    from backend.schematic_routing import RoutingConflict
    try:return _make_diagram(circuit,connectivity,wires)
    except RoutingConflict:
        if not connectivity:raise
        # A crowded photograph must not make an otherwise valid circuit
        # impossible to export. Preserve netlist identities in a clear layout.
        diagram=_make_diagram(circuit)
        diagram['warnings'].append('Símbolos reorganizados no desenho para manter as redes separadas.')
        diagram['source_layout']='reorganized'
        return diagram

def _make_diagram(circuit,connectivity=None,wires=None):
    connectivity=connectivity or {};wires=wires or {};sources=connectivity.get('components',[])
    source_index={}
    for s in sources:
        for key in [s.get('component_id'),s.get('detection_id'),s.get('reference',{}).get('text')]:
            if key:source_index[key]=s
    selected=[];used=set();all_points=[];warnings=[]
    for i,c in enumerate(circuit['components']):
        origin=c.get('metadata',{}).get('source_component_id');s=source_index.get(origin) or source_index.get(c['id'])
        raw={t.get('pin'):point(t['position']) for t in (s or {}).get('terminals',[]) if t.get('position')}
        # Never silently reuse a footprint after a logical component changes type.
        spatial=s and s.get('type')==c['type'] and set(c['pins'])<=set(raw) and s.get('component_id') not in used
        if spatial:used.add(s.get('component_id'));all_points.extend(raw[p] for p in c['pins'])
        selected.append((c,s if spatial else None,raw if spatial else {}))
    minx=min((p[0] for p in all_points),default=0);miny=min((p[1] for p in all_points),default=0)
    from backend.schematic_routing import aligned_pins,orthogonal
    # Remove the common camera tilt, then use CircuitJS's 8 pixel grid.
    slopes=[]
    for c,s,raw in selected:
        if len(raw)!=2:continue
        a,b=raw.values();dx,dy=b[0]-a[0],b[1]-a[1]
        if math.hypot(dx,dy)<10:continue
        angle=math.atan2(dy,dx);angle=(angle+math.pi/4)%(math.pi/2)-math.pi/4
        if abs(angle)<math.radians(20):slopes.append(angle)
    tilt=sorted(slopes)[len(slopes)//2] if slopes else 0
    # The GND's sole observed lead enters from the opposite side to its bars.
    # Rotate every pin and captured wire together before layout/grid rounding.
    votes=[]
    for c,s,raw in selected:
        if not s or c['type']!='gnd' or 'GND' not in raw:continue
        b=s.get('bbox') or {}
        if not all(k in b for k in ('x1','y1','x2','y2')):continue
        p=raw['GND'];dx=(b['x1']+b['x2'])/2-p[0];dy=(b['y1']+b['y2'])/2-p[1]
        if math.hypot(dx,dy)<max(3,min(b['x2']-b['x1'],b['y2']-b['y1'])*.12):continue
        direction=round((math.atan2(dy,dx)-tilt)/(math.pi/2))%4
        votes.append((direction-1)%4)
    ground_turn=Counter(votes).most_common(1)[0] if votes else (0,0)
    turns=ground_turn[0] if ground_turn[1]>=max(1,math.ceil(len(votes)*.6)) else 0
    photo_tilt=tilt;tilt+=turns*math.pi/2
    cos,sin=math.cos(tilt),math.sin(tilt)
    def rotated(p):return (p[0]-minx)*cos+(p[1]-miny)*sin,-(p[0]-minx)*sin+(p[1]-miny)*cos
    rotated_points=[rotated(p) for p in all_points]
    lowx=min((p[0] for p in rotated_points),default=0);lowy=min((p[1] for p in rotated_points),default=0)
    def convert(p):
        x,y=rotated(p)
        return 2*round((20+(x-lowx)*.32)/2),2*round((20+(y-lowy)*.32)/2)
    components=[];positions={};source_members={}
    for i,(c,s,raw) in enumerate(selected):
        if s:
            pins={p:convert(raw[p]) for p in c['pins']}
            for p in c['pins']:source_members[s['component_id']+':'+p]=c['id']+'.'+p
        else:
            from backend.kicad_export import layout_pins
            # New manually added parts get their own area beside the captured drawing.
            x=40+(i%5)*48+(max((convert(p)[0] for p in all_points),default=0)+35 if all_points else 0);y=35+(i//5)*45
            pins={p:(round(x+px,3),round(y-py,3)) for p,px,py,_ in layout_pins(c['type'])}
            warnings.append(c['id']+': posição original indisponível; colocado na área de componentes adicionais.')
        pins={pin:(2*round(p[0]/2),2*round(p[1]/2)) for pin,p in pins.items()}
        pins=aligned_pins(c['type'],pins)
        transform=frame(c['type'],pins);lines,circles,texts=shapes(c['type'])
        digital_lines=[[transform(*p) for p in line] for line in lines]
        # Circles become polygons under an affine transform, retaining source orientation.
        for x,y,r in circles:digital_lines.append([transform(x+r*math.cos(t*math.pi/16),y+r*math.sin(t*math.pi/16)) for t in range(33)])
        digital_text=[(*transform(x,y),text) for x,y,text in texts]
        center=transform(0,0)
        components.append(dict(id=c['id'],type=c['type'],pins=pins,center=center,lines=digital_lines,texts=digital_text,spatial=bool(s)))
        positions.update({c['id']+'.'+p:pos for p,pos in pins.items()})
    member_net={m:n['name'] for n in circuit['nets'] for m in n['members']}
    from backend.schematic_routing import separate_presentation
    components,adjustments=separate_presentation(components,member_net)
    warnings.extend(adjustments)
    positions={c['id']+'.'+pin:p for c in components for pin,p in c['pins'].items()}
    source_nets={}
    for n in connectivity.get('nets',[]):
        members=[source_members.get(ref) for ref in n.get('terminal_refs',[])]
        names={member_net.get(m) for m in members}
        if members and None not in names and len(names)==1:
            source_nets.setdefault(next(iter(names)),[]).extend(n.get('wire_ids',[]))
    segments={s['wire_id']:s for s in wires.get('segments',[])};routes=[];junctions=[]
    wire_net={}
    for net in connectivity.get('nets',[]):
        mapped={member_net.get(source_members.get(ref)) for ref in net.get('terminal_refs',[])}
        if len(mapped)==1 and None not in mapped:
            wire_net.update({wid:next(iter(mapped)) for wid in net.get('wire_ids',[])})
    for index,lead in enumerate(connectivity.get('recovered_observed_leads',[])):
        names=set()
        for anchor in lead.get('anchors',[]):
            if anchor.startswith('T:'):names.add(member_net.get(source_members.get(anchor[2:])))
            elif anchor.startswith('W:'):names.add(wire_net.get(anchor[2:]))
        if len(names)!=1 or None in names:continue
        source_name=next(iter(names))
        sid='OBSERVED_'+str(index);segments[sid]=dict(points=lead['points'])
        source_nets.setdefault(source_name,[]).append(sid)
    ground_members={c['id']+'.GND' for c in circuit['components'] if c['type']=='gnd'}
    for net in circuit['nets']:
        points=[positions[m] for m in net['members']];paths=[]
        grounds={positions[m] for m in net['members'] if m in ground_members}
        # Ground markers share node zero electrically. Draw one local tree per
        # marker, never the imaginary bus between separate photographed grounds.
        groups=[{p} for p in points]
        from backend.photo_routes import captured_paths
        observed=captured_paths(source_nets.get(net['name'],[]),segments,convert,points,grounds)
        if observed:
            paths.extend(observed)
            groups=[]
            for path in observed:
                joined={p for p in path}
                for old in list(groups):
                    if joined & old:joined|=old;groups.remove(old)
                groups.append(joined)
            for p in points:
                if not any(p in g for g in groups):groups.append({p})
        while len(groups)>1:
            best=None
            for i,a in enumerate(groups):
                for j in range(i+1,len(groups)):
                    if grounds & a and grounds & groups[j]:continue
                    for p in a:
                        for q in groups[j]:
                            candidate=(math.dist(p,q),i,j,p,q)
                            if best is None or candidate<best:best=candidate
            if best is None:break
            _,i,j,p,q=best
            if p!=q:paths.append(orthogonal(p,q,components,[point for member,point in positions.items() if member_net[member]!=net['name']]))
            groups[i]|=groups[j];groups.pop(j)
        routes.append(dict(net=net['name'],members=list(net['members']),paths=paths))
    protect_routes(routes,positions,member_net,warnings)
    for route in routes:
        degree=Counter();edges=set()
        for path in route['paths']:
            for a,b in zip(path,path[1:]):
                edge=tuple(sorted((a,b)))
                if a==b or edge in edges:continue
                edges.add(edge);degree[a]+=1;degree[b]+=1
        junctions.extend(dict(net=route['net'],point=p) for p,d in degree.items() if d>=3)
    extent=list(positions.values())+[p for r in routes for path in r['paths'] for p in path]+[p for c in components for line in c['lines'] for p in line]
    return dict(version=VERSION,components=components,routes=routes,junctions=junctions,
                width=max(p[0] for p in extent)+38,height=max(p[1] for p in extent)+25,
                source_layout='image' if all(c['spatial'] for c in components) else 'mixed' if all_points else 'logical',photo_tilt_degrees=round(math.degrees(photo_tilt),2),ground_rotation_degrees=turns*90,ground_orientation_votes=len(votes),warnings=warnings)

def diagram_svg(diagram,circuit):
    esc=lambda x:html.escape(str(x),quote=True)
    w,h=diagram['width'],diagram['height'];svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.3f} {h:.3f}" role="img" aria-label="Esquemático digital conectado"><rect width="100%" height="100%" fill="#ffffff"/>']
    for r in diagram['routes']:
        svg.append(f'<g data-net="{esc(r["net"])}" fill="none" stroke="#357a68" stroke-width=".65" stroke-linejoin="round">')
        for path in r['paths']:svg.append('<polyline points="'+' '.join(f'{x:.3f},{y:.3f}' for x,y in path)+'"/>')
        svg.append('</g>')
    for j in diagram['junctions']:
        x,y=j['point'];svg.append(f'<circle cx="{x}" cy="{y}" r="1" fill="#357a68" data-net="{esc(j["net"])}"/>')
    originals={c['id']:c for c in circuit['components']}
    for c in diagram['components']:
        svg.append(f'<g data-component="{esc(c["id"])}"><title>{esc(c["id"])} — {esc(c["type"])}</title><g fill="none" stroke="#243647" stroke-width=".65" stroke-linecap="round" stroke-linejoin="round">')
        for line in c['lines']:svg.append('<polyline points="'+' '.join(f'{x:.3f},{y:.3f}' for x,y in line)+'"/>')
        svg.append('</g>')
        for x,y,text in c['texts']:svg.append(f'<text x="{x:.3f}" y="{y:.3f}" font-family="Arial" font-size="3.2">{esc(text)}</text>')
        original=originals[c['id']];model=original.get('metadata',{}).get('model') or {};model=model.get('text','') if isinstance(model,dict) else str(model)
        value=original.get('value_text') or original.get('value') or '';x,y=c['center'];maxx=max(p[0] for p in c['pins'].values())
        vertical=len(c['pins'])==1 or max(p[1] for p in c['pins'].values())-min(p[1] for p in c['pins'].values())>max(p[0] for p in c['pins'].values())-min(p[0] for p in c['pins'].values())
        lx,ly=(maxx+6,y-4) if vertical else (x,y-13)
        anchor='start' if vertical else 'middle'
        svg.append(f'<text x="{lx:.3f}" y="{ly:.3f}" text-anchor="{anchor}" font-family="Arial" font-size="3.8" font-weight="bold">{esc(c["id"])}</text>')
        for i,text in enumerate(filter(None,[value,model])):svg.append(f'<text x="{lx:.3f}" y="{ly+5*(i+1):.3f}" text-anchor="{anchor}" font-family="Arial" font-size="3.2">{esc(text)}</text>')
        svg.append('</g>')
    return '\n'.join(svg+['</svg>'])
