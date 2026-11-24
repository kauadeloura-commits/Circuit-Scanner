"""Stages 15/16: self-contained KiCad schematic and optional routed PCB."""
import json
import math
import re
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL
from backend.circuit_document import validar_circuito
from backend.circuit_review import revision
from backend.circuit_types import PIN_SCHEMAS
from backend.digital_twin import make_diagram, diagram_svg, read_geometry, VERSION

ARTIFACTS=('schematic.kicad_sch','schematic.kicad_pro','CircuitScanner.kicad_sym',
           'sym-lib-table','fp-lib-table','board.kicad_pcb','schematic.kicad_pcb','kicad.json','schematic.svg','board.svg','schematic_layout.json','circuitjs.txt','circuitjs.json','pcb_routing.json')
PRESETS={
 'terminal_1':dict(label='Ponto de alimentação/terra',count=1,positions=[(0,0)],size=1.7,drill=1.0,types=['gnd','vcc']),
 'terminal_2':dict(label='Conector de 2 sinais — passo 2,54 mm',count=2,positions=[(0,0),(2.54,0)],size=1.7,drill=1.0),
 'terminal_3':dict(label='Conector de 3 sinais — passo 2,54 mm',count=3,positions=[(0,0),(2.54,0),(5.08,0)],size=1.7,drill=1.0),
 'axial_7_62':dict(label='Axial de 2 terminais — passo 7,62 mm',count=2,positions=[(0,0),(7.62,0)],size=2.0,drill=.8),
 'radial_5':dict(label='Radial de 2 terminais — passo 5 mm',count=2,positions=[(0,0),(5,0)],size=2.0,drill=.8),
 'to92_cbe':dict(label='TO-92: C, B, E — confira o encapsulamento',count=3,positions=[(0,0),(2.54,0),(5.08,0)],pin_numbers={'C':'1','B':'2','E':'3'},size=1.7,drill=.8,types=['transistor_npn','transistor_pnp']),
 'to92_ebc':dict(label='TO-92: E, B, C — confira o encapsulamento',count=3,positions=[(0,0),(2.54,0),(5.08,0)],pin_numbers={'E':'1','B':'2','C':'3'},size=1.7,drill=.8,types=['transistor_npn','transistor_pnp']),
 'to220_gds':dict(label='TO-220: G, D, S — confira o encapsulamento',count=3,positions=[(0,0),(2.54,0),(5.08,0)],size=2.0,drill=1.0,types=['mosfet_n','mosfet_p']),
}
def quote(text):
    return json.dumps(str(text),ensure_ascii=False)
def uid(text):return str(uuid5(NAMESPACE_URL,'circuit-scanner/'+text))
def effects(size=1.0):return f'(effects (font (size {size} {size})))'
def model_text(c):
    value=c.get('metadata',{}).get('model',{})
    return value.get('text','') if isinstance(value,dict) else str(value or '')
def confirmed(circuit):
    validation=validar_circuito(circuit)
    if not validation['valid']:raise ValueError('; '.join(validation['errors']))
    if circuit.get('confirmed_by_user') is not True or circuit.get('status')!='confirmed':
        raise ValueError('Revise e confirme o circuito antes de gerar os arquivos.')
    if len(circuit['components'])>200:raise ValueError('A exportação inicial aceita até 200 componentes por projeto.')
    for c in circuit['components']:
        if c['type'] not in PIN_SCHEMAS or set(c['pins'])!=set(PIN_SCHEMAS[c['type']]):raise ValueError('Confira os pinos de '+c['id'])
def layout_pins(kind):
    pins=PIN_SCHEMAS[kind]
    if len(pins)==1:return [(pins[0],-7.62,0,0)]
    if len(pins)==2:return [(pins[0],-7.62,0,0),(pins[1],7.62,0,180)]
    if kind=='opamp':return [('+',-7.62,2.54,0),('-',-7.62,-2.54,0),('OUT',7.62,0,180)]
    return [(pins[0],-7.62,0,0),(pins[1],7.62,3.81,180),(pins[2],7.62,-3.81,180)]
def symbol_definition(kind,key=None,pin_numbers=None):
    pins=layout_pins(kind)
    key=key or kind;pin_numbers=pin_numbers or {}
    # Standalone symbols contain explicit logical pin names and stable numbered pins.
    graphics='(rectangle (start -3.81 5.08) (end 3.81 -5.08) (stroke (width 0.254) (type default)) (fill (type background)))'
    if kind=='resistor':graphics='(rectangle (start -3.81 1.27) (end 3.81 -1.27) (stroke (width 0.254) (type default)) (fill (type none)))'
    elif kind in {'capacitor','capacitor_polarized'}:
        graphics=' '.join(f'(polyline (pts (xy {x} 3.81) (xy {x} -3.81)) (stroke (width .254) (type default)) (fill (type none)))' for x in (-.635,.635))
    elif kind=='opamp':graphics='(polyline (pts (xy -3.81 5.08) (xy 3.81 0) (xy -3.81 -5.08) (xy -3.81 5.08)) (stroke (width .254) (type default)) (fill (type background)))'
    elif kind in {'source_dc','battery'}:graphics='(circle (center 0 0) (radius 3.81) (stroke (width .254) (type default)) (fill (type none)))'
    s=[f'(symbol {quote("CircuitScanner:"+key)} (pin_names (offset .5)) (in_bom yes) (on_board yes)',
       f'(property "Reference" "X" (at 0 7.62 0) {effects()})',
       f'(property "Value" {quote(kind)} (at 0 -7.62 0) {effects()})',
       f'(symbol {quote(key+"_0_1")} {graphics})',f'(symbol {quote(key+"_1_1")}']
    for number,(pin,x,y,angle) in enumerate(pins,1):
        number=pin_numbers.get(pin,str(number))
        s.append(f'(pin passive line (at {x} {y} {angle}) (length 3.81) (name {quote(pin)} {effects()}) (number "{number}" {effects()}))')
    return '\n'.join(s)+'))'
def export_options(circuit,options):
    options=options or {}
    if not isinstance(options,dict) or set(options)-{'pcb','width_mm','height_mm','footprints'}:raise ValueError('Opções de exportação inválidas.')
    pcb=options.get('pcb',False)
    if not isinstance(pcb,bool):raise ValueError('A opção PCB deve ser verdadeira ou falsa.')
    physical=list(circuit['components'])
    columns=min(8,max(1,len(physical)));rows=math.ceil(len(physical)/columns)
    dims=[]
    for key,default,minimum in [('width_mm',max(40,columns*22+12),30),('height_mm',max(30,rows*18+12),20)]:
        value=options.get(key,default)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not minimum<=value<=500:raise ValueError('Dimensões de placa inválidas (limite 500 mm).')
        dims.append(float(value))
    if pcb and (dims[0]<columns*22+12 or dims[1]<rows*18+12):raise ValueError('A placa é pequena para o posicionamento inicial. Aumente suas dimensões.')
    chosen=options.get('footprints',{})
    if not isinstance(chosen,dict) or set(chosen)-{c['id'] for c in physical}:raise ValueError('Seleção de footprints inválida.')
    assignments={}
    for c in physical:
        name=chosen.get(c['id'],'terminal_'+str(len(c['pins'])))
        p=PRESETS.get(name)
        if not p or p['count']!=len(c['pins']) or ('types' in p and c['type'] not in p['types']):raise ValueError('Footprint incompatível com '+c['id'])
        assignments[c['id']]=name
    return dict(pcb=pcb,width_mm=dims[0],height_mm=dims[1],footprints=assignments,columns=columns)
def spatial_symbol(component,key,pin_numbers):
    x,y=component['center'];graphics=[]
    for line in component['lines']:
        pts=' '.join(f'(xy {px-x:.6f} {y-py:.6f})' for px,py in line)
        graphics.append(f'(polyline (pts {pts}) (stroke (width .254) (type default)) (fill (type none)))')
    for tx,ty,text in component['texts']:
        graphics.append(f'(text {quote(text)} (at {tx-x:.6f} {y-ty:.6f} 0) {effects(1.5)})')
    pins=[]
    for i,(pin,(px,py)) in enumerate(component['pins'].items(),1):
        num=pin_numbers.get(pin,str(i))
        pins.append(f'(pin passive line (at {px-x:.6f} {y-py:.6f} 0) (length 0) (name {quote(pin)} {effects(.8)}) (number {quote(num)} {effects(.8)}))')
    return f'(symbol {quote("CircuitScanner:"+key)} (pin_names (offset 0) hide) (pin_numbers hide) (in_bom yes) (on_board yes) (property "Reference" "X" (at 0 10 0) {effects()}) (property "Value" "" (at 0 -10 0) {effects()}) (symbol {quote(key+"_0_1")} '+ ' '.join(graphics)+f') (symbol {quote(key+"_1_1")} '+ ' '.join(pins)+'))'

def build_kicad(circuit,options=None,geometry=None):
    confirmed(circuit);opts=export_options(circuit,options);rev=revision(circuit)
    diagram=make_diagram(circuit,*(geometry or ({},{})))
    placement={c['id']:c for c in diagram['components']}
    root=uid(rev+'/root');members={m:n['name'] for n in circuit['nets'] for m in n['members']}
    valid_ref=re.compile(r'^[A-Za-z][A-Za-z_]*[0-9]+$')
    reference_map={c['id']:c['id'] for c in circuit['components'] if valid_ref.fullmatch(c['id'])}
    reserved=set(reference_map.values());prefixes={'resistor':'R','capacitor':'C','capacitor_polarized':'C','inductor':'L','diode':'D','led':'D','source_dc':'V','battery':'BT','opamp':'U','transistor_npn':'Q','transistor_pnp':'Q','mosfet_n':'Q','mosfet_p':'Q','gnd':'GND','vcc':'PWR'}
    for c in circuit['components']:
        if c['id'] in reference_map:continue
        prefix=prefixes[c['type']];number=1
        while prefix+str(number) in reserved:number+=1
        reference_map[c['id']]=prefix+str(number);reserved.add(prefix+str(number))
    variants={};symbol_keys={}
    for i,c in enumerate(circuit['components']):
        fp=opts['footprints'].get(c['id']);numbers=PRESETS.get(fp,{}).get('pin_numbers',{})
        key=c['type']+('_'+fp if numbers else '')+'_spatial_'+str(i);symbol_keys[c['id']]=key
        variants[key]=spatial_symbol(placement[c['id']],key,numbers)
    definitions=[variants[key] for key in sorted(variants)]
    sch=[f'(kicad_sch (version 20230121) (generator "circuit_scanner") (uuid {root}) (paper "A0")',
         '(lib_symbols '+ '\n'.join(definitions)+')']
    bindings=[];warnings=diagram['warnings']+(['PCB com roteamento inicial: confira encapsulamentos, dimensões e regras no KiCad antes de fabricar.'] if opts['pcb'] else [])
    for index,c in enumerate(circuit['components']):
        place=placement[c['id']];x,y=place['center'];kind=c['type'];symbol_uid=uid(rev+'/'+c['id'])
        physical=kind not in {'gnd','vcc'};footprint=opts['footprints'].get(c['id']);value=' | '.join(dict.fromkeys(filter(None,[c.get('value_text') or c.get('value'),model_text(c)]))) or kind;ref=reference_map[c['id']]
        sch += [f'(symbol (lib_id {quote("CircuitScanner:"+symbol_keys[c["id"]])}) (at {x} {y} 0) (unit 1) (in_bom {"yes" if physical else "no"}) (on_board {"yes" if physical else "no"}) (uuid {symbol_uid})',
                f'(property "Reference" {quote(ref)} (at {x} {y-10.16} 0) {effects()})',
                f'(property "Value" {quote(value)} (at {x} {y+10.16} 0) {effects()})',
                f'(property "Footprint" {quote("CircuitScanner:"+footprint if footprint and opts["pcb"] else "")} (at {x} {y} 0) (effects (font (size 1 1)) hide))',
                f'(property "CircuitScannerType" {quote(kind)} (at {x} {y} 0) (effects (font (size 1 1)) hide))',
                f'(property "CircuitScannerId" {quote(c["id"])} (at {x} {y} 0) (effects (font (size 1 1)) hide))']
        pins={}
        for n,(pin,px,py,angle) in enumerate(layout_pins(kind),1):
            n=PRESETS.get(footprint,{}).get('pin_numbers',{}).get(pin,str(n))
            sch.append(f'(pin "{n}" (uuid {uid(symbol_uid+pin)}))')
            pins[pin]=str(n)
        sch.append(f'(instances (project "schematic" (path "/{root}" (reference {quote(ref)}) (unit 1)))))')
        for pin,(sx,sy) in place['pins'].items():
            net=members[c['id']+'.'+pin]
            sch.append(f'(label {quote(net)} (at {sx} {sy} 0) (effects (font (size .8 .8)) (justify left bottom)) (uuid {uid(symbol_uid+pin+"label")}))')
        bindings.append(dict(component_id=c['id'],reference=ref,type=kind,symbol_uuid=symbol_uid,pins=pins,footprint=footprint,
                             nets={pin:members[c['id']+'.'+pin] for pin in c['pins']}))
    seen=set()
    for route in diagram['routes']:
        for path in route['paths']:
            for a,b in zip(path,path[1:]):
                pair=tuple(sorted((a,b)))
                if a==b or pair in seen:continue
                seen.add(pair)
                sch.append(f'(wire (pts (xy {a[0]} {a[1]}) (xy {b[0]} {b[1]})) (stroke (width 0) (type default)) (uuid {uid(rev+route["net"]+str(pair))}))')
    for j in diagram['junctions']:
        px,py=j['point'];sch.append(f'(junction (at {px} {py}) (diameter 0) (color 0 0 0 0) (uuid {uid(rev+str(j))}))')
    sch.append('(sheet_instances (path "/" (page "1"))))')
    files={'schematic.kicad_sch':'\n'.join(sch),'schematic.svg':diagram_svg(diagram,circuit),
           'schematic_layout.json':json.dumps(diagram,ensure_ascii=False,indent=2),
           'CircuitScanner.kicad_sym':'(kicad_symbol_lib (version 20220914) (generator "circuit_scanner") '+ '\n'.join(d.replace('CircuitScanner:','') for d in definitions)+')',
           'sym-lib-table':'(sym_lib_table (lib (name "CircuitScanner") (type "KiCad") (uri "${KIPRJMOD}/CircuitScanner.kicad_sym") (options "") (descr "Logical circuit symbols")))',
           'fp-lib-table':'(fp_lib_table (lib (name "CircuitScanner") (type "KiCad") (uri "${KIPRJMOD}/CircuitScanner.pretty") (options "") (descr "Initial footprints")))',
           'schematic.kicad_pro':json.dumps({'meta':{'filename':'schematic.kicad_pro','version':1}},indent=2)}
    routing=None
    if opts['pcb']:
        pads=[]
        board=['(kicad_pcb (version 20221018) (generator "circuit_scanner") (general (thickness 1.6)) (paper "A4")',
               '(layers (0 "F.Cu" signal) (31 "B.Cu" signal) (36 "B.SilkS" user "b.silkscreen") (37 "F.SilkS" user "f.silkscreen") (38 "B.Mask" user) (39 "F.Mask" user) (44 "Edge.Cuts" user))',
               '(setup (pad_to_mask_clearance 0)) (net 0 "")']
        codes={net['name']:i+1 for i,net in enumerate(circuit['nets'])}
        for name,code in codes.items():board.append(f'(net {code} {quote(name)})')
        physical=[b for b in bindings if b['footprint']]
        for i,b in enumerate(physical):
            p=PRESETS[b['footprint']];x=10+(i%opts['columns'])*22;y=10+(i//opts['columns'])*18
            fp=[f'(footprint {quote("CircuitScanner:"+b["footprint"])} (layer "F.Cu") (at {x} {y}) (tstamp {b["symbol_uuid"]}) (path "/{root}/{b["symbol_uuid"]}") (attr through_hole)',
                f'(fp_text reference {quote(b["reference"])} (at 2.54 -3) (layer "F.SilkS") {effects()})',
                f'(fp_text value {quote(b["type"])} (at 2.54 3) (layer "F.SilkS") {effects(.8)})']
            for pin,num in b['pins'].items():
                px,py=p['positions'][int(num)-1];net=b['nets'][pin]
                pads.append(dict(member=b['component_id']+'.'+pin,x=x+px,y=y+py,size=p['size'],net=net,rect=num=='1'))
                fp.append(f'(pad {quote(num)} thru_hole {"rect" if num=="1" else "circle"} (at {px} {py}) (size {p["size"]} {p["size"]}) (drill {p["drill"]}) (layers "*.Cu" "*.Mask") (net {codes[net]} {quote(net)}))')
                fp.append(f'(fp_text user {quote(pin)} (at {px} {py-1.7}) (layer "F.SilkS") {effects(.8)})')
            board.append('\n'.join(fp)+')')
            # Local library footprint has no instance reference or nets.
            library=[f'(footprint {quote(b["footprint"])} (version 20221018) (generator "circuit_scanner") (layer "F.Cu") (attr through_hole)']
            for number,(px,py) in enumerate(p['positions'],1):
                library.append(f'(pad "{number}" thru_hole {"rect" if number==1 else "circle"} (at {px} {py}) (size {p["size"]} {p["size"]}) (drill {p["drill"]}) (layers "*.Cu" "*.Mask"))')
            files['CircuitScanner.pretty/'+b['footprint']+'.kicad_mod']='\n'.join(library)+')'
            if b['footprint'].startswith('terminal_') and len(b['pins'])>2:
                warnings.append(b['component_id']+': PCB usa conector dos sinais lógicos, não o encapsulamento comercial. A alimentação física do AOP deve ser acrescentada no KiCad.' if b['type']=='opamp' else b['component_id']+': PCB usa conector dos sinais; escolha o encapsulamento físico para montar o transistor.')
        w=opts['width_mm'];h=opts['height_mm']
        from backend.pcb_router import route_board
        tracks,vias,routing=route_board(pads,w,h)
        for i,t in enumerate(tracks):
            a,b=t['start'],t['end'];board.append(f'(segment (start {a[0]} {a[1]}) (end {b[0]} {b[1]}) (width {t["width"]}) (layer {quote(t["layer"])}) (net {codes[t["net"]]}) (tstamp {uid(rev+"track"+str(i))}))')
        for i,v in enumerate(vias):
            x,y=v['position'];board.append(f'(via (at {x} {y}) (size {v["size"]}) (drill {v["drill"]}) (layers "F.Cu" "B.Cu") (net {codes[v["net"]]}) (tstamp {uid(rev+"via"+str(i))}))')
        files['pcb_routing.json']=json.dumps(dict(routing,pads=pads,segments=tracks,via_positions=vias),ensure_ascii=False,indent=2)
        for n in routing['nets']:
            if n['unrouted_members']:warnings.append(n['net']+': complete no KiCad as ligações '+', '.join(n['unrouted_members']))
        for start,end in [((0,0),(w,0)),((w,0),(w,h)),((w,h),(0,h)),((0,h),(0,0))]:
            board.append(f'(gr_line (start {start[0]} {start[1]}) (end {end[0]} {end[1]}) (stroke (width .05) (type default)) (layer "Edge.Cuts") (tstamp {uid(rev+str(start))}))')
        board.append(')');files['board.kicad_pcb']='\n'.join(board)
        # KiCad's project manager finds the board with the schematic basename.
        files['schematic.kicad_pcb']=files['board.kicad_pcb']
    manifest=dict(schema_version='stage15-16-kicad-1.0',source_revision=rev,options=opts,components=bindings,
                  warnings=warnings,schematic=True,pcb=opts['pcb'],routed=bool(routing and routing['complete']),routing=routing,format_version='KiCad 7+',files=sorted(files),layout_version=VERSION,source_layout=diagram['source_layout'])
    return manifest,files

def write_kicad(circuit,folder,options=None,geometry=None):
    folder=Path(folder)
    manifest,files=build_kicad(circuit,options,geometry=read_geometry(folder) if geometry is None else geometry)
    from backend.circuitjs_export import export_circuitjs
    try:
        falstad=export_circuitjs(circuit,json.loads(files['schematic_layout.json']))
        files['circuitjs.txt']=falstad['circuit'];files['circuitjs.json']=json.dumps(falstad,ensure_ascii=False,indent=2)
        manifest['files']=sorted(files)
    except ValueError as error:manifest['warnings'].append('Falstad: '+str(error))
    for name,text in files.items():
        path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf8')
    (folder/'kicad.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    return manifest
