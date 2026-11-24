"""Stage 18: reproducible electrical/export reference circuits, not detector metrics."""
import copy
import json
import math
from pathlib import Path
from backend.circuit_review import revision
from backend.kicad_export import write_kicad
from backend.project_export import generate_export
from backend.spice_netlist import build_netlist
from backend.spice_simulation import run_simulation,deck_hash

def component(cid,kind,value=None,params=None,model=''):
    from backend.circuit_types import PIN_SCHEMAS
    c=dict(id=cid,type=kind,value=value,pins=PIN_SCHEMAS[kind])
    if params:c['metadata']=dict(model={'text':model},electrical_parameters=params)
    return c
def circuit(components,nets):
    return dict(schema_version='circuit-1.0',status='confirmed',confirmed_by_user=True,
                components=components,nets=[dict(name=name,members=members) for name,members in nets])
def reference_cases():
    def divider():
        return circuit([component('VCC1','source_dc','5V'),component('R_top','resistor','1k'),component('R_bottom','resistor','1k')],
                       [('VIN',['VCC1.+','R_top.1']),('VOUT',['R_top.2','R_bottom.1']),('GND',['VCC1.-','R_bottom.2'])])
    # Independent expected values come from Ohm's law, RC response and simple transistor equations.
    cases=[]
    def add(cid,name,c,net,expected,tolerance,options=None):
        cases.append(dict(id=cid,name=name,circuit=c,net=net,expected=expected,tolerance=tolerance,options=options))
    for cid,name,kind,vf in [('T01','Fonte, resistor e LED','led',1.9),('T04','Diodo e resistor','diode',.697)]:
        c=divider();c['components'][2].update(type=kind,value=None,pins=['A','K'])
        c['nets'][1]['members'][-1]='R_bottom.A';c['nets'][2]['members'][-1]='R_bottom.K'
        add(cid,name,c,'VOUT',vf,.25)
    add('T02','Divisor resistivo',divider(),'VOUT',2.5,1e-6)
    c=divider();c['components'][2].update(type='capacitor',value='1uF')
    add('T03','Carga RC em cinco constantes de tempo',c,'VOUT',5*(1-math.exp(-5)),.005,dict(analysis='tran',step='10u',stop='5m',initial_zero=True))
    c=circuit([component('V1','source_dc','5'),component('R1','resistor','1k'),component('R2','resistor','2k')],
              [('VIN',['V1.+','R1.1','R2.1']),('GND',['V1.-','R1.2','R2.2'])])
    add('T05','Dois ramos paralelos',c,'VIN',5,1e-8)
    c=circuit([component('V1','source_dc','5'),component('V2','source_dc','3'),component('R1','resistor','1k'),component('R2','resistor','1k')],
              [('BRANCH1',['V1.+','R1.1']),('BRANCH2',['V2.+','R2.1']),('GND',['V1.-','V2.-','R1.2','R2.2'])])
    add('T06','Ramos que se cruzam sem conexão lógica',c,'BRANCH2',3,1e-8)
    for cid,kind,name,expected in [('T07','transistor_npn','BJT NPN',4.55),('T08','transistor_pnp','BJT PNP',.45)]:
        c=circuit([component('V1','source_dc','5'),component('RB','resistor','1M'),component('RL','resistor','1k'),component('Q1',kind,params={'hfe':100})],
                  [('VIN',['V1.+']+(['RL.1','RB.1'] if kind.endswith('npn') else ['Q1.E'])),
                   ('BASE',['RB.2','Q1.B']),('OUT',['Q1.C','RL.2' if kind.endswith('npn') else 'RL.1']),
                   ('GND',['V1.-']+(['Q1.E'] if kind.endswith('npn') else ['RB.1','RL.2']))])
        add(cid,name,c,'OUT',expected,.12)
    c=circuit([component('V1','source_dc','5'),component('VG','source_dc','3'),component('RL','resistor','1k'),
               component('M1','mosfet_n',params={'vgs_threshold':1,'mos_kp':.02})],
              [('VIN',['V1.+','RL.1']),('GATE',['VG.+','M1.G']),('OUT',['RL.2','M1.D']),('GND',['V1.-','VG.-','M1.S'])])
    add('T09','MOSFET canal N com KP explícito',c,'OUT',( .041-math.sqrt(.041**2-4*.01*.005))/.02,.003)
    c=divider();c['components'].append(component('U1','opamp',params={'open_loop_gain':100000,'gbw':700000},model='LM358'))
    c['nets'][1]['members'].append('U1.+');c['nets'].append(dict(name='BUFFER',members=['U1.OUT','U1.-']))
    add('T10','AOP seguidor com ganho finito',c,'BUFFER',2.5*100000/100001,1e-5)
    return sorted(cases,key=lambda c:c['id'])
def _run_reference_suite(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);results=[]
    for case in reference_cases():
        folder=root/case['id'];folder.mkdir(exist_ok=True);c=copy.deepcopy(case['circuit'])
        (folder/'circuit.json').write_text(json.dumps(c,indent=2),encoding='utf8')
        outcome=dict(id=case['id'],name=case['name'],expected_v=case['expected'],tolerance_v=case['tolerance'],passed=False)
        try:
            manifest=build_netlist(c,case['options']);manifest['netlist_sha256']=deck_hash(manifest['netlist'])
            (folder/'spice.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
            simulation=run_simulation(manifest,folder)
            if not simulation['success']:raise ValueError(simulation['message'])
            measured=simulation['last']['voltages_v'][case['net']]
            kicad=write_kicad(c,folder,dict(pcb=True));export=generate_export(folder,c,kicad)
            outcome.update(measured_v=measured,error_v=abs(measured-case['expected']),
                           passed=abs(measured-case['expected'])<=case['tolerance'],schematic=True,pcb=True,zip=True,
                           source_revision=revision(c),warnings=simulation['warnings']+kicad['warnings'])
        except Exception as error:outcome['error']=str(error)[:500]
        results.append(outcome)
    report=dict(schema_version='stage18-validation-1.0',total=len(results),passed=sum(c['passed'] for c in results),cases=results,
                scope='Circuitos lógicos confirmados, simulação ngspice, geração KiCad, PCB e ZIP. Não mede a precisão do detector ou OCR.',
                recognition_metrics=None)
    (root/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return report


_REFERENCE_LOCK=__import__('threading').RLock()

def run_reference_suite(root):
    with _REFERENCE_LOCK:
        return _run_reference_suite(root)

def run_reference_suite_cached(root,*,force=False):
    """Reuse only a passing report for these sources and this simulator runtime."""
    from backend.performance import runtime_signature,file_hash,digest,read_json,write_json
    from backend.spice_simulation import locate_ngspice
    from time import perf_counter
    base=Path(__file__).resolve().parent.parent;root=Path(root)
    engine=locate_ngspice();engine_folder=Path(engine).parent if engine else None
    engine_files=[Path(engine)] + list(engine_folder.glob('*.dll')) if engine_folder else []
    key=digest(dict(runtime=runtime_signature(base,include_assets=False),cases=reference_cases(),
                    engine={str(p):file_hash(p) for p in engine_files}))
    started=perf_counter()
    outputs=['validation_report.json']+[f'T{i:02d}/project.zip' for i in range(1,11)]
    with _REFERENCE_LOCK:
        cache=read_json(root/'reference_cache.json');report=read_json(root/'validation_report.json')
        valid=not force and cache.get('key')==key and report.get('passed')==report.get('total')==10
        hashes=cache.get('outputs')
        valid=valid and isinstance(hashes,dict) and set(hashes)==set(outputs) and all(value is not None and file_hash(root/name)==value for name,value in hashes.items())
        if valid:return dict(report,cached=True,elapsed_seconds=round(perf_counter()-started,4))
        report=run_reference_suite(root)
        if report.get('passed')==report.get('total')==10:
            hashes={name:file_hash(root/name) for name in outputs}
            if all(hashes.values()):write_json(root/'reference_cache.json',dict(key=key,outputs=hashes))
        return dict(report,cached=False,elapsed_seconds=round(perf_counter()-started,4))
