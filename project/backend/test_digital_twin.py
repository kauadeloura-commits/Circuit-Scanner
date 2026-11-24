import copy,json,subprocess,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from backend.digital_twin import make_diagram,diagram_svg
from backend.circuitjs_export import export_circuitjs
from backend.kicad_export import build_kicad
from backend.spice_netlist import build_netlist
from backend.spice_simulation import run_simulation,locate_ngspice
from backend.power_rails import rail_sources
from backend.test_spice_stages import divider
from backend.production_validation import reference_cases
from backend import main,test_spice_api as fixture

def geometry(circuit):
    # Distinct spatial orientation, not the old horizontal export layout.
    components=[]
    for i,c in enumerate(circuit['components']):
        x=100+(i%3)*180;y=80+(i//3)*180
        pts=[(x,y),(x,y+90)] if len(c['pins'])==2 else [(x,y),(x+80,y-40),(x+80,y+40)] if len(c['pins'])==3 else [(x,y)]
        components.append(dict(component_id=c['id'],type=c['type'],reference={},terminals=[dict(pin=p,position=dict(x=px,y=py)) for p,(px,py) in zip(c['pins'],pts)]))
    nets=[dict(name=n['name'],terminal_refs=[m.replace('.',':') for m in n['members']],wire_ids=[]) for n in circuit['nets']]
    return dict(components=components,nets=nets),dict(segments=[])

class TwinTests(unittest.TestCase):
    def test_photo_positions_and_renamed_source_preserved(self):
        c=divider();g,w=geometry(c);c['components'][1]['metadata']={'source_component_id':'R_top'}
        c['components'][1]['id']='R_REVISED'
        for n in c['nets']:n['members']=[m.replace('R_top.','R_REVISED.') for m in n['members']]
        d=make_diagram(c,g,w);self.assertEqual(d['source_layout'],'image')
        pins=d['components'][1]['pins'];self.assertEqual(pins['1'][0],pins['2'][0]);self.assertGreater(pins['2'][1],pins['1'][1])
        self.assertEqual({r['net']:set(r['members']) for r in d['routes']},{n['name']:set(n['members']) for n in c['nets']})
        from backend.kicad_export import write_kicad
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'connectivity.json').write_text(json.dumps(g),encoding='utf8');(folder/'wires.json').write_text(json.dumps(w),encoding='utf8')
            manifest=write_kicad(c,folder);self.assertEqual(manifest['source_layout'],'image');self.assertIn('circuitjs.txt',manifest['files']);self.assertTrue((folder/'circuitjs.txt').exists())

    def test_added_part_gets_fallback_without_reusing_other_part(self):
        c=divider();g,w=geometry(c);c['components'][1]['metadata']={'source_component_id':'VCC1'}
        d=make_diagram(c,g,w);self.assertEqual(d['source_layout'],'mixed');self.assertFalse(d['components'][1]['spatial']);self.assertTrue(d['warnings'])

    def test_ocr_markup_cannot_enter_svg_and_value_model_both_visible(self):
        c=divider();c['components'][1]['metadata']={'model':{'text':'<script>alert(1)</script>'}}
        d=make_diagram(c);svg=diagram_svg(d,c);self.assertNotIn('<script>',svg);self.assertIn('&lt;script&gt;',svg);self.assertIn('1kΩ',svg)
        _,files=build_kicad(c);self.assertIn('1kΩ | <script>',files['schematic.kicad_sch'])

    def test_changed_net_does_not_reuse_old_photo_wires(self):
        c=divider();g,w=geometry(c);g['nets'][0]['wire_ids']=['OLD'];w['segments']=[dict(wire_id='OLD',points=[dict(x=999,y=999),dict(x=888,y=888)])]
        c['nets'][0]['members'].remove('R_top.1');c['nets'].append(dict(name='NEW',members=['R_top.1']))
        d=make_diagram(c,g,w);self.assertLess(d['width'],300)

    def test_unreferenced_ink_fragments_not_exported(self):
        c=divider();g,w=geometry(c);w['segments']=[dict(wire_id='TEXT',points=[dict(x=8000,y=8000),dict(x=9000,y=9000)])]
        self.assertLess(make_diagram(c,g,w)['width'],300)

    def test_wire_detours_foreign_pin_instead_of_shorting_it(self):
        c=divider();g,w=geometry(c)
        g['components'][0]['terminals'][0]['position']=dict(x=100,y=100)
        g['components'][0]['terminals'][1]['position']=dict(x=100,y=200)
        g['components'][1]['terminals'][0]['position']=dict(x=100,y=300)
        g['components'][1]['terminals'][1]['position']=dict(x=200,y=300)
        d=make_diagram(c,g,w)
        from backend.digital_twin import on_segment
        foreign=d['components'][0]['pins']['-'];vin=next(r for r in d['routes'] if r['net']=='VIN')
        self.assertFalse(any(on_segment(foreign,a,b) for path in vin['paths'] for a,b in zip(path,path[1:])))
        self.assertTrue(any('traçado ajustado' in warning for warning in d['warnings']))

    def test_native_families_reference_cases_export_and_no_wire_cycle(self):
        for case in reference_cases():
            c=case['circuit'];g,w=geometry(c);d=make_diagram(c,g,w)
            with self.subTest(case=case['id']):
                native=export_circuitjs(c,d);self.assertEqual(len(native['components']),len(c['components']))
                reordered=copy.deepcopy(c)
                for part in reordered['components']:part['pins'].reverse()
                self.assertEqual(export_circuitjs(reordered,d)['circuit'],native['circuit'])
                parent={}
                def find(a):
                    parent.setdefault(a,a)
                    while parent[a]!=a:a=parent[a]
                    return a
                for line in native['circuit'].splitlines():
                    t=line.split()
                    if t[0]!='w':continue
                    a,b=tuple(t[1:3]),tuple(t[3:5]);ra,rb=find(a),find(b);self.assertNotEqual(ra,rb);parent[ra]=rb
                for comp in native['components']:
                    original=next(x for x in c['components'] if x['id']==comp['component_id'])
                    self.assertEqual(set(comp['pins']),set(original['pins']))
                    if original['type'] in {'battery','source_dc'}:
                        self.assertEqual(comp['pins']['+']['post_index'],1);self.assertEqual(comp['pins']['-']['post_index'],0)
                    if original['type']=='mosfet_n':self.assertEqual(comp['pins']['D']['post_index'],2)
                    if original['type']=='opamp':self.assertEqual(comp['pins']['-']['post_index'],0)

    def test_coincident_terminals_different_nets_rejected(self):
        c=divider();d=make_diagram(c);d['components'][0]['pins']['+']=d['components'][0]['pins']['-']
        with self.assertRaisesRegex(ValueError,'coincidem'):export_circuitjs(c,d)

    def test_missing_ground_marker_gets_physical_ground(self):
        c=divider();text=export_circuitjs(c,make_diagram(c))['circuit'];self.assertEqual(sum(line.startswith('g ') for line in text.splitlines()),1)

    def test_nested_labels_and_duplicate_paths_do_not_make_wire_loop(self):
        c=divider();d=make_diagram(c);d['routes'][0]['paths']*=3
        text=export_circuitjs(c,d)['circuit'];self.assertEqual(sum(line.startswith('207 ') for line in text.splitlines()),3)

class RailAndResultTests(unittest.TestCase):
    def rail_divider(self):
        c=divider();c['components'][0].update(type='vcc',pins=['VCC']);c['nets'][0]['members'][0]='VCC1.VCC';c['nets'][-1]['members'].remove('VCC1.-');return c

    def test_explicit_vcc_energizes_divider_in_actual_ngspice(self):
        if not locate_ngspice():self.skipTest('ngspice indisponível')
        with TemporaryDirectory() as folder:
            result=run_simulation(build_netlist(self.rail_divider()),folder)
            self.assertTrue(result['success'],result);self.assertAlmostEqual(result['last']['voltages_v']['VOUT'],2.5)

    def test_identical_rails_deduplicated_conflict_rejected(self):
        c=self.rail_divider();c['components'].append(dict(id='VCC2',type='vcc',value='5V',pins=['VCC']));c['nets'][0]['members'].append('VCC2.VCC')
        self.assertEqual(len(rail_sources(c)),1);c['components'][-1]['value']='12V'
        with self.assertRaisesRegex(ValueError,'diferentes'):rail_sources(c)

    def test_unvalued_vcc_not_invented_or_duplicate_of_grounded_source(self):
        c=divider();c['components'].append(dict(id='PWR',type='vcc',value=None,pins=['VCC']));c['nets'][0]['members'].append('PWR.VCC')
        self.assertEqual(rail_sources(c),[]);c['components'][-1]['value']='5V';self.assertEqual(rail_sources(c),[])
        c['components'][-1]['value']='12V'
        with self.assertRaisesRegex(ValueError,'conflita'):rail_sources(c)

    def test_zero_exit_aborted_analysis_not_success(self):
        with TemporaryDirectory() as folder,patch('backend.spice_simulation.subprocess.run',return_value=subprocess.CompletedProcess([],0,'run simulation(s) aborted','')):
            result=run_simulation(build_netlist(divider()),folder,command='mock')
            self.assertFalse(result['success']);self.assertEqual(result['error_code'],'simulator_error');self.assertFalse((Path(folder)/'simulation.csv').exists())

    def test_partial_transient_raw_not_success(self):
        m=build_netlist(divider(),dict(analysis='tran',step='10u',stop='10m'))
        partial=dict(last=dict(time_s=1e-8),samples=[],point_count=1)
        with TemporaryDirectory() as folder,patch('backend.spice_simulation.subprocess.run',return_value=subprocess.CompletedProcess([],0,'','')),patch('backend.spice_simulation.parse_raw',return_value=([],[])),patch('backend.spice_simulation.measurements',return_value=partial):
            result=run_simulation(m,folder,command='mock');self.assertFalse(result['success']);self.assertIn('duração solicitada',result['message'])

class TwinAPITests(unittest.TestCase):
    setUp=fixture.SpiceAPItests.setUp
    tearDown=fixture.SpiceAPItests.tearDown
    prepare=fixture.SpiceAPItests.prepare
    update=fixture.SpiceAPItests.update

    def test_confirmed_api_preview_native_download_and_package_files(self):
        self.prepare()
        for route in ['digital-twin','circuitjs','circuitjs/download']:
            response=self.client.get('/projects/TEST/'+route);self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(self.client.get('/projects/MISSING/'+route).status_code,404)
        response=self.client.post('/projects/TEST/kicad',json={});self.assertEqual(response.status_code,200,response.text)
        self.assertIn('circuitjs.txt',response.json()['files']);self.assertIn('schematic_layout.json',response.json()['files'])
        self.assertEqual(self.client.get('/simulators/circuitjs/circuitjs.html').status_code,200)
        self.assertFalse((self.folder/'simulation.json').exists())
        self.assertEqual(self.client.post('/projects/TEST/export',json={}).status_code,200)
        path=self.folder/'export.json';manifest=json.loads(path.read_text(encoding='utf8'));manifest.pop('layout_version');path.write_text(json.dumps(manifest),encoding='utf8')
        self.assertEqual(self.client.get('/projects/TEST/export/download').status_code,409)

    def test_draft_diagram_available_but_simulator_requires_confirmation(self):
        self.client.post('/projects/TEST/connectivity');self.client.post('/projects/TEST/circuit')
        response=self.client.get('/projects/TEST/digital-twin');self.assertEqual(response.status_code,200);self.assertFalse(response.json()['confirmed'])
        self.assertEqual(self.client.get('/projects/TEST/circuitjs').status_code,409)

from backend import test_guided_workflow as guided_fixture
class TwinWorkflowTests(unittest.TestCase):
    setUp=guided_fixture.GuidedTests.setUp
    tearDown=guided_fixture.GuidedTests.tearDown
    update=guided_fixture.GuidedTests.update
    prepare=guided_fixture.GuidedTests.prepare
    status=guided_fixture.GuidedTests.status
    step=guided_fixture.GuidedTests.step

    def test_falstad_choice_reaches_files_without_false_ngspice_result(self):
        self.prepare();self.assertEqual(self.step(dict(engine='falstad',analysis='op'))['phase'],'simulation')
        with patch.object(main,'simular_projeto',side_effect=AssertionError('ngspice não selecionado')):
            self.assertEqual(self.step()['phase'],'production');self.assertEqual(self.status()['phase'],'production')
            self.assertEqual(self.step(dict(pcb=False))['phase'],'kicad');self.assertEqual(self.step()['phase'],'export')
            self.assertEqual(self.step()['phase'],'validation')
        self.assertTrue((self.folder/'project.zip').is_file());self.assertFalse((self.folder/'simulation.json').exists())
        flow=json.loads((self.folder/'guided_workflow.json').read_text(encoding='utf8'));self.assertTrue(flow['falstad']['import_ready'])
        (self.folder/'circuitjs.txt').write_text('changed',encoding='utf8');self.assertEqual(self.status()['phase'],'simulation')

    def test_unknown_simulator_cannot_advance(self):
        self.prepare();s=self.status();response=self.client.post('/projects/TEST/guided/step',json=dict(s,options=dict(engine='other')))
        self.assertEqual(response.status_code,422);self.assertEqual(self.status()['phase'],'circuit')

    def test_ngspice_choice_still_runs_real_calculation(self):
        if not locate_ngspice():self.skipTest('ngspice indisponível')
        self.prepare();self.step(dict(engine='ngspice',analysis='op'));self.assertEqual(self.step()['phase'],'production')
        self.assertAlmostEqual(self.client.get('/projects/TEST/simulation').json()['last']['voltages_v']['VOUT'],2.5)

if __name__=='__main__':unittest.main()
