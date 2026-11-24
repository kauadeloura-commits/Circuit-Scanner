import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile
from backend import main,test_spice_api as spice_fixture
from backend.circuit_review import revision
from backend.kicad_export import build_kicad,write_kicad,PRESETS
from backend.project_export import generate_export,export_is_current,base_address,qr_svg
from backend.production_validation import reference_cases,run_reference_suite
from backend.spice_netlist import build_netlist
from backend.spice_simulation import run_simulation
from backend.test_spice_stages import divider

class ProductionGenerationTests(unittest.TestCase):
    def test_all_logical_classes_have_embedded_symbols_and_exact_pin_map(self):
        from backend.circuit_types import PIN_SCHEMAS
        c=divider()
        for i,(kind,pins) in enumerate(PIN_SCHEMAS.items()):
            cid='X'+str(i);c['components'].append(dict(id=cid,type=kind,pins=pins,value=None))
            for pin in pins:c['nets'].append(dict(name='NET_'+str(i)+'_'+str(pins.index(pin)),members=[cid+'.'+pin]))
        manifest,files=build_kicad(c,dict(pcb=True))
        for b in manifest['components']:
            self.assertEqual(set(b['pins']),set(PIN_SCHEMAS[b['type']]))
            self.assertEqual(len(set(b['pins'].values())),len(b['pins']))
            self.assertIn('CircuitScanner:'+b['type'],files['schematic.kicad_sch'])
        self.assertIn('(net 0 "")',files['board.kicad_pcb'])
        self.assertNotIn('(segment ',files['board.kicad_pcb'])
    def test_invalid_reference_is_mapped_with_original_id_retained(self):
        m,f=build_kicad(divider())
        refs={b['component_id']:b['reference'] for b in m['components']}
        self.assertEqual(refs['VCC1'],'VCC1');self.assertRegex(refs['R_top'],r'^R\d+$')
        self.assertIn('(property "CircuitScannerId" "R_top"',f['schematic.kicad_sch'])
        self.assertEqual(len(set(refs.values())),len(refs))
    def test_to92_physical_numbers_keep_bce_nets_for_both_orders(self):
        c=next(case['circuit'] for case in reference_cases() if case['id']=='T07')
        for preset,expected in [('to92_cbe',{'C':'1','B':'2','E':'3'}),('to92_ebc',{'E':'1','B':'2','C':'3'})]:
            manifest,files=build_kicad(c,dict(pcb=True,footprints={'Q1':preset}))
            b=next(b for b in manifest['components'] if b['component_id']=='Q1')
            self.assertEqual(b['pins'],expected)
            self.assertIn('transistor_npn_'+preset,files['schematic.kicad_sch'])
            self.assertIn('(path ',files['board.kicad_pcb'])
    def test_generation_requires_confirmation_and_compatible_physical_choices(self):
        c=divider();c['confirmed_by_user']=False
        with self.assertRaises(ValueError):build_kicad(c)
        for opts in ({'pcb':'yes'},{'pcb':True,'width_mm':30},{'footprints':{'R_top':'terminal_3'}},{'footprints':{'../x':'terminal_2'}},{'width_mm':float('nan')}):
            with self.subTest(options=opts),self.assertRaises(ValueError):build_kicad(divider(),opts)
    def test_quoted_component_model_cannot_escape_kicad_string(self):
        c=divider();c['components'][1]['metadata']={'model':{'text':'a") (evil "x'}}
        _,files=build_kicad(c);self.assertIn('a\\\") (evil \\\"x',files['schematic.kicad_sch'])
    def test_zip_only_has_selected_artifacts_and_changes_make_it_stale(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);c=divider();(folder/'circuit.json').write_text(json.dumps(c))
            (folder/'private.db').write_text('private');(folder/'secret.txt').write_text('secret')
            m=write_kicad(c,folder,dict(pcb=True));e=generate_export(folder,c,m)
            with zipfile.ZipFile(folder/'project.zip') as z:
                self.assertIsNone(z.testzip());self.assertIn('schematic.kicad_sch',z.namelist());self.assertIn('board.kicad_pcb',z.namelist())
                self.assertNotIn('private.db',z.namelist());self.assertNotIn('share.json',z.namelist());self.assertNotIn('secret.txt',z.namelist())
            self.assertTrue(export_is_current(folder,c));(folder/'schematic.kicad_sch').write_text('changed')
            self.assertIsNone(export_is_current(folder,c));self.assertTrue(e['local_address'])
    def test_qr_encodes_share_link_and_server_addresses_are_validated(self):
        import xml.etree.ElementTree as ET
        url='http://192.168.1.10:8010/share/'+'a'*32
        self.assertTrue(ET.fromstring(qr_svg(url)).tag.endswith('svg'))
        self.assertFalse(base_address('http://192.168.1.10:8010')[1])
        self.assertTrue(base_address('http://[::1]:8010')[1])
        for address in ('javascript:alert(1)','https://a:b@example.com','http://x/path','http://x:bad','https://x?q=1'):
            with self.assertRaises(ValueError):base_address(address)
    def test_old_simulation_is_not_included_in_a_current_package(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);c=divider();m=write_kicad(c,folder)
            (folder/'simulation.cir').write_text('old deck')
            (folder/'simulation.json').write_text(json.dumps(dict(success=True,source_revision='old')))
            e=generate_export(folder,c,m)
            self.assertFalse(e['report']['simulation_success'])
            self.assertNotIn('simulation.cir',e['files']);self.assertNotIn('simulation.json',e['files'])
    def test_reference_suite_records_ten_expected_vs_observed_cases(self):
        with TemporaryDirectory() as tmp:
            report=run_reference_suite(tmp)
            self.assertEqual((report['passed'],report['total']),(10,10),report)
            self.assertIsNone(report['recognition_metrics'])
            self.assertTrue((Path(tmp)/'T10/project.zip').is_file())

class ExtendedSimulationTests(unittest.TestCase):
    def simulate(self,c):
        with TemporaryDirectory() as tmp:
            result=run_simulation(build_netlist(c),tmp)
            self.assertTrue(result['success'],result);return result
    def test_explicit_manual_policy_requires_missing_parameters(self):
        for case in reference_cases():
            if case['id'] not in {'T07','T08','T09','T10'}:continue
            c=copy.deepcopy(case['circuit']);c['components'][-1]['metadata']={'parameter_policy':{k:'manual' for k in __import__('backend.component_models',fromlist=['FIELDS']).FIELDS[c['components'][-1]['type']]}}
            with self.assertRaises(ValueError):build_netlist(c)
    def test_pmos_current_direction_and_threshold_polarity(self):
        c=next(c['circuit'] for c in reference_cases() if c['id']=='T09');c=copy.deepcopy(c)
        c['components'][-1].update(type='mosfet_p',metadata={'electrical_parameters':{'vgs_threshold':-1,'mos_kp':.02}})
        c['components'][1]['value']='2'
        c['nets']=[dict(name='VIN',members=['V1.+','M1.S']),dict(name='GATE',members=['VG.+','M1.G']),
                   dict(name='OUT',members=['RL.1','M1.D']),dict(name='GND',members=['V1.-','VG.-','RL.2'])]
        result=self.simulate(c);self.assertAlmostEqual(result['last']['voltages_v']['OUT'],4.874188,delta=.003)
        self.assertLess(result['last']['currents_a']['M1'],0)
        direction=next(d for d in result['current_directions'] if d['component_id']=='M1')
        self.assertEqual((direction['from_pin'],direction['to_pin']),('D','S'))
    def test_datasheet_opt_in_and_manual_bjt_gain_change_the_deck(self):
        c=copy.deepcopy(next(c['circuit'] for c in reference_cases() if c['id']=='T07'))
        c['components'][-1]['metadata']=dict(model={'text':'BC547B'},parameter_policy={'hfe':'datasheet'})
        deck=build_netlist(c);self.assertIn('BF=325',deck['netlist'])
        c['components'][-1]['metadata']['electrical_parameters']={'hfe':'150'}
        self.assertIn('BF=150',build_netlist(c)['netlist'])
    def test_mos_kp_fit_requires_rds_and_drive_and_does_not_treat_gm_as_gain(self):
        c=copy.deepcopy(next(c['circuit'] for c in reference_cases() if c['id']=='T09'))
        c['components'][-1]['metadata']=dict(electrical_parameters={'vgs_threshold':1,'gm':.32},parameter_policy={'mos_kp':'manual','rds_on':'manual','gate_drive_voltage':'manual'})
        with self.assertRaises(ValueError):build_netlist(c)
        c['components'][-1]['metadata']['electrical_parameters'].update(rds_on=1,gate_drive_voltage=3)
        self.assertIn('KP=0.5',build_netlist(c)['netlist'])
    def test_inductor_series_resistance_is_used_in_simulation(self):
        c=divider();c['components'][2].update(type='inductor',value='10mH',metadata=dict(electrical_parameters={'series_resistance':100}))
        result=self.simulate(c);self.assertAlmostEqual(result['last']['voltages_v']['VOUT'],5*100/1100,places=6)

class ProductionAPITests(unittest.TestCase):
    setUp=spice_fixture.SpiceAPItests.setUp
    tearDown=spice_fixture.SpiceAPItests.tearDown
    update=spice_fixture.SpiceAPItests.update
    prepare=spice_fixture.SpiceAPItests.prepare
    def test_current_simulation_and_deck_survive_export_on_windows(self):
        self.prepare()
        self.assertTrue(self.client.post('/projects/TEST/simulation',json={}).json()['success'])
        self.client.post('/projects/TEST/kicad',json={})
        result=self.client.post('/projects/TEST/export',json={})
        self.assertEqual(result.status_code,200,result.text)
        self.assertTrue(result.json()['report']['simulation_success'])
        package=self.client.get('/projects/TEST/export/download')
        with zipfile.ZipFile(io.BytesIO(package.content)) as archive:
            self.assertTrue({'simulation.cir','simulation.json','simulation.csv','spice.json'}<=set(archive.namelist()))
    def test_confirmed_project_exports_downloads_shares_and_revokes(self):
        c=self.prepare()
        r=self.client.post('/projects/TEST/kicad',json={'base_revision':c['revision'],'pcb':True});self.assertEqual(r.status_code,200,r.text)
        self.assertTrue(self.client.get('/projects/TEST/stage15-status').json()['criterion_met'])
        self.assertTrue(self.client.get('/projects/TEST/stage16-status').json()['criterion_met'])
        self.assertEqual(self.client.get('/projects/TEST/kicad/download/schematic').status_code,200)
        self.assertEqual(self.client.get('/projects/TEST/kicad/download/pcb').status_code,200)
        e=self.client.post('/projects/TEST/export',json={'base_url':'http://192.168.1.10:8010'});self.assertEqual(e.status_code,200,e.text)
        token=e.json()['token'];url='/share/'+token
        self.assertEqual(self.client.get(url).status_code,200)
        package=self.client.get(url+'/download');self.assertEqual(package.status_code,200)
        with zipfile.ZipFile(io.BytesIO(package.content)) as z:self.assertIn('circuit.json',z.namelist())
        self.assertEqual(self.client.get('/projects/TEST/export/qr').status_code,200)
        self.assertTrue(self.client.get('/projects/TEST/stage17-status').json()['criterion_met'])
        self.assertEqual(self.client.post('/projects/TEST/share/revoke').status_code,200)
        self.assertEqual(self.client.get(url).status_code,404)
        new=self.client.post('/projects/TEST/export',json={}).json();self.assertNotEqual(token,new['token'])
    def test_unconfirmed_revision_conflict_unknown_artifacts_and_tampering_are_blocked(self):
        self.client.post('/projects/TEST/connectivity');self.client.post('/projects/TEST/circuit')
        self.assertEqual(self.client.post('/projects/TEST/kicad',json={}).status_code,422)
        c=self.prepare()
        self.assertEqual(self.client.post('/projects/TEST/kicad',json={'base_revision':'old'}).status_code,409)
        self.client.post('/projects/TEST/kicad',json={});self.client.post('/projects/TEST/export',json={})
        self.assertEqual(self.client.get('/projects/TEST/kicad/download/private').status_code,404)
        (self.folder/'schematic.kicad_sch').write_text('changed')
        self.assertEqual(self.client.get('/projects/TEST/kicad/download/schematic').status_code,409)
        self.assertEqual(self.client.get('/projects/TEST/export/download').status_code,409)
    def test_review_edit_invalidates_final_stage_files_and_keeps_image_geometry(self):
        current=self.prepare();before=(self.folder/'terminals.json').read_bytes()
        self.client.post('/projects/TEST/kicad',json={'pcb':True});self.client.post('/projects/TEST/export',json={})
        payload=dict(base_revision=current['revision'],components=[dict(c,original_id=c['id']) for c in current['circuit']['components']],nets=current['circuit']['nets'])
        payload['components'][1]['value']='2k';self.assertEqual(self.client.post('/projects/TEST/circuit-review',json=payload).status_code,200)
        for name in ('kicad.json','board.kicad_pcb','project.zip','export.json'):self.assertFalse((self.folder/name).exists(),name)
        self.assertEqual((self.folder/'terminals.json').read_bytes(),before)
    def test_final_pages_are_in_workflow_and_reference_report_is_downloadable(self):
        self.prepare()
        self.assertEqual(self.client.get('/projects/TEST/production-view').status_code,200)
        self.assertIn('Gerar arquivos e concluir',self.client.get('/projects/TEST/workflow').text)
        self.assertIn('GuidedStep',self.client.get('/projects/TEST/guided/editor/production').text)
        self.assertIn('Esquemático, PCB e ZIP',self.client.get('/projects/TEST/simulation-view').text)
        with patch.object(main,'BASE_DIR',self.root):
            report=self.client.post('/validation/run',json={});self.assertEqual(report.status_code,200,report.text)
            self.assertEqual(report.json()['passed'],10);self.assertEqual(self.client.get('/validation/report').status_code,200)

if __name__=='__main__':unittest.main()
