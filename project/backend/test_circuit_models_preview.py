import copy,json,unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
from backend.component_models import automatic_model_fields,FIELDS,CATALOG
from backend.circuit_review import apply_review,revision
from backend.production_validation import reference_cases
from backend.spice_netlist import build_netlist
from backend.spice_simulation import run_simulation
from backend.test_circuit_review import HumanReviewTests
from backend import main

class AutomaticModelsTests(unittest.TestCase):
    def test_blank_model_and_parameters_get_defaults_with_evidence(self):
        for kind in ('transistor_npn','transistor_pnp','mosfet_n','mosfet_p','opamp','inductor'):
            with self.subTest(kind=kind):
                c=automatic_model_fields(dict(type=kind))
                self.assertEqual(c['model']['source'],'suggested')
                self.assertTrue(c['electrical_parameters'])
                for p in c['electrical_parameters'].values():
                    self.assertTrue(p['estimated']);self.assertTrue(p['datasheet']['url'])
                if kind.startswith('mosfet'):
                    self.assertGreater(c['electrical_parameters']['mos_kp']['value'],0)
                    self.assertEqual(c['electrical_parameters']['mos_kp']['source'],'datasheet_derived')
                    self.assertEqual(c['electrical_parameters']['vgs_threshold']['value']>0,kind=='mosfet_n')

    def test_manual_values_win_and_derived_kp_uses_them(self):
        c=automatic_model_fields(dict(type='mosfet_n',model={'text':'IRLZ44N'},electrical_parameters={'vgs_threshold':1,'rds_on':1,'gate_drive_voltage':3}))
        self.assertEqual(c['electrical_parameters']['mos_kp']['value'],.5)
        c['electrical_parameters']['mos_kp']={'value':.07,'source':'manual'}
        self.assertEqual(automatic_model_fields(c)['electrical_parameters']['mos_kp']['value'],.07)
        self.assertEqual(c['model']['text'],'IRLZ44N')

    def test_every_catalog_mosfet_has_threshold_and_current_coefficient(self):
        for name,item in CATALOG.items():
            if not item['type'].startswith('mosfet'):continue
            with self.subTest(model=name):
                c=automatic_model_fields(dict(type=item['type'],model={'text':name}))
                self.assertIn('vgs_threshold',c['electrical_parameters'])
                self.assertGreater(c['electrical_parameters']['mos_kp']['value'],0)

    def test_unknown_model_retains_code_and_labels_reference(self):
        c=automatic_model_fields(dict(type='mosfet_n',model={'text':'CUSTOM123'}))
        self.assertEqual(c['model']['text'],'CUSTOM123')
        self.assertTrue(c['electrical_parameters']['mos_kp']['datasheet']['reference_only'])
        with self.assertRaises(ValueError):automatic_model_fields(dict(type='mosfet_p',model={'text':'IRLZ44N'}))

    def test_opt_out_is_preserved_and_passive_values_are_not_invented(self):
        c=automatic_model_fields(dict(type='mosfet_p',parameter_policy={'mos_kp':'manual'}))
        self.assertNotIn('mos_kp',c['electrical_parameters'])
        c=dict(type='resistor',value=None)
        self.assertEqual(automatic_model_fields(c),c)

    def test_blank_semiconductor_parameters_run_in_actual_ngspice(self):
        for case in reference_cases():
            if case['id'] not in {'T07','T08','T09','T10'}:continue
            c=copy.deepcopy(case['circuit']);c['components'][-1].pop('metadata')
            deck=build_netlist(c)
            self.assertEqual(len(deck['components']),len(c['components']))
            with TemporaryDirectory() as tmp:
                result=run_simulation(deck,tmp);self.assertTrue(result['success'],result)
        c=copy.deepcopy(next(x['circuit'] for x in reference_cases() if x['id']=='T09'))
        c['components'][-1].update(type='mosfet_p',metadata={})
        c['components'][1]['value']='0'
        c['nets']=[dict(name='VIN',members=['V1.+','M1.S']),dict(name='GATE',members=['VG.+','M1.G']),dict(name='OUT',members=['RL.1','M1.D']),dict(name='GND',members=['V1.-','VG.-','RL.2'])]
        with TemporaryDirectory() as tmp:self.assertTrue(run_simulation(build_netlist(c),tmp)['success'])

class CircuitPreviewTests(unittest.TestCase):
    setUp=HumanReviewTests.setUp
    tearDown=HumanReviewTests.tearDown
    update=HumanReviewTests.update
    prepare=HumanReviewTests.prepare
    def fixture(self):
        current,payload=self.prepare()
        from backend.test_spice_stages import divider
        c=divider();c['components'].append(dict(id='M_AUTO',type='mosfet_p',value=None,pins=['G','D','S']))
        for i,pin in enumerate(['G','D','S']):c['nets'][i]['members'].append('M_AUTO.'+pin)
        payload.update(components=c['components'],nets=c['nets'])
        return payload
    def test_preview_reads_draft_without_saving_or_simulating(self):
        payload=self.fixture();before={p.name:p.read_bytes() for p in self.folder.iterdir() if p.is_file()}
        with patch.object(main,'run_simulation') as run:
            response=self.client.post('/projects/TEST/circuit-review/preview',json=payload);run.assert_not_called()
        self.assertEqual(response.status_code,200,response.text);data=response.json();self.assertTrue(data['ready'],data)
        self.assertIn('PMOS',data['netlist']);self.assertIn('not confirmed',data['netlist'])
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.folder.iterdir() if p.is_file()})
        self.assertFalse(self.client.get('/projects/TEST/circuit').json()['confirmed_by_user'])
        payload['components'][-1].update(model='IRF4905',electrical_parameters={'mos_kp':.08})
        data=self.client.post('/projects/TEST/circuit-review/preview',json=payload).json()
        self.assertIn('KP=0.08',data['netlist'])
        saved=self.client.post('/projects/TEST/circuit-review',json=payload).json()['circuit']
        self.assertEqual(saved['components'][-1]['metadata']['model']['text'],'IRF4905')
        self.assertEqual(saved['components'][-1]['metadata']['electrical_parameters']['mos_kp']['source'],'manual')
        self.assertEqual(self.client.post('/projects/TEST/circuit-review/preview',json=payload).status_code,409)

    def test_get_enriches_legacy_copy_and_missing_values_show_clear_preview(self):
        payload=self.fixture();self.client.post('/projects/TEST/circuit-review',json=payload)
        path=self.folder/'circuit.json';c=json.loads(path.read_text(encoding='utf8'));c['components'][-1]['metadata']={};path.write_text(json.dumps(c),encoding='utf8');before=path.read_bytes()
        data=self.client.get('/projects/TEST/circuit-review').json()
        self.assertEqual(data['revision'],revision(c));self.assertTrue(data['circuit']['components'][-1]['metadata']['electrical_parameters']['mos_kp'])
        self.assertEqual(before,path.read_bytes())
        payload['base_revision']=data['revision'];payload['components'][0]['value']=None
        response=self.client.post('/projects/TEST/circuit-review/preview',json=payload)
        self.assertEqual(response.status_code,200);self.assertFalse(response.json()['ready']);self.assertTrue(response.json()['message'])

    def test_type_change_discards_incompatible_model_and_parameters(self):
        payload=self.fixture();saved=self.client.post('/projects/TEST/circuit-review',json=payload).json()['circuit']
        p=copy.deepcopy(payload);p['components'][-1].update(original_id='M_AUTO',type='transistor_pnp',pins=['B','C','E'])
        for n in p['nets']:n['members']=[m.replace('M_AUTO.G','M_AUTO.B').replace('M_AUTO.D','M_AUTO.C').replace('M_AUTO.S','M_AUTO.E') for m in n['members']]
        changed=apply_review(saved,p)['components'][-1]['metadata']
        self.assertEqual(changed['model']['text'],'BC557B');self.assertNotIn('mos_kp',changed['electrical_parameters']);self.assertIn('hfe',changed['electrical_parameters'])

    def test_review_page_includes_pin_schemas_and_preview_controls(self):
        self.prepare();page=self.client.get('/projects/TEST/circuit-review-view').text
        self.assertIn('"types":',page);self.assertIn('"mosfet_p":',page)
        self.assertIn('Modelo e parâmetros',page);self.assertIn('netlistParameters',page)
        from backend.circuit_review_page import review_page
        self.assertNotIn('</script><script>alert',review_page('</script><script>alert'))
