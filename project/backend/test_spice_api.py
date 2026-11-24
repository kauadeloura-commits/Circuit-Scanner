import copy
import json
from unittest.mock import patch
from backend import main
from backend import test_circuit_api as api_fixture
from backend.test_spice_stages import divider
from backend.spice_simulation import locate_ngspice
import unittest


class SpiceAPItests(unittest.TestCase):
    setUp=api_fixture.CircuitAPItests.setUp
    tearDown=api_fixture.CircuitAPItests.tearDown
    update=api_fixture.CircuitAPItests.update

    def prepare(self):
        self.assertEqual(self.client.post('/projects/TEST/connectivity').status_code,200)
        self.assertEqual(self.client.post('/projects/TEST/circuit').status_code,200)
        draft=self.client.get('/projects/TEST/circuit-review').json();c=divider()
        payload=dict(base_revision=draft['revision'],components=c['components'],nets=c['nets'])
        response=self.client.post('/projects/TEST/circuit-review',json=payload)
        self.assertEqual(response.status_code,200,response.text)
        response=self.client.post('/projects/TEST/circuit-review/confirm',json=dict(base_revision=response.json()['revision'],acknowledge_warnings=True))
        self.assertEqual(response.status_code,200,response.text)
        return response.json()

    def test_confirmed_circuit_generates_netlist_simulates_and_downloads(self):
        current=self.prepare()
        if not locate_ngspice():self.skipTest('ngspice indisponível')
        generated=self.client.post('/projects/TEST/spice',json=dict(base_revision=current['revision']))
        self.assertEqual(generated.status_code,200,generated.text)
        self.assertIn('DC 5',self.client.get('/projects/TEST/spice/download').text)
        self.assertTrue(self.client.get('/projects/TEST/stage13-status').json()['criterion_met'])
        response=self.client.post('/projects/TEST/simulation',json={'analysis':'op'})
        self.assertEqual(response.status_code,200,response.text);self.assertTrue(response.json()['success'],response.text)
        self.assertAlmostEqual(response.json()['last']['voltages_v']['VOUT'],2.5)
        self.assertTrue(self.client.get('/projects/TEST/stage14-status').json()['criterion_met'])
        self.assertIn('V(VOUT)',self.client.get('/projects/TEST/simulation/download/csv').text)
        self.assertTrue(self.client.get('/projects/TEST/simulation/download/json').json()['success'])
        self.assertEqual(self.client.get('/projects/TEST/simulation/download/log').status_code,200)
        self.assertEqual(self.client.get('/projects/TEST/simulation/download/raw').status_code,200)
        self.assertEqual(self.client.get('/projects/TEST/simulation/download/private').status_code,404)

    def test_unconfirmed_and_unknown_projects_cannot_simulate(self):
        self.client.post('/projects/TEST/connectivity');self.client.post('/projects/TEST/circuit')
        for route in ('spice','simulation'):
            self.assertEqual(self.client.post('/projects/TEST/'+route,json={}).status_code,409)
            self.assertEqual(self.client.post('/projects/MISSING/'+route,json={}).status_code,404)
        self.assertFalse(self.client.get('/projects/TEST/stage13-status').json()['criterion_met'])

    def test_saved_edit_and_network_rebuild_remove_old_stage13_14_artifacts(self):
        current=self.prepare();self.client.post('/projects/TEST/simulation',json={})
        payload=dict(base_revision=current['revision'],components=[dict(c,original_id=c['id']) for c in current['circuit']['components']],nets=current['circuit']['nets'])
        payload['components'][0]['value']='10k'
        self.assertEqual(self.client.post('/projects/TEST/circuit-review',json=payload).status_code,200)
        for name in main.SPICE_ARTIFACTS:self.assertFalse((self.folder/name).exists(),name)
        self.assertEqual(self.client.get('/projects/TEST/simulation').status_code,409)
        self.prepare();self.client.post('/projects/TEST/spice',json={})
        self.client.post('/projects/TEST/connectivity')
        for name in main.SPICE_ARTIFACTS:self.assertFalse((self.folder/name).exists(),name)

    def test_new_analysis_invalidates_previous_results_and_wrong_revision_is_blocked(self):
        current=self.prepare();self.client.post('/projects/TEST/simulation',json={})
        self.assertEqual(self.client.post('/projects/TEST/spice',json={'base_revision':'old'}).status_code,409)
        response=self.client.post('/projects/TEST/spice',json=dict(analysis='tran',step='10u',stop='10m'))
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.client.get('/projects/TEST/simulation').status_code,409)
        self.assertFalse(self.client.get('/projects/TEST/stage14-status').json()['criterion_met'])

    def test_altered_netlist_cannot_be_downloaded_or_claimed_as_current(self):
        self.prepare();self.client.post('/projects/TEST/spice',json={})
        with (self.folder/'simulation.cir').open('a') as f:f.write('* changed\n')
        self.assertEqual(self.client.get('/projects/TEST/spice/download').status_code,409)
        self.assertFalse(self.client.get('/projects/TEST/stage13-status').json()['criterion_met'])

    def test_failure_is_saved_and_cannot_expose_previous_csv(self):
        from subprocess import TimeoutExpired
        self.prepare();self.client.post('/projects/TEST/simulation',json={})
        with patch('backend.spice_simulation.subprocess.run',side_effect=TimeoutExpired('ngspice',45)):
            response=self.client.post('/projects/TEST/simulation',json={})
        self.assertEqual(response.status_code,200,response.text);self.assertFalse(response.json()['success'])
        self.assertEqual(response.json()['error_code'],'timeout')
        self.assertEqual(self.client.get('/projects/TEST/simulation/download/csv').status_code,409)
        self.assertFalse(self.client.get('/projects/TEST/stage14-status').json()['criterion_met'])
        self.assertEqual(self.client.get('/projects/TEST/simulation/download/json').status_code,200)

    def test_circuit_changed_during_execution_discards_results(self):
        self.prepare();real=main.run_simulation
        def change_after_running(manifest,folder):
            result=real(manifest,folder)
            circuit=json.loads((self.folder/'circuit.json').read_text(encoding='utf8'))
            circuit['name']='Mudou durante a execução'
            (self.folder/'circuit.json').write_text(json.dumps(circuit),encoding='utf8')
            return result
        with patch.object(main,'run_simulation',side_effect=change_after_running):
            response=self.client.post('/projects/TEST/simulation',json={})
        self.assertEqual(response.status_code,409,response.text)
        self.assertFalse((self.folder/'simulation.json').exists())

    def test_manual_missing_parameter_fails_before_running_and_page_is_linked(self):
        self.prepare();c=json.loads((self.folder/'circuit.json').read_text(encoding='utf8'))
        c['components'].append(dict(id='U1',type='opamp',value=None,pins=['+','-','OUT'],metadata={'parameter_policy':{'open_loop_gain':'manual'}}))
        for i,pin in enumerate(['+','-','OUT']):c['nets'][i]['members'].append('U1.'+pin)
        (self.folder/'circuit.json').write_text(json.dumps(c),encoding='utf8')
        with patch.object(main,'run_simulation') as run:
            response=self.client.post('/projects/TEST/simulation',json={})
        self.assertEqual(response.status_code,422,response.text);run.assert_not_called()
        self.assertIn('U1',response.json()['detail'])
        for route in ('simulation-view','workflow','circuit-view','circuit-review-view'):
            response=self.client.get('/projects/TEST/'+route)
            self.assertEqual(response.status_code,200)
            self.assertIn('Simulação' if route=='workflow' else 'etapas 13 e 14',response.text)
