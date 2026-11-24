import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import cv2
from fastapi.testclient import TestClient
from backend import main
from backend.circuit_document import validar_circuito
from backend.test_circuit_connectivity import series_case


class CircuitAPItests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.root=Path(self.temp.name);self.folder=self.root/'TEST';self.folder.mkdir()
        image,wires,terms,_=series_case();cv2.imwrite(str(self.folder/'original.png'),image)
        (self.folder/'wires.json').write_text(json.dumps(wires),encoding='utf8')
        (self.folder/'terminals.json').write_text(json.dumps(terms),encoding='utf8')
        self.project=dict(project_id='TEST',name='Circuito de teste',created_at='2026-09-30',status='juncoes_resolvidas',
                          path=str(self.folder),original_filename='original.png',stored_filename='original.png',
                          wires_info=json.dumps(wires),terminals_info=json.dumps(terms))
        self.patches=[patch.object(main,'PROJECTS_DIR',self.root),patch.object(main,'buscar_projeto',side_effect=lambda id:self.project if id=='TEST' else None),
                      patch.object(main,'atualizar_fios',side_effect=self.update)]
        for p in self.patches:p.start()
        self.client=TestClient(main.app)
    def update(self,**kwargs):self.project.update({k:v for k,v in kwargs.items() if k!='project_id'})
    def tearDown(self):
        self.client.close()
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    def build(self):
        response=self.client.post('/projects/TEST/connectivity');self.assertEqual(response.status_code,200,response.text)
        return response.json()

    def test_networks_json_status_and_download_work_end_to_end(self):
        network=self.build();self.assertEqual(network['net_count'],3)
        self.assertEqual(self.client.get('/projects/TEST/stage10-status').json()['generated'],True)
        self.assertFalse(self.client.get('/projects/TEST/stage11-status').json()['generated'])
        response=self.client.post('/projects/TEST/circuit');self.assertEqual(response.status_code,200,response.text)
        circuit=self.client.get('/projects/TEST/circuit').json();self.assertTrue(validar_circuito(circuit)['valid'])
        download=self.client.get('/projects/TEST/circuit/download')
        self.assertEqual(download.status_code,200)
        self.assertIn('circuit.json',download.headers['content-disposition'])
        self.assertEqual(download.json(),circuit)

    def test_circuit_generation_does_not_need_the_photo_after_connectivity(self):
        self.build();(self.folder/'original.png').unlink()
        response=self.client.post('/projects/TEST/circuit');self.assertEqual(response.status_code,200,response.text)
        self.assertTrue(validar_circuito(self.client.get('/projects/TEST/circuit').json())['valid'])

    def test_changed_terminal_invalidates_links_and_download(self):
        self.build();self.client.post('/projects/TEST/circuit')
        terms=json.loads((self.folder/'terminals.json').read_text());terms['components'][0]['terminals'][0]['position']['x']+=1
        (self.folder/'terminals.json').write_text(json.dumps(terms),encoding='utf8');self.project['terminals_info']=json.dumps(terms)
        response=self.client.get('/projects/TEST').json()
        self.assertIsNone(response['connectivity_url']);self.assertIsNone(response['circuit_json_url'])
        self.assertEqual(self.client.get('/projects/TEST/circuit/download').status_code,409)
        self.assertEqual(self.client.post('/projects/TEST/circuit').status_code,409)

    def test_changed_junction_invalidates_json_even_with_same_pin_positions(self):
        self.build();self.client.post('/projects/TEST/circuit')
        wires=json.loads((self.folder/'wires.json').read_text());wires['stage9']['nodes'].append({'node_id':'NEW','electrical_status':'review_required'})
        (self.folder/'wires.json').write_text(json.dumps(wires),encoding='utf8')
        self.assertEqual(self.client.get('/projects/TEST/circuit').status_code,409)

    def test_rebuilding_networks_removes_the_previous_circuit(self):
        self.build();self.client.post('/projects/TEST/circuit');self.assertTrue((self.folder/'circuit.json').exists())
        self.build();self.assertFalse((self.folder/'circuit.json').exists())

    def test_upstream_invalidation_clears_new_stage_files(self):
        self.build();self.client.post('/projects/TEST/circuit')
        main._remover_artefatos_desatualizados(self.folder,('terminals.json',))
        for filename in ('connectivity.json','connectivity.png','circuit.json'):self.assertFalse((self.folder/filename).exists())

    def test_missing_prerequisites_and_unknown_projects_are_clear_errors(self):
        self.assertEqual(self.client.post('/projects/TEST/circuit').status_code,409)
        (self.folder/'wires.json').unlink()
        self.assertEqual(self.client.post('/projects/TEST/connectivity').status_code,409)
        self.assertEqual(self.client.get('/projects/MISSING/stage10-status').status_code,404)

    def test_page_and_flow_have_the_new_controls(self):
        page=self.client.get('/projects/TEST/circuit-view');self.assertEqual(page.status_code,200)
        self.assertIn('Gerar circuit.json',page.text)
        self.assertIn('Salvar e continuar',self.client.get('/projects/TEST/workflow').text)
        self.assertIn('Redes e circuit.json',self.client.get('/projects/TEST/workflow-advanced').text)

    def test_automatic_vision_pipeline_publishes_both_electrical_stages(self):
        from backend.test_terminal_identity import fixture
        image,bbox,_=fixture('opamp')
        cv2.imwrite(str(self.folder/'original.png'),image)
        component=dict(component_id='U1',type='opamp',bbox=bbox,visual_confidence=.99)
        with patch.object(main,'atualizar_terminais'),patch.object(main,'atualizar_terminais_pos_fios'):
            main._gerar_terminais_automaticamente('TEST',self.project,
                {'components':[component]},trigger='automatic_detection')
        wires=json.loads((self.folder/'wires.json').read_text(encoding='utf8'))
        self.assertNotIn('stage10_11_warning',wires)
        self.assertIn('stage10',wires);self.assertIn('stage11',wires)
        self.assertEqual(wires['electrical_pipeline_order'],['components','wires','terminals','junctions','connectivity','circuit'])
        for filename in ('connectivity.json','connectivity.png','circuit.json'):
            self.assertTrue((self.folder/filename).is_file(),filename)
        circuit=json.loads((self.folder/'circuit.json').read_text(encoding='utf8'))
        self.assertTrue(validar_circuito(circuit)['valid'])


if __name__=='__main__':unittest.main()
