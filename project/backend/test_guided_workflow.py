import copy
import json
import threading
import unittest
from unittest.mock import patch
from backend import main
from backend import test_circuit_api as api_fixture
from backend import test_spice_api as spice_fixture
from backend.spice_simulation import locate_ngspice

class GuidedTests(unittest.TestCase):
    setUp=api_fixture.CircuitAPItests.setUp
    tearDown=api_fixture.CircuitAPItests.tearDown
    update=api_fixture.CircuitAPItests.update
    prepare=spice_fixture.SpiceAPItests.prepare
    def status(self):
        r=self.client.get('/projects/TEST/guided/status');self.assertEqual(r.status_code,200,r.text);return r.json()
    def step(self,options=None):
        s=self.status();r=self.client.post('/projects/TEST/guided/step',json=dict(phase=s['phase'],cursor=s['cursor'],options=options))
        self.assertEqual(r.status_code,200,r.text);self.assertFalse(r.json()['busy']);return r.json()
    def seed(self):
        if not hasattr(self,'seed_terms'):self.seed_terms=json.loads((self.folder/'terminals.json').read_text())
        terms=copy.deepcopy(self.seed_terms)
        for c in terms['components']:
            # The test fixture's terminals lie exactly on its boundary; move only the box.
            b=c['bbox'];b['x1']-=2;b['y1']-=2;b['x2']+=2;b['y2']+=2
        (self.folder/'terminals.json').write_text(json.dumps(terms))
        self.project['terminals_info']=json.dumps(terms)
        (self.folder/'components.json').write_text(json.dumps(dict(components=[{'detection_id':c['component_id'],'type':c['type']} for c in terms['components']])))
        (self.folder/'detections.json').write_text(json.dumps({'detections':[]}))
    def test_resume_existing_vision_stops_at_review_and_does_not_reread(self):
        self.seed()
        with patch.object(main,'ler_referencias_e_valores') as ocr,patch.object(main,'reconhecer_componentes') as detector:
            s=self.status();self.assertEqual(s['phase'],'terminals');self.assertEqual(self.status()['cursor'],s['cursor'])
            self.assertEqual(self.step()['phase'],'models');ocr.assert_not_called();detector.assert_not_called()
    def test_fast_project_navigation_never_loads_vision_or_electrical_details(self):
        items=[dict(project_id='TEST',name='Circuito',status='criado',created_at='2026-10-05')]
        with patch.object(main,'listar_resumos_projetos',return_value=items),patch.object(main,'preparar_resposta_projeto') as full,patch.object(main,'_recuperar_indice_projetos_do_disco') as recovery:
            r=self.client.get('/projects/summary');self.assertEqual(r.status_code,200);self.assertEqual(r.json(),items)
            full.assert_not_called();recovery.assert_not_called()
    def test_status_reuses_unchanged_fingerprints_but_catches_file_changes(self):
        self.seed();self.status()
        with patch.object(main,'_json_projeto',wraps=main._json_projeto) as reads:
            self.status();self.assertTrue(all(call.args[1]=='guided_workflow.json' for call in reads.call_args_list))
        self.step();self.step();self.assertEqual(self.status()['phase'],'junctions')
        terms=json.loads((self.folder/'terminals.json').read_text());terms['components'][0]['terminals'][0]['position']['x']+=1
        (self.folder/'terminals.json').write_text(json.dumps(terms));self.assertEqual(self.status()['phase'],'terminals')
    def test_new_vision_steps_order_and_failure_resume(self):
        calls=[]
        def stage(name):
            def run(pid):
                calls.append(name)
                if name=='ocr':self.seed()
            return run
        with patch.object(main,'processar_projeto',side_effect=stage('process')),patch.object(main,'reconhecer_componentes',side_effect=stage('detect')),patch.object(main,'ler_referencias_e_valores',side_effect=stage('ocr')):
            self.step();self.step();self.assertEqual(self.step()['phase'],'terminals')
        self.assertEqual(calls,['process','detect','ocr'])
    def test_detection_failure_keeps_phase_and_does_not_claim_ready(self):
        self.step_process=patch.object(main,'processar_projeto',return_value={})
        with self.step_process:self.step()
        from fastapi import HTTPException
        with patch.object(main,'reconhecer_componentes',side_effect=HTTPException(422,'Modelo indisponível')):
            s=self.status();r=self.client.post('/projects/TEST/guided/step',json=s)
        self.assertEqual(r.status_code,422);self.assertEqual(self.status()['phase'],'detect');self.assertIn('Modelo',self.status()['error'])
    def test_ocr_with_missing_point_opens_terminal_editor_for_repair(self):
        with patch.object(main,'processar_projeto'):self.step()
        with patch.object(main,'reconhecer_componentes'):self.step()
        def ocr(pid):
            self.seed();terms=json.loads((self.folder/'terminals.json').read_text());terms['components'][0]['terminals']=[]
            (self.folder/'terminals.json').write_text(json.dumps(terms))
        with patch.object(main,'ler_referencias_e_valores',side_effect=ocr):self.assertEqual(self.step()['phase'],'terminals')
        self.assertEqual(self.client.post('/projects/TEST/guided/step',json=self.status()).status_code,422)
    def test_invalid_terminal_and_unresolved_junction_stop_progress(self):
        self.seed();s=self.status();terms=json.loads((self.folder/'terminals.json').read_text());terms['components'][0]['terminals'][0]['position']={'x':-500,'y':0}
        (self.folder/'terminals.json').write_text(json.dumps(terms))
        self.assertEqual(self.client.post('/projects/TEST/guided/step',json=s).status_code,422)
        self.seed();self.step();self.step();s=self.status();w=json.loads((self.folder/'wires.json').read_text());w['stage9']['review_count']=1
        (self.folder/'wires.json').write_text(json.dumps(w))
        self.assertEqual(self.client.post('/projects/TEST/guided/step',json=s).status_code,422)
        self.assertEqual(self.status()['phase'],'junctions')
    def test_model_metadata_does_not_force_terminal_review_but_moved_points_do(self):
        self.seed();self.step();terms=json.loads((self.folder/'terminals.json').read_text());terms['components'][0]['model']={'text':'TEST'}
        (self.folder/'terminals.json').write_text(json.dumps(terms))
        self.assertEqual(self.step()['phase'],'junctions')
        s=self.status();self.client.post('/projects/TEST/guided/back',json=dict(cursor=s['cursor'],phase='models'))
        terms['components'][0]['terminals'][0]['position']['x']+=1
        (self.folder/'terminals.json').write_text(json.dumps(terms))
        self.assertEqual(self.step()['phase'],'terminals');self.assertEqual(self.step()['phase'],'junctions')
    def test_circuit_must_be_confirmed_and_existing_edits_are_preserved(self):
        current=self.prepare();saved=(self.folder/'circuit.json').read_bytes();s=self.status();self.assertEqual(s['phase'],'circuit')
        self.assertEqual((self.folder/'circuit.json').read_bytes(),saved)
        c=json.loads(saved);c['confirmed_by_user']=False;c['status']='draft';(self.folder/'circuit.json').write_text(json.dumps(c))
        self.assertEqual(self.client.post('/projects/TEST/guided/step',json=s).status_code,409)
        self.assertEqual(self.status()['phase'],'circuit')
    def test_cursor_blocks_duplicate_and_out_of_order_requests(self):
        self.seed();s=self.status();self.step()
        self.assertEqual(self.client.post('/projects/TEST/guided/step',json=s).status_code,409)
        current=self.status();current['phase']='done'
        self.assertEqual(self.client.post('/projects/TEST/guided/step',json=current).status_code,409)
        self.assertEqual(self.status()['phase'],'models')
    def test_second_tab_cannot_duplicate_running_step(self):
        self.seed();s=self.status();self.step();self.step();self.step();self.assertEqual(self.status()['phase'],'circuit')
        self.prepare();s=self.status();self.step()
        entered=threading.Event();release=threading.Event()
        def slow(*args):entered.set();release.wait(10);return {'success':True}
        results=[]
        with patch.object(main,'simular_projeto',side_effect=slow):
            thread=threading.Thread(target=lambda:results.append(self.client.post('/projects/TEST/guided/step',json=self.status())));thread.start()
            self.assertTrue(entered.wait(10));self.assertTrue(self.status()['busy'])
            self.assertEqual(self.client.post('/projects/TEST/guided/step',json=self.status()).status_code,423)
            release.set();thread.join(10)
        self.assertEqual(results[0].status_code,200);self.assertEqual(results[0].json()['phase'],'production')
    def test_upstream_changes_return_to_relevant_review(self):
        self.seed();self.step();self.step();self.step();self.assertEqual(self.status()['phase'],'circuit')
        values=json.loads((self.folder/'components.json').read_text());values['components'][0]['value']={'text':'10k'}
        (self.folder/'components.json').write_text(json.dumps(values))
        self.assertEqual(self.status()['phase'],'models');self.assertIn('alterados',self.status()['error'])
    def test_full_confirmed_flow_real_simulation_exports_and_validation(self):
        if not locate_ngspice():self.skipTest('ngspice indisponível')
        self.prepare();self.assertEqual(self.step({'analysis':'op'})['phase'],'simulation')
        self.assertEqual(self.step()['phase'],'production')
        self.assertEqual(self.step({'pcb':True,'base_url':'http://127.0.0.1:8010'})['phase'],'kicad')
        self.assertEqual(self.step()['phase'],'export');self.assertEqual(self.step()['phase'],'validation')
        with patch.object(main,'BASE_DIR',self.root):self.assertEqual(self.step()['phase'],'done')
        self.assertTrue((self.folder/'project.zip').is_file());self.assertTrue((self.folder/'board.kicad_pcb').is_file())
        self.assertAlmostEqual(self.client.get('/projects/TEST/simulation').json()['last']['voltages_v']['VOUT'],2.5)
        self.assertEqual(self.status()['phase'],'done');self.assertEqual(self.client.get('/projects/TEST/export/download').status_code,200)
        (self.folder/'project.zip').write_bytes(b'changed')
        self.assertEqual(self.status()['phase'],'production')
        (self.folder/'simulation.json').unlink()
        self.assertEqual(self.status()['phase'],'simulation')
    def test_reference_failure_cannot_mark_workflow_complete(self):
        if not locate_ngspice():self.skipTest('ngspice indisponível')
        self.prepare();self.step();self.step();s=self.status()
        flow=json.loads((self.folder/'guided_workflow.json').read_text());flow['phase']='validation';(self.folder/'guided_workflow.json').write_text(json.dumps(flow))
        from backend import production_validation
        with patch.object(production_validation,'run_reference_suite_cached',return_value={'passed':9,'total':10}):
            r=self.client.post('/projects/TEST/guided/step',json=self.status())
        self.assertEqual(r.status_code,422);self.assertEqual(self.status()['phase'],'validation')
    def test_missing_parameter_does_not_launch_simulation(self):
        self.prepare();c=json.loads((self.folder/'circuit.json').read_text());c['components'].append(dict(id='Q1',type='transistor_npn',value=None,pins=['B','C','E'],metadata={'parameter_policy':{'hfe':'manual'}}))
        for i,pin in enumerate(['B','C','E']):c['nets'][i]['members'].append('Q1.'+pin)
        (self.folder/'circuit.json').write_text(json.dumps(c));s=self.status()
        with patch.object(main,'simular_projeto') as run:r=self.client.post('/projects/TEST/guided/step',json=s)
        self.assertEqual(r.status_code,422);run.assert_not_called();self.assertEqual(self.status()['phase'],'circuit')
    def test_pages_reuse_editors_without_external_step_links(self):
        self.prepare();page=self.client.get('/projects/TEST/workflow')
        self.assertIn('Salvar e continuar',page.text);self.assertIn('e.source!==',page.text)
        for phase in ('models','junctions','circuit','production','done','components'):
            page=self.client.get('/projects/TEST/guided/editor/'+phase);self.assertEqual(page.status_code,200,page.text);self.assertIn('GuidedStep',page.text)
        self.assertEqual(self.client.get('/projects/MISSING/guided/status').status_code,404)
        self.assertEqual(self.client.get('/projects/TEST/guided/editor/private').status_code,404)
        self.assertEqual(self.client.get('/guided-assets/backend/main.py').status_code,404)

    def test_upload_entry_creates_project_then_runs_vision_before_reviews(self):
        # Exercise the real multipart upload/storage route, with an isolated index.
        project={};calls=[]
        def finalize(internal_id,pid,path):
            project.update(self.project,project_id=pid,path=path,status='criado',
                           stored_filename='original.png',detections_info=None,ocr_info=None,terminals_info=None,wires_info=None)
        def ocr(pid):
            calls.append('ocr');self.seed()
            for name in ('components.json','terminals.json','detections.json'):
                (self.root/pid/name).write_bytes((self.folder/name).read_bytes())
        with patch.object(main,'inserir_projeto_temporario',return_value=417),patch.object(main,'finalizar_projeto',side_effect=finalize),patch.object(main,'buscar_projeto',side_effect=lambda pid:project if pid=='PRJ_0417' else self.project if pid=='TEST' else None):
            r=self.client.post('/projects',data={'name':'Circuito novo'},files={'file':('circuito.png',(self.folder/'original.png').read_bytes(),'image/png')})
            self.assertEqual(r.status_code,200,r.text);pid=r.json()['project_id'];self.assertEqual(pid,'PRJ_0417')
            self.assertTrue((self.root/pid/'original.png').is_file())
            with patch.object(main,'processar_projeto',side_effect=lambda pid:calls.append('process')),patch.object(main,'reconhecer_componentes',side_effect=lambda pid:calls.append('detect')),patch.object(main,'ler_referencias_e_valores',side_effect=ocr):
                for expected in ('process','detect','ocr'):
                    s=self.client.get(f'/projects/{pid}/guided/status').json();self.assertEqual(s['phase'],expected)
                    r=self.client.post(f'/projects/{pid}/guided/step',json=s);self.assertEqual(r.status_code,200,r.text)
            self.assertEqual(r.json()['phase'],'terminals');self.assertEqual(calls,['process','detect','ocr'])
            self.assertEqual(self.client.get(f'/projects/{pid}/workflow').status_code,200)

if __name__=='__main__':unittest.main()
