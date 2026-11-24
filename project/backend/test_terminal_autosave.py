import copy,json,re,unittest
from unittest.mock import patch
from backend import main
from backend import test_guided_workflow as guided_fixture
from backend.terminal_review_draft import restore,load,FILENAME

class TerminalAutoSaveTests(unittest.TestCase):
    setUp=guided_fixture.GuidedTests.setUp
    tearDown=guided_fixture.GuidedTests.tearDown
    seed=guided_fixture.GuidedTests.seed
    update=guided_fixture.api_fixture.CircuitAPItests.update
    def prepare(self):
        self.seed();self.components=json.loads((self.folder/'terminals.json').read_text())['components'];self.c=self.components[0]
        self.points={t['pin']:t['position'] for t in self.c['terminals']}
    def save(self,points=None,items=None):
        return self.client.put('/projects/TEST/terminals/draft',json={'components':items or [dict(component_id=self.c['component_id'],points=self.points if points is None else points)]})
    def restored(self,components=None):
        components=components or self.components;editable=[dict(component_id=c['component_id'],points={},pin_schema=[]) for c in components]
        restore(self.folder,self.folder/'original.png',editable,components);return editable
    def test_partial_edits_and_clearing_saved_without_changing_electrical_results(self):
        self.prepare();before=(self.folder/'terminals.json').read_bytes();pin=next(iter(self.points));points={pin:self.points[pin]}
        with patch.object(main,'_continuar_automaticamente_para_linhas') as wires,patch.object(main,'save_generic_terminal_training_sample') as training:
            response=self.save(points);self.assertEqual(response.status_code,200,response.text);wires.assert_not_called();training.assert_not_called()
        self.assertEqual((self.folder/'terminals.json').read_bytes(),before);self.assertEqual(self.restored()[0]['points'],points)
        self.assertTrue(self.restored()[0]['draft_restored'])
        self.assertEqual(self.save({}).status_code,200);self.assertEqual(self.restored()[0]['points'],{})
    def test_back_navigation_accepts_fresh_cursor_after_save(self):
        self.prepare();s=self.client.get('/projects/TEST/guided/status').json();self.assertEqual(s['phase'],'terminals')
        self.assertEqual(self.save({}).status_code,200);s=self.client.get('/projects/TEST/guided/status').json()
        r=self.client.post('/projects/TEST/guided/back',json={'cursor':s['cursor'],'phase':'components'})
        self.assertEqual(r.status_code,200,r.text);self.assertEqual(r.json()['phase'],'components')
        page=self.client.get('/projects/TEST/terminal-trainer');self.assertEqual(page.status_code,200);self.assertIn('"draft_restored": true',page.text)
    def test_changed_image_or_box_does_not_restore_old_marks(self):
        self.prepare();self.assertEqual(self.save({}).status_code,200)
        changed=copy.deepcopy(self.components);changed[0]['bbox']['x1']-=1
        self.assertNotIn('draft_restored',self.restored(changed)[0])
        with (self.folder/'original.png').open('ab') as f:f.write(b'changed')
        self.assertEqual(load(self.folder,self.folder/'original.png')['components'],[])
    def test_invalid_batch_is_atomic_and_points_must_be_in_box(self):
        self.prepare();self.save({});before=(self.folder/FILENAME).read_bytes();pin=next(iter(self.points))
        for edits in ([None],[dict(component_id='missing',points={})],[dict(component_id=self.c['component_id'],points={pin:{'x':-10,'y':2}})],
                      [dict(component_id=self.c['component_id'],points={}),dict(component_id='missing',points={})],
                      [dict(component_id=self.c['component_id'],points={'invalid':{'x':20,'y':20}})]):
            r=self.save(items=edits);self.assertEqual(r.status_code,400,r.text);self.assertEqual((self.folder/FILENAME).read_bytes(),before)
    def test_corrupt_draft_does_not_break_editor(self):
        self.prepare();self.save({});path=self.folder/FILENAME;data=json.loads(path.read_text());data['components']=[None,{'points':{'A':{}},'pin_schema':[None]}];path.write_text(json.dumps(data))
        self.assertEqual(self.client.get('/projects/TEST/terminal-trainer').status_code,200)
    def test_complete_save_removes_draft_and_retains_pin_positions(self):
        self.prepare();self.save({})
        with patch.object(main,'save_generic_terminal_training_sample',return_value={'sample_id':'test'}),patch.object(main,'atualizar_terminais',side_effect=self.update),patch.object(main,'_continuar_automaticamente_para_linhas',return_value={}),patch.object(main,'preparar_resposta_projeto',return_value={}),patch.object(main,'salvar_imagem_terminais'):
            r=self.client.put('/projects/TEST/terminals/corrections',json={'components':[dict(component_id=self.c['component_id'],points=self.points)]})
            self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(load(self.folder,self.folder/'original.png')['components'],[])
        saved=json.loads((self.folder/'terminals.json').read_text())['components'][0]
        self.assertEqual({t['pin']:t['position'] for t in saved['terminals']},self.points)
