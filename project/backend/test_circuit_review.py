import copy
import json
import unittest
from backend import test_circuit_api as api_fixture
from backend.circuit_review import apply_review, confirm_review, revision
from backend.circuit_document import validar_circuito


class HumanReviewTests(unittest.TestCase):
    setUp=api_fixture.CircuitAPItests.setUp
    tearDown=api_fixture.CircuitAPItests.tearDown
    update=api_fixture.CircuitAPItests.update

    def prepare(self):
        self.assertEqual(self.client.post('/projects/TEST/connectivity').status_code,200)
        self.assertEqual(self.client.post('/projects/TEST/circuit').status_code,200)
        response=self.client.get('/projects/TEST/circuit-review')
        self.assertEqual(response.status_code,200)
        data=response.json();c=data['circuit']
        return c,dict(base_revision=data['revision'],components=[dict(x,original_id=x['id']) for x in c['components']],nets=copy.deepcopy(c['nets']))

    def save(self,payload):return self.client.post('/projects/TEST/circuit-review',json=payload)

    def test_correct_component_value_and_connection_then_confirm_and_download(self):
        original,payload=self.prepare()
        c=payload['components'][0];old=c['id'];c.update(id='C99',type='capacitor',value='100u',unit='F')
        for n in payload['nets']:n['members']=[member.replace(old+'.','C99.') for member in n['members']]
        response=self.save(payload);self.assertEqual(response.status_code,200,response.text)
        saved=response.json()['circuit'];self.assertEqual(saved['components'][0]['id'],'C99')
        self.assertEqual(saved['components'][0]['value'],'100u');self.assertFalse(saved['confirmed_by_user'])
        self.assertTrue(validar_circuito(saved)['valid'])
        for c in payload['components']:
            c['original_id']=c['id']
            if c['type'] in {'resistor','capacitor','source_dc','battery'}:c['value']='10'
        payload['base_revision']=response.json()['revision']
        # Move a pin to another existing network, without leaving duplicate membership.
        moved=payload['nets'][0]['members'].pop();payload['nets'][1]['members'].append(moved)
        payload['nets']=[n for n in payload['nets'] if n['members']]
        response=self.save(payload);self.assertEqual(response.status_code,200,response.text)
        confirmation=self.client.post('/projects/TEST/circuit-review/confirm',json=dict(base_revision=response.json()['revision'],acknowledge_warnings=True))
        self.assertEqual(confirmation.status_code,200,confirmation.text)
        downloaded=self.client.get('/projects/TEST/circuit/download').json()
        self.assertTrue(downloaded['confirmed_by_user']);self.assertEqual(downloaded['status'],'confirmed')
        self.assertTrue(self.client.get('/projects/TEST/stage12-status').json()['criterion_met'])

    def test_editing_a_confirmed_circuit_revokes_confirmation(self):
        current,payload=self.prepare()
        for c in payload['components']:c['value']='100'
        draft=apply_review(current,payload);confirmed=confirm_review(draft,True)
        edited=apply_review(confirmed,payload)
        self.assertFalse(edited['confirmed_by_user']);self.assertEqual(edited['status'],'draft')
        self.assertIsNone(edited['review']['confirmed_at'])

    def test_conflicting_browser_version_does_not_overwrite(self):
        current,payload=self.prepare();response=self.save(payload);self.assertEqual(response.status_code,200)
        before=(self.folder/'circuit.json').read_bytes()
        payload['components'][0]['value']='99'
        self.assertEqual(self.save(payload).status_code,409)
        self.assertEqual(before,(self.folder/'circuit.json').read_bytes())

    def test_invalid_members_or_duplicate_ids_do_not_modify_saved_json(self):
        current,payload=self.prepare();before=(self.folder/'circuit.json').read_bytes()
        for mutate in (lambda p:p['nets'][0]['members'].append('UNKNOWN.9'),
                       lambda p:p['components'][1].update(id=p['components'][0]['id']),
                       lambda p:p['components'][0].update(pins=['1']),
                       lambda p:p['components'][0].update(type='invented'),
                       lambda p:p['nets'][0].update(name='<script>'),
                       lambda p:p['components'][0].update(pins=[{}])):
            candidate=copy.deepcopy(payload);mutate(candidate)
            self.assertEqual(self.save(candidate).status_code,422)
            self.assertEqual(before,(self.folder/'circuit.json').read_bytes())

    def test_missing_values_and_pins_block_confirmation(self):
        current,payload=self.prepare()
        for c in payload['components']:c['value']=None
        saved=self.save(payload);self.assertEqual(saved.status_code,200)
        confirmation=self.client.post('/projects/TEST/circuit-review/confirm',json=dict(base_revision=saved.json()['revision'],acknowledge_warnings=True))
        self.assertEqual(confirmation.status_code,422)
        self.assertFalse(self.client.get('/projects/TEST/circuit').json()['confirmed_by_user'])

    def test_warning_acknowledgement_is_required(self):
        current,payload=self.prepare()
        for c in payload['components']:c['value']='100'
        draft=apply_review(current,payload)
        draft['warnings']=[dict(message='Confira o circuito')]
        with self.assertRaises(ValueError):confirm_review(draft,False)
        self.assertTrue(confirm_review(draft,True)['confirmed_by_user'])

    def test_add_remove_components_and_change_pin_schema(self):
        current,payload=self.prepare();removed=payload['components'].pop(0)['id']
        for n in payload['nets']:n['members']=[m for m in n['members'] if not m.startswith(removed+'.')]
        payload['components'].append(dict(id='BAT99',original_id=None,type='battery',pins=['+','-'],value='9',unit='V'))
        payload['nets'].extend([dict(name='NEW_P',members=['BAT99.+']),dict(name='NEW_N',members=['BAT99.-'])])
        payload['nets']=[n for n in payload['nets'] if n['members']]
        saved=self.save(payload);self.assertEqual(saved.status_code,200,saved.text)
        self.assertTrue(validar_circuito(saved.json()['circuit'])['valid'])

    def test_geometry_is_unchanged_and_review_history_is_preserved(self):
        current,payload=self.prepare();before={name:(self.folder/name).read_bytes() for name in ('terminals.json','connectivity.json','connectivity.png')}
        response=self.save(payload);self.assertEqual(response.status_code,200)
        history=list((self.folder/'circuit_review_history').glob('*.json'));self.assertEqual(len(history),1)
        self.assertEqual(json.loads(history[0].read_text(encoding='utf8')),current)
        for name,data in before.items():self.assertEqual(data,(self.folder/name).read_bytes())
        self.assertEqual(self.client.post('/projects/TEST/circuit').status_code,409)

    def test_upstream_changes_invalidate_and_archive_review(self):
        current,payload=self.prepare();self.assertEqual(self.save(payload).status_code,200)
        response=self.client.post('/projects/TEST/connectivity');self.assertEqual(response.status_code,200)
        self.assertEqual(self.client.get('/projects/TEST/circuit-review').status_code,409)
        self.assertGreaterEqual(len(list((self.folder/'circuit_review_history').glob('*.json'))),2)

    def test_review_page_escapes_project_id_and_has_edit_controls(self):
        from backend.circuit_review_page import review_page
        self.assertNotIn('</script><script>alert',review_page('</script><script>alert'))
        page=self.client.get('/projects/TEST/circuit-review-view')
        self.assertEqual(page.status_code,200);self.assertIn('Confirmar circuito',page.text)
        self.assertIn('Salvar revisão',page.text)
        self.assertEqual(self.client.get('/projects/MISSING/circuit-review-view').status_code,404)

    def test_no_photo_is_required_for_manual_review(self):
        current,payload=self.prepare();(self.folder/'original.png').unlink()
        self.assertEqual(self.save(payload).status_code,200)

if __name__=='__main__':unittest.main()

