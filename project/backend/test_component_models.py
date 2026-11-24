import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import cv2
import numpy as np
from unittest.mock import patch
from backend.component_models import (CATALOG,FIELDS,PARAMETERS,resolve_parameters,model_edit,
                                      model_classes,parameter_number,preserve_manual_models)
from backend.ocr import criar_componentes_logicos,aplicar_correcoes_manuais,_limpar_texto
from backend import ocr
from backend import main,test_circuit_api as api_fixture
from backend.circuit_document import gerar_circuito
from backend.circuit_review import apply_review

def detection(cid,kind,x):
    return dict(detection_id=cid,**{'class':kind},confidence=.95,
                bbox=dict(x1=x,y1=50,x2=x+60,y2=110,width=60,height=60,center_x=x+30,center_y=80))
def token(text,x,y=30,confidence=.95):
    return dict(text=text,confidence=confidence,engine='easyocr',token_id=text+str(x),
                bbox=dict(x1=x,y1=y,x2=x+40,y2=y+12,width=40,height=12,center_x=x+20,center_y=y+6))

class ModelParametersTests(unittest.TestCase):
    def test_suggested_models_auto_datasheets_and_ideal_components(self):
        missing=dict(type='transistor_npn',model={'text':'','source':'missing'})
        result=model_edit(missing,{})
        self.assertEqual(result['model']['text'],'BC547B');self.assertEqual(result['model']['source'],'suggested')
        self.assertEqual(result['electrical_parameters']['hfe']['value'],325)
        disabled=model_edit(missing,{'parameter_policy':{'hfe':'manual'}})
        self.assertNotIn('hfe',disabled['electrical_parameters'])
        for kind in ('resistor','capacitor','capacitor_polarized','battery','source_dc'):
            result=model_edit(dict(type=kind,model={'text':'OLD'},electrical_parameters={'tolerance':{'value':30},'series_resistance':{'value':1}}),{})
            self.assertEqual(result['model']['text'],'');self.assertEqual(result['electrical_parameters'],{})
    def test_simplified_fields_and_derived_capacitance_estimates(self):
        from backend.component_models import UI_FIELDS
        self.assertEqual(UI_FIELDS['inductor'],['saturation_current','initial_current'])
        self.assertEqual(len(UI_FIELDS['mosfet_n']),5)
        result=resolve_parameters('mosfet_n','BS170',policies={k:'datasheet' for k in UI_FIELDS['mosfet_n']})
        self.assertAlmostEqual(result['parameters']['gate_source_capacitance']['value'],17e-12)
        self.assertEqual(result['parameters']['gate_source_capacitance']['source'],'datasheet_derived')
        self.assertEqual(result['parameters']['gate_drain_capacitance']['source'],'datasheet_typical')
        self.assertAlmostEqual(result['parameters']['mos_kp']['value'],1/(1.2*7.9))
        self.assertNotIn('channel_lambda',result['parameters'])
        forged=model_edit(dict(type='mosfet_n',model={'text':'BS170'}),dict(electrical_parameters={'mos_kp':{'value':999,'source':'datasheet_derived'}},parameter_policy={'mos_kp':'datasheet'}))
        self.assertAlmostEqual(forged['electrical_parameters']['mos_kp']['value'],1/(1.2*7.9))
    def test_datasheet_requires_opt_in_and_manual_values_win(self):
        result=resolve_parameters('transistor_npn','BC547B')
        self.assertEqual(result['parameters'],{})
        result=resolve_parameters('transistor_npn','BC547B',{'hfe':'175'},{'hfe':'datasheet'})
        self.assertEqual(result['parameters']['hfe']['value'],175)
        self.assertEqual(result['parameters']['hfe']['source'],'manual')

    def test_midpoint_typical_and_source_conditions_are_retained(self):
        result=resolve_parameters('transistor_npn','BC547B',policies={'hfe':'datasheet','vbe_on':'datasheet'})
        self.assertEqual(result['parameters']['hfe']['value'],325)
        self.assertEqual(result['parameters']['hfe']['source'],'datasheet_midpoint')
        self.assertEqual(result['parameters']['vbe_on']['value'],.66)
        self.assertIn('2 mA',result['parameters']['hfe']['datasheet']['condition'])
        self.assertTrue(result['parameters']['hfe']['datasheet']['url'].startswith('https://'))
        result=resolve_parameters('mosfet_n','2N7000',policies={'vgs_threshold':'datasheet'})
        self.assertEqual(result['parameters']['vgs_threshold']['value'],2.1)
        self.assertEqual(result['parameters']['vgs_threshold']['source'],'datasheet_typical')

    def test_gain_group_and_p_channel_sign_are_not_lost(self):
        for model,expected in [('BC547A',165),('BC547B',325),('BC547C',610)]:
            self.assertEqual(resolve_parameters('transistor_npn',model,policies={'hfe':'datasheet'})['parameters']['hfe']['value'],expected)
        result=resolve_parameters('mosfet_p','IRF9540N',policies={'vgs_threshold':'datasheet'})
        self.assertEqual(result['parameters']['vgs_threshold']['value'],-3)
        with self.assertRaises(ValueError):resolve_parameters('mosfet_p','IRF9540N',{'vgs_threshold':'3'})
        with self.assertRaises(ValueError):resolve_parameters('transistor_pnp','BC547B')

    def test_unknown_models_and_one_sided_limits_never_invent_estimates(self):
        unknown=resolve_parameters('mosfet_n','CUSTOM-FET-123',policies={'gm':'datasheet'})
        self.assertIsNone(unknown['datasheet']);self.assertEqual(unknown['parameters'],{})
        result=resolve_parameters('mosfet_n','IRF540N',policies={'gm':'datasheet','rds_on':'datasheet'})
        self.assertEqual(result['parameters'],{});self.assertEqual(len(result['pending']),2)
        self.assertEqual(result['pending'][0]['reason'],'only_limit')
        self.assertEqual(result['pending'][1]['bounds']['max'],.044)

    def test_engineering_units_and_invalid_parameters(self):
        for value,key,expected in [('320mS','gm',.32),('2,1V','vgs_threshold',2.1),('100k','open_loop_gain',100000),('700kHz','gbw',700000),('10µH','inductance',10e-6)]:
            self.assertAlmostEqual(parameter_number(value,key),expected)
        for value in ('nan','1e309','1; shell','-100','True'):
            with self.assertRaises(ValueError):parameter_number(value,'hfe')
        with self.assertRaises(ValueError):resolve_parameters('inductor','L123',{'hfe':'100'})
        with self.assertRaises(ValueError):resolve_parameters('opamp','LM358',policies={'gbw':'guess'})

    def test_new_model_recomputes_estimates_but_preserves_manual_parameters(self):
        component=dict(type='transistor_npn',model=dict(text='BC547A',source='manual'),
                       electrical_parameters=resolve_parameters('transistor_npn','BC547A',policies={'hfe':'datasheet'})['parameters'],
                       parameter_policy={'hfe':'datasheet'})
        updated=model_edit(component,{'model':'BC547C'})
        self.assertEqual(updated['electrical_parameters']['hfe']['value'],610)
        updated=model_edit(component,{'model':'BC547C','electrical_parameters':{'hfe':{'value':123,'source':'manual'}}})
        self.assertEqual(updated['electrical_parameters']['hfe']['value'],123)

    def test_catalog_bounds_have_consistent_units_and_types(self):
        self.assertGreaterEqual(len(CATALOG),100)
        for name,item in CATALOG.items():
            self.assertEqual(name,item['model'])
            self.assertTrue(item['url'].startswith('https://'))
            for key,spec in item['parameters'].items():
                self.assertIn(key,FIELDS[item['type']]);self.assertEqual(spec['unit'],PARAMETERS[key]['unit'])
                if spec['min'] is not None and spec['max'] is not None:
                    self.assertLessEqual(spec['min'],spec['max'])
                    if spec['typ'] is not None:self.assertLessEqual(spec['min'],spec['typ']);self.assertLessEqual(spec['typ'],spec['max'])

    def test_ocr_assigns_models_once_to_nearest_compatible_components(self):
        detections=[detection('A','transistor_npn',20),detection('B','transistor_npn',180),detection('C','opamp',340),detection('D','inductor',500)]
        tokens=[token('BC547B',25),token('BC547C',185),token('LM358',345),token('SRR6028-100Y',505),token('Q1',22,115)]
        result=criar_componentes_logicos(tokens,detections)
        self.assertEqual([c['model']['text'] for c in result['components']],['BC547B','BC547C','LM358','SRR6028-100Y'])
        self.assertEqual(result['model_ocr_count'],4)
        self.assertEqual(result['components'][0]['reference']['text'],'Q1')
        self.assertEqual(result['components'][0]['value']['source'],'not_applicable')
        self.assertTrue(all(not c['electrical_parameters'] for c in result['components']))

    def test_cleaned_parameter_ocr_and_passive_value_remain_separate(self):
        dets=[detection('A','transistor_npn',20),detection('B','inductor',200)]
        tokens=[token(_limpar_texto('hFE = 120'),22),token('SRR6028-100Y',205),token('10uH',205,115)]
        result=criar_componentes_logicos(tokens,dets)
        self.assertEqual(result['components'][0]['electrical_parameters']['hfe']['value'],120)
        self.assertEqual(result['components'][0]['electrical_parameters']['hfe']['source'],'ocr')
        self.assertEqual(result['components'][1]['model']['text'],'SRR6028-100Y')
        self.assertEqual(result['components'][1]['value']['text'],'10uH')

    def test_model_corrections_can_teach_the_ocr_library(self):
        image=np.full((170,260,3),255,np.uint8)
        cv2.putText(image,'BC547B',(25,42),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        dets=[detection('A','transistor_npn',20)];text_token=token('BC547B',22)
        text_token['bbox']=dict(x1=22,y1=17,x2=155,y2=47,width=133,height=30,center_x=88.5,center_y=32)
        result=criar_componentes_logicos([text_token],dets)
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);path=folder/'image.png';cv2.imwrite(str(path),image)
            with patch.object(ocr,'OCR_SAMPLES_DIR',folder/'samples'),patch.object(ocr,'OCR_SAMPLES_INDEX',folder/'samples/index.json'):
                count=ocr.salvar_amostras_ocr(path,result,[dict(detection_id='A',reference=result['components'][0]['reference']['text'],value='',model='BC547C')])
                self.assertEqual(count,1)
                records=ocr._carregar_indice_amostras();self.assertEqual(records[0]['kind'],'model')
                updated=ocr._aplicar_biblioteca_ocr(image,[text_token],dets)
                self.assertEqual(updated[0]['text'],'BC547C')

    def test_removing_datasheet_policy_clears_estimate_and_spice_is_unchanged(self):
        c=dict(type='mosfet_n',model={'text':'2N7000','source':'manual'},parameter_policy={'vgs_threshold':'datasheet'},
               electrical_parameters=resolve_parameters('mosfet_n','2N7000',policies={'vgs_threshold':'datasheet'})['parameters'])
        edit=model_edit(c,dict(parameter_policy={'vgs_threshold':'manual'}))
        self.assertNotIn('vgs_threshold',edit['electrical_parameters'])
        from backend.spice_netlist import build_netlist
        from backend.test_spice_stages import divider
        before=build_netlist(divider())['netlist'];circuit=divider()
        circuit['components'][1]['metadata']={'model':{'text':'CUSTOM'},'electrical_parameters':{'hfe':{'value':999}}}
        self.assertEqual(build_netlist(circuit)['netlist'],before)

    def test_ocr_reread_preserves_manual_edits_only_for_same_component(self):
        dets=[detection('A','transistor_npn',20)];original=criar_componentes_logicos([token('BC547A',25)],dets)
        edited=aplicar_correcoes_manuais(original,[dict(detection_id='A',reference='Q1',value='',model='BC547B',
                                electrical_parameters={'hfe':'172'},parameter_policy={'vbe_on':'datasheet'})],dict(detections=dets))
        reread=criar_componentes_logicos([token('BC547C',25)],dets);preserve_manual_models(reread,edited)
        self.assertEqual(reread['components'][0]['model']['text'],'BC547B')
        self.assertEqual(reread['components'][0]['electrical_parameters']['hfe']['value'],172)
        moved=criar_componentes_logicos([token('BC547C',45)],[detection('A','transistor_npn',40)])
        preserve_manual_models(moved,edited);self.assertEqual(moved['components'][0]['model']['text'],'BC547C')
        self.assertEqual(original['components'][0]['model']['text'],'BC547A')

class ModelOCRAPITests(unittest.TestCase):
    setUp=api_fixture.CircuitAPItests.setUp
    tearDown=api_fixture.CircuitAPItests.tearDown
    update=api_fixture.CircuitAPItests.update
    def seed(self):
        dets=[detection('A','transistor_npn',20)]
        result=criar_componentes_logicos([token('BC547B',25),token('Q1',22,115)],dets)
        result.update(tokens=[],count=1)
        self.project.update(detections_info=json.dumps(dict(detections=dets)),ocr_info=json.dumps(result),ocr_filename='ocr.png')
        return result
    def test_model_lookup_api_page_and_shared_editor(self):
        self.assertGreaterEqual(self.client.get('/component-models').json()['model_count'],100)
        response=self.client.post('/component-models/resolve',json=dict(type='mosfet_n',model='BS170',parameter_policy={'gm':'datasheet'}))
        self.assertEqual(response.status_code,200);self.assertEqual(response.json()['parameters']['gm']['value'],.32)
        self.assertEqual(self.client.get('/projects/TEST/ocr-models-view').status_code,200)
        self.assertEqual(self.client.get('/projects/MISSING/ocr-models-view').status_code,404)
        self.assertEqual(self.client.get('/model-editor.js').status_code,200)
        self.assertIn('Valores e modelos',self.client.get('/projects/TEST/workflow').text)
        self.assertIn('GuidedStep',self.client.get('/projects/TEST/guided/editor/models').text)
    def test_save_model_parameters_without_changing_wire_or_terminal_points(self):
        original=self.seed();before=(self.folder/'terminals.json').read_bytes()
        terminals=json.loads(before);terminals['components'][0]['detection_id']='A';terminals['components'][0]['type']='transistor_npn'
        (self.folder/'terminals.json').write_text(json.dumps(terminals));positions=copy.deepcopy(terminals['components'][0]['terminals'])
        with patch.object(main,'atualizar_ocr',side_effect=self.update),patch.object(main,'atualizar_terminais',side_effect=self.update),patch.object(main,'salvar_amostras_ocr',return_value=0),patch.object(main,'_gerar_terminais_automaticamente') as generate:
            response=self.client.put('/projects/TEST/ocr/components',json={'components':[dict(detection_id='A',reference='Q1',value='',model='BC547C',electrical_parameters={},parameter_policy={'hfe':'datasheet'})]})
        self.assertEqual(response.status_code,200,response.text);generate.assert_not_called()
        saved=json.loads((self.folder/'components.json').read_text(encoding='utf8'))['components'][0]
        self.assertEqual(saved['electrical_parameters']['hfe']['value'],610)
        after=json.loads((self.folder/'terminals.json').read_text(encoding='utf8'))
        self.assertEqual(after['components'][0]['terminals'],positions)
        self.assertEqual(after['components'][0]['model']['text'],'BC547C')
    def test_invalid_model_does_not_save_or_train(self):
        original=self.seed()
        with patch.object(main,'atualizar_ocr') as save,patch.object(main,'salvar_amostras_ocr') as train:
            response=self.client.put('/projects/TEST/ocr/components',json={'components':[dict(detection_id='A',reference='Q1',value='',model='IRF9540N')]})
        self.assertEqual(response.status_code,400,response.text);save.assert_not_called();train.assert_not_called()
        self.assertFalse((self.folder/'components.json').exists())
    def test_circuit_and_review_keep_model_metadata_for_future_simulation(self):
        source=dict(schema_version='stage10-connectivity-1.0',components=[dict(component_id='Q1',type='transistor_npn',
                    reference={'text':'Q1'},value=None,terminals=[dict(pin=p) for p in ['B','C','E']],
                    **model_edit(dict(type='transistor_npn'),dict(model='BC547B',electrical_parameters={'hfe':'150'})))],
                    warnings=[],nets=[dict(name=p,terminal_refs=[p]) for p in ['B','C','E']],
                    terminal_connections=[dict(terminal_ref=p,component_id='Q1',pin=p) for p in ['B','C','E']])
        circuit=gerar_circuito(source);c=circuit['components'][0]
        reviewed=apply_review(circuit,dict(components=[dict(c,original_id='Q1')],nets=circuit['nets']))
        self.assertEqual(reviewed['components'][0]['metadata']['model']['text'],'BC547B')
        self.assertEqual(reviewed['components'][0]['metadata']['electrical_parameters']['hfe']['value'],150)

if __name__=='__main__':unittest.main()
