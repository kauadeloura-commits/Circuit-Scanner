import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from backend import main,terminal_detection as td,recognition
from backend.terminal_wire_contacts import constrain_terminal_positions,refine_all_terminal_contacts
from backend.test_terminal_identity import fixture,rotate_fixture
from backend.test_npn_wire_contacts import fixture as bjt_fixture,rotate_case
from backend.wire_detection import detectar_fios


def assert_inside(test,result,image=None):
    for c in result['components']:
        box=c['bbox']
        for t in c.get('terminals') or []:
            p=t['position']
            test.assertTrue(box['x1']<p['x']<box['x2'] and box['y1']<p['y']<box['y2'],(c.get('type'),box,t))
            if image is not None and t.get('wire_contact_confirmed'):
                test.assertLess(int(image[round(p['y']),round(p['x'])].min()),100,t)


class WireFirstPipelineTests(unittest.TestCase):
    def test_all_supported_classes_have_bounded_geometry_without_an_image(self):
        for kind in td.PIN_LIBRARY:
            with self.subTest(kind=kind):
                c={'component_id':'X','type':kind,'bbox':dict(x1=10.25,y1=15.8,x2=190.7,y2=130.6)}
                result=td.detectar_terminais_componentes({'components':[c]},220,180)
                assert_inside(self,result)

    def test_model_points_outside_the_box_are_bounded_on_original_ink(self):
        image=np.full((180,220,3),255,np.uint8)
        cv2.line(image,(0,90),(219,90),(20,20,20),3)
        result={'components':[{'type':'resistor','bbox':dict(x1=55.2,y1=60.3,x2=160.7,y2=120.8),
            'terminals':[dict(pin='1',position=dict(x=-20,y=90)),dict(pin='2',position=dict(x=250,y=90))]}]}
        constrain_terminal_positions(result,image)
        assert_inside(self,result)
        for t in result['components'][0]['terminals']:
            p=t['position'];self.assertLess(int(image[int(p['y']),int(p['x'])].min()),60)

    def test_invalid_box_has_no_terminals(self):
        image=np.full((100,100,3),255,np.uint8)
        for box in (dict(x1=20,y1=20,x2=20,y2=60),dict(x1=float('nan'),y1=20,x2=60,y2=60)):
            result={'components':[{'bbox':box,'terminals':[dict(pin='1',position=dict(x=25,y=25))]}]}
            constrain_terminal_positions(result,image)
            self.assertEqual(result['terminal_count'],0)
            self.assertTrue(result['components'][0]['review_required'])

    def test_nonfinite_prediction_does_not_escape_the_box(self):
        result={'components':[{'bbox':dict(x1=20,y1=20,x2=80,y2=80),
            'terminals':[dict(pin='1',position=dict(x=float('nan'),y=1000))]}]}
        constrain_terminal_positions(result,np.full((100,100,3),255,np.uint8))
        assert_inside(self,result)
        self.assertTrue(result['components'][0]['review_required'])

    def test_old_manual_correction_outside_box_is_bounded_and_reviewed(self):
        result={'components':[{'bbox':dict(x1=20,y1=20,x2=80,y2=80),
            'terminal_detection':{'status':'manual_ground_truth'},
            'terminals':[dict(pin='1',position=dict(x=10,y=10),source='manual_ground_truth')]}]}
        constrain_terminal_positions(result,np.full((100,100,3),255,np.uint8))
        assert_inside(self,result)
        self.assertIn('correcao_manual_antiga_fora_da_caixa',result['components'][0]['review_reasons'])

    def test_wires_ignore_old_terminal_positions(self):
        image,bbox,_=fixture('opamp')
        c=dict(component_id='U',type='opamp',bbox=bbox,visual_confidence=.99,terminals=[])
        info={'components':[c],'terminal_pipeline_phase':'before_terminals'}
        first=detectar_fios(image,components_info=info)
        bad=copy.deepcopy(info)
        bad['components'][0]['terminals']=[dict(position=dict(x=275,y=270))]
        second=detectar_fios(image,components_info=bad)
        self.assertTrue(np.array_equal(first[2],second[2]))
        self.assertFalse(first[0]['preprocessing']['terminal_seed_filter_applied'])

    def test_disk_wires_exist_before_terminal_detection_and_junctions_follow(self):
        image,bbox,_=fixture('opamp')
        c=dict(component_id='U',type='opamp',bbox=bbox,visual_confidence=.99)
        events=[]
        wire_detector=main.detectar_fios;terminal_detector=main.executar_deteccao_terminais;junction_resolver=main.resolver_juncoes
        with TemporaryDirectory() as folder,patch.object(main,'PROJECTS_DIR',Path(folder)):
            p=Path(folder)/'CASE';p.mkdir();cv2.imwrite(str(p/'original.png'),image)
            def wires(*args,**kwargs):
                events.append('wires')
                self.assertFalse((p/'terminals.json').exists())
                # Internal legacy masking anchors are allowed, final contact
                # positions have not yet been computed or published.
                return wire_detector(*args,**kwargs)
            def terminals(*args,**kwargs):
                events.append('terminals')
                self.assertTrue((p/'wires.json').is_file())
                self.assertTrue((p/'wires_skeleton.png').is_file())
                self.assertIsNotNone(kwargs['skeleton'])
                return terminal_detector(*args,**kwargs)
            def junctions(*args,**kwargs):
                events.append('junctions');return junction_resolver(*args,**kwargs)
            with patch('backend.performance.shared_restore',return_value=False), \
                 patch.object(main,'detectar_fios',side_effect=wires),patch.object(main,'executar_deteccao_terminais',side_effect=terminals), \
                 patch.object(main,'resolver_juncoes',side_effect=junctions),patch.object(main,'atualizar_terminais'), \
                 patch.object(main,'atualizar_terminais_pos_fios'),patch.object(main,'atualizar_fios'):
                result=main._gerar_terminais_automaticamente('CASE',{'stored_filename':'original.png'},
                    {'components':[c]},trigger='automatic_detection')
            self.assertEqual(events,['wires','terminals','junctions'])
            self.assertEqual(result['pipeline_order'],['components','wires','terminals','junctions'])
            assert_inside(self,result,image)

    def test_wire_failure_never_runs_or_publishes_new_terminals(self):
        with TemporaryDirectory() as folder,patch.object(main,'PROJECTS_DIR',Path(folder)):
            p=Path(folder)/'CASE';p.mkdir();cv2.imwrite(str(p/'original.png'),np.full((100,100,3),255,np.uint8))
            (p/'terminals.json').write_text('old')
            c=dict(component_id='R',type='resistor',bbox=dict(x1=20,y1=20,x2=80,y2=80))
            with patch.object(main,'detectar_fios',side_effect=ValueError('wire failure')), \
                 patch.object(main,'executar_deteccao_terminais') as terminals:
                with self.assertRaisesRegex(ValueError,'wire failure'):
                    main._gerar_terminais_automaticamente('CASE',{'stored_filename':'original.png'},{'components':[c]})
                terminals.assert_not_called()
            self.assertFalse((p/'terminals.json').exists())

    def test_all_polarized_classes_keep_names_inside_boxes_after_wires_at_four_rotations(self):
        for kind in ('source_dc','capacitor_polarized','battery','opamp','mosfet_n','mosfet_p'):
            for k in range(4):
                with self.subTest(kind=kind,k=k):
                    image,bbox,expected=rotate_fixture(*fixture(kind),k)
                    c=dict(component_id='X',type=kind,bbox=bbox,visual_confidence=.99)
                    info={'components':[c],'terminal_pipeline_phase':'before_terminals'}
                    wires,_,skeleton,_=detectar_fios(image,components_info=info)
                    result=td.detectar_terminais_componentes(info,280,280,image)
                    final=refine_all_terminal_contacts(image,result,wires,skeleton)
                    assert_inside(self,final,image)
                    pins={t['pin']:t for t in final['components'][0]['terminals']}
                    self.assertEqual(set(pins),set(expected))
                    for pin,p in expected.items():
                        target=np.clip(p,[bbox['x1']+1,bbox['y1']+1],[bbox['x2']-1,bbox['y2']-1])
                        self.assertLess(np.linalg.norm(np.array(list(pins[pin]['position'].values()))-target),8,(kind,k,pin))
                    self.assertTrue(all(t.get('wire_contact_confirmed') for t in pins.values()),final['wire_terminal_contacts'])

    def test_npn_and_pnp_share_arrow_rule_in_rotations_and_different_sizes(self):
        from backend.transistor_wire_refinement import refine_transistor_terminals_from_wires
        for kind,direction in (('transistor_npn','outward'),('transistor_pnp','inward')):
            for scale,k in ((.75,0),(1.,1),(1.25,2),(1.,3)):
                with self.subTest(kind=kind,scale=scale,k=k):
                    image,sk,c,expected=rotate_case(*bjt_fixture(mark=direction),k)
                    image=cv2.resize(image,None,fx=scale,fy=scale)
                    # Area sampling preserves a one-pixel lead when shrinking;
                    # nearest sampling can erase an entire conductor row.
                    sk=(cv2.resize(sk,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_AREA)>0).astype(np.uint8)*255
                    c['bbox']={key:value*scale for key,value in c['bbox'].items()};c['type']=kind
                    result,_=refine_transistor_terminals_from_wires(image,{'components':[c]}, {},sk)
                    final=refine_all_terminal_contacts(image,result,{},sk)
                    assert_inside(self,final,image)
                    self.assertEqual(len(final['components'][0]['terminals']),3)
                    for t in final['components'][0]['terminals']:
                        target=expected[t['pin']]*scale
                        self.assertLess(np.linalg.norm(np.array(list(t['position'].values()))-target),7,t)
                    self.assertFalse(final['components'][0]['terminal_detection']['review_required'])

    def test_missing_lead_keeps_the_other_observed_terminal(self):
        image,bbox,_=fixture('source_dc')
        c=dict(component_id='V',type='source_dc',bbox=bbox,visual_confidence=.99)
        result=td.detectar_terminais_componentes({'components':[c]},280,280,image)
        image[:,:62]=255
        sk=np.zeros((280,280),np.uint8);cv2.line(sk,(220,140),(245,140),255,1)
        final=refine_all_terminal_contacts(image,result,{},sk)
        assert_inside(self,final)
        pins=final['components'][0]['terminals']
        self.assertEqual(len(pins),1)
        self.assertTrue(pins[0]['wire_contact_confirmed'])
        self.assertGreater(pins[0]['position']['x'],200)

    def test_manual_click_outside_component_is_rejected_before_training(self):
        from fastapi import HTTPException
        with TemporaryDirectory() as folder,patch.object(main,'PROJECTS_DIR',Path(folder)):
            p=Path(folder)/'CASE';p.mkdir();cv2.imwrite(str(p/'original.png'),np.full((100,100,3),255,np.uint8))
            c=dict(component_id='R',type='resistor',bbox=dict(x1=20,y1=20,x2=80,y2=80))
            with patch.object(main,'buscar_projeto',return_value={'stored_filename':'original.png'}), \
                 patch.object(main,'_terminal_result_for_manual_edit',return_value={'components':[c]}), \
                 patch.object(main,'save_generic_terminal_training_sample') as training:
                with self.assertRaises(HTTPException) as failure:
                    main.corrigir_terminais_e_salvar_treino('CASE',{'components':[{'component_id':'R',
                        'points':{'1':dict(x=10,y=50),'2':dict(x=70,y=50)}}]})
                self.assertEqual(failure.exception.status_code,400)
                training.assert_not_called()

    def test_preview_terminal_marker_and_search_ring_do_not_leave_box(self):
        with TemporaryDirectory() as folder:
            source=Path(folder)/'source.png';output=Path(folder)/'preview.png'
            cv2.imwrite(str(source),np.full((100,100,3),255,np.uint8))
            result={'components':[{'component_id':'R','type':'resistor','bbox':dict(x1=20,y1=20,x2=80,y2=80),
                'terminals':[dict(pin='1',position=dict(x=79,y=50),search_radius_px=12)]}]}
            td.salvar_imagem_terminais(source,result,output)
            image=cv2.imread(str(output))
            self.assertTrue(np.all(image[40:65,82:98]==255))


class GeneralRecognitionTests(unittest.TestCase):
    def predict_case(self,image,predict):
        with TemporaryDirectory() as folder:
            source=Path(folder)/'input.png';cv2.imwrite(str(source),image)
            with patch.object(recognition,'_carregar_modelo',return_value=object()), \
                 patch.object(recognition,'_hash_modelo',return_value='test'), \
                 patch.object(recognition,'_executar_predicao',side_effect=predict):
                return recognition.detectar_componentes(source,Path(folder)/'preview.png',Path(folder)/'result.json')

    def test_rotated_model_predictions_are_considered_for_source_dc_not_only_resistors(self):
        image,box,_=fixture('source_dc')
        for rotation in ('cw','ccw','half'):
            with self.subTest(rotation=rotation):
                expected_source=f"rotated_{180 if rotation=='half' else 90}_{rotation}_all_classes"
                if rotation=='cw':
                    rotated=dict(x1=280-box['y2'],x2=280-box['y1'],y1=box['x1'],y2=box['x2'])
                elif rotation=='ccw':
                    rotated=dict(x1=box['y1'],x2=box['y2'],y1=280-box['x2'],y2=280-box['x1'])
                else:
                    rotated=dict(x1=280-box['x2'],x2=280-box['x1'],y1=280-box['y2'],y2=280-box['y1'])
                def predict(model,view,confidence,imgsz,source):
                    if source!=expected_source:return []
                    return [dict(class_id=6,**{'class':'source_dc'},confidence=.85,bbox=rotated,inference_sources=[source])]
                result=self.predict_case(image,predict)
                self.assertEqual(result['count'],1)
                for key in ('x1','y1','x2','y2'):
                    self.assertAlmostEqual(result['detections'][0]['bbox'][key],box[key],delta=.1)

    def test_repeated_weak_rotations_do_not_promote_false_components(self):
        image=np.full((280,280,3),255,np.uint8)
        def predict(model,view,confidence,imgsz,source):
            if not source.startswith('rotated_90_'):return []
            return [dict(class_id=6,**{'class':'source_dc'},confidence=.13,
                bbox=dict(x1=90,y1=90,x2=190,y2=190),inference_sources=[source])]
        self.assertEqual(self.predict_case(image,predict)['count'],0)

    def test_notebook_rules_do_not_become_mosfets_after_rotation(self):
        image=np.full((280,280,3),245,np.uint8)
        for y in range(25,280,35):cv2.line(image,(0,y),(279,y),(90,90,90),2)
        def predict(model,view,confidence,imgsz,source):
            if not source.startswith('rotated_90_'):return []
            return [dict(class_id=11,**{'class':'mosfet_n'},confidence=.99,
                bbox=dict(x1=80,y1=80,x2=180,y2=180),inference_sources=[source])]
        self.assertEqual(self.predict_case(image,predict)['count'],0)

    def test_validated_component_color_produces_a_clean_inference_view(self):
        image,box,_=fixture('source_dc')
        image[np.all(image==20,axis=2)]=(155,45,15)
        for x in (15,45,260):cv2.line(image,(x,0),(x,279),(35,35,35),2)
        observed={}
        def predict(model,view,confidence,imgsz,source):
            if source=='component_ink_normalized':observed['view']=view.copy()
            if source!='original_color':return []
            return [dict(class_id=6,**{'class':'source_dc'},confidence=.95,bbox=box,inference_sources=[source])]
        self.predict_case(image,predict)
        self.assertIn('view',observed)
        view=observed['view']
        self.assertGreater(int(view[140,15].min()),220)
        self.assertLess(int(view[140,100].min()),100)


if __name__=='__main__':unittest.main()
