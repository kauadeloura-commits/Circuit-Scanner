import copy
import itertools
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend.detection_boxes import fully_contains, remove_nested_boxes, validate_no_nested_boxes


def detection(kind, xy, confidence=.8, name='candidate'):
    x1, y1, x2, y2 = xy
    return dict(detection_id=name, **{'class':kind}, class_id={'resistor':0,'capacitor':1,'mosfet_p':2}.get(kind,3),
                confidence=confidence,bbox=dict(x1=x1,y1=y1,x2=x2,y2=y2,width=x2-x1,height=y2-y1,
                                                center_x=(x1+x2)/2,center_y=(y1+y2)/2))


class NestedBoxesTests(unittest.TestCase):
    def test_mosfet_contains_more_confident_capacitor(self):
        items=[detection('capacitor',(138,161,193,195),.99),detection('mosfet_p',(40,46,284,281),.65)]
        kept,rejected=remove_nested_boxes(items)
        self.assertEqual([d['class'] for d in kept],['mosfet_p'])
        self.assertEqual(rejected[0]['enclosing_detection_id'],kept[0]['detection_id'])
        self.assertEqual(rejected[0]['rejection_reason'],'component_box_fully_inside_another')

    def test_nested_resistors_keep_full_symbol(self):
        kept,rejected=remove_nested_boxes([detection('resistor',(22,0,123,145)),detection('resistor',(57,38,96,130))])
        self.assertEqual(len(kept),1);self.assertEqual(len(rejected),1)
        self.assertEqual(kept[0]['bbox']['x1'],22)

    def test_partial_overlap_even_nearly_contained_is_allowed(self):
        outer=detection('mosfet_p',(0,0,100,100))
        for coords in [(90,40,120,60),(10,10,100.000001,30),(-.000001,10,30,30)]:
            with self.subTest(coords=coords):
                items=[outer,detection('capacitor',coords)]
                self.assertEqual(len(remove_nested_boxes(items)[0]),2)
                validate_no_nested_boxes(items)

    def test_shared_edges_count_as_containment(self):
        outer=detection('resistor',(0,0,100,100))
        inner=detection('capacitor',(0,10,100,20))
        self.assertTrue(fully_contains(outer,inner));self.assertEqual(len(remove_nested_boxes([inner,outer])[0]),1)

    def test_identical_boxes_choose_confidence_and_then_manual(self):
        low=detection('resistor',(0,0,40,40),.5);high=detection('capacitor',(0,0,40,40),.9)
        self.assertEqual(remove_nested_boxes([low,high])[0][0]['class'],'capacitor')
        low['manual_reviewed']=True
        self.assertEqual(remove_nested_boxes([low,high])[0][0]['class'],'resistor')

    def test_nested_chain_and_order_do_not_change_result(self):
        items=[detection('mosfet_p',(0,0,200,200)),detection('capacitor',(30,30,100,100)),detection('resistor',(40,40,60,60))]
        for perm in itertools.permutations(items):
            kept,rejected=remove_nested_boxes(perm)
            self.assertEqual([d['class'] for d in kept],['mosfet_p']);self.assertEqual(len(rejected),2)
            self.assertTrue(all(r['enclosing_detection_id']=='DET_001' for r in rejected))

    def test_no_mutation_and_audit_ids_follow_final_spatial_order(self):
        items=[detection('mosfet_p',(100,100,200,200),name='old1'),detection('capacitor',(120,120,140,140),name='old2'),detection('resistor',(0,0,20,20),name='old3')]
        before=copy.deepcopy(items);kept,rejected=remove_nested_boxes(items)
        self.assertEqual(items,before);self.assertEqual(rejected[0]['enclosing_detection_id'],'DET_002')
        self.assertEqual(kept[1]['class'],'mosfet_p')

    def test_random_output_always_has_no_containment_and_is_idempotent(self):
        rng=random.Random(531)
        items=[]
        for _ in range(100):
            x,y=rng.randrange(200),rng.randrange(200)
            items.append(detection('resistor',(x,y,x+rng.randrange(1,150),y+rng.randrange(1,150))))
        kept,_=remove_nested_boxes(items);validate_no_nested_boxes(kept)
        again,rejected=remove_nested_boxes(kept);self.assertEqual(again,kept);self.assertEqual(rejected,[])

    def test_manual_nesting_and_equal_boxes_are_rejected(self):
        for xy in [(10,10,20,20),(0,0,100,100)]:
            with self.subTest(xy=xy),self.assertRaisesRegex(ValueError,'sobreposição parcial'):
                validate_no_nested_boxes([detection('resistor',(0,0,100,100)),detection('capacitor',xy)])

    def test_invalid_coordinates_cannot_bypass_constraint(self):
        for coords in [(0,0,0,10),(0,0,float('nan'),10),(0,0,float('inf'),10)]:
            with self.subTest(coords=coords),self.assertRaises(ValueError):
                remove_nested_boxes([detection('resistor',coords)])

    def test_empty_and_single_box(self):
        self.assertEqual(remove_nested_boxes([]),([],[]))
        self.assertEqual(len(remove_nested_boxes([detection('resistor',(0,0,20,30))])[0]),1)

    def test_actual_detector_saves_and_draws_only_survivors_for_all_strategies(self):
        from backend import recognition
        candidates=[detection('mosfet_p',(10,10,190,190),.7),detection('capacitor',(60,60,80,80),.95),detection('capacitor',(180,100,215,125),.8)]
        for strategy in ('color','grayscale','auto'):
            with self.subTest(strategy=strategy),tempfile.TemporaryDirectory() as folder:
                folder=Path(folder);image=folder/'input.png';output=folder/'detections.png';meta=folder/'detections.json'
                cv2.imwrite(str(image),np.full((240,240,3),255,dtype=np.uint8))
                def accepted(_,items):return items,[]
                with (patch.object(recognition,'_carregar_modelo',return_value=object()),
                      patch.object(recognition,'_hash_modelo',return_value='test'),
                      patch.object(recognition,'_executar_predicao',return_value=copy.deepcopy(candidates)),
                      patch.object(recognition,'consolidar_multiview',side_effect=lambda items,**kwargs:copy.deepcopy(candidates)),
                      patch.object(recognition,'filtrar_falsos_positivos_geometricos',side_effect=accepted),
                      patch.object(recognition,'_deteccoes_resistores_geometricos',return_value=[]),
                      patch.object(recognition,'_deteccoes_resistores_retangulares',return_value=[]),
                      patch.object(recognition,'_desenhar_deteccoes',wraps=recognition._desenhar_deteccoes) as draw):
                    result=recognition.detectar_componentes(image,output,meta,confidence=.2,strategy=strategy)
                self.assertEqual(result['count'],2);self.assertEqual(result['nested_box_rejected_count'],1)
                self.assertEqual(json.loads(meta.read_text(encoding='utf8'))['detections'],result['detections'])
                self.assertEqual(draw.call_args.args[1],result['detections']);self.assertTrue(output.exists())
                validate_no_nested_boxes(result['detections'])


class NestedBoxesAPITests(unittest.TestCase):
    def setUp(self):
        from backend.test_circuit_api import CircuitAPItests
        self.fixture=CircuitAPItests();self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def setup_api(self):
        from backend import main
        self.fixture.project['detections_info']=json.dumps({'image_width':320,'image_height':240,'detections':[]})
        return main

    def test_manual_rejection_leaves_project_and_downstream_files_untouched(self):
        main=self.setup_api();folder=self.fixture.folder
        for name in ('components.json','detections.json','terminals.png'):(folder/name).write_bytes(b'original artifact')
        before={p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}
        items=[detection('resistor',(10,10,200,200)),detection('resistor',(30,30,50,50))]
        with (patch.object(main,'load_real_classes',return_value=[{'id':0,'key':'resistor'}]),
              patch.object(main,'atualizar_deteccoes') as update,
              patch.object(main,'_gerar_terminais_automaticamente') as terminals):
            response=self.fixture.client.put('/projects/TEST/detections',json={'detections':items})
        self.assertEqual(response.status_code,400);self.assertIn('inteiramente',response.json()['detail'])
        self.assertEqual(before,{p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()})
        update.assert_not_called();terminals.assert_not_called()

    def test_manual_partial_overlap_can_be_saved(self):
        main=self.setup_api();items=[detection('resistor',(10,10,100,100)),detection('resistor',(90,20,180,110))]
        with (patch.object(main,'load_real_classes',return_value=[{'id':0,'key':'resistor'}]),
              patch.object(main,'atualizar_deteccoes',side_effect=self.fixture.update),
              patch.object(main,'_gerar_terminais_automaticamente',return_value=None)):
            response=self.fixture.client.put('/projects/TEST/detections',json={'detections':items})
        self.assertEqual(response.status_code,200,response.text)
        saved=json.loads((self.fixture.folder/'detections.json').read_text(encoding='utf8'))
        self.assertEqual(len(saved['detections']),2);validate_no_nested_boxes(saved['detections'])
