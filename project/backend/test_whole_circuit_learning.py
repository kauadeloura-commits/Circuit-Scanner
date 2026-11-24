import unittest
from unittest.mock import patch
import numpy as np
from backend.context_detection import separate_predictions,AUXILIARY_NAMES
from backend import recognition
from backend.sequence_ocr import prepare,compatible,decode

class WholeCircuitLearningTests(unittest.TestCase):
    def test_auxiliary_text_is_not_a_component(self):
        boxes=[dict(class_id=i,**{'class':name},confidence=.9,bbox=dict(x1=2,y1=2,x2=18,y2=18)) for i,name in ((0,'resistor'),(15,'text_aux'),(18,'other_symbol_aux'))]
        components,text=separate_predictions(np.full((24,24,3),255,np.uint8),boxes)
        self.assertEqual([d['class'] for d in components],['resistor']);self.assertEqual(len(text),1)

    def test_generic_transistor_is_never_assigned_an_arbitrary_subtype(self):
        det=dict(class_id=17,**{'class':'transistor_unspecified_aux'},confidence=.9,bbox=dict(x1=2,y1=2,x2=18,y2=18))
        with patch('backend.context_detection.transistor_subtype',return_value=None):
            self.assertEqual(separate_predictions(np.zeros((24,24,3),np.uint8),[det])[0],[])

    def test_mapping_accepts_only_the_named_auxiliary_extension(self):
        core=recognition._classes_esperadas()
        self.assertTrue(recognition.validar_mapeamento_classes(core)[0])
        self.assertTrue(recognition.validar_mapeamento_classes({**core,**AUXILIARY_NAMES})[0])
        self.assertFalse(recognition.validar_mapeamento_classes({**core,15:'resistor'})[0])

    def test_reference_and_measurement_units_follow_component_kind(self):
        for text,kind,expected in [('R9','resistor',True),('R9','capacitor',False),('C10','capacitor_polarized',True),('10mH','inductor',True),('10mH','capacitor',False),('5mV','vcc',True),('100kΩ','resistor',True),('100uF','resistor',False)]:
            with self.subTest(text=text,kind=kind):self.assertEqual(bool(compatible(text,kind)),expected)

    def test_ctc_blank_preserves_repeated_zeros(self):
        import torch
        alphabet='10k';indices=[1,1,0,2,2,0,2,0,3,3]
        logits=torch.full((len(indices),1,4),-20.)
        for t,i in enumerate(indices):logits[t,0,i]=20.
        self.assertEqual(decode(logits,alphabet)[0]['text'],'100k')

    def test_empty_and_faint_crop_are_safe(self):
        self.assertIsNone(prepare(np.zeros((0,0),np.uint8)))
        self.assertIsNone(prepare(np.zeros((12,40),np.uint8)))
        mask=np.zeros((30,60),np.uint8);mask[4:25,20:24]=255
        result=prepare(mask);self.assertEqual(result.shape,(32,192));self.assertTrue(result.any())

    def test_missing_upright_shape_does_not_prevent_rotated_validation(self):
        image=np.zeros((40,60,3),np.uint8);box=dict(x1=10,y1=10,x2=30,y2=30)
        with patch.object(recognition,'_validar_classe_geometrica_upright',side_effect=[(False,0,dict(hard_text_rejection=True)),(True,.9,dict())]):
            valid,_,evidence=recognition._validar_classe_geometrica(image,box,'battery')
        self.assertTrue(valid);self.assertEqual(evidence['validation_rotation_degrees'],90)

    def test_global_assignment_avoids_greedy_reference_theft(self):
        from backend.text_assignment import globally_assign
        tokens=[{'text':'R1'},{'text':'R2'}]
        pairs=[(3.,-.9,'t1','A',tokens[0]),(12.,-.9,'t1','B',tokens[0]),(8.,-.9,'t2','A',tokens[1]),(32.,-.9,'t2','B',tokens[1])]
        selected=globally_assign(pairs)
        self.assertEqual({p[2]:p[3] for p in selected},{'t1':'B','t2':'A'})

    def test_unit_priority_is_preserved_and_unmatched_component_has_no_invented_text(self):
        from backend.text_assignment import globally_assign
        explicit={'text':'10kΩ'};bare={'text':'10'}
        selected=globally_assign([(0,30.,-.9,'unit','A',explicit),(2,1.,-.99,'bare','A',bare)],value_pairs=True)
        self.assertEqual(len(selected),1);self.assertEqual(selected[0][-1]['text'],'10kΩ')

    def test_triangle_ground_requires_exactly_one_external_lead(self):
        import cv2
        from backend.ground_symbols import triangle_ground_evidence
        for leads in (1,2,3):
            image=np.full((200,220,3),255,np.uint8)
            cv2.polylines(image,[np.array([(80,100),(140,100),(110,145)],np.int32)],True,(20,20,20),3)
            cv2.line(image,(110,100),(110,35),(20,20,20),3)
            if leads>=2:cv2.line(image,(110,145),(110,190),(20,20,20),3)
            if leads>=3:cv2.line(image,(80,100),(20,100),(20,20,20),3)
            evidence=triangle_ground_evidence(image,dict(x1=72,y1=85,x2=148,y2=153))
            self.assertEqual(bool(evidence),leads==1)

    def test_resistor_consensus_requires_opposed_leads_and_a_wide_body(self):
        import cv2
        from backend.physical_consensus import opposed_resistor_leads
        from backend.resistor_geometry import symbol_mask
        box=dict(x1=75,y1=65,x2=145,y2=135)
        for shape,expected in [('wire',False),('open_body',True),('text',False),('one_lead',False)]:
            image=np.full((200,240,3),255,np.uint8)
            if shape=='text':cv2.putText(image,'R2',(80,115),cv2.FONT_HERSHEY_SIMPLEX,1,(20,20,20),2)
            else:
                cv2.line(image,(15,100),(90,100),(20,20,20),3)
                if shape!='one_lead':cv2.line(image,(130,100),(220,100),(20,20,20),3)
                if shape=='wire':cv2.line(image,(90,100),(130,100),(20,20,20),3)
                else:cv2.polylines(image,[np.array([(90,100),(90,82),(130,82),(130,118),(94,118)],np.int32)],False,(20,20,20),3)
            self.assertEqual(bool(opposed_resistor_leads(image,box,symbol_mask(image))),expected,shape)

    def test_context_crop_preserves_all_detected_anchors_and_nearby_writing(self):
        from backend.inference_regions import circuit_context_roi
        detections=[dict(confidence=.9,**{'class':'resistor'},bbox=dict(x1=x,y1=300,x2=x+40,y2=380)) for x in (300,450,600)]
        text=[dict(bbox=dict(x1=300,y1=240,x2=380,y2=280))]
        region=circuit_context_roi((1400,1100,3),detections,text)
        self.assertIsNotNone(region)
        for b in [d['bbox'] for d in detections]+[text[0]['bbox']]:
            self.assertLessEqual(region['x1'],b['x1']);self.assertLessEqual(region['y1'],b['y1']);self.assertGreaterEqual(region['x2'],b['x2']);self.assertGreaterEqual(region['y2'],b['y2'])

    def test_unusual_capacitance_prefix_requests_recheck_without_rewriting_digits(self):
        from backend.handwriting_values import prefix_needs_photographic_recheck as check
        reading=dict(text='8kF',confidence=61.)
        self.assertTrue(check(reading,'capacitor'));self.assertEqual(reading['text'],'8kF')
        self.assertFalse(check(dict(text='8kF',confidence=99.),'capacitor'))
        self.assertFalse(check(dict(text='10uF',confidence=70.),'capacitor'))
        self.assertFalse(check(dict(text='10kΩ',confidence=70.),'resistor'))

    def test_interrupted_power_ring_is_verified_in_original_pixels_and_serializable(self):
        import cv2,json
        from backend.power_ring import faint_ring_with_one_lead
        image=np.full((240,260,3),225,np.uint8)
        cv2.circle(image,(130,90),20,(90,90,90),3);cv2.line(image,(130,110),(130,200),(90,90,90),3)
        image[87:90,103:112]=225
        proof=faint_ring_with_one_lead(image,dict(x1=102,y1=63,x2=157,y2=135))
        self.assertIsNotNone(proof);json.dumps(proof)

    def test_voltage_text_recovers_one_lead_power_bar_but_not_unlabeled_bar_or_junction(self):
        import cv2
        from backend.voltage_text_recovery import recover_voltage_ports
        image=np.full((320,400,3),255,np.uint8)
        cv2.line(image,(150,80),(230,80),(20,20,20),3);cv2.line(image,(190,80),(190,220),(20,20,20),3)
        proposal=dict(class_id=9,**{'class':'vcc'},bbox=dict(x1=140,y1=65,x2=240,y2=105),confidence=.9)
        token=dict(text='12V',confidence=.95,bbox=dict(x1=168,y1=25,x2=220,y2=55))
        self.assertEqual(len(recover_voltage_ports(image,[],[proposal],[token])),1)
        self.assertEqual(recover_voltage_ports(image,[],[proposal],[]),[])
        self.assertEqual(recover_voltage_ports(image,[],[proposal],[dict(token,text='12uF')]),[])
        cv2.line(image,(20,80),(380,80),(20,20,20),3)
        self.assertEqual(recover_voltage_ports(image,[],[proposal],[token]),[])

    def test_power_ring_cannot_borrow_a_separate_digit_as_its_wire(self):
        import cv2
        from backend.power_ring import faint_ring_with_one_lead
        image=np.full((250,260,3),235,np.uint8)
        cv2.ellipse(image,(105,85),(18,20),0,0,360,(35,35,70),4)
        cv2.line(image,(87,85),(87,145),(35,35,70),4)
        cv2.line(image,(105,106),(126,145),(35,35,70),4)
        cv2.line(image,(140,110),(190,160),(35,35,70),4)
        self.assertIsNone(faint_ring_with_one_lead(image,dict(x1=77,y1=55,x2=150,y2=153)))

    def test_generic_transistor_can_resolve_to_mosfet_without_becoming_a_bjt(self):
        from unittest.mock import patch
        from backend.context_detection import separate_predictions
        image=np.full((100,120,3),255,np.uint8)
        proposal=dict(class_id=17,**{'class':'transistor_unspecified_aux'},confidence=.9,bbox=dict(x1=20,y1=20,x2=70,y2=80))
        with patch('backend.context_detection.transistor_subtype',return_value=(12,.8)):
            components,text=separate_predictions(image,[proposal])
        self.assertEqual(components[0]['class'],'mosfet_n');self.assertEqual(components[0]['class_id'],12)
        self.assertEqual(text,[])

    def test_geometry_confirmation_preserves_learned_component_box_and_leads(self):
        from backend.recognition import consolidar_multiview
        learned=dict(class_id=0,**{'class':'resistor'},confidence=.8,bbox=dict(x1=40,y1=20,x2=100,y2=150),inference_sources=['original_color'])
        body=dict(class_id=0,**{'class':'resistor'},confidence=.92,bbox=dict(x1=54,y1=45,x2=86,y2=125),inference_sources=['resistor_iec_geometry'])
        for det in (learned,body):
            b=det['bbox'];b.update(width=b['x2']-b['x1'],height=b['y2']-b['y1'],center_x=(b['x1']+b['x2'])/2,center_y=(b['y1']+b['y2'])/2)
        result=consolidar_multiview([learned,body])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['bbox'],learned['bbox'])
        self.assertEqual(result[0]['geometry_body_bbox'],body['bbox'])

    def test_complete_reference_and_calibrated_voltage_are_not_overwritten_by_digit_retry(self):
        from backend.handwriting_values import complete_reading_has_priority as protected
        self.assertTrue(protected(dict(text='R1',confidence=80.,unresolved=False),'resistor'))
        self.assertFalse(protected(dict(text='R1',confidence=80.,unresolved=True),'resistor'))
        self.assertTrue(protected(dict(text='5V',confidence=99.5,min_character_confidence=.97,reading_method='trained_whole_inscription_ctc'),'vcc'))
        self.assertFalse(protected(dict(text='5V',confidence=80.,min_character_confidence=.7,reading_method='trained_whole_inscription_ctc'),'vcc'))
        self.assertFalse(protected(dict(text='5V',confidence=80.,reading_method='legacy_numeric'),'vcc'))

if __name__=='__main__':unittest.main()
