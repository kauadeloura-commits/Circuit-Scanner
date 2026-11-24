import copy
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from backend.resistor_contacts import lead_contacts
from backend.terminal_wire_contacts import _power_lead_contacts
from backend.verified_power_contacts import recover_power


class ExternalContactRecoveryTests(unittest.TestCase):
    def setUp(self):
        from backend.drawing_mode import using
        self.domain=using('handwritten');self.domain.__enter__()
        self.addCleanup(self.domain.__exit__,None,None,None)

    def test_resistor_keeps_an_off_center_bent_lead_but_rejects_side_wall_contacts(self):
        body=dict(center=[100.,100.],evidence=dict(body_axis=[1.,0.],body_size=[80.,30.]))
        c=dict(type='resistor',bbox=dict(x1=40,y1=60,x2=160,y2=140))
        ports=[dict(point=np.array(p,float),stage8_distance_px=0) for p in [(41,100),(159,115),(140,60),(159,130)]]
        with patch('backend.resistor_contacts.body_for_box',return_value=body):
            selected,proof=lead_contacts(np.full((200,200,3),255,np.uint8),c,ports)
        self.assertEqual({tuple(p['point']) for p in selected},{(41,100),(159,115)})

    def test_power_stem_beats_multiple_false_contacts_on_a_faint_ring(self):
        image=np.full((180,180,3),255,np.uint8)
        cv2.circle(image,(90,55),20,(60,60,60),3)
        cv2.line(image,(90,75),(90,135),(20,20,20),4)
        c=dict(type='vcc',bbox=dict(x1=65,y1=30,x2=115,y2=125))
        ports=[dict(point=np.array(p,float),source_ink_continuity=True) for p in [(67,55),(80,32),(90,124)]]
        chosen,proof=_power_lead_contacts(image,c,ports)
        self.assertEqual(len(chosen),1)
        np.testing.assert_allclose(chosen[0]['point'],[90,124])

    def test_radial_power_stem_near_a_body_is_found_in_all_four_rotations(self):
        image=np.full((240,300,3),255,np.uint8)
        cv2.circle(image,(150,60),13,(35,35,35),4)
        cv2.line(image,(150,73),(150,105),(35,35,35),4)
        cv2.rectangle(image,(135,105),(165,155),(35,35,35),4)
        cv2.line(image,(150,155),(150,205),(35,35,35),4)
        detection=dict(detection_id='R',**{'class':'resistor'},bbox=dict(x1=125,y1=98,x2=175,y2=175))
        from backend.handwriting_orientation import transform_box
        from backend.drawing_mode import using
        with using('handwritten'):
            for k in range(4):
                view=np.rot90(image,k).copy();d=copy.deepcopy(detection)
                d['bbox']=transform_box(d['bbox'],k,*image.shape[:2])
                found=recover_power(view,[d],9)
                self.assertEqual(len(found),1,k)

    def test_detached_ring_or_a_letter_without_a_radial_symbol_lead_is_not_power(self):
        image=np.full((240,300,3),255,np.uint8)
        cv2.putText(image,'R10',(30,100),cv2.FONT_HERSHEY_SIMPLEX,1.4,(20,20,20),3)
        cv2.circle(image,(210,150),13,(20,20,20),3)
        self.assertEqual(recover_power(image,[],9),[])

    def test_source_sign_outside_the_circle_box_names_existing_contacts_in_all_rotations(self):
        from backend.test_terminal_identity import fixture,rotate_fixture,plus
        from backend.test_polarity_after_wires import contacts
        from backend.terminal_identity import finalize_polarity_on_contacts
        image,_,_=fixture('source_dc',mark=False)
        plus(image,204,83)
        bbox=dict(x1=92,y1=95,x2=188,y2=186)
        expected={'+':(187,140),'-':(93,140)}
        for k in range(4):
            view,box,pins=rotate_fixture(image,bbox,expected,k)
            c=contacts('source_dc',box,pins)
            proof=finalize_polarity_on_contacts(view,c,components=[c])
            self.assertTrue(proof['confirmed'],(k,proof))
            self.assertEqual({t['pin']:t['position'] for t in c['terminals']},
                {p:dict(x=float(v[0]),y=float(v[1])) for p,v in pins.items()})

    def test_reference_alternative_needs_a_distinct_value_owned_by_the_same_component(self):
        from backend.annotation_semantics import recover_reference_readings
        d=dict(detection_id='R',**{'class':'resistor'})
        token=dict(text='4R7',confidence=.99,bbox=dict(x1=10,y1=10,x2=60,y2=30),
            component_context=dict(detection_id='R',ambiguous=False),
            spatial_alternatives=[dict(text='R7',confidence=.999,bbox=dict(x1=20,y1=10,x2=60,y2=30))])
        value=dict(text='2K2',confidence=.95,bbox=dict(x1=10,y1=40,x2=60,y2=60),
            component_context=dict(detection_id='R',ambiguous=False))
        self.assertEqual(recover_reference_readings([token],[d])[0]['text'],'4R7')
        self.assertEqual(recover_reference_readings([token,value],[d])[0]['text'],'R7')
        from backend.drawing_mode import using
        with using('digital'):
            self.assertEqual(recover_reference_readings([token,value],[d])[0]['text'],'4R7')
        value['component_context']['detection_id']='OTHER'
        self.assertEqual(recover_reference_readings([token,value],[d])[0]['text'],'4R7')
