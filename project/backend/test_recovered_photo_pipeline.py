import copy
import unittest
import cv2
import numpy as np
from unittest.mock import patch

from backend.external_conductor_ink import local_color_mask
from backend.handwriting_deskew import estimate_angle,transform_box,restore
from backend.inscription_guard import separate
from backend.terminal_wire_contacts import refine_all_terminal_contacts
from backend.wire_detection import _skeletonize


class RecoveredPhotoPipelineTests(unittest.TestCase):
    def test_desaturated_middle_of_a_real_pen_stroke_survives_but_white_gap_does_not(self):
        hsv=np.zeros((100,220,3),np.uint8);hsv[:,:,2]=255
        hsv[47:54,20:200]=(125,110,110)
        hsv[47:54,80:140]=(125,12,145)
        image=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        profile=dict(active=True,chromatic=True,reference_hsv=[125,110,110],hue_tolerance=12)
        rules=np.zeros(image.shape[:2],np.uint8)
        mask=local_color_mask(image,profile,rules)
        _,labels=cv2.connectedComponents(mask,8)
        self.assertGreater(labels[50,25],0)
        self.assertEqual(labels[50,25],labels[50,195])
        image[:,108:112]=255
        _,labels=cv2.connectedComponents(local_color_mask(image,profile,rules),8)
        self.assertNotEqual(labels[50,25],labels[50,195])

    def test_foreign_hue_and_masked_notebook_rules_cannot_join_the_pen(self):
        hsv=np.zeros((100,220,3),np.uint8);hsv[:,:,2]=255
        hsv[47:54,20:80]=(125,110,110);hsv[47:54,140:200]=(125,110,110)
        hsv[47:54,80:140]=(85,110,110)
        image=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        profile=dict(active=True,chromatic=True,reference_hsv=[125,110,110],hue_tolerance=12)
        rules=np.zeros(image.shape[:2],np.uint8)
        mask=local_color_mask(image,profile,rules)
        self.assertFalse(mask[50,100])
        hsv[47:54,80:140]=(125,110,110);image=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        rules[:,108:112]=255
        _,labels=cv2.connectedComponents(local_color_mask(image,profile,rules),8)
        self.assertNotEqual(labels[50,25],labels[50,195])

    def test_page_deskew_needs_a_consensus_of_real_word_outlines(self):
        def row(angle):return dict(text_angle_degrees=angle,text_aspect_ratio=2.,confidence=.96)
        self.assertIsNone(estimate_angle([row(32)]*3))
        self.assertIsNone(estimate_angle([row(0),row(1),row(-1),row(2)]))
        angle=estimate_angle([row(a) for a in [33,34,35,36,0,-2]])
        self.assertAlmostEqual(angle,34.5,places=1)
        self.assertIsNone(estimate_angle([row(a) for a in [-38,-20,0,20,38]]))

    def test_deskew_maps_evidence_back_to_the_original_photograph(self):
        box=dict(x1=30.,y1=50.,x2=90.,y2=70.)
        matrix=cv2.getRotationMatrix2D((100,100),33,1)
        mapped,polygon=transform_box(box,matrix)
        inverse=cv2.invertAffineTransform(matrix)
        token=dict(text='12k',bbox=mapped,text_polygon=polygon,text_rotation_degrees=90,
                   component_context=dict(detection_id='D'))
        restored=restore([token],inverse,33)[0]
        for key in box:self.assertAlmostEqual(restored['bbox'][key],box[key],places=6)
        self.assertEqual(restored['component_context']['detection_id'],'D')
        self.assertEqual(restored['text_rotation_degrees'],123)

    def test_two_visible_passive_contacts_restore_a_missing_proposal_without_inventing_a_lead(self):
        image=np.full((200,300,3),255,np.uint8)
        cv2.rectangle(image,(100,88),(200,112),(20,20,20),3)
        cv2.line(image,(20,100),(100,100),(20,20,20),3)
        cv2.line(image,(200,100),(280,100),(20,20,20),3)
        sk=_skeletonize(cv2.inRange(image,(0,0,0),(100,100,100)))
        component=dict(component_id='R',type='resistor',bbox=dict(x1=59,y1=60,x2=241,y2=140),
                       terminal_detection={},terminals=[dict(pin='2',position=dict(x=240.,y=100.))])
        result=refine_all_terminal_contacts(image,dict(components=[copy.deepcopy(component)]),{},sk)
        terms=result['components'][0]['terminals']
        self.assertEqual({t['pin'] for t in terms},{'1','2'})
        self.assertTrue(all(t['wire_contact_confirmed'] for t in terms))
        image[95:106,50:105]=255
        sk=_skeletonize(cv2.inRange(image,(0,0,0),(100,100,100)))
        result=refine_all_terminal_contacts(image,dict(components=[component]),{},sk)
        self.assertEqual({t['pin'] for t in result['components'][0]['terminals']},{'2'})

    def test_oversized_detection_frame_does_not_veto_an_independently_read_word(self):
        image=np.full((100,200,3),255,np.uint8)
        detection=dict(bbox=dict(x1=10,y1=10,x2=190,y2=90))
        token=dict(text='R12',bbox=dict(x1=40,y1=30,x2=90,y2=60),component_context=dict(distance=0),confidence=.99)
        self.assertFalse(separate(image,[detection],[token])[0])
        verified=dict(token,word_evidence=[dict(engine='general_word'),dict(engine='circuit_word')])
        self.assertEqual(separate(image,[detection],[verified])[0],[verified])

    def test_power_circle_next_to_a_resistor_uses_the_stem_without_the_resistor_body(self):
        from backend.power_recovery import recover_power
        image=np.full((240,300,3),255,np.uint8)
        cv2.circle(image,(150,60),15,(20,20,20),3)
        cv2.line(image,(150,75),(150,110),(20,20,20),3)
        cv2.rectangle(image,(135,110),(165,160),(20,20,20),3)
        cv2.line(image,(150,160),(150,210),(20,20,20),3)
        detection=dict(detection_id='R',**{'class':'resistor'},bbox=dict(x1=125,y1=100,x2=175,y2=175),confidence=.99)
        found=recover_power(image,[detection],9)
        self.assertEqual(len(found),1)
        self.assertEqual(found[0]['class'],'vcc')
        blank=np.full((240,300,3),255,np.uint8)
        cv2.circle(blank,(150,60),15,(20,20,20),3)
        self.assertEqual(recover_power(blank,[],9),[])

    def test_rectangular_body_and_lead_cannot_become_a_low_confidence_value(self):
        image=np.full((130,260,3),255,np.uint8)
        cv2.rectangle(image,(80,50),(155,85),(20,20,20),4)
        cv2.line(image,(155,68),(225,68),(20,20,20),3)
        detection=dict(bbox=dict(x1=70,y1=40,x2=165,y2=95))
        token=dict(text='05uΩ',bbox=dict(x1=65,y1=35,x2=230,y2=100),component_context=dict(distance=0),confidence=.56)
        self.assertFalse(separate(image,[detection],[token])[0])

    def test_complete_word_can_reject_a_false_frame_around_only_some_of_its_letters(self):
        from backend.ocr import remover_falsos_positivos_textuais
        image=np.full((180,300,3),255,np.uint8)
        cv2.putText(image,'39K',(45,110),cv2.FONT_HERSHEY_SIMPLEX,1.4,(20,20,20),3)
        detection=dict(detection_id='C',**{'class':'capacitor'},bbox=dict(x1=85,y1=70,x2=125,y2=120),confidence=.95)
        token=dict(text='39K',bbox=dict(x1=40,y1=65,x2=160,y2=125),confidence=.999,
                   min_character_confidence=.999,word_evidence=[dict(engine='circuit_word')],independent_readers=1)
        kept,rejected=remover_falsos_positivos_textuais([detection],[token],image)
        self.assertFalse(kept);self.assertEqual(len(rejected),1)

    def test_power_contact_is_on_the_stem_instead_of_a_tight_ring_frame(self):
        from backend.terminal_wire_contacts import _power_lead_contacts
        image=np.full((140,220,3),255,np.uint8)
        cv2.circle(image,(100,50),15,(20,20,20),3)
        cv2.line(image,(100,65),(100,120),(20,20,20),3)
        c=dict(type='vcc',bbox=dict(x1=80,y1=30,x2=120,y2=100))
        ports=[dict(point=np.array(p,dtype=float)) for p in [(83,50),(117,50),(100,99)]]
        selected,proof=_power_lead_contacts(image,c,ports)
        self.assertEqual(len(selected),1);np.testing.assert_allclose(selected[0]['point'],[100,99])

    def test_open_resistor_body_uses_two_real_opposed_leads_and_rejects_a_page_rule(self):
        from backend.resistor_contacts import lead_contacts
        c=dict(type='resistor',bbox=dict(x1=60,y1=60,x2=240,y2=140),
            learned_terminal_geometry=dict(accepted=True,keypoint_confidence=.95,
                points={'1':dict(x=50,y=100),'2':dict(x=250,y=100)}))
        ports=[dict(point=np.array(p,dtype=float),stage8_distance_px=0) for p in [(61,100),(239,100),(90,61)]]
        with patch('backend.resistor_contacts.body_for_box',return_value=None):
            selected,proof=lead_contacts(np.full((180,300,3),255,np.uint8),c,ports)
        self.assertEqual({tuple(p['point']) for p in selected},{(61,100),(239,100)})


if __name__=='__main__':unittest.main()
