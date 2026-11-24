import copy
import random
import unittest
import cv2
import numpy as np
from backend.dataset_generator import _draw_symbol, _bbox_from_mask
from backend.recognition import filtrar_falsos_positivos_geometricos
from backend.ocr import remover_falsos_positivos_textuais


def detection(image, cls, manual=False):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    y,x=np.where(gray<150)
    return dict(detection_id='DET_001',class_id=0,**{'class':cls},confidence=1.,
                consensus_votes=4,consensus_source_count=4,best_view_confidence=1.,manual_reviewed=manual,
                bbox=dict(x1=float(x.min()-5),y1=float(y.min()-5),x2=float(x.max()+6),y2=float(y.max()+6)))


class SymbolTextGuardTests(unittest.TestCase):
    def test_values_and_reference_letters_never_become_coils_by_consensus(self):
        for text in ('100nF','200','10k','L1','WWW','MMMM','ED','0000'):
            page=np.full((260,420,3),255,np.uint8)
            cv2.putText(page,text,(55,145),cv2.FONT_HERSHEY_SIMPLEX,1.4,(20,20,20),3,cv2.LINE_AA)
            for turn in range(4):
                with self.subTest(text=text,rotation=90*turn):
                    image=np.ascontiguousarray(np.rot90(page,turn))
                    kept,rejected=filtrar_falsos_positivos_geometricos(image,[detection(image,'inductor')])
                    self.assertEqual(kept,[])
                    self.assertTrue(rejected[0]['geometry_validation']['evidence']['hard_text_rejection'])

    def test_zero_and_letter_o_are_not_sources(self):
        for text in ('0','O','8','B','VCC'):
            image=np.full((240,360,3),255,np.uint8)
            cv2.putText(image,text,(65,155),cv2.FONT_HERSHEY_SIMPLEX,2.,(20,20,20),3,cv2.LINE_AA)
            kept,_=filtrar_falsos_positivos_geometricos(image,[detection(image,'source_dc')])
            self.assertEqual(kept,[],text)

    def test_isolated_aligned_letters_rejected_for_all_structural_classes(self):
        image=np.full((260,420,3),255,np.uint8)
        cv2.putText(image,'ABC123',(40,145),cv2.FONT_HERSHEY_SIMPLEX,1.4,(20,20,20),3,cv2.LINE_AA)
        for cls in ('resistor','capacitor','capacitor_polarized','battery','gnd','vcc','diode','led',
                    'opamp','transistor_npn','transistor_pnp','mosfet_n','mosfet_p','source_dc','inductor'):
            with self.subTest(cls=cls):
                kept,rejected=filtrar_falsos_positivos_geometricos(image,[detection(image,cls)])
                self.assertEqual(kept,[])
                self.assertEqual(rejected[0]['rejection_reason'],'isolated_letter_number_group')

    def test_real_coils_and_sources_with_adjacent_values_survive_rotations(self):
        for cls in ('inductor','source_dc'):
            drawn,mask=_draw_symbol(cls,256,random.Random(34))
            page=np.full((400,400,3),255,np.uint8)
            page[72:328,72:328]=cv2.cvtColor(drawn,cv2.COLOR_GRAY2BGR)
            cv2.putText(page,'100mH' if cls=='inductor' else '10V',(140,100),cv2.FONT_HERSHEY_SIMPLEX,.7,(20,20,20),2)
            for angle in (0,45,90,135,180,270):
                with self.subTest(cls=cls,angle=angle):
                    matrix=cv2.getRotationMatrix2D((200,200),angle,1.)
                    image=cv2.warpAffine(page,matrix,(400,400),borderValue=(255,255,255))
                    rotated_mask=cv2.warpAffine(np.pad(mask,((72,72),(72,72))),matrix,(400,400))
                    x1,y1,x2,y2=_bbox_from_mask(rotated_mask)
                    d=detection(image,cls); d['bbox']=dict(x1=x1,y1=y1,x2=x2,y2=y2)
                    kept,rejected=filtrar_falsos_positivos_geometricos(image,[d])
                    self.assertEqual(len(kept),1,rejected)

    def test_ocr_can_remove_a_single_digit_or_letter(self):
        for text in ('0','8','R','A','100'):
            image=np.full((220,300,3),255,np.uint8)
            cv2.putText(image,text,(70,145),cv2.FONT_HERSHEY_SIMPLEX,1.4,(20,20,20),3,cv2.LINE_AA)
            d=detection(image,'inductor')
            token=dict(text=text,confidence=.95,engine='easyocr',bbox=d['bbox'],evidence_count=2)
            kept,rejected=remover_falsos_positivos_textuais([d],[token],image)
            self.assertEqual(kept,[])
            self.assertEqual(len(rejected),1)

    def test_ocr_misreading_a_real_source_as_zero_preserves_it(self):
        drawn,mask=_draw_symbol('source_dc',256,random.Random(34))
        image=cv2.cvtColor(drawn,cv2.COLOR_GRAY2BGR)
        d=detection(image,'source_dc')
        token=dict(text='0',confidence=.95,engine='easyocr',bbox=d['bbox'],evidence_count=2)
        kept,rejected=remover_falsos_positivos_textuais([d],[token],image)
        self.assertEqual(len(kept),1)
        self.assertEqual(rejected,[])

    def test_manual_corrections_are_preserved(self):
        image=np.full((220,300,3),255,np.uint8)
        cv2.putText(image,'L1',(70,145),cv2.FONT_HERSHEY_SIMPLEX,1.4,(20,20,20),3)
        d=detection(image,'inductor',True)
        kept,rejected=filtrar_falsos_positivos_geometricos(image,[copy.deepcopy(d)])
        self.assertEqual(len(kept),1)
        self.assertEqual(rejected,[])
        token=dict(text='L1',confidence=.95,engine='easyocr',bbox=d['bbox'])
        kept,rejected=remover_falsos_positivos_textuais([d],[token],image)
        self.assertEqual(len(kept),1)
        self.assertEqual(rejected,[])

    def test_bent_wires_next_to_real_coil_do_not_destroy_its_body(self):
        drawn,mask=_draw_symbol('inductor',256,random.Random(34))
        image=cv2.cvtColor(drawn,cv2.COLOR_GRAY2BGR)
        cv2.line(image,(28,128),(28,210),(20,20,20),3)
        cv2.line(image,(228,128),(228,55),(20,20,20),3)
        x1,y1,x2,y2=_bbox_from_mask(mask)
        d=detection(image,'inductor');d['bbox']=dict(x1=x1,y1=y1,x2=x2,y2=y2)
        kept,rejected=filtrar_falsos_positivos_geometricos(image,[d])
        self.assertEqual(len(kept),1,rejected)
        token=dict(text='L1',confidence=.95,engine='easyocr',bbox=d['bbox'])
        kept,rejected=remover_falsos_positivos_textuais([d],[token],image)
        self.assertEqual(len(kept),1)
        self.assertEqual(rejected,[])
