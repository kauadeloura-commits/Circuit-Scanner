import unittest
from unittest.mock import patch,Mock
import cv2
import numpy as np
from backend import ocr
from backend.ocr_context import repair_gain_label,prioritized_regions,equals_number_box,soft_text_page
from backend.test_component_models import token,detection
from backend.thinning import thin_binary


class ContextOCRTests(unittest.TestCase):
    def test_pe58_recovers_beta_only_beside_a_bjt_and_preserves_raw(self):
        t=token('Pe58',20);t.update(confidence=.7676,engine='easyocr')
        detections=[detection('A','transistor_npn',20)]
        corrected=repair_gain_label(t,detections)
        self.assertEqual(corrected['text'],'β=58');self.assertEqual(corrected['raw_text'],'Pe58')
        self.assertEqual(t['text'],'Pe58')
        c=ocr.criar_componentes_logicos([corrected],detections)['components'][0]
        self.assertEqual(c['electrical_parameters']['hfe']['value'],58)
        self.assertTrue(c['electrical_parameters']['hfe']['needs_review'])

    def test_numeric_confusions_repair_only_after_the_explicit_label(self):
        t=repair_gain_label(token('P=I2O',20),[detection('A','transistor_pnp',20)])
        self.assertEqual(t['text'],'β=120')
        for text in ('BC547B','1N4001','P=Ixo','P=120mW','PE','58'):
            with self.subTest(text=text):self.assertEqual(repair_gain_label(token(text,20),[detection('A','transistor_npn',20)])['text'],text)

    def test_no_beta_repair_on_a_resistor_mosfet_or_remote_text(self):
        for kind in ('resistor','mosfet_n','capacitor'):
            self.assertEqual(repair_gain_label(token('Pe58',20),[detection('A',kind,20)])['text'],'Pe58')
        self.assertEqual(repair_gain_label(token('Pe58',900),[detection('A','transistor_npn',20)])['text'],'Pe58')

    def test_mixed_overlapping_component_classes_do_not_justify_repair(self):
        self.assertEqual(repair_gain_label(token('Pe58',20),[detection('A','transistor_npn',20),detection('B','resistor',20)])['text'],'Pe58')

    def test_component_neighbor_receives_priority_over_top_left_decoration(self):
        page=np.full((220,700),255,np.uint8)
        far=dict(x1=10,y1=10,x2=180,y2=65);near=dict(x1=260,y1=110,x2=440,y2=165)
        for b in (far,near):cv2.putText(page,'100mH',(b['x1']+5,b['y2']-10),cv2.FONT_HERSHEY_SIMPLEX,.7,0,2)
        ordered,_=prioritized_regions(page,[far,near],[detection('A','inductor',260)])
        self.assertEqual(ordered[0]['x1'],near['x1'])

    def test_single_number_or_unit_near_a_component_is_not_silently_dropped(self):
        page=np.full((150,300),255,np.uint8);box=dict(x1=50,y1=45,x2=75,y2=90)
        cv2.putText(page,'H',(50,82),cv2.FONT_HERSHEY_SIMPLEX,1,0,2)
        ordered,_=prioritized_regions(page,[box],[detection('A','inductor',20)])
        self.assertEqual(len(ordered),1)

    def test_equal_geometry_produces_only_a_numeric_crop(self):
        page=np.full((90,250),255,np.uint8)
        cv2.putText(page,'B',(10,62),cv2.FONT_HERSHEY_SIMPLEX,1.3,0,3)
        cv2.line(page,(55,35),(85,35),0,3);cv2.line(page,(55,49),(85,49),0,3)
        cv2.putText(page,'120',(105,62),cv2.FONT_HERSHEY_SIMPLEX,1.3,0,3)
        box=dict(x1=0,y1=10,x2=240,y2=75)
        cropped=equals_number_box(page,box)
        self.assertIsNotNone(cropped);self.assertGreater(cropped['x1'],85);self.assertLess(cropped['x1'],105)
        cv2.line(page,(55,35),(55,49),0,3)
        self.assertIsNone(equals_number_box(page,box))

    def test_gain_digits_are_reread_with_an_equal_and_bjt_context(self):
        page=np.full((90,250),255,np.uint8)
        cv2.putText(page,'B',(10,62),cv2.FONT_HERSHEY_SIMPLEX,1.3,0,3)
        cv2.line(page,(55,35),(85,35),0,3);cv2.line(page,(55,49),(85,49),0,3)
        cv2.putText(page,'120',(105,62),cv2.FONT_HERSHEY_SIMPLEX,1.3,0,3)
        box=dict(x1=0,y1=10,x2=240,y2=75,component_context=dict(kind='transistor_npn',distance=0,ambiguous=False))
        reader=Mock();reader.recognize.side_effect=[[(None,'P= Ixo',.3)],[(None,'120',.96)]]
        with patch.object(ocr,'_carregar_easyocr',return_value=reader):result=ocr._executar_easyocr_linhas(page,[box])
        self.assertEqual(result[-1]['text'],'β=120');self.assertEqual(result[-1]['raw_text'],'P= Ixo')
        self.assertEqual(reader.recognize.call_args.kwargs['allowlist'],'0123456789.,')

    def test_soft_rendering_keeps_stroke_shades_and_white_background(self):
        image=np.full((50,100,3),225,np.uint8);cv2.line(image,(15,25),(80,25),(110,65,40),3)
        binary=np.where(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<150,0,255).astype(np.uint8)
        page=soft_text_page(image,binary)
        self.assertEqual(page[0,0],255);self.assertLess(page[25,40],60)

    def test_lookup_thinning_handles_empty_and_border_masks(self):
        for shape in ((0,0),(1,10),(10,1),(2,2)):
            self.assertFalse(np.any(thin_binary(np.ones(shape,np.uint8))))


if __name__=='__main__':unittest.main()
