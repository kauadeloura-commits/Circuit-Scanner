import random, unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from backend import recognition as r
import cv2
import numpy as np
from backend.dataset_generator import _draw_symbol,_bbox_from_mask
from backend.test_symbol_text_guard import detection
from backend.recognition import filtrar_falsos_positivos_geometricos,_deteccoes_capacitores_geometricos
from backend.ocr import remover_falsos_positivos_textuais

class CapacitorTextGuardTests(unittest.TestCase):
    def test_single_letters_and_digits_rejected_despite_high_consensus(self):
        for text in ('h','H','n','N','11','II','1','4','F','00','OO','22','33'):
            for angle in (0,23,45,90,180,270):
                with self.subTest(text=text,angle=angle):
                    im=np.full((260,320,3),255,np.uint8)
                    cv2.putText(im,text,(100,150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(20,20,20),3,cv2.LINE_AA)
                    im=cv2.warpAffine(im,cv2.getRotationMatrix2D((160,130),angle,1),(320,260),borderValue=(255,255,255))
                    for cls in ('capacitor','capacitor_polarized'):
                        kept,rejected=filtrar_falsos_positivos_geometricos(im,[detection(im,cls)])
                        self.assertEqual(kept,[],(text,angle,cls))
                        self.assertTrue(rejected[0]['geometry_validation']['evidence']['hard_text_rejection'])
    def test_real_capacitors_with_values_color_and_rotations_survive(self):
        for cls in ('capacitor','capacitor_polarized'):
            drawn,mask=_draw_symbol(cls,256,random.Random(34));page=np.full((400,400,3),255,np.uint8)
            page[72:328,72:328]=cv2.cvtColor(drawn,cv2.COLOR_GRAY2BGR)
            cv2.putText(page,'10nF',(145,100),cv2.FONT_HERSHEY_SIMPLEX,.7,(20,20,20),2)
            for blue in (False,True):
                colored=page.copy()
                if blue:colored[np.min(page,axis=2)<200]=(160,50,25)
                for angle in (0,18,45,90,135,180,270):
                    with self.subTest(cls=cls,blue=blue,angle=angle):
                        m=cv2.getRotationMatrix2D((200,200),angle,1)
                        im=cv2.warpAffine(colored,m,(400,400),borderValue=(255,255,255))
                        mm=cv2.warpAffine(np.pad(mask,((72,72),(72,72))),m,(400,400))
                        x1,y1,x2,y2=_bbox_from_mask(mm);d=detection(im,cls);d['bbox']=dict(x1=x1,y1=y1,x2=x2,y2=y2)
                        kept,rejected=filtrar_falsos_positivos_geometricos(im,[d]);self.assertEqual(len(kept),1,rejected)
    def test_blue_capacitor_over_dark_notebook_rule_survives(self):
        drawn,mask=_draw_symbol('capacitor',256,random.Random(34));im=np.full((256,256,3),255,np.uint8)
        cv2.line(im,(0,128),(255,128),(100,100,100),1);im[drawn<180]=(170,45,20)
        x1,y1,x2,y2=_bbox_from_mask(mask);d=detection(im,'capacitor');d['bbox']=dict(x1=x1,y1=y1,x2=x2,y2=y2)
        self.assertEqual(len(filtrar_falsos_positivos_geometricos(im,[d])[0]),1)
    def test_ocr_h_removes_glyph_but_preserves_real_capacitor(self):
        glyph=np.full((256,256,3),255,np.uint8);cv2.putText(glyph,'h',(90,150),cv2.FONT_HERSHEY_SIMPLEX,2,(20,20,20),3)
        drawn,_=_draw_symbol('capacitor',256,random.Random(34))
        for im,expected in ((glyph,0),(cv2.cvtColor(drawn,cv2.COLOR_GRAY2BGR),1)):
            d=detection(im,'capacitor');token=dict(text='h',confidence=.99,engine='easyocr',bbox=d['bbox'],evidence_count=3)
            kept,_=remover_falsos_positivos_textuais([d],[token],im);self.assertEqual(len(kept),expected)
    def test_manual_component_is_not_removed(self):
        im=np.full((256,256,3),255,np.uint8);cv2.putText(im,'h',(90,150),cv2.FONT_HERSHEY_SIMPLEX,2,(20,20,20),3)
        self.assertEqual(len(filtrar_falsos_positivos_geometricos(im,[detection(im,'capacitor',True)])[0]),1)

    def test_capacitor_paths_may_reconnect_outside_detection_box(self):
        drawn,_=_draw_symbol('capacitor',256,random.Random(34));im=cv2.cvtColor(drawn,cv2.COLOR_GRAY2BGR)
        cv2.line(im,(28,128),(28,65),(20,20,20),3);cv2.line(im,(28,65),(228,65),(20,20,20),3);cv2.line(im,(228,65),(228,128),(20,20,20),3)
        d=detection(im,'capacitor');d['bbox']=dict(x1=88,y1=80,x2=168,y2=177)
        self.assertEqual(len(filtrar_falsos_positivos_geometricos(im,[d])[0]),1)
    def test_model_prediction_on_h_is_removed_by_full_pipeline(self):
        im=np.full((260,320,3),255,np.uint8);cv2.putText(im,'h',(100,150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(20,20,20),3)
        with tempfile.TemporaryDirectory() as directory:
            f=Path(directory);cv2.imwrite(str(f/'original.png'),im)
            with patch.object(r,'_carregar_modelo',return_value=object()),patch.object(r,'_executar_predicao',return_value=[detection(im,'capacitor')]):
                result=r.detectar_componentes(f/'original.png',f/'annotated.png',f/'result.json')
            self.assertEqual(result['detections'],[])
