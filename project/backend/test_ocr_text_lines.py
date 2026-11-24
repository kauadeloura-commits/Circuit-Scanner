import copy
import unittest
from unittest.mock import Mock,patch
import cv2
import numpy as np
from backend import ocr
from backend.ocr_regions import text_ink_image,text_line_regions,line_crop
from backend.resistor_geometry import cached_vision,symbol_analysis_session
from backend.test_component_models import token,detection


class TextLineReadingTests(unittest.TestCase):
    def test_units_models_and_beta_labels_stay_relevant(self):
        for text in ('1N4001','BC547B','B=58','B=120','100mH','R2 10KOHM','1KOHM','C2 10uF','Led1'):
            with self.subTest(text=text):
                t=token(text,20);t.update(engine='easyocr',confidence=.95)
                self.assertTrue(ocr._token_relevante(t))
        self.assertEqual(ocr._limpar_texto('BC547B'),'BC547B')
        self.assertEqual(ocr._limpar_texto('1N4001'),'1N4001')

    def test_reference_and_value_in_same_phrase_fill_both_fields(self):
        for phrase,kind,name,expected in [('R2 10KOHM','resistor','R2','10000'),('C2 10uF','capacitor','C2','1e-05')]:
            c=ocr.criar_componentes_logicos([token(phrase,20)],[detection('A',kind,20)])['components'][0]
            self.assertEqual(c['reference']['text'],name)
            self.assertEqual(c['value']['normalized'],expected)

    def test_two_beta_labels_assign_to_their_own_transistors(self):
        out=ocr.criar_componentes_logicos([token('B=58',20),token('B=120',230)],
                                         [detection('A','transistor_npn',20),detection('B','transistor_pnp',230)])
        by_id={c['detection_id']:c for c in out['components']}
        self.assertEqual(by_id['A']['electrical_parameters']['hfe']['value'],58)
        self.assertEqual(by_id['B']['electrical_parameters']['hfe']['value'],120)

    def test_three_nested_model_readings_become_one_box_with_evidence(self):
        readings=[]
        for text,confidence,box in [('BC547B',.8,(20,20,180,65)),('8C5478',.7,(27,23,173,63)),('C547',.9,(43,25,154,61))]:
            t=token(text,20,confidence=confidence);t['bbox']=dict(zip(('x1','y1','x2','y2'),box));readings.append(t)
        original=copy.deepcopy(readings);out=ocr._consolidate_text_boxes(readings)
        self.assertEqual(len(out),1);self.assertEqual(out[0]['text'],'BC547B')
        self.assertEqual(len(out[0]['spatial_alternatives']),2);self.assertEqual(readings,original)

    def test_model_and_beta_on_adjacent_lines_remain_separate(self):
        a=token('BC547B',20);a['bbox']=dict(x1=20,y1=20,x2=200,y2=70)
        b=token('B=58',20);b['bbox']=dict(x1=40,y1=62,x2=180,y2=108)
        self.assertEqual(len(ocr._consolidate_text_boxes([a,b])),2)

    def test_distinct_identical_labels_at_different_locations_remain_separate(self):
        self.assertEqual(len(ocr._consolidate_text_boxes([token('10uF',20),token('10uF',230)])),2)

    def test_single_box_keeps_conflicting_units_for_review(self):
        boxes=ocr._consolidate_text_boxes([token('1mOHM',20,confidence=.95),token('1MOHM',20,confidence=.90)])
        self.assertEqual(len(boxes),1)
        c=ocr.criar_componentes_logicos(boxes,[detection('A','resistor',20)])['components'][0]
        self.assertTrue(c['value']['needs_review'])
        self.assertEqual(set(c['value']['alternatives']),{'1mΩ','1MΩ'})

    def test_learned_word_cannot_overwrite_a_conflicting_confident_value(self):
        sample=np.ones((ocr.OCR_SAMPLE_HEIGHT,ocr.OCR_SAMPLE_WIDTH),np.uint8)
        t=token('10uF',20,confidence=.95)
        with patch.object(ocr,'_carregar_indice_amostras',return_value=[dict(file='word.png',label='100nF',kind='value')]), \
                patch.object(ocr.cv2,'imread',return_value=sample),patch.object(ocr,'_normalizar_amostra_ocr',return_value=sample), \
                patch.object(ocr.cv2,'matchTemplate',return_value=np.array([[.95]])):
            result=ocr._aplicar_biblioteca_ocr(np.zeros((200,200,3),np.uint8),[t],[])
        self.assertEqual(result[0]['text'],'10uF')
        self.assertEqual(set(result[0]['numeric_alternatives']),{'10uF','100nF'})

    def test_blue_text_regions_ignore_black_notebook_rules(self):
        image=np.full((480,800,3),210,np.uint8)
        for y in range(60,460,45):cv2.line(image,(0,y),(799,y),(80,80,80),1)
        words=[('1N4001',(40,130)),('100mH',(340,130)),('BC547B',(40,240)),('B=58',(340,240)),('10uF',(40,350))]
        expected=[]
        for text,point in words:
            width,height=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1.1,3)[0]
            cv2.putText(image,text,point,cv2.FONT_HERSHEY_SIMPLEX,1.1,(115,40,35),3)
            expected.append((point[0]+width/2,point[1]-height/2))
        page=text_ink_image(image);regions=text_line_regions(page)
        for point in expected:
            self.assertTrue(any(b['x1']<=point[0]<=b['x2'] and b['y1']<=point[1]<=b['y2'] for b in regions),(point,regions))
        self.assertLess(len(regions),12)

    def test_source_boxes_are_preserved_while_crops_are_deskewed(self):
        page=np.full((110,250),255,np.uint8)
        cv2.putText(page,'100mH',(15,68),cv2.FONT_HERSHEY_SIMPLEX,1.1,0,3)
        page=cv2.warpAffine(page,cv2.getRotationMatrix2D((125,55),8,1),(250,110),borderValue=255)
        box=dict(x1=3,y1=12,x2=230,y2=98);before=dict(box)
        crop,angle=line_crop(page,box)
        self.assertEqual(box,before);self.assertGreater(np.count_nonzero(crop<128),100)
        self.assertLessEqual(abs(angle),15)

    def test_black_annotation_survives_a_blue_component_pen(self):
        image=np.full((300,800,3),230,np.uint8)
        cv2.rectangle(image,(40,50),(100,150),(130,45,35),4)
        cv2.line(image,(0,190),(799,190),(100,100,100),1)
        cv2.putText(image,'100mH',(220,130),cv2.FONT_HERSHEY_SIMPLEX,1.1,(20,20,20),3)
        regions=text_line_regions(text_ink_image(image))
        self.assertTrue(any(b['x1']<300<b['x2'] and b['y1']<120<b['y2'] for b in regions))

    def test_line_recognition_does_not_repeat_the_text_detector(self):
        page=np.full((100,400),255,np.uint8)
        boxes=[dict(x1=10,y1=10,x2=140,y2=50),dict(x1=190,y1=10,x2=330,y2=50)]
        reader=Mock();reader.recognize.side_effect=[[(None,'1N4001',.95)],[(None,'B=120',.95)]]
        with patch.object(ocr,'_carregar_easyocr',return_value=reader):readings=ocr._executar_easyocr_linhas(page,boxes)
        reader.readtext.assert_not_called();reader.detect.assert_not_called()
        self.assertEqual(reader.recognize.call_count,2)
        self.assertEqual([t['bbox'] for t in readings],boxes)

    def test_unclear_reading_is_archived_along_with_beam_result(self):
        reader=Mock();reader.recognize.side_effect=[[(None,'1Nyoo1',.15)],[(None,'1N4001',.82)]]
        page=np.full((100,200),255,np.uint8);boxes=[dict(x1=10,y1=10,x2=140,y2=60)]
        with patch.object(ocr,'_carregar_easyocr',return_value=reader):readings=ocr._executar_easyocr_linhas(page,boxes)
        self.assertEqual([t['raw_text'] for t in readings],['1Nyoo1','1N4001'])

    def test_one_readable_value_does_not_disable_low_coverage_recovery(self):
        image=np.full((300,600,3),255,np.uint8)
        regions=[dict(x1=10+i*50,y1=20,x2=45+i*50,y2=60) for i in range(10)]
        t=token('9V',20);t.update(engine='easyocr',line_region_id=0)
        with patch.object(ocr,'easyocr_disponivel',return_value=True), \
                patch('backend.ocr_regions.text_line_regions',return_value=regions), \
                patch('backend.ocr_context.prioritized_regions',return_value=(regions,[])), \
                patch.object(ocr,'_executar_easyocr_linhas',return_value=[t]), \
                patch.object(ocr,'_executar_easyocr',return_value=[token('10uF',230)]) as global_read, \
                patch.object(ocr,'_detectar_tokens_por_amostras',return_value=[]), \
                patch.object(ocr,'_aplicar_biblioteca_ocr',side_effect=lambda image,tokens,dets:tokens):
            result,_=ocr.extrair_tokens_ocr(image,None,[detection('A','battery',20)])
        global_read.assert_called_once()
        self.assertEqual({t['text'] for t in result},{'9V','10uF'})

    def test_pure_vision_cache_is_scoped_copy_safe_and_detects_mutation(self):
        calls=[]
        @cached_vision
        def analyze(image):calls.append(1);return dict(mask=image.copy())
        @symbol_analysis_session
        def run(image):
            expected=int(image[0,0])
            a=analyze(image);a['mask'][0,0]=99
            b=analyze(image);self.assertEqual(b['mask'][0,0],expected)
            image[0,0]=expected+1;self.assertEqual(analyze(image)['mask'][0,0],expected+1)
        image=np.ones((12,15),np.uint8);run(image);self.assertEqual(len(calls),2)
        run(image);self.assertEqual(len(calls),4)


if __name__=='__main__':unittest.main()
