import random
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from backend import recognition as r
from backend.dataset_generator import _draw_symbol,_bbox_from_mask


def iec_case(angle=0,size=(64,34),color=(20,20,20),paper=255,scale=1):
    image=np.full((320,420,3),paper,np.uint8);cx,cy=210,160;length,width=size
    cv2.rectangle(image,(cx-length//2,cy-width//2),(cx+length//2,cy+width//2),color,3)
    cv2.line(image,(cx-length//2-45,cy),(cx-length//2,cy),color,3)
    cv2.line(image,(cx+length//2,cy),(cx+length//2+45,cy),color,3)
    matrix=cv2.getRotationMatrix2D((cx,cy),angle,scale)
    return cv2.warpAffine(image,matrix,(420,320),borderValue=(paper,paper,paper))


class ResistorVCCRecognitionTests(unittest.TestCase):
    def test_auto_pipeline_recovers_both_resistor_shapes_when_model_returns_none(self):
        images=[iec_case(18)]
        zigzag=np.full((320,420,3),255,np.uint8)
        points=np.array([(50,160),(90,160),(105,135),(125,185),(145,135),(165,185),(185,135),(205,160),(260,160)],np.int32)
        cv2.polylines(zigzag,[points],False,(20,20,20),3)
        images.append(zigzag)
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            with patch.object(r,'_carregar_modelo',return_value=object()),patch.object(r,'_executar_predicao',return_value=[]):
                for index,image in enumerate(images):
                    path=folder/f'input{index}.png';cv2.imwrite(str(path),image)
                    result=r.detectar_componentes(path,folder/f'annotated{index}.png',folder/f'result{index}.json')
                    self.assertEqual(result['count'],1)
                    self.assertEqual(result['detections'][0]['class'],'resistor')
                    self.assertTrue(result['detections'][0]['geometry_validation']['valid'])

    def test_iec_square_and_inclined_bodies_have_two_opposed_leads(self):
        for size in ((42,42),(72,30),(40,24)):
            for angle in (0,18,-32,90,137):
                with self.subTest(size=size,angle=angle):
                    image=iec_case(angle,size)
                    candidates=r._deteccoes_resistores_retangulares(image)
                    self.assertTrue(candidates)
                    self.assertTrue(r._validar_geometria_resistor(image,candidates[0]['bbox'])[0])

    def test_blue_and_light_pencil_under_gray_paper_are_recognized(self):
        for color,paper in (((125,35,35),170),((70,70,70),145),((175,175,175),230)):
            image=iec_case(14,(60,30),color,paper)
            self.assertTrue(r._deteccoes_resistores_retangulares(image),(color,paper))

    def test_small_page_component_is_not_limited_by_page_size(self):
        small=iec_case(0,(40,22));image=np.full((1500,2000,3),255,np.uint8)
        image[600:920,800:1220]=small
        candidates=r._deteccoes_resistores_retangulares(image)
        self.assertTrue(candidates)
        self.assertLess(candidates[0]['bbox']['width'],90)

    def test_zigzag_is_detected_without_a_model_at_rotations_and_scales(self):
        for angle,scale in ((0,1),(90,1),(18,.8),(-25,1.2)):
            image=np.full((320,420,3),255,np.uint8)
            points=np.array([(50,160),(90,160),(105,135),(125,185),(145,135),(165,185),(185,135),(205,160),(260,160)],np.int32)
            cv2.polylines(image,[points],False,(20,20,20),3)
            image=cv2.warpAffine(image,cv2.getRotationMatrix2D((210,160),angle,scale),(420,320),borderValue=(255,255,255))
            with self.subTest(angle=angle,scale=scale):
                self.assertTrue(r._deteccoes_resistores_geometricos(image))

    def test_text_circles_triangles_and_bus_loops_are_not_resistors(self):
        for shape in ('text','circle','triangle','loop','one_lead','plain_wire'):
            image=np.full((400,600,3),255,np.uint8)
            if shape=='text':cv2.putText(image,'R W M ED 10k',(70,200),cv2.FONT_HERSHEY_SIMPLEX,1.5,(20,20,20),3)
            elif shape=='circle':
                cv2.circle(image,(230,200),28,(20,20,20),3)
                cv2.line(image,(120,200),(202,200),(20,20,20),3);cv2.line(image,(258,200),(350,200),(20,20,20),3)
            elif shape=='triangle':
                cv2.polylines(image,[np.array([(180,150),(180,250),(270,200)],np.int32)],True,(20,20,20),3)
                cv2.line(image,(100,200),(180,200),(20,20,20),3);cv2.line(image,(270,200),(350,200),(20,20,20),3)
            elif shape=='loop':
                cv2.rectangle(image,(150,40),(300,360),(20,20,20),3)
                cv2.line(image,(80,200),(150,200),(20,20,20),3);cv2.line(image,(300,200),(380,200),(20,20,20),3)
            elif shape=='one_lead':
                cv2.rectangle(image,(180,170),(250,230),(20,20,20),3);cv2.line(image,(90,200),(180,200),(20,20,20),3)
            else:cv2.line(image,(80,200),(400,200),(20,20,20),3)
            with self.subTest(shape=shape):
                self.assertEqual(r._deteccoes_resistores_retangulares(image),[])
                self.assertEqual(r._deteccoes_resistores_geometricos(image),[])

    def test_noisy_rule_grid_does_not_create_resistors(self):
        image=np.full((600,800,3),215,np.uint8)
        for x in range(30,800,55):cv2.line(image,(x,0),(x,599),(165,165,165),2)
        for y in range(30,600,55):cv2.line(image,(0,y),(799,y),(165,165,165),2)
        self.assertEqual(r._deteccoes_resistores_retangulares(image),[])

    def test_oval_symbols_split_by_a_rule_do_not_become_resistors(self):
        for axes in ((30,20),(40,18),(50,15),(20,20)):
            image=np.full((320,420,3),230,np.uint8)
            cv2.line(image,(210,0),(210,319),(180,180,180),2)
            cv2.ellipse(image,(210,160),axes,0,0,360,(20,20,20),3)
            cv2.line(image,(100,160),(210-axes[0],160),(20,20,20),3)
            cv2.line(image,(210+axes[0],160),(320,160),(20,20,20),3)
            self.assertEqual(r._deteccoes_resistores_retangulares(image),[],axes)

    def test_body_tolerates_small_inward_hand_drawn_notches(self):
        image=np.full((320,420,3),230,np.uint8)
        points=np.array([(174,141),(245,141),(241,150),(245,176),(174,178),(177,167),(174,155)],np.int32)
        cv2.polylines(image,[points],True,(20,20,20),3)
        cv2.line(image,(110,160),(175,160),(20,20,20),3)
        cv2.line(image,(245,160),(310,160),(20,20,20),3)
        self.assertTrue(r._deteccoes_resistores_retangulares(image))

    def test_three_branch_node_is_rejected_at_every_rotation_and_confidence(self):
        for angle in (0,90,180,270):
            image=np.full((260,260,3),255,np.uint8)
            for end in ((30,130),(230,130),(130,230)):cv2.line(image,(130,130),end,(20,20,20),3)
            cv2.circle(image,(130,130),7,(20,20,20),-1)
            image=np.rot90(image,angle//90).copy();box=dict(x1=116,y1=116,x2=144,y2=144)
            with self.subTest(angle=angle):
                kept,rejected=r.filtrar_falsos_positivos_geometricos(image,[dict(class_id=0,**{'class':'vcc'},bbox=box,confidence=1,consensus_votes=12,consensus_source_count=8)])
                self.assertEqual(kept,[]);self.assertTrue(rejected[0]['geometry_validation']['evidence']['hard_topology_rejection'])

    def test_hollow_power_terminal_accepts_one_lead_and_rejects_three(self):
        for number in (1,2,3):
            image=np.full((260,260,3),255,np.uint8);cv2.circle(image,(130,130),20,(20,20,20),3)
            for end,start in list(zip(((30,130),(230,130),(130,230)),((110,130),(150,130),(130,150))))[:number]:cv2.line(image,start,end,(20,20,20),3)
            ok,_,_=r._validar_classe_geometrica(image,dict(x1=105,y1=105,x2=155,y2=155),'vcc')
            self.assertEqual(ok,number==1)

    def test_bjt_and_opamp_with_three_leads_never_become_vcc(self):
        for kind in ('transistor_npn','transistor_pnp','opamp'):
            image,mask=_draw_symbol(kind,256,random.Random(7));image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
            x1,y1,x2,y2=_bbox_from_mask(mask);box=dict(x1=x1,y1=y1,x2=x2,y2=y2)
            for turn in range(4):
                corners=np.array([[x,y] for x in (x1,x2) for y in (y1,y2)])
                for _ in range(turn):corners=np.column_stack((corners[:,1],255-corners[:,0]))
                box=dict(x1=corners[:,0].min(),y1=corners[:,1].min(),x2=corners[:,0].max(),y2=corners[:,1].max())
                with self.subTest(kind=kind,turn=turn):
                    self.assertFalse(r._validar_classe_geometrica(np.rot90(image,turn).copy(),box,'vcc')[0])

    def test_vcc_pin_library_contains_only_one_pin(self):
        from backend.terminal_detection import PIN_LIBRARY,detectar_terminais_componentes
        self.assertEqual(len(PIN_LIBRARY['vcc']),1)
        result=detectar_terminais_componentes({'components':[dict(component_id='P',type='vcc',bbox=dict(x1=50,y1=50,x2=100,y2=100))]},150,150)
        self.assertEqual(len(result['components'][0]['terminals']),1)

if __name__=='__main__':unittest.main()
