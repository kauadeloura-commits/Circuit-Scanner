"""Regression cases for terminal placement from photographed border leads."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import cv2
import numpy as np
from backend.terminal_wire_contacts import refine_all_terminal_contacts,constrain_terminal_positions
from backend.wire_detection import _skeletonize
from backend.test_npn_wire_contacts import fixture
from backend.npn_wire_contacts import detect_npn_wire_contacts
from backend import main
from backend.test_terminal_identity import fixture as polar_fixture


class BoundaryWireContactsTests(unittest.TestCase):
    def test_recalculating_wires_ignores_stale_final_automatic_points(self):
        image,bbox,_=polar_fixture('opamp')
        component=dict(component_id='U',type='opamp',bbox=bbox,visual_confidence=.99)
        info={'components':[component]}
        with TemporaryDirectory() as folder,patch.object(main,'PROJECTS_DIR',Path(folder)), \
             patch.object(main,'atualizar_terminais'),patch.object(main,'atualizar_terminais_pos_fios'),patch.object(main,'atualizar_fios'):
            project=Path(folder)/'CASE';project.mkdir();cv2.imwrite(str(project/'original.png'),image)
            state=dict(stored_filename='original.png',ocr_info=json.dumps(info))
            first=main._executar_etapa8_fios('CASE',state)
            mask=(project/'wires_mask.png').read_bytes();sk=(project/'wires_skeleton.png').read_bytes()
            stale=json.loads((project/'terminals.json').read_text())
            for c in stale['components']:
                for terminal in c['terminals']:terminal['position']=dict(x=270,y=270)
            state['terminals_info']=json.dumps(stale)
            second=main._executar_etapa8_fios('CASE',state)
            self.assertEqual((project/'wires_mask.png').read_bytes(),mask)
            self.assertEqual((project/'wires_skeleton.png').read_bytes(),sk)
            self.assertEqual(first['segments'],second['segments'])

    def test_curved_leads_replace_empty_model_points_in_rotations(self):
        base=np.full((240,300,3),245,np.uint8)
        cv2.rectangle(base,(115,90),(180,160),(20,20,20),3)
        cv2.polylines(base,[np.array([(20,100),(70,100),(95,135),(115,135)],np.int32)],False,(20,20,20),3)
        cv2.polylines(base,[np.array([(180,135),(205,135),(225,115),(280,115)],np.int32)],False,(20,20,20),3)
        b=dict(x1=80,y1=75,x2=215,y2=180)
        points=[np.array([81.,115.]),np.array([214.,126.])]
        for k in range(4):
            with self.subTest(rotation=k):
                image=np.ascontiguousarray(np.rot90(base,k));h,w=base.shape[:2]
                def rotate(p):
                    x,y=p;hh,ww=h,w
                    for _ in range(k):x,y=y,ww-1-x;hh,ww=ww,hh
                    return np.array([x,y])
                corners=np.array([rotate((x,y)) for x in (b['x1'],b['x2']) for y in (b['y1'],b['y2'])])
                box=dict(x1=float(corners[:,0].min()),x2=float(corners[:,0].max()),y1=float(corners[:,1].min()),y2=float(corners[:,1].max()))
                anchors=[rotate((95,110)),rotate((200,155))]
                c=dict(type='resistor',component_id='R',bbox=box,terminal_detection={},
                       terminals=[dict(pin=str(i+1),position=dict(x=float(p[0]),y=float(p[1]))) for i,p in enumerate(anchors)])
                before={'segments':[{'path':[[20,100],[70,100]]}]};unchanged=copy.deepcopy(before)
                sk=_skeletonize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<100);original=sk.copy()
                final=refine_all_terminal_contacts(image,{'components':[c]},before,sk)
                self.assertEqual(before,unchanged);self.assertTrue(np.array_equal(sk,original))
                terms=final['components'][0]['terminals'];self.assertEqual(len(terms),2)
                for t in terms:
                    p=t['position'];self.assertLess(int(image[round(p['y']),round(p['x'])].min()),100)
                    self.assertTrue(box['x1']<p['x']<box['x2'] and box['y1']<p['y']<box['y2'])

    def test_a_filtered_graph_lead_is_recovered_from_the_source_ink(self):
        image,sk,c,_=fixture(notebook=False)
        sk[:,240:]=0
        found,meta=detect_npn_wire_contacts(image,c['bbox'],sk)
        self.assertIsNotNone(found,meta)
        ports,_=found
        self.assertEqual(len(ports),3)
        self.assertTrue(any(not p['stage8_context_confirmed'] for p in ports))
        for port in ports:
            x,y=map(round,port['point']);self.assertLess(int(image[y,x].min()),100)

    def test_final_point_in_empty_paper_is_snapped_even_when_already_inside(self):
        image=np.full((100,120,3),255,np.uint8)
        cv2.line(image,(5,50),(115,50),(20,20,20),3)
        info={'components':[dict(type='resistor',bbox=dict(x1=20,y1=20,x2=100,y2=80),
            terminals=[dict(pin='1',position=dict(x=25,y=57))])]}
        constrain_terminal_positions(info,image,require_ink=True)
        p=info['components'][0]['terminals'][0]['position']
        self.assertLess(int(image[round(p['y']),round(p['x'])].min()),100)

    def test_final_point_with_no_observed_ink_is_not_published(self):
        info={'components':[dict(type='resistor',bbox=dict(x1=20,y1=20,x2=100,y2=80),
            terminals=[dict(pin='1',position=dict(x=25,y=57))])]}
        constrain_terminal_positions(info,np.full((100,120,3),255,np.uint8),require_ink=True)
        self.assertEqual(info['terminal_count'],0)

    def test_black_rule_cannot_supply_a_blue_circuit_terminal(self):
        image=np.full((140,200,3),255,np.uint8)
        blue=(130,35,20)
        cv2.line(image,(0,80),(199,80),blue,4)
        cv2.rectangle(image,(80,65),(120,95),blue,4)
        cv2.line(image,(0,40),(199,40),(20,20,20),2)
        c=dict(component_id='R',type='resistor',visual_confidence=.99,terminal_detection={},
            bbox=dict(x1=60,y1=25,x2=140,y2=110),terminals=[
                dict(pin='1',position=dict(x=62,y=40)),dict(pin='2',position=dict(x=138,y=40))])
        final=refine_all_terminal_contacts(image,{'components':[c]}, {},_skeletonize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<150))
        terms=final['components'][0]['terminals'];self.assertEqual(len(terms),2)
        for t in terms:self.assertAlmostEqual(t['position']['y'],80,delta=3)


if __name__=='__main__':unittest.main()
