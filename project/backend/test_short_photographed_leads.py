import unittest
import cv2,numpy as np
from backend.circuit_connectivity import construir_conectividade
from backend.test_circuit_connectivity import component

class ShortPhotographedLeadTests(unittest.TestCase):
    def test_overlapping_boxes_recover_only_the_visible_short_external_lead(self):
        image=np.full((200,320,3),255,np.uint8)
        cv2.line(image,(150,100),(165,100),(20,20,20),5)
        a=component('A','transistor_npn',(150,40,260,160),{'B':(151,100),'C':(205,41),'E':(205,159)})
        b=component('B','transistor_pnp',(50,40,166,160),{'B':(51,100),'C':(165,100),'E':(110,159)})
        graph=dict(nodes=[],segments=[],stage9={'nodes':[]})
        def linked():
            result=construir_conectividade(image,graph,{'components':[a,b]})
            mapping={r['terminal_ref']:r['net'] for r in result['terminal_connections']}
            return mapping['A:B']==mapping['B:C']
        self.assertTrue(linked())
        image[94:107,156:160]=255;self.assertFalse(linked())

    def test_a_thick_elbow_is_continuous_but_blank_paper_is_not(self):
        image=np.full((240,280,3),255,np.uint8)
        cv2.line(image,(100,30),(100,150),(20,20,20),6);cv2.line(image,(100,150),(240,150),(20,20,20),6)
        graph=dict(nodes=[],segments=[dict(wire_id='V',points=[dict(x=100,y=30),dict(x=100,y=142)]),dict(wire_id='H',points=[dict(x=103,y=151),dict(x=240,y=150)])],stage9={'nodes':[]})
        terms={'components':[component('A','vcc',(80,15,120,31),{'VCC':(100,30)})]}
        self.assertEqual(len(construir_conectividade(image,graph,terms)['recovered_wire_bridges']),1)
        image[145:149,94:108]=255
        self.assertFalse(construir_conectividade(image,graph,terms)['recovered_wire_bridges'])

if __name__=='__main__':unittest.main()
