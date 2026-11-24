"""Glyph ink and glyph skeleton endpoints must not become wire contacts."""
import copy
import unittest
import cv2
import numpy as np
from backend.terminal_wire_contacts import refine_all_terminal_contacts
from backend.test_terminal_identity import fixture,rotate_fixture,plus
from backend.wire_detection import _skeletonize


class PolarityGlyphContactTests(unittest.TestCase):
    def detect(self,image,box,anchors):
        component=dict(component_id='P',type=self.kind,visual_confidence=.99,bbox=box,
            terminal_detection={},terminals=[dict(pin=pin,position=dict(x=float(p[0]),y=float(p[1])))
                                            for pin,p in anchors.items()])
        # Deliberately include all glyph pixels in the graph, reproducing the
        # failure even when the glyph has apparently perfect wire evidence.
        sk=_skeletonize((cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<150).astype(np.uint8)*255)
        before=sk.copy();wire={'segments':[]};saved=copy.deepcopy(wire)
        result=refine_all_terminal_contacts(image,{'components':[component]},wire,sk)
        self.assertTrue(np.array_equal(sk,before));self.assertEqual(wire,saved)
        return result

    def assert_on_leads(self,image,box,anchors,expected):
        result=self.detect(image,box,anchors)
        terminals=result['components'][0]['terminals']
        self.assertEqual(len(terminals),2,result)
        pins={t['pin']:t['position'] for t in terminals}
        for pin,p in expected.items():
            self.assertLess(np.linalg.norm(np.array(list(pins[pin].values()))-p),5,(pin,pins,expected))
            q=pins[pin]
            self.assertTrue(box['x1']<q['x']<box['x2'] and box['y1']<q['y']<box['y2'])
        return result

    def test_plus_crossing_each_box_face_is_not_a_contact(self):
        for self.kind in ('source_dc','capacitor_polarized','battery'):
            for side in ('left','right'):
                image,box,_=fixture(self.kind,mark=False)
                x=box['x1']+4 if side=='left' else box['x2']-4
                y=box['y1']+22
                plus(image,round(x),round(y))
                expected={'+':(box['x1']+1 if side=='left' else box['x2']-1,140 if self.kind!='capacitor_polarized' else 145),
                          '-':(box['x2']-1 if side=='left' else box['x1']+1,140 if self.kind!='capacitor_polarized' else 145)}
                anchors={'+':(x,y),'-':expected['-']}
                for k in range(4):
                    with self.subTest(kind=self.kind,side=side,rotation=k):
                        im,b,points=rotate_fixture(image,box,expected,k)
                        _,_,wrong=rotate_fixture(image,box,anchors,k)
                        result=self.assert_on_leads(im,b,wrong,points)
                        self.assertGreater(result['wire_terminal_contacts']['components'][0]['rejected_polarity_glyph_contacts'],0)

    def test_detached_plus_endpoints_inside_box_are_rejected(self):
        for self.kind in ('source_dc','capacitor_polarized','battery'):
            image,box,_=fixture(self.kind,mark=False)
            x=box['x2']-12;y=box['y1']+24
            plus(image,round(x),round(y))
            lead_y=145 if self.kind=='capacitor_polarized' else 140
            expected={'+':(box['x2']-1,lead_y),'-':(box['x1']+1,lead_y)}
            anchors={'+':(x+8,y),'-':expected['-']}
            for k in range(4):
                with self.subTest(kind=self.kind,rotation=k):
                    im,b,pins=rotate_fixture(image,box,expected,k)
                    _,_,wrong=rotate_fixture(image,box,anchors,k)
                    self.assert_on_leads(im,b,wrong,pins)

    def test_plus_on_a_short_spur_is_not_the_wire_contact(self):
        self.kind='source_dc'
        image,box,_=fixture(self.kind,mark=False)
        plus(image,216,106)
        cv2.line(image,(216,106),(210,140),(20,20,20),3)
        expected={'+':(219,140),'-':(61,140)}
        anchors={'+':(219,106),'-':expected['-']}
        for k in range(4):
            with self.subTest(rotation=k):
                im,b,pins=rotate_fixture(image,box,expected,k)
                _,_,wrong=rotate_fixture(image,box,anchors,k)
                self.assert_on_leads(im,b,wrong,pins)

    def test_plus_drawn_over_a_continuous_wire_does_not_remove_the_wire(self):
        self.kind='source_dc'
        image,box,_=fixture(self.kind,mark=False)
        cv2.line(image,(180,140),(265,140),(20,20,20),4)
        plus(image,210,140)
        expected={'+':(219,140),'-':(61,140)}
        for k in range(4):
            with self.subTest(rotation=k):
                im,b,pins=rotate_fixture(image,box,expected,k)
                self.assert_on_leads(im,b,pins,pins)


if __name__=='__main__':unittest.main()
