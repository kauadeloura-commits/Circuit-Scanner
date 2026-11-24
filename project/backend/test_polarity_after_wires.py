"""Polarities belong to visible signs/plates, independently of contact placement."""
import copy
import unittest
import cv2
import numpy as np
from backend.terminal_identity import finalize_polarity_on_contacts
from backend.terminal_wire_contacts import refine_all_terminal_contacts
from backend.test_terminal_identity import fixture,rotate_fixture,plus
from backend.wire_detection import _skeletonize


def contacts(kind,box,expected):
    # Deliberately reverse the old names. Coordinates and wire evidence must
    # survive the correction exactly, rather than being regenerated from a bbox.
    return dict(component_id='X',type=kind,bbox=box,visual_confidence=.99,
        terminal_detection={'review_required':True,'review_reasons':['confirmar_identidade_eletrica_dos_lados']},
        terminals=[dict(pin='-' if pin=='+' else '+',position=dict(x=float(p[0]),y=float(p[1])),
                        source='wire_contact_inside_component_bbox',wire_contact_confirmed=True,
                        stage8_context_confirmed=True,identity_uncertain=True) for pin,p in expected.items()])


class PolarityAfterWiresTests(unittest.TestCase):
    def check(self,image,c,expected):
        before=copy.deepcopy(c['terminals'])
        meta=finalize_polarity_on_contacts(image,c)
        self.assertTrue(meta['confirmed'],meta)
        pins={t['pin']:t for t in c['terminals']}
        self.assertEqual(set(pins),{'+','-'})
        for pin,p in expected.items():
            self.assertEqual(pins[pin]['position'],dict(x=float(p[0]),y=float(p[1])))
            self.assertEqual(pins[pin]['electrical_role'],'positive' if pin=='+' else 'negative')
            self.assertEqual(pins[pin]['pin_id'],'X.'+pin)
            self.assertFalse(pins[pin]['identity_uncertain'])
        self.assertEqual(sorted(tuple(t['position'].values()) for t in before),
                         sorted(tuple(t['position'].values()) for t in c['terminals']))

    def test_dc_and_capacitor_follow_plus_on_both_sides_in_four_rotations(self):
        for kind in ('source_dc','capacitor_polarized'):
            for swap in (False,True):
                for k in range(4):
                    with self.subTest(kind=kind,swap=swap,k=k):
                        image,bbox,expected=rotate_fixture(*fixture(kind,swap),k)
                        self.check(image,contacts(kind,bbox,expected),expected)

    def test_source_with_open_circle_still_follows_visible_plus(self):
        image,box,expected=fixture('source_dc')
        cv2.line(image,(128,98),(145,98),(255,255,255),12)
        self.check(image,contacts('source_dc',box,expected),expected)

    def test_plus_side_uses_body_center_with_unequal_wire_lengths(self):
        image,box,_=fixture('source_dc',mark=False)
        plus(image,153,135)
        cv2.line(image,(180,140),(277,140),(20,20,20),4)
        box=dict(x1=60,y1=75,x2=275,y2=205)
        expected={'+':(274,140),'-':(61,140)}
        # '+' lies to the right of the source center (140), although it is
        # left of the midpoint between the unequal wire ends (167.5).
        self.check(image,contacts('source_dc',box,expected),expected)

    def test_battery_long_plate_without_plus_follows_rotations_and_mirror(self):
        for mirror in (False,True):
            for k in range(4):
                with self.subTest(mirror=mirror,k=k):
                    image,box,expected=fixture('battery',mark=False)
                    if mirror:
                        image=np.ascontiguousarray(image[:,::-1]);box=dict(x1=279-box['x2'],x2=279-box['x1'],y1=box['y1'],y2=box['y2'])
                        expected={pin:(279-p[0],p[1]) for pin,p in expected.items()}
                    image,box,expected=rotate_fixture(image,box,expected,k)
                    self.check(image,contacts('battery',box,expected),expected)

    def test_multiple_battery_cells_use_outer_plate_lengths_not_split_segments(self):
        image=np.full((280,280,3),255,np.uint8);ink=(20,20,20)
        box=dict(x1=65,y1=55,x2=225,y2=230)
        cv2.line(image,(140,35),(140,85),ink,4)
        for y,width in ((85,100),(120,50),(160,100),(195,50)):
            cv2.line(image,(140-width//2,y),(140+width//2,y),ink,4)
        cv2.line(image,(140,195),(140,250),ink,4)
        expected={'+':(140,56),'-':(140,229)}
        for k in range(4):
            rotated,bbox,pins=rotate_fixture(image,box,expected,k)
            with self.subTest(k=k):self.check(rotated,contacts('battery',bbox,pins),pins)

    def test_explicit_battery_plus_wins_over_a_longer_opposite_plate(self):
        for k in range(4):
            image,box,expected=rotate_fixture(*fixture('battery',mark=True),k)
            with self.subTest(k=k):self.check(image,contacts('battery',box,expected),expected)

    def test_neighbor_plus_does_not_override_a_battery_without_a_sign(self):
        image,box,expected=fixture('battery',mark=False)
        plus(image,235,200)
        c=contacts('battery',box,expected)
        neighbor=dict(component_id='NEXT',type='opamp',bbox=dict(x1=228,y1=180,x2=275,y2=230))
        meta=finalize_polarity_on_contacts(image,c,components=[c,neighbor])
        self.assertTrue(meta['confirmed'],meta)
        self.assertEqual(meta['reason'],'battery_long_plate_positive_after_wire_contacts')
        self.assertEqual(next(t['position'] for t in c['terminals'] if t['pin']=='+'),dict(x=60.,y=140.))

    def test_polarity_correction_does_not_modify_wire_pixels_or_contact_set(self):
        image,box,expected=fixture('source_dc',swap=True)
        c=contacts('source_dc',box,expected)
        sk=_skeletonize((cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<150).astype(np.uint8)*255)
        wire_info={'segments':[{'path':[[50,140],[100,140]]}]}
        saved=copy.deepcopy(wire_info);pixels=sk.copy()
        final=refine_all_terminal_contacts(image,{'components':[c]},wire_info,sk)
        self.assertEqual(wire_info,saved);self.assertTrue(np.array_equal(sk,pixels))
        pins={t['pin']:t for t in final['components'][0]['terminals']}
        self.assertLess(pins['+']['position']['x'],100)
        self.assertGreater(pins['-']['position']['x'],200)
        self.assertTrue(all(t['wire_contact_confirmed'] for t in pins.values()))


if __name__=='__main__':unittest.main()
