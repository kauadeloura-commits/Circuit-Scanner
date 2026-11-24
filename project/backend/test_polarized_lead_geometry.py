"""Polarity signs must identify opposed real leads, never plate tips or T nodes."""
import copy
import unittest
import cv2
import numpy as np
from backend.terminal_identity import _crop, _marks, _polarized_plate_body, finalize_polarity_on_contacts
from backend.terminal_wire_contacts import refine_all_terminal_contacts, _select_polarized_contacts
from backend.test_terminal_identity import fixture, rotate_fixture, plus
from backend.test_polarity_after_wires import contacts
from backend.wire_detection import _skeletonize


class PolarizedLeadGeometryTests(unittest.TestCase):
    def test_plate_lead_t_junctions_are_never_plus_marks(self):
        for kind in ('battery', 'capacitor_polarized'):
            for k in range(4):
                image, box, _ = rotate_fixture(*fixture(kind, mark=False), k)
                mask, origin = _crop(image, box)
                with self.subTest(kind=kind, rotation=k):
                    marks, _ = _marks(mask, origin, box, include_attached=True)
                    self.assertFalse([m for m in marks if m['kind'] == '+'])

    def test_small_attached_plus_retains_four_arms(self):
        for k in range(4):
            image, box, expected = fixture('capacitor_polarized', mark=False)
            plus(image,159,105)
            image, box, expected = rotate_fixture(image, box, expected, k)
            with self.subTest(rotation=k):
                c = contacts('capacitor_polarized', box, expected)
                self.assertTrue(finalize_polarity_on_contacts(image,c)['confirmed'])
                self.assertEqual(c['terminals'][0]['position'],dict(x=float(expected['+'][0]),y=float(expected['+'][1])))

    def test_detached_plus_is_not_an_extra_plate(self):
        for kind in ('battery', 'capacitor_polarized'):
            image, box, expected = fixture(kind)
            body = _polarized_plate_body(image,box,np.asarray(list(expected.values())),kind)
            with self.subTest(kind=kind):
                self.assertEqual(len(body['plates']),2)
                self.assertLess(abs(body['center'][1]-(140 if kind=='battery' else 145)),2)

    def test_unequal_leads_do_not_change_capacitor_plus_side(self):
        image, box, _ = fixture('capacitor_polarized',mark=False)
        cv2.line(image,(155,145),(277,145),(20,20,20),4)
        plus(image,149,103)
        box.update(x2=275)
        expected={'+':(274,145),'-':(66,145)}
        c=contacts('capacitor_polarized',box,expected)
        self.assertTrue(finalize_polarity_on_contacts(image,c)['confirmed'])
        self.assertEqual(c['terminals'][0]['position'],dict(x=274.,y=145.))

    def test_plate_tips_do_not_win_over_opposed_wire_contacts(self):
        for kind in ('battery','capacitor_polarized'):
            image,box,expected=fixture(kind)
            c=contacts(kind,box,expected)
            c['terminals'][0]['position']=dict(x=115.,y=115.)
            c['terminals'][1]['position']=dict(x=155.,y=115.)
            observed=[dict(point=np.array(p,float),face_axis=face,stage8_distance_px=0)
                      for p,face in (((box['x1']+1,140 if kind=='battery' else 145),0),
                                     ((box['x2']-1,140 if kind=='battery' else 145),0),
                                     ((115,box['y1']+1),1),((155,box['y2']-1),1))]
            before=copy.deepcopy(observed)
            selected,meta,body=_select_polarized_contacts(image,c,observed)
            with self.subTest(kind=kind):
                self.assertEqual(meta['selected'],2)
                self.assertEqual([p['face_axis'] for p in selected],[0,0])
                self.assertEqual(sorted(p['point'][0] for p in selected),[box['x1']+1,box['x2']-1])
                for a,b in zip(before,observed):self.assertTrue(np.array_equal(a['point'],b['point']))

    def test_missing_opposite_contact_is_not_invented(self):
        image,box,expected=fixture('battery',mark=False)
        c=contacts('battery',box,expected)
        observed=[dict(point=np.array([61,140.]),face_axis=0,stage8_distance_px=0)]
        selected,meta,_=_select_polarized_contacts(image,c,observed)
        self.assertEqual(meta['selected'],1)
        self.assertIs(selected[0],observed[0])

    def test_unconfirmed_capacitor_clears_stale_confirmed_identity(self):
        image,box,expected=fixture('capacitor_polarized',mark=False)
        c=contacts('capacitor_polarized',box,expected)
        c['terminal_identity_geometry']={'confirmed':True}
        for t in c['terminals']:t['identity_uncertain']=False
        meta=finalize_polarity_on_contacts(image,c)
        self.assertFalse(meta['confirmed'])
        self.assertFalse(c['terminal_identity_geometry']['confirmed'])
        self.assertTrue(all(t['identity_uncertain'] for t in c['terminals']))
        self.assertTrue(c['terminal_detection']['review_required'])

    def test_incorrect_old_anchors_recover_leads_in_four_rotations(self):
        for kind in ('battery','capacitor_polarized'):
            for k in range(4):
                image,box,expected=rotate_fixture(*fixture(kind),k)
                c=contacts(kind,box,expected)
                for t in c['terminals']:t['position']=dict(x=(box['x1']+box['x2'])/2,y=box['y1']+2)
                sk=_skeletonize((cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<150).astype(np.uint8)*255)
                before=sk.copy()
                result=refine_all_terminal_contacts(image,{'components':[c]},{},sk)['components'][0]
                pins={t['pin']:t for t in result['terminals']}
                with self.subTest(kind=kind,rotation=k):
                    self.assertEqual(set(pins),{'+','-'})
                    self.assertTrue(result['terminal_identity_geometry']['confirmed'])
                    for pin,p in expected.items():
                        p=np.clip(p,[box['x1']+1,box['y1']+1],[box['x2']-1,box['y2']-1])
                        actual=np.array(list(pins[pin]['position'].values()))
                        self.assertLess(np.linalg.norm(actual-p),3)
                    self.assertTrue(np.array_equal(sk,before))

    def test_manual_polarity_and_positions_are_preserved(self):
        image,box,expected=fixture('battery')
        c=contacts('battery',box,expected)
        for t in c['terminals']:
            t['source']='manual_ground_truth_persisted'
            t['position']['x']=float(np.clip(t['position']['x'],box['x1']+1,box['x2']-1))
        before=copy.deepcopy(c['terminals'])
        out=refine_all_terminal_contacts(image,{'components':[c]},{},None)
        self.assertEqual(out['components'][0]['terminals'],before)

    def test_inclined_scaled_symbols_on_gray_paper_and_blue_ink(self):
        for kind in ('battery','capacitor_polarized'):
            for angle,scale in ((15,.8),(30,1),(45,1.1),(70,.9),(125,1)):
                image,box,expected=fixture(kind,mark=kind!='battery')
                ink=np.min(image,axis=2)<150
                image[:]=(185,185,185);image[ink]=(145,55,35)
                matrix=cv2.getRotationMatrix2D((140,140),angle,scale)
                rotated=cv2.warpAffine(image,matrix,(280,280),borderValue=(185,185,185))
                corners=np.asarray([[x,y,1] for x in (box['x1'],box['x2']) for y in (box['y1'],box['y2'])])@matrix.T
                box=dict(x1=corners[:,0].min(),y1=corners[:,1].min(),x2=corners[:,0].max(),y2=corners[:,1].max())
                pins={pin:matrix@np.r_[p,1] for pin,p in expected.items()}
                c=contacts(kind,box,pins)
                with self.subTest(kind=kind,angle=angle,scale=scale):
                    meta=finalize_polarity_on_contacts(rotated,c)
                    self.assertTrue(meta['confirmed'],meta)
                    self.assertEqual(c['terminals'][0]['position'],dict(x=float(pins['+'][0]),y=float(pins['+'][1])))

    def test_lighter_same_hue_leads_survive_global_dark_ink_profile(self):
        for kind in ('battery','capacitor_polarized'):
            image,box,expected=fixture(kind,mark=kind!='battery')
            color=np.array([170,85,60],np.uint8)
            image[np.min(image,axis=2)<150]=color
            dark=np.rint(color*.3).astype(np.uint8)
            hsv=cv2.cvtColor(dark.reshape(1,1,3),cv2.COLOR_BGR2HSV)[0,0]
            profile=dict(active=True,chromatic=True,reference_rgb=dark[::-1].tolist(),
                         reference_hsv=hsv.tolist(),channel_tolerance=20.,hue_tolerance=12.)
            c=contacts(kind,box,expected)
            sk=_skeletonize((cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<150).astype(np.uint8)*255)
            result=refine_all_terminal_contacts(image,{'components':[c]},{'ink_color_profile':profile},sk)['components'][0]
            with self.subTest(kind=kind):
                self.assertEqual(len(result['terminals']),2)
                self.assertTrue(result['terminal_identity_geometry']['confirmed'])
                for t in result['terminals']:
                    self.assertLess(abs(t['position']['y']-(140 if kind=='battery' else 145)),2)

    def test_one_observed_pole_cannot_retain_complete_identity_confirmation(self):
        image,box,expected=fixture('battery',mark=False)
        c=contacts('battery',box,expected)
        c['terminals']=c['terminals'][:1]
        c['terminal_identity_geometry']={'confirmed':True}
        meta=finalize_polarity_on_contacts(image,c)
        self.assertFalse(meta['confirmed'])
        self.assertEqual(meta['reason'],'opposite_polarized_lead_not_observed')
        self.assertEqual(len(c['terminals']),1)
        self.assertTrue(c['terminals'][0]['identity_uncertain'])


if __name__=='__main__':unittest.main()
