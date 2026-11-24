import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from backend.npn_wire_contacts import detect_npn_wire_contacts
from backend import transistor_wire_refinement as twr


def fixture(mark='outward', notebook=True):
    image=np.full((280,280,3),245,np.uint8)
    bbox=dict(x1=55.,y1=60.,x2=250.,y2=215.)
    sk=np.zeros((280,280),np.uint8)
    if notebook:
        for x in (47,151,246):
            cv2.line(image,(x,0),(x,279),(140,140,140),1)
            cv2.line(sk,(x,0),(x,59),255,1)
            cv2.line(sk,(x,216),(x,279),255,1)
    ink=(20,20,20)
    for a,b in (((80,95),(205,95)),((145,0),(145,95)),((110,95),(60,190)),
                ((165,95),(230,155)),((230,155),(279,155)),((60,190),(60,279))):
        cv2.line(image,a,b,ink,4)
    if mark in ('outward','inward'):
        start=np.array([110.,95.]); unit=np.array([-50.,95.]);unit/=np.linalg.norm(unit)
        normal=np.array([-unit[1],unit[0]])
        tip=start+unit*(93 if mark=='outward' else 72)
        tail=start+unit*(74 if mark=='outward' else 93)
        vertices=np.array([tip,tail+normal*12,tail-normal*12],np.int32)
        cv2.fillPoly(image,[vertices],ink)
    elif mark=='dot':
        cv2.circle(image,(69,173),11,ink,-1)
    cv2.line(sk,(145,0),(145,60),255,1)
    cv2.line(sk,(250,155),(279,155),255,1)
    cv2.line(sk,(60,215),(60,279),255,1)
    expected={'B':np.array([145.,60.]),'C':np.array([250.,155.]),'E':np.array([60.,215.])}
    component=dict(component_id='Q1',type='transistor_npn',bbox=bbox,visual_confidence=.99,terminals=[])
    return image,sk,component,expected


def rotate_case(image,sk,component,expected,k):
    def rotate(p):
        x,y=p
        for _ in range(k):x,y=y,279-x
        return np.array([x,y],float)
    component=copy.deepcopy(component)
    b=component['bbox']
    corners=np.array([rotate((x,y)) for x in (b['x1'],b['x2']) for y in (b['y1'],b['y2'])])
    component['bbox']=dict(x1=float(corners[:,0].min()),x2=float(corners[:,0].max()),
                           y1=float(corners[:,1].min()),y2=float(corners[:,1].max()))
    return np.ascontiguousarray(np.rot90(image,k)),np.ascontiguousarray(np.rot90(sk,k)),component,{pin:rotate(p) for pin,p in expected.items()}


class NpnOriginalInkContactTests(unittest.TestCase):
    def run_case(self,image,sk,component):
        result,summary=twr.refine_transistor_terminals_from_wires(image,{'components':[component]}, {},sk)
        self.assertEqual(summary['refined_count'],1,summary)
        return result['components'][0]

    def assert_physical_points(self,c,image,expected,tolerance=5):
        pins={t['pin']:t for t in c['terminals']}
        self.assertEqual(set(pins),set(expected))
        for pin,p in expected.items():
            actual=np.array([pins[pin]['position']['x'],pins[pin]['position']['y']])
            self.assertLess(np.linalg.norm(actual-p),tolerance,(pin,actual,p))
            x,y=np.rint(actual).astype(int)
            self.assertLess(int(image[y,x].min()) if image.ndim==3 else int(image[y,x]),100)
            self.assertTrue(pins[pin]['original_ink_confirmed'])

    def test_rotated_npn_with_notebook_rules_keeps_three_true_leads(self):
        for k in range(4):
            with self.subTest(k=k):
                image,sk,c,expected=rotate_case(*fixture(),k)
                result=self.run_case(image,sk,c)
                self.assert_physical_points(result,image,expected)
                self.assertFalse(result['terminal_detection']['review_required'])

    def test_mirrored_npn_keeps_names_and_ink_centers(self):
        image,sk,c,expected=fixture()
        image=np.ascontiguousarray(image[:,::-1]);sk=np.ascontiguousarray(sk[:,::-1])
        b=c['bbox'];c['bbox']=dict(x1=279-b['x2'],x2=279-b['x1'],y1=b['y1'],y2=b['y2'])
        expected={pin:np.array([279-p[0],p[1]]) for pin,p in expected.items()}
        result=self.run_case(image,sk,c)
        self.assert_physical_points(result,image,expected)
        self.assertFalse(result['terminal_detection']['review_required'])

    def test_phantom_skeleton_rules_cannot_replace_real_collector(self):
        image,sk,c,expected=fixture()
        cv2.line(sk,(248,0),(248,279),255,1)
        cv2.line(sk,(58,0),(58,279),255,1)
        result=self.run_case(image,sk,c)
        self.assert_physical_points(result,image,expected)

    def test_blue_npn_with_black_notebook_rules_keeps_only_blue_contacts(self):
        image,sk,c,expected=fixture()
        blue=np.all(image==20,axis=2)
        image[np.all(image==140,axis=2)]=(30,30,30)
        image[blue]=(155,45,15)
        result=self.run_case(image,sk,c)
        self.assert_physical_points(result,image,expected)
        self.assertFalse(result['terminal_detection']['review_required'])
        for terminal in result['terminals']:
            p=terminal['position'];pixel=image[int(p['y']),int(p['x'])]
            self.assertGreater(int(pixel[0])-int(pixel[2]),100)

    def test_wrong_old_pins_and_model_do_not_override_visible_leads(self):
        image,sk,c,expected=fixture()
        c['terminals']=[dict(pin=pin,position=dict(x=260,y=260),source='trained_terminal_keypoint_model') for pin in ('B','C','E')]
        with patch.object(twr,'_model_evidence',side_effect=AssertionError('Clear original ink must take priority')):
            self.assert_physical_points(self.run_case(image,sk,c),image,expected)

    def test_missing_arrow_keeps_physical_points_but_requires_identity_review(self):
        image,sk,c,expected=fixture(mark=None)
        result=self.run_case(image,sk,c)
        self.assertTrue(result['terminal_detection']['review_required'])
        self.assertEqual(len(result['terminals']),3)
        self.assertTrue(all(t['original_ink_confirmed'] for t in result['terminals']))
        pins={t['pin']:t['position'] for t in result['terminals']}
        self.assertLess(np.linalg.norm(np.array([pins['B']['x'],pins['B']['y']])-expected['B']),5)

    def test_dot_on_branch_is_not_certified_as_an_arrow(self):
        image,sk,c,_=fixture(mark='dot')
        result=self.run_case(image,sk,c)
        self.assertTrue(result['terminal_detection']['review_required'])

    def test_inward_arrow_in_npn_does_not_move_emitter_to_opposite_branch(self):
        image,sk,c,expected=fixture(mark='inward')
        result=self.run_case(image,sk,c)
        self.assert_physical_points(result,image,expected)
        self.assertTrue(result['terminal_detection']['review_required'])
        self.assertIn('classe_bjt_conflita_com_sentido_estimado_da_seta',result['terminal_detection']['review_reasons'])

    def test_manual_npn_is_preserved(self):
        image,sk,c,_=fixture()
        c['terminals']=[dict(pin=pin,position=dict(x=150+i,y=160+i),source='manual_ground_truth') for i,pin in enumerate(('B','C','E'))]
        saved=copy.deepcopy(c)
        result,summary=twr.refine_transistor_terminals_from_wires(image,{'components':[c]}, {},sk)
        self.assertEqual(result['components'][0],saved)
        self.assertEqual(summary['manual_preserved_count'],1)

    def test_pnp_does_not_enter_new_npn_detector(self):
        image,sk,c,_=fixture(mark='inward')
        c['type']='transistor_pnp'
        with patch.object(twr,'detect_npn_wire_contacts',side_effect=AssertionError('PNP flow must stay unchanged')):
            self.run_case(image,sk,c)

    def test_two_external_wires_do_not_invent_a_third_terminal(self):
        image,sk,c,_=fixture(notebook=False)
        sk[:,240:]=0
        image[:,240:]=245
        found,meta=detect_npn_wire_contacts(image,c['bbox'],sk)
        self.assertIsNone(found,meta)
        result,summary=twr.refine_transistor_terminals_from_wires(image,{'components':[c]}, {},sk)
        self.assertEqual(result['components'][0]['terminals'],[])
        self.assertEqual(summary['review_count'],1)

    def test_small_skeleton_offset_does_not_move_pin_off_original_ink(self):
        image,sk,c,expected=fixture(notebook=False)
        sk=cv2.warpAffine(sk,np.float32([[1,0,7],[0,1,-3]]),(280,280))
        result=self.run_case(image,sk,c)
        self.assert_physical_points(result,image,expected)

    def test_grayscale_original_is_supported(self):
        image,sk,c,expected=fixture()
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        self.assert_physical_points(self.run_case(image,sk,c),image,expected)

    def test_tilted_photo_keeps_pins_on_the_correct_original_leads(self):
        for angle in (-15,15):
            with self.subTest(angle=angle):
                image,sk,c,expected=fixture()
                matrix=cv2.getRotationMatrix2D((139.5,139.5),angle,1)
                image=cv2.warpAffine(image,matrix,(280,280),borderValue=(245,245,245))
                sk=cv2.warpAffine(sk,matrix,(280,280),flags=cv2.INTER_NEAREST)
                b=c['bbox']
                corners=np.array([[x,y,1] for x in (b['x1'],b['x2']) for y in (b['y1'],b['y2'])])@matrix.T
                c['bbox']=dict(x1=float(corners[:,0].min()),x2=float(corners[:,0].max()),
                               y1=float(corners[:,1].min()),y2=float(corners[:,1].max()))
                expected={pin:matrix@np.r_[p,1] for pin,p in expected.items()}
                result=self.run_case(image,sk,c)
                self.assert_physical_points(result,image,expected,tolerance=32)
                self.assertFalse(result['terminal_detection']['review_required'])

    def test_known_base_is_reserved_when_emitter_arrow_is_missing(self):
        tri=[dict(point=(100.,5.)),dict(point=(150.,90.)),dict(point=(180.,110.))]
        evidence={'source':'npn_original_ink_bar_branches_and_external_leads','hypotheses':[
            {'source':'physical_arrow_then_base_bar','physical':True,
             'points':{'B':dict(x=100.,y=5.)},'keypoint_confidences':{'B':.94}}]}
        roles,meta=twr._assign_roles_arrow_base_leftover(tri,evidence,dict(x1=80.,y1=0.,x2=200.,y2=120.))
        self.assertEqual(roles['B'],0)
        self.assertTrue(meta['identity_uncertain'])

    def test_automatic_pipeline_saves_correct_npn_points_after_wires(self):
        from backend import main
        image,_,c,expected=fixture()
        with TemporaryDirectory() as folder, patch.object(main,'PROJECTS_DIR',Path(folder)):
            project=Path(folder)/'TEST_NPN_CONTACTS';project.mkdir()
            cv2.imwrite(str(project/'original.png'),image)
            def current_project(_):
                return {'stored_filename':'original.png','terminals_info':(project/'terminals.json').read_text(encoding='utf-8')}
            with patch.object(main,'buscar_projeto',side_effect=current_project), \
                 patch.object(main,'atualizar_terminais'),patch.object(main,'atualizar_terminais_pos_fios'), \
                 patch.object(main,'atualizar_fios'):
                result=main._gerar_terminais_automaticamente('TEST_NPN_CONTACTS',{'stored_filename':'original.png'},
                    {'components':[c]},trigger='manual_detection_edit')
            self.assertTrue(result['automatic_stage8']['executed'],result['automatic_stage8'])
            saved=json.loads((project/'terminals.json').read_text(encoding='utf-8'))
            self.assert_physical_points(saved['components'][0],image,expected)
            self.assertFalse(saved['components'][0]['terminal_detection']['review_required'])
            self.assertEqual(saved['review_count'],0)
            self.assertTrue((project/'junctions.json').is_file())


if __name__=='__main__':unittest.main()
