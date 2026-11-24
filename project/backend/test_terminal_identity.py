import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from backend import terminal_detection as td


def plus(image, x, y):
    cv2.line(image, (x-8, y), (x+8, y), (20,20,20), 3)
    cv2.line(image, (x, y-8), (x, y+8), (20,20,20), 3)


def rotate_fixture(image, bbox, expected, k):
    def rotate(point):
        x, y = point
        for _ in range(k): x, y = y, 279-x
        return np.array([x,y], float)
    corners = np.array([rotate((x,y)) for x in (bbox['x1'],bbox['x2']) for y in (bbox['y1'],bbox['y2'])])
    rotated_bbox = dict(x1=float(corners[:,0].min()),y1=float(corners[:,1].min()),
                        x2=float(corners[:,0].max()),y2=float(corners[:,1].max()))
    return np.ascontiguousarray(np.rot90(image,k)), rotated_bbox, {pin:rotate(p) for pin,p in expected.items()}


def fixture(kind, swap=False, mark=True):
    image = np.full((280,280,3),255,np.uint8)
    ink = (20,20,20)
    bbox = dict(x1=60,y1=55,x2=230,y2=225)
    if kind=='source_dc':
        bbox = dict(x1=60,y1=75,x2=220,y2=205)
        cv2.circle(image,(140,140),40,ink,4)
        cv2.line(image,(50,140),(100,140),ink,4)
        cv2.line(image,(180,140),(235,140),ink,4)
        if mark: plus(image,80 if swap else 200,108)
        expected = {'+':(60 if swap else 220,140),'-':(220 if swap else 60,140)}
    elif kind=='capacitor_polarized':
        bbox = dict(x1=65,y1=85,x2=225,y2=205)
        cv2.line(image,(50,145),(115,145),ink,4)
        cv2.line(image,(115,115),(115,175),ink,4)
        cv2.line(image,(155,115),(155,175),ink,4)
        cv2.line(image,(155,145),(240,145),ink,4)
        if mark: plus(image,95 if swap else 175,103)
        expected = {'+':(50 if swap else 240,145),'-':(240 if swap else 50,145)}
    elif kind=='battery':
        bbox = dict(x1=60,y1=75,x2=220,y2=205)
        cv2.line(image,(50,140),(100,140),ink,4)
        cv2.line(image,(100,98),(100,182),ink,4)
        cv2.line(image,(155,115),(155,165),ink,4)
        cv2.line(image,(155,140),(235,140),ink,4)
        if mark: plus(image,185,103)
        expected = {'+':(220 if mark else 60,140),'-':(60 if mark else 220,140)}
    elif kind=='opamp':
        cv2.polylines(image,[np.array([(100,70),(100,210),(210,140)],np.int32)],True,ink,4)
        cv2.line(image,(50,105),(100,105),ink,4)
        cv2.line(image,(50,175),(100,175),ink,4)
        cv2.line(image,(210,140),(245,140),ink,4)
        if mark:
            plus(image,125,175 if swap else 105)
            cv2.line(image,(117,105 if swap else 175),(133,105 if swap else 175),ink,3)
        expected = {'+':(60,175 if swap else 105),'-':(60,105 if swap else 175),'OUT':(230,140)}
    else:
        cv2.line(image,(50,140),(90,140),ink,4)
        cv2.line(image,(90,75),(90,205),ink,4)
        for y1,y2 in [(70,100),(126,154),(180,210)]:
            cv2.line(image,(120,y1),(120,y2),ink,4)
        cv2.line(image,(120,70),(245,70),ink,4)
        cv2.line(image,(120,210),(245,210),ink,4)
        if mark:
            cv2.line(image,(120,140),(155,140),ink,4)
            cv2.line(image,(155,140),(155,210),ink,4)
            cv2.fillPoly(image,[np.array([(117,140),(135,130),(135,150)],np.int32)],ink)
        expected = {'G':(60,140),'D':(230,70),'S':(230,210)}
    return image, bbox, expected


class VisibleTerminalIdentityTests(unittest.TestCase):
    def detect(self, kind, image, bbox, extra=None):
        component = dict(component_id='X', type=kind, bbox=bbox, visual_confidence=.98)
        component.update(extra or {})
        with patch.object(td,'predict_component_terminals',return_value=None):
            return td.detectar_terminais_componentes({'components':[component]},280,280,image)['components'][0]

    def assert_points(self, component, expected, tolerance=9):
        pins = {t['pin']:t for t in component['terminals']}
        self.assertEqual(set(pins),set(expected))
        self.assertTrue(component['terminal_identity_geometry']['confirmed'], component['terminal_identity_geometry'])
        self.assertFalse(component['terminal_detection']['review_required'])
        for pin, point in expected.items():
            box=component['bbox']
            point=np.clip(point,[box['x1']+1,box['y1']+1],[box['x2']-1,box['y2']-1])
            actual = np.array([pins[pin]['position']['x'],pins[pin]['position']['y']])
            self.assertLess(np.linalg.norm(actual-point),tolerance,(pin,actual,point))
            self.assertFalse(pins[pin]['identity_uncertain'])

    def test_capacitor_plus_follows_either_side_in_all_four_rotations(self):
        for swap in (False,True):
            for k in range(4):
                with self.subTest(swap=swap,k=k):
                    image,bbox,expected=rotate_fixture(*fixture('capacitor_polarized',swap),k)
                    self.assert_points(self.detect('capacitor_polarized',image,bbox),expected)

    def test_dc_source_plus_follows_either_side_in_all_four_rotations(self):
        for swap in (False,True):
            for k in range(4):
                with self.subTest(swap=swap,k=k):
                    image,bbox,expected=rotate_fixture(*fixture('source_dc',swap),k)
                    self.assert_points(self.detect('source_dc',image,bbox),expected)

    def test_battery_long_plate_positive_in_all_four_rotations(self):
        for k in range(4):
            with self.subTest(k=k):
                image,bbox,expected=rotate_fixture(*fixture('battery',mark=False),k)
                self.assert_points(self.detect('battery',image,bbox),expected)

    def test_explicit_battery_plus_takes_precedence_over_plate_length(self):
        image,bbox,expected=fixture('battery',mark=True)
        self.assert_points(self.detect('battery',image,bbox),expected)

    def test_opamp_plus_minus_and_output_follow_triangle_and_marks(self):
        for swap in (False,True):
            for k in range(4):
                with self.subTest(swap=swap,k=k):
                    image,bbox,expected=rotate_fixture(*fixture('opamp',swap),k)
                    component=self.detect('opamp',image,bbox)
                    self.assert_points(component,expected)
                    roles={t['pin']:t['electrical_role'] for t in component['terminals']}
                    self.assertEqual(roles, {'+':'non_inverting_input','-':'inverting_input','OUT':'output'})

    def test_mosfet_gate_and_source_connector_follow_both_classes_and_rotations(self):
        for kind in ('mosfet_n','mosfet_p'):
            for k in range(4):
                with self.subTest(kind=kind,k=k):
                    image,bbox,expected=rotate_fixture(*fixture(kind),k)
                    self.assert_points(self.detect(kind,image,bbox),expected)

    def test_absent_polarity_sign_is_not_certified(self):
        for kind in ('source_dc','capacitor_polarized','opamp'):
            with self.subTest(kind=kind):
                image,bbox,_=fixture(kind,mark=False)
                component=self.detect(kind,image,bbox)
                self.assertFalse(component['terminal_identity_geometry']['confirmed'])
                self.assertTrue(component['terminal_detection']['review_required'])
                self.assertTrue(all(t['identity_uncertain'] for t in component['terminals']))

    def test_source_drain_without_body_connector_require_review(self):
        image,bbox,_=fixture('mosfet_n',mark=False)
        component=self.detect('mosfet_n',image,bbox)
        self.assertFalse(component['terminal_identity_geometry']['confirmed'])
        self.assertTrue(component['terminal_detection']['review_required'])

    def test_manual_ground_truth_is_preserved(self):
        image,bbox,_=fixture('opamp')
        manual=[{'pin':pin,'position':{'x':85+i,'y':105+i},'source':'manual_ground_truth_persisted'}
                for i,pin in enumerate(('+','-','OUT'))]
        component=self.detect('opamp',image,bbox,{'terminals':copy.deepcopy(manual)})
        self.assertEqual(component['terminals'],manual)
        self.assertFalse(component['terminal_detection']['review_required'])

    def test_polarity_confirmation_overrides_wrong_model_pin_names(self):
        image,bbox,expected=fixture('capacitor_polarized')
        fake={'accepted':True,'review_required':True,'keypoint_confidence':.7,
              'points':{'+':{'x':65,'y':145},'-':{'x':225,'y':145}}}
        with patch.dict(td.TRAINED_TERMINAL_SCHEMAS,{'capacitor_polarized':('+','-')}), \
             patch.object(td,'predict_component_terminals',return_value=fake):
            component=td.detectar_terminais_componentes({'components':[{'component_id':'X','type':'capacitor_polarized',
                        'bbox':bbox,'visual_confidence':.98}]},280,280,image)['components'][0]
        self.assert_points(component,expected)

    def test_mirrored_symbols_follow_visible_marks_and_body_connector(self):
        for kind in ('capacitor_polarized','source_dc','battery','opamp','mosfet_n','mosfet_p'):
            with self.subTest(kind=kind):
                image,bbox,expected=fixture(kind)
                image=np.ascontiguousarray(image[:,::-1])
                bbox=dict(x1=279-bbox['x2'], x2=279-bbox['x1'], y1=bbox['y1'], y2=bbox['y2'])
                expected={pin:(279-point[0],point[1]) for pin,point in expected.items()}
                self.assert_points(self.detect(kind,image,bbox),expected)

    def test_plus_touching_capacitor_plate_is_still_read(self):
        image,bbox,expected=fixture('capacitor_polarized',mark=False)
        plus(image,159,105)
        self.assert_points(self.detect('capacitor_polarized',image,bbox),expected)

    def test_crossing_on_capacitor_wire_is_not_a_polarity_sign(self):
        image,bbox,_=fixture('capacitor_polarized',mark=False)
        cv2.line(image,(195,130),(195,160),(20,20,20),3)
        component=self.detect('capacitor_polarized',image,bbox)
        self.assertFalse(component['terminal_identity_geometry']['confirmed'])
        self.assertTrue(component['terminal_detection']['review_required'])

    def test_source_with_gap_in_circle_uses_visible_opposite_leads(self):
        image,bbox,expected=fixture('source_dc')
        cv2.line(image,(130,98),(142,98),(255,255,255),9)
        self.assert_points(self.detect('source_dc',image,bbox),expected)

    def test_thick_mosfet_branch_without_arrow_does_not_certify_source(self):
        image,bbox,_=fixture('mosfet_n',mark=False)
        cv2.line(image,(120,210),(245,210),(20,20,20),12)
        cv2.line(image,(120,140),(155,140),(20,20,20),12)
        cv2.line(image,(155,140),(155,210),(20,20,20),12)
        component=self.detect('mosfet_n',image,bbox)
        self.assertFalse(component['terminal_identity_geometry']['confirmed'])
        self.assertTrue(component['terminal_detection']['review_required'])

    def test_automatic_pipeline_keeps_correct_opamp_inputs_and_output(self):
        from backend import main
        image,bbox,expected=rotate_fixture(*fixture('opamp',swap=True),1)
        with TemporaryDirectory() as folder, patch.object(main,'PROJECTS_DIR',Path(folder)), \
             patch.object(td,'predict_component_terminals',return_value=None):
            project=Path(folder)/'TEST_VISIBLE_PINS'
            project.mkdir()
            cv2.imwrite(str(project/'original.png'),image)
            def current_project(_):
                return {'stored_filename':'original.png',
                        'terminals_info':(project/'terminals.json').read_text(encoding='utf-8')}
            with patch.object(main,'buscar_projeto',side_effect=current_project), \
                 patch.object(main,'atualizar_terminais'), patch.object(main,'atualizar_terminais_pos_fios'), \
                 patch.object(main,'atualizar_fios'):
                result=main._gerar_terminais_automaticamente('TEST_VISIBLE_PINS',{'stored_filename':'original.png'},
                    {'components':[dict(component_id='X',type='opamp',bbox=bbox,visual_confidence=.98)]},
                    trigger='manual_detection_edit')
            self.assertTrue(result['automatic_stage8']['executed'],result['automatic_stage8'])
            self.assert_points(result['components'][0],expected)
            saved=json.loads((project/'terminals.json').read_text(encoding='utf-8'))
            self.assert_points(saved['components'][0],expected)
            self.assertEqual(saved['review_count'],0)
            self.assertTrue((project/'wires.json').is_file())

    def test_automatic_pipeline_counts_missing_polarity_for_review(self):
        from backend import main
        image,bbox,_=fixture('source_dc',mark=False)
        with TemporaryDirectory() as folder, patch.object(main,'PROJECTS_DIR',Path(folder)), \
             patch.object(td,'predict_component_terminals',return_value=None):
            project=Path(folder)/'TEST_MISSING_SIGN'
            project.mkdir()
            cv2.imwrite(str(project/'original.png'),image)
            def current_project(_):
                return {'stored_filename':'original.png',
                        'terminals_info':(project/'terminals.json').read_text(encoding='utf-8')}
            with patch.object(main,'buscar_projeto',side_effect=current_project), \
                 patch.object(main,'atualizar_terminais'), patch.object(main,'atualizar_terminais_pos_fios'), \
                 patch.object(main,'atualizar_fios'):
                result=main._gerar_terminais_automaticamente('TEST_MISSING_SIGN',{'stored_filename':'original.png'},
                    {'components':[dict(component_id='X',type='source_dc',bbox=bbox,visual_confidence=.98)]},
                    trigger='manual_detection_edit')
            self.assertTrue(result['automatic_stage8']['executed'],result['automatic_stage8'])
            self.assertEqual(result['review_count'],1)
            saved=json.loads((project/'terminals.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['review_count'],1)
            self.assertFalse(saved['components'][0]['terminal_identity_geometry']['confirmed'])


if __name__=='__main__': unittest.main()
