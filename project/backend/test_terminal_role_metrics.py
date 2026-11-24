import copy
import unittest
import cv2
import numpy as np
from backend import terminal_identity as ti
from backend.terminal_wire_contacts import refine_all_terminal_contacts
from backend.transistor_wire_refinement import refine_transistor_terminals_from_wires
from backend.test_terminal_identity import fixture as opamp_fixture,rotate_fixture
from backend.test_npn_wire_contacts import fixture as bjt_fixture,rotate_case
from backend.npn_wire_contacts import detect_bjt_wire_contacts


class TerminalRoleMetricTests(unittest.TestCase):
    def test_opamp_final_roles_follow_original_marks_when_old_names_are_wrong(self):
        for swapped in (False,True):
            for rotation in range(4):
                with self.subTest(swapped=swapped,rotation=rotation):
                    image,box,expected=rotate_fixture(*opamp_fixture('opamp',swap=swapped),rotation)
                    points=[expected['-'],expected['OUT'],expected['+']]
                    component=dict(component_id='A',type='opamp',bbox=box,
                        terminals=[dict(pin=pin,position=dict(x=float(p[0]),y=float(p[1])),original_ink_confirmed=True) for pin,p in zip(('+','-','OUT'),points)])
                    meta=ti.finalize_opamp_on_contacts(image,component)
                    self.assertTrue(meta['confirmed'],meta)
                    self.assertGreater(meta['assignment_margin_normalized'],.08)
                    for term in component['terminals']:
                        np.testing.assert_allclose([term['position']['x'],term['position']['y']],expected[term['pin']])

    def test_opamp_after_wires_replaces_three_wrong_roles_without_moving_off_contacts(self):
        image,box,expected=opamp_fixture('opamp',swap=True)
        sk=np.zeros(image.shape[:2],np.uint8)
        for start,end in (((0,105),(100,105)),((0,175),(100,175)),((210,140),(279,140))):cv2.line(sk,start,end,255,1)
        component=dict(component_id='A',type='opamp',bbox=box,terminal_detection={},
            terminals=[dict(pin=pin,position=dict(x=float(p[0]),y=float(p[1]))) for pin,p in zip(('+','-','OUT'),(expected['-'],expected['OUT'],expected['+']))])
        result=refine_all_terminal_contacts(image,{'components':[component]}, {},sk)['components'][0]
        self.assertTrue(result['terminal_identity_geometry']['confirmed'])
        for term in result['terminals']:
            actual=np.array([term['position']['x'],term['position']['y']])
            self.assertLess(np.linalg.norm(actual-expected[term['pin']]),5)
            self.assertTrue(term['original_ink_confirmed'])

    def test_pnp_with_head_next_to_base_and_bent_collector_keeps_ce_at_all_rotations(self):
        for rotation in range(4):
            image,sk,component,expected=bjt_fixture(mark=None,notebook=False)
            component['type']='transistor_pnp'
            root=np.array([165.,95.]);vector=np.array([65.,60.]);unit=vector/np.linalg.norm(vector);normal=np.array([-unit[1],unit[0]])
            tip=root+unit*8;tail=root+unit*29
            cv2.fillPoly(image,[np.array([tip,tail+normal*11,tail-normal*9],np.int32)],(20,20,20))
            expected={'B':expected['B'],'C':expected['E'],'E':expected['C']}
            image,sk,component,expected=rotate_case(image,sk,component,expected,rotation)
            result,_=refine_transistor_terminals_from_wires(image,{'components':[component]}, {},sk)
            result=result['components'][0]
            with self.subTest(rotation=rotation):
                self.assertEqual(len(result['terminals']),3)
                for term in result['terminals']:
                    self.assertLess(np.linalg.norm(np.array([term['position']['x'],term['position']['y']])-expected[term['pin']]),6)
                self.assertTrue(result['transistor_geometry']['original_ink_geometry']['arrow_confirmed'])

    def test_missing_opamp_marks_do_not_certify_positional_guess(self):
        image,box,expected=opamp_fixture('opamp',mark=False)
        component=dict(type='opamp',bbox=box,terminals=[dict(pin=pin,position=dict(x=p[0],y=p[1])) for pin,p in expected.items()])
        saved=copy.deepcopy(component['terminals'])
        self.assertFalse(ti.finalize_opamp_on_contacts(image,component)['confirmed'])
        self.assertEqual(saved,component['terminals'])

    def test_arrow_on_a_neighboring_wire_outside_box_cannot_identify_emitter(self):
        image,sk,component,_=bjt_fixture(mark=None,notebook=False)
        component['type']='transistor_pnp'
        cv2.fillPoly(image,[np.array([(269,155),(254,146),(254,164)],np.int32)],(20,20,20))
        found,meta=detect_bjt_wire_contacts(image,component['bbox'],sk,component['type'])
        self.assertIsNotNone(found)
        self.assertFalse(meta['arrow_confirmed'],meta)
