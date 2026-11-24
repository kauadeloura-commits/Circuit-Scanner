"""Rule colors are evidence, never a global instruction to erase pencil."""
import copy
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from backend.notebook_rulings import detect_notebook_rule_profile,apply_notebook_rule_profile
from backend.ink_color import estimate_component_ink
from backend.terminal_wire_contacts import refine_all_terminal_contacts
from backend.wire_detection import detectar_fios
from backend.junction_detection import resolver_juncoes


def ruled_scene(rule=(204,204,204),ink=(25,25,25)):
    image=np.full((360,480,3),255,np.uint8)
    for y in range(35,360,40):cv2.line(image,(0,y),(479,y),rule,2)
    cv2.rectangle(image,(165,110),(240,165),ink,4)
    cv2.line(image,(100,138),(165,138),ink,4)
    cv2.line(image,(240,138),(310,138),ink,4)
    cv2.line(image,(310,138),(310,270),ink,4)
    component=dict(component_id='R',type='resistor',confidence=.99,visual_confidence=.99,
        bbox=dict(x1=140,y1=95,x2=265,y2=180),terminal_detection={},terminals=[
            dict(pin='1',position=dict(x=141,y=115)),dict(pin='2',position=dict(x=264,y=115))])
    return image,{'components':[component]}


class NotebookColorRuleTests(unittest.TestCase):
    def test_dark_rules_under_nonuniform_shadow_are_removed_without_breaking_ink(self):
        image,info=ruled_scene(rule=(145,145,145))
        exposure=np.linspace(.34,.8,image.shape[1])[None,:,None]
        image=np.rint(image*exposure).astype(np.uint8)
        rules=detect_notebook_rule_profile(image)
        self.assertTrue(rules['active'])
        clean,mask=apply_notebook_rule_profile(image,rules)
        self.assertGreater(np.count_nonzero(mask[75]),image.shape[1]*.85)
        self.assertFalse(np.any(mask[190:250,310]))
        self.assertTrue(np.array_equal(clean[190:250,310],image[190:250,310]))

    def test_repeated_real_buses_with_the_same_component_ink_are_not_erased(self):
        image,info=ruled_scene(rule=(255,255,255))
        for y in (35,75,235,275):cv2.line(image,(0,y),(479,y),(25,25,25),2)
        ink=estimate_component_ink(image,info)
        rules=detect_notebook_rule_profile(image,ink)
        clean,mask=apply_notebook_rule_profile(image,rules,ink)
        self.assertFalse(np.any(mask[75]))
        self.assertTrue(np.array_equal(clean[75],image[75]))

    def test_observed_contact_reached_by_retained_wire_survives_an_ambiguous_rule_color(self):
        image,info=ruled_scene(rule=(255,255,255))
        profile=estimate_component_ink(image,info)
        exclusions=np.zeros(image.shape[:2],np.uint8);exclusions[138,135:270]=255
        skeleton=np.zeros_like(exclusions)
        cv2.line(skeleton,(100,138),(165,138),255,1)
        cv2.line(skeleton,(240,138),(310,138),255,1)
        with patch('backend.terminal_wire_contacts.apply_notebook_rule_profile',return_value=(image,exclusions)):
            without=refine_all_terminal_contacts(image,info,{'ink_color_profile':profile},None)
            final=refine_all_terminal_contacts(image,info,{'ink_color_profile':profile},skeleton)
        self.assertEqual(without['terminal_count'],0)
        self.assertEqual(final['terminal_count'],2)
        for terminal in final['components'][0]['terminals']:
            self.assertAlmostEqual(terminal['position']['y'],138,delta=2)

    def test_reference_gray_and_cyan_rules_removed_with_shadows_and_rotations(self):
        for rule in ((179,179,179),(204,204,204),(217,217,217),(230,230,230),(230,230,179),(255,255,153)):
            for k in range(4):
                for exposure in (1.,.6):
                    with self.subTest(rule=rule,rotation=k,exposure=exposure):
                        image,info=ruled_scene(rule)
                        expected=np.zeros(image.shape[:2],np.uint8);cv2.line(expected,(0,75),(479,75),255,1)
                        image=np.rint(image*exposure).astype(np.uint8)
                        image=np.ascontiguousarray(np.rot90(image,k));expected=np.ascontiguousarray(np.rot90(expected,k))
                        profile=detect_notebook_rule_profile(image)
                        self.assertTrue(profile['active'],profile)
                        clean,mask=apply_notebook_rule_profile(image,profile)
                        self.assertGreater(np.count_nonzero(mask[expected>0]),np.count_nonzero(expected)*.90)
                        self.assertTrue(np.array_equal(image[mask==0],clean[mask==0]))

    def test_pencil_in_a_reference_gray_range_is_preserved(self):
        for shade in (179,190,204,217):
            with self.subTest(shade=shade):
                image,info=ruled_scene(ink=(shade,)*3)
                # 217 is outside the old dark-ink estimator: an explicit
                # reliable reference reproduces a genuinely light pencil.
                profile=estimate_component_ink(image,info)
                if shade==217:
                    profile=dict(active=True,reference_rgb=[217]*3,reference_hsv=[0,0,217],
                                 chromatic=False,channel_tolerance=22,hue_tolerance=12)
                self.assertTrue(profile['active'],profile)
                rules=detect_notebook_rule_profile(image,profile)
                clean,mask=apply_notebook_rule_profile(image,rules,profile)
                self.assertEqual(int(mask[235,310]),0)
                self.assertTrue(np.array_equal(clean[235,310],image[235,310]))
                self.assertTrue(np.array_equal(clean[138,255],image[138,255]))

    def test_one_long_wire_and_pale_symbols_are_not_a_notebook_family(self):
        image=np.full((360,480,3),255,np.uint8)
        cv2.line(image,(0,130),(479,130),(190,190,190),3)
        cv2.circle(image,(200,220),35,(204,204,204),3)
        cv2.circle(image,(300,270),6,(230,230,179),-1)
        rules=detect_notebook_rule_profile(image)
        self.assertFalse(rules['active'])
        clean,mask=apply_notebook_rule_profile(image,rules)
        self.assertTrue(np.array_equal(image,clean));self.assertFalse(np.any(mask))

    def test_slanted_and_grid_rules_survive_jpeg_as_rules(self):
        image,info=ruled_scene(rule=(204,204,204))
        for x in range(30,480,40):cv2.line(image,(x,0),(x,359),(204,204,204),1)
        transform=cv2.getRotationMatrix2D((240,180),12,1)
        image=cv2.warpAffine(image,transform,(480,360),borderValue=(255,255,255))
        image=cv2.imdecode(cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])[1],cv2.IMREAD_COLOR)
        rules=detect_notebook_rule_profile(image)
        self.assertTrue(rules['active']);self.assertGreaterEqual(len(rules['families']),2,rules)
        self.assertGreater(rules['removed_pixel_count'],2000)

    def test_wire_terminals_and_saved_junctions_share_the_same_rule_profile(self):
        image,info=ruled_scene(ink=(190,190,190))
        wires,mask,sk,_=detectar_fios(image,components_info=info)
        self.assertTrue(wires['notebook_rule_profile']['active'])
        self.assertEqual(int(mask[115,300]),0)
        # The existing component ROI ends at y=228 in this fixture. Check
        # continuity at a ruling crossing inside that intentional crop.
        self.assertGreater(np.count_nonzero(mask[180:220,310]),35)
        before=copy.deepcopy(wires);pixels=sk.copy()
        final=refine_all_terminal_contacts(image,info,wires,sk)
        self.assertTrue(np.array_equal(sk,pixels));self.assertEqual(wires,before)
        terms=final['components'][0]['terminals'];self.assertEqual(len(terms),2,final)
        for terminal in terms:self.assertAlmostEqual(terminal['position']['y'],138,delta=3)
        junctions=resolver_juncoes(image,wires,final)
        self.assertEqual(junctions['ink_color_profile'],wires['ink_color_profile'])
        self.assertEqual(junctions['notebook_rule_profile'],wires['notebook_rule_profile'])

    def test_old_saved_projects_without_rule_profile_still_work(self):
        image,info=ruled_scene()
        clean,mask=apply_notebook_rule_profile(image,None)
        self.assertTrue(np.array_equal(clean,image));self.assertFalse(np.any(mask))

    def test_missing_color_consensus_still_preserves_darker_crossing_pencil(self):
        image,info=ruled_scene(ink=(190,190,190))
        profile=detect_notebook_rule_profile(image)
        clean,mask=apply_notebook_rule_profile(image,profile)
        self.assertEqual(int(mask[235,310]),0)
        self.assertTrue(np.array_equal(clean[235,310],image[235,310]))

    def test_shadow_paper_does_not_become_a_wire_with_rule_shaped_holes(self):
        image,info=ruled_scene(ink=(100,100,100))
        cv2.rectangle(image,(280,60),(470,290),(100,100,100),-1)
        for y in range(75,280,40):cv2.line(image,(280,y),(470,y),(76,76,76),2)
        profile=detect_notebook_rule_profile(image,estimate_component_ink(image,info))
        self.assertTrue(profile['active'])
        with patch('backend.wire_detection.detect_notebook_rule_profile',return_value={'active':False,'removed_pixel_count':0}):
            previous,old_mask,old_sk,_=detectar_fios(image,components_info=info)
        current,mask,sk,_=detectar_fios(image,components_info=info)
        # This flat patch is shaded paper, not a filled electrical conductor.
        self.assertEqual(int(mask[180,400]),0)
        self.assertEqual(int(sk[180,400]),0)
        self.assertEqual(np.count_nonzero(mask[150:210,350:430]),0)
        # The actual visible component lead outside the shadow remains intact.
        self.assertGreater(np.count_nonzero(mask[138,100:140]),30)


if __name__=='__main__':unittest.main()
