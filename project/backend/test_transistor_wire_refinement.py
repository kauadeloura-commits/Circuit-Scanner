import unittest
from unittest.mock import patch
import numpy as np
import cv2

from backend.transistor_wire_refinement import (
    _assign_roles_arrow_base_leftover,
    _detect_symbol_role_evidence,
    refine_transistor_terminals_from_wires,
)


def comp_transistor(ctype="transistor_npn"):
    return {
        "component_id": "Q1",
        "detection_id": "Q1",
        "type": ctype,
        "bbox": {"x1": 60, "y1": 50, "x2": 140, "y2": 150, "width": 80, "height": 100, "center_x": 100, "center_y": 100},
        "terminals": [
            {"pin":"B","position":{"x":90,"y":95},"confidence":0.4,"source":"trained_terminal_keypoint_model"},
            {"pin":"C","position":{"x":95,"y":95},"confidence":0.4,"source":"trained_terminal_keypoint_model"},
            {"pin":"E","position":{"x":105,"y":135},"confidence":0.8,"source":"trained_terminal_keypoint_model"},
        ],
        "terminal_detection": {"method":"trained_terminal_keypoint_model"},
        "review_required": True,
        "review_reasons": ["old_transistor_error"],
    }


def skeleton_three_paths():
    sk = np.zeros((220,220), np.uint8)
    cv2.line(sk, (20,100), (60,100), 255, 1)   # base left
    cv2.line(sk, (120,15), (120,50), 255, 1)   # collector top
    cv2.line(sk, (120,150), (120,190), 255, 1) # emitter bottom
    return sk


class TransistorWireRefinementTests(unittest.TestCase):
    @patch("backend.transistor_wire_refinement._detect_transistor_fork")
    @patch("backend.transistor_wire_refinement._symbol_image_variants")
    def test_multi_pass_prefers_consensus_arrow_and_base(self, variants, detect):
        image = np.full((220, 220, 3), 255, np.uint8)
        variants.return_value = [("original", image), ("clahe", image), ("sharp", image)]
        common_meta = {
            "success": True, "branch_distinct": True, "base_candidates": 3,
            "expected_arrow_direction": "outward", "detected_arrow_direction": "outward",
            "class_arrow_consistent": True, "base_axis": "vertical", "fork_side": "right",
            "upper_arrow_score": 4.0, "lower_arrow_score": 12.0,
        }
        agree_points = {"B": (60.0, 100.0), "C": (120.0, 50.0), "E": (120.0, 150.0)}
        disagree_points = {"B": (61.0, 100.0), "C": (120.0, 150.0), "E": (120.0, 50.0)}
        detect.side_effect = [
            (agree_points, {**common_meta, "source": "pass1", "arrow_branch": "lower"}),
            (agree_points, {**common_meta, "source": "pass2", "arrow_branch": "lower"}),
            (disagree_points, {**common_meta, "source": "pass3", "arrow_branch": "upper", "upper_arrow_score": 12.0, "lower_arrow_score": 4.0}),
        ]
        component = comp_transistor()
        summary, hypothesis = _detect_symbol_role_evidence(image, component)
        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["consensus_count"], 2)
        self.assertFalse(summary["multi_pass_uncertain"])
        self.assertEqual(hypothesis["points"]["E"], {"x": 120.0, "y": 150.0})

    def _draw_bjt_and_external_skeleton(self, ctype, arrow_direction):
        image = np.full((220, 220, 3), 245, np.uint8)
        cv2.line(image, (95, 65), (95, 155), (20, 20, 20), 4)
        cv2.line(image, (40, 110), (95, 110), (20, 20, 20), 3)
        upper = ((95, 90), (155, 55)); lower = ((95, 130), (155, 165))
        cv2.line(image, *upper, (20, 20, 20), 3)
        cv2.line(image, *lower, (20, 20, 20), 3)
        (ax, ay), (bx, by) = lower
        tip_t, tail_t = (0.74, 0.54) if arrow_direction == "outward" else (0.38, 0.60)
        tip = (int(ax + (bx-ax)*tip_t), int(ay + (by-ay)*tip_t))
        cx, cy = ax + (bx-ax)*tail_t, ay + (by-ay)*tail_t
        dx, dy = bx-ax, by-ay; norm = (dx*dx+dy*dy)**0.5
        px, py = -dy/norm, dx/norm
        cv2.line(image, (int(cx+px*9), int(cy+py*9)), tip, (20,20,20), 2)
        cv2.line(image, (int(cx-px*9), int(cy-py*9)), tip, (20,20,20), 2)
        skeleton = np.zeros((220, 220), np.uint8)
        cv2.line(skeleton, (20,110), (58,110), 255, 1)
        cv2.line(skeleton, (155,55), (195,32), 255, 1)
        cv2.line(skeleton, (155,165), (195,192), 255, 1)
        component = comp_transistor(ctype)
        component["bbox"] = {"x1":55,"y1":45,"x2":170,"y2":180,"width":115,"height":135}
        component["terminals"] = []
        component["terminal_detection"] = {"method":"transistor_points_after_stage8_wires"}
        return image, skeleton, component

    def test_real_npn_arrow_sets_E_bar_sets_B_and_leftover_sets_C(self):
        image, skeleton, component = self._draw_bjt_and_external_skeleton("transistor_npn", "outward")
        out, summary = refine_transistor_terminals_from_wires(image, {"components":[component]}, {}, skeleton)
        self.assertEqual(summary["refined_count"], 1)
        pins = {t["pin"]: t["position"] for t in out["components"][0]["terminals"]}
        self.assertLess(pins["B"]["x"], 75)
        self.assertGreater(pins["E"]["y"], 145)
        self.assertLess(pins["C"]["y"], 75)
        td = out["components"][0]["terminal_detection"]
        self.assertEqual(td["emitter_identity_source"], "physical_arrow_then_base_bar")
        self.assertEqual(td["base_identity_source"], "physical_arrow_then_base_bar")
        self.assertEqual(td["collector_identity_source"], "remaining_wire_after_E_and_B")

    def test_real_pnp_uses_inward_arrow_before_base_and_collector(self):
        image, skeleton, component = self._draw_bjt_and_external_skeleton("transistor_pnp", "inward")
        out, summary = refine_transistor_terminals_from_wires(image, {"components":[component]}, {}, skeleton)
        self.assertEqual(summary["refined_count"], 1)
        pins = {t["pin"]: t["position"] for t in out["components"][0]["terminals"]}
        self.assertLess(pins["B"]["x"], 75)
        self.assertGreater(pins["E"]["y"], 145)
        self.assertLess(pins["C"]["y"], 75)

    @patch("backend.transistor_wire_refinement._model_evidence")
    def test_three_stage8_paths_define_distinct_bce(self, evidence):
        evidence.return_value = {
            "source":"trained_terminal_keypoint_model",
            "accepted":True,
            "points": {
                "B":{"x":58,"y":100},
                "C":{"x":120,"y":48},
                "E":{"x":120,"y":152},
            },
            "keypoint_confidences":{"B":0.8,"C":0.7,"E":0.9},
            "geometry_sane":True,
        }
        info={"components":[comp_transistor()]}
        out,summary=refine_transistor_terminals_from_wires(np.full((220,220,3),255,np.uint8),info,{},skeleton_three_paths())
        self.assertEqual(summary["refined_count"],1)
        terms={t["pin"]:t for t in out["components"][0]["terminals"]}
        self.assertEqual(set(terms),{"B","C","E"})
        self.assertLess(terms["B"]["position"]["x"],75)
        self.assertLess(terms["C"]["position"]["y"],70)
        self.assertGreater(terms["E"]["position"]["y"],135)
        pts=[(terms[p]["position"]["x"],terms[p]["position"]["y"]) for p in ("B","C","E")]
        self.assertTrue(all(np.hypot(pts[i][0]-pts[j][0],pts[i][1]-pts[j][1])>20 for i in range(3) for j in range(i+1,3)))
        self.assertEqual(out["components"][0]["terminal_detection"]["method"],"stage8_wires_then_arrow_E_then_base_B_then_leftover_C")

    def test_training_fallback_follows_emitter_then_base_then_leftover(self):
        tri = [
            {"point": (60.0, 100.0), "angle_degrees": 0.0},
            {"point": (120.0, 50.0), "angle_degrees": 0.0},
            {"point": (140.0, 100.0), "angle_degrees": 90.0},
        ]
        bbox = {"x1": 60, "y1": 50, "x2": 140, "y2": 150}
        evidence = {
            "accepted": True,
            "hypotheses": [{
                "source": "known_training_ground_truth",
                "points": {
                    "B": {"x": 58, "y": 100},
                    "C": {"x": 120, "y": 48},
                    "E": {"x": 142, "y": 100},
                },
                "keypoint_confidences": {"B": 1.0, "C": 1.0, "E": 1.0},
                "weight": 5.0,
            }],
        }
        mapping, meta = _assign_roles_arrow_base_leftover(tri, evidence=evidence, bbox=bbox)
        self.assertEqual(mapping, {"B": 0, "C": 1, "E": 2})
        self.assertFalse(meta["identity_uncertain"])
        self.assertEqual(meta["collector"]["source"], "remaining_wire_after_E_and_B")

    def test_physical_arrow_and_bar_override_completely_swapped_pose_names(self):
        tri = [
            {"point": (60.0, 100.0), "angle_degrees": 0.0},
            {"point": (120.0, 50.0), "angle_degrees": 90.0},
            {"point": (120.0, 150.0), "angle_degrees": 90.0},
        ]
        bbox = {"x1": 60, "y1": 50, "x2": 140, "y2": 150}
        physical = {
            "B": {"x": 60, "y": 100},
            "E": {"x": 120, "y": 150},
        }
        swapped = {
            "B": {"x": 120, "y": 50},
            "C": {"x": 120, "y": 150},
            "E": {"x": 60, "y": 100},
        }
        evidence = {"hypotheses": [
            {"source": "trained_terminal_keypoint_model", "points": swapped,
             "keypoint_confidences": {"B": .65, "C": .65, "E": .65}, "weight": 2.35},
            {"source": "physical_arrow_then_base_bar", "points": physical,
             "keypoint_confidences": {"B": .91, "E": .96}, "weight": 10.0, "physical": True},
        ]}
        mapping, meta = _assign_roles_arrow_base_leftover(tri, evidence=evidence, bbox=bbox)
        self.assertEqual(mapping, {"B": 0, "C": 1, "E": 2})
        self.assertTrue(meta["emitter"]["physical_arrow"])
        self.assertTrue(meta["base"]["physical_base_bar"])
        self.assertEqual(meta["sequence"], ["stage8_wires", "detector_NPN_PNP", "arrow_emitter", "base_bar", "collector_leftover"])

    def test_collector_is_never_selected_by_model(self):
        tri = [
            {"point": (60.0, 100.0), "angle_degrees": 0.0},
            {"point": (120.0, 50.0), "angle_degrees": 90.0},
            {"point": (120.0, 150.0), "angle_degrees": 90.0},
        ]
        bbox = {"x1": 60, "y1": 50, "x2": 140, "y2": 150}
        evidence = {"hypotheses": [{
            "source": "physical_arrow_then_base_bar", "physical": True,
            "points": {"E": {"x": 120, "y": 150}, "B": {"x": 60, "y": 100}},
            "keypoint_confidences": {"E": .96, "B": .91},
        }, {
            "source": "trained_terminal_keypoint_model",
            "points": {"B": {"x": 120, "y": 50}, "C": {"x": 60, "y": 100}, "E": {"x": 120, "y": 150}},
            "keypoint_confidences": {"B": .99, "C": .99, "E": .99},
        }]}
        mapping, meta = _assign_roles_arrow_base_leftover(tri, evidence=evidence, bbox=bbox)
        self.assertEqual(mapping["E"], 2)
        self.assertEqual(mapping["B"], 0)
        self.assertEqual(mapping["C"], 1)
        self.assertEqual(meta["collector"]["source"], "remaining_wire_after_E_and_B")

    @patch("backend.transistor_wire_refinement._model_evidence")
    def test_two_paths_do_not_invent_transistor_points(self, evidence):
        evidence.return_value={"source":"trained_terminal_keypoint_model","accepted":False,"points":{},"keypoint_confidences":{}}
        sk=skeleton_three_paths(); sk[150:,:]=0
        info={"components":[comp_transistor()]}
        out,summary=refine_transistor_terminals_from_wires(np.full((220,220,3),255,np.uint8),info,{},sk)
        self.assertEqual(summary["refined_count"],0)
        self.assertEqual(out["components"][0]["terminals"],[])
        self.assertTrue(out["components"][0]["review_required"])

    def test_manual_transistor_is_never_overwritten(self):
        c=comp_transistor()
        for t in c["terminals"]:
            t["source"]="manual_ground_truth"
        c["terminal_detection"]={"method":"user_keypoint_correction","status":"manual_ground_truth"}
        before=[dict(t) for t in c["terminals"]]
        out,summary=refine_transistor_terminals_from_wires(np.full((220,220,3),255,np.uint8),{"components":[c]}, {}, skeleton_three_paths())
        self.assertEqual(summary["manual_preserved_count"],1)
        self.assertEqual(out["components"][0]["terminals"],before)


    @patch("backend.transistor_wire_refinement._model_evidence")
    def test_bad_old_bce_do_not_control_stage8_wire_extraction(self, evidence):
        from backend.wire_detection import detectar_fios
        evidence.return_value = {
            "source":"trained_terminal_keypoint_model", "accepted":True,
            "points":{"B":{"x":60,"y":100},"C":{"x":120,"y":50},"E":{"x":120,"y":150}},
            "keypoint_confidences":{"B":0.8,"C":0.8,"E":0.9}, "geometry_sane":True,
        }
        image=np.full((220,220,3),255,np.uint8)
        # Real external conductor paths.
        cv2.line(image,(15,100),(63,100),(0,0,0),3)
        cv2.line(image,(120,10),(120,53),(0,0,0),3)
        cv2.line(image,(120,147),(120,205),(0,0,0),3)
        c=comp_transistor()
        # Deliberately collapse all previous automatic points in the wrong place.
        for t in c["terminals"]:
            t["position"]={"x":90,"y":90}
        info={"components":[c]}
        wires,mask,skel,_=detectar_fios(image,components_info=info,ocr_info=None,min_segment_length=5)
        out,summary=refine_transistor_terminals_from_wires(image,info,wires,skel)
        self.assertEqual(summary["refined_count"],1)
        terms={t["pin"]:t["position"] for t in out["components"][0]["terminals"]}
        self.assertLess(terms["B"]["x"],75)
        self.assertLess(terms["C"]["y"],70)
        self.assertGreater(terms["E"]["y"],135)

    def test_non_transistor_is_untouched(self):
        r={"component_id":"R1","type":"resistor","bbox":{"x1":10,"y1":10,"x2":30,"y2":80},"terminals":[{"pin":"1","position":{"x":20,"y":10}},{"pin":"2","position":{"x":20,"y":80}}]}
        out,summary=refine_transistor_terminals_from_wires(np.full((100,100,3),255,np.uint8),{"components":[r]}, {}, np.zeros((100,100),np.uint8))
        self.assertEqual(summary["transistor_count"],0)
        self.assertEqual(out["components"][0],r)

if __name__ == "__main__":
    unittest.main()
