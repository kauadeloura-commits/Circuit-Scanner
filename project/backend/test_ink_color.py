import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend.ink_color import estimate_component_ink, component_ink_mask
from backend.wire_detection import detectar_fios
from backend.junction_detection import resolver_juncoes


BLUE = (160, 65, 25)  # BGR


def symbol(image, x, y, color=BLUE, confidence=0.96):
    cv2.rectangle(image, (x, y), (x + 35, y + 45), color, 4)
    return {"component_id": f"R_{x}_{y}", "type": "resistor", "confidence": confidence,
            "bbox": {"x1": x - 5, "y1": y - 5, "x2": x + 40, "y2": y + 50}}


def scene(color=BLUE):
    image = np.full((260, 400, 3), 255, np.uint8)
    components = {"components": [symbol(image, 70, 50, color), symbol(image, 270, 150, color)]}
    cv2.line(image, (105, 73), (180, 73), color, 4)
    cv2.line(image, (180, 73), (180, 172), color, 4)
    cv2.line(image, (180, 172), (270, 172), color, 4)
    return image, components


def crossing_graph():
    branches = [[(200, 120), (145, 120)], [(200, 120), (255, 120)],
                [(200, 120), (200, 65)], [(200, 120), (200, 175)]]
    return {"nodes": [{"node_id": "N", "position": {"x": 200, "y": 120}}],
            "segments": [{"wire_id": f"W{i}", "start_node": "N", "end_node": None,
                          "points": [{"x": x, "y": y} for x, y in branch]}
                         for i, branch in enumerate(branches)]}


class ComponentInkColorTests(unittest.TestCase):
    def test_project_pipeline_and_saved_junction_rerun_use_original_color(self):
        from backend import main
        image, components = scene()
        cv2.line(image, (0, 115), (399, 115), (20, 20, 20), 1)
        detections = {"detections": components["components"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project_dir = root / "COLOR_TEST"
            project_dir.mkdir()
            cv2.imwrite(str(project_dir / "original.png"), image)
            cv2.imwrite(str(project_dir / "processed.png"), cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            project = {"stored_filename": "original.png", "processed_filename": "processed.png",
                       "detections_info": json.dumps(detections)}
            with patch.object(main, "PROJECTS_DIR", root), patch.object(main, "atualizar_fios"):
                for source in ("original", "processed"):
                    result = main._executar_etapa8_fios("COLOR_TEST", project, source=source)
                    self.assertTrue(result["ink_color_profile"]["active"])
                    self.assertEqual(result["stage9"]["ink_color_profile"], result["ink_color_profile"])
                    mask = cv2.imread(str(project_dir / "wires_mask.png"), cv2.IMREAD_GRAYSCALE)
                    self.assertEqual(int(mask[115, 230]), 255)
                    self.assertLess(int(mask[150, 180]), 100)
                    project["wires_info"] = json.dumps(result)
                    junctions = main._executar_etapa9_juncoes("COLOR_TEST", project)
                    self.assertEqual(junctions["ink_color_profile"], result["ink_color_profile"])

    def test_shadowed_photo_with_muted_blue_and_tilted_vertical_gray_rulings(self):
        image = np.full((480, 400, 3), 170, np.uint8)
        for x in range(25, 400, 45):
            cv2.line(image, (x, 0), (x + 20, 479), (122, 122, 122), 2)
        ink = (112, 76, 83)
        components = {"detections": [symbol(image, 75, 100, ink), symbol(image, 255, 330, ink)]}
        cv2.line(image, (110, 122), (180, 122), ink, 3)
        cv2.line(image, (180, 122), (180, 352), ink, 3)
        encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
        photo = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        profile = estimate_component_ink(photo, components)
        self.assertTrue(profile["active"])
        self.assertTrue(profile["chromatic"])
        mask = component_ink_mask(photo, profile)
        self.assertGreater(np.count_nonzero(mask[160:310, 179:182]), 230)
        self.assertLess(np.count_nonzero(mask[:70]), 15)

    def test_mismatched_reference_dimensions_disable_color_filter(self):
        image, components = scene()
        result, *_ = detectar_fios(image, components_info=components,
                                   color_reference_image=image[:200])
        self.assertFalse(result["ink_color_profile"]["active"])
        self.assertEqual(result["ink_color_profile"]["reason"], "color_reference_dimension_mismatch")

    def test_terminal_bridge_in_a_different_color_is_not_a_junction(self):
        image, components = scene()
        cv2.line(image, (130, 120), (270, 120), BLUE, 4)
        cv2.line(image, (200, 112), (200, 120), (20, 20, 20), 3)
        wires = {"nodes": [{"node_id": "L", "position": {"x": 130, "y": 120}},
                           {"node_id": "R", "position": {"x": 270, "y": 120}}],
                 "segments": [{"wire_id": "H", "start_node": "L", "end_node": "R",
                               "points": [{"x": 130, "y": 120}, {"x": 270, "y": 120}]}]}
        terminals = {"components": [{"component_id": "D", "terminals": [
            {"pin": "K", "position": {"x": 200, "y": 112}}]}]}
        self.assertEqual(resolver_juncoes(image, wires, terminals)["synthesized_terminal_t_junction_count"], 1)
        wires["ink_color_profile"] = estimate_component_ink(image, components)
        self.assertEqual(resolver_juncoes(image, wires, terminals)["synthesized_terminal_t_junction_count"], 0)

    def test_ten_blue_symbols_average_ink_not_white_paper_or_black_rulings(self):
        image = np.full((400, 700, 3), 255, np.uint8)
        for y in range(30, 400, 35):
            cv2.line(image, (0, y), (699, y), (20, 20, 20), 2)
        components = {"detections": [symbol(image, 40 + (i % 5) * 130, 70 + (i // 5) * 190)
                                     for i in range(10)]}
        profile = estimate_component_ink(image, components)
        self.assertTrue(profile["active"])
        self.assertEqual(profile["consensus_component_count"], 10)
        self.assertLess(np.max(np.abs(np.array(profile["reference_rgb"]) - BLUE[::-1])), 5)
        json.dumps(profile)  # profile is persisted in wires.json

    def test_blue_wires_survive_black_notebook_lines_without_false_branches(self):
        image, components = scene()
        for y in range(25, 240, 30):
            cv2.line(image, (0, y), (399, y), (20, 20, 20), 1)
        result, mask, skeleton, _ = detectar_fios(image, components_info=components)
        self.assertTrue(result["ink_color_profile"]["active"])
        self.assertEqual(int(mask[115, 220]), 0)
        self.assertGreater(np.count_nonzero(mask[:, 180]), 85)
        self.assertGreater(np.count_nonzero(skeleton), 140)
        self.assertEqual(result["junction_candidate_count"], 0)

    def test_light_blue_circuit_is_not_erased_as_pale_blue_paper(self):
        color = (210, 135, 70)
        image, components = scene(color)
        cv2.line(image, (0, 115), (399, 115), (30, 30, 30), 1)
        result, mask, *_ = detectar_fios(image, components_info=components)
        self.assertTrue(result["ink_color_profile"]["active"])
        self.assertGreater(np.count_nonzero(mask[:, 180]), 90)
        self.assertEqual(int(mask[115, 230]), 0)

    def test_black_rgb_code_rejects_a_different_gray_in_both_directions(self):
        for symbol_gray, ruling_gray in [(20, 85), (85, 20)]:
            with self.subTest(symbol_gray=symbol_gray):
                image, components = scene((symbol_gray,) * 3)
                cv2.line(image, (0, 115), (399, 115), (ruling_gray,) * 3, 1)
                result, mask, *_ = detectar_fios(image, components_info=components)
                self.assertFalse(result["ink_color_profile"]["chromatic"])
                self.assertEqual(int(mask[115, 230]), 0)
                self.assertGreater(np.count_nonzero(mask[:, 180]), 85)

    def test_same_black_code_keeps_real_wires(self):
        image, components = scene((20, 20, 20))
        profile = estimate_component_ink(image, components)
        mask = component_ink_mask(image, profile)
        self.assertEqual(int(mask[73, 150]), 255)
        self.assertEqual(profile["reference_hex"], "#141414")

    def test_uncertain_symbols_cannot_set_the_reference(self):
        image, components = scene()
        for c in components["components"]:
            c["confidence"] = 0.30
        self.assertFalse(estimate_component_ink(image, components)["active"])
        components["components"][0]["manual_reviewed"] = True
        self.assertTrue(estimate_component_ink(image, components)["active"])

    def test_visual_review_is_excluded_even_at_high_confidence(self):
        image, components = scene()
        for c in components["components"]:
            c["review_required"] = True
        self.assertFalse(estimate_component_ink(image, components)["active"])

    def test_one_outlier_does_not_mix_black_into_blue_mean(self):
        image = np.full((270, 540, 3), 255, np.uint8)
        comps = [symbol(image, 30 + 100 * i, 60, BLUE if i < 4 else (20, 20, 20))
                 for i in range(5)]
        profile = estimate_component_ink(image, {"components": comps})
        self.assertTrue(profile["active"])
        self.assertEqual(profile["consensus_component_count"], 4)
        self.assertTrue(profile["chromatic"])

    def test_incompatible_colors_without_consensus_do_not_guess(self):
        image = np.full((240, 400, 3), 255, np.uint8)
        comps = [symbol(image, 50, 50), symbol(image, 270, 140, (20, 20, 160))]
        self.assertEqual(estimate_component_ink(image, {"components": comps})["reason"],
                         "component_colors_without_consensus")

    def test_original_color_reference_is_used_for_processed_monochrome_image(self):
        image, components = scene()
        cv2.line(image, (0, 115), (399, 115), (20, 20, 20), 1)
        processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result, mask, *_ = detectar_fios(processed, components_info=components,
                                        color_reference_image=image)
        self.assertTrue(result["ink_color_profile"]["active"])
        self.assertEqual(int(mask[115, 230]), 0)
        self.assertGreater(np.count_nonzero(mask[:, 180]), 85)

    def test_grayscale_or_missing_certainty_retains_legacy_detection(self):
        image = np.full((180, 280), 255, np.uint8)
        cv2.line(image, (30, 90), (250, 90), 20, 4)
        result, *_ = detectar_fios(image)
        self.assertFalse(result["ink_color_profile"]["active"])
        self.assertGreaterEqual(result["segment_count"], 1)

    def test_black_dot_cannot_connect_blue_crossing_but_blue_dot_can(self):
        for dot_color, connected in [((20, 20, 20), False), (BLUE, True)]:
            with self.subTest(dot_color=dot_color):
                image, components = scene()
                cv2.line(image, (145, 120), (255, 120), BLUE, 4)
                cv2.line(image, (200, 65), (200, 175), BLUE, 4)
                cv2.circle(image, (200, 120), 10, dot_color, -1)
                wires = crossing_graph()
                wires["ink_color_profile"] = estimate_component_ink(image, components)
                result = resolver_juncoes(image, wires)
                self.assertEqual(result["nodes"][0]["connected"], connected)

    def test_black_bridge_cannot_synthesize_t_between_blue_wires(self):
        image, components = scene()
        cv2.line(image, (130, 120), (270, 120), BLUE, 4)
        cv2.line(image, (200, 70), (200, 112), BLUE, 4)
        cv2.line(image, (200, 113), (200, 120), (20, 20, 20), 3)
        wires = {"nodes": [{"node_id": "L", "position": {"x": 130, "y": 120}},
                           {"node_id": "R", "position": {"x": 270, "y": 120}},
                           {"node_id": "TOP", "position": {"x": 200, "y": 70}},
                           {"node_id": "TIP", "position": {"x": 200, "y": 112}}],
                 "segments": [{"wire_id": "H", "start_node": "L", "end_node": "R",
                               "points": [{"x": 130, "y": 120}, {"x": 270, "y": 120}]},
                              {"wire_id": "V", "start_node": "TOP", "end_node": "TIP",
                               "points": [{"x": 200, "y": 70}, {"x": 200, "y": 112}]}]}
        self.assertEqual(resolver_juncoes(image, wires)["synthesized_t_junction_count"], 1)
        wires["ink_color_profile"] = estimate_component_ink(image, components)
        self.assertEqual(resolver_juncoes(image, wires)["synthesized_t_junction_count"], 0)


if __name__ == "__main__":
    unittest.main()
