import unittest

import cv2
import numpy as np

from backend.junction_detection import (
    analisar_clique_juncao,
    aplicar_correcoes_juncoes,
    resolver_juncoes,
)
from backend.wire_detection import detectar_fios


def _wire(wire_id, node_id, points):
    return {
        "wire_id": wire_id,
        "start_node": node_id,
        "end_node": None,
        "points": [{"x": x, "y": y} for x, y in points],
        "semantic_status": "unresolved_until_stage9",
    }


def _graph(branches):
    node_id = "NODE_0001"
    return {
        "schema_version": "stage8-wires-1.0",
        "nodes": [{
            "node_id": node_id,
            "position": {"x": 120, "y": 120},
            "segment_degree": len(branches),
            "kind_candidate": "junction_candidate",
        }],
        "segments": [
            _wire(f"WIRE_{index:04d}", node_id, branch)
            for index, branch in enumerate(branches, start=1)
        ],
    }


class JunctionDetectionStage9Tests(unittest.TestCase):
    def test_gap_between_endpoint_and_wire_is_recovered_as_t(self):
        image = np.full((240, 280, 3), 255, np.uint8)
        cv2.line(image, (30, 120), (250, 120), (20, 20, 20), 4)
        cv2.line(image, (140, 45), (140, 120), (20, 20, 20), 4)
        wires = {
            "schema_version": "stage8-wires-1.0",
            "nodes": [
                {"node_id": "N_LEFT", "position": {"x": 30, "y": 120}, "segment_degree": 1},
                {"node_id": "N_RIGHT", "position": {"x": 250, "y": 120}, "segment_degree": 1},
                {"node_id": "N_STEM", "position": {"x": 140, "y": 112}, "segment_degree": 1},
                {"node_id": "N_TOP", "position": {"x": 140, "y": 45}, "segment_degree": 1},
            ],
            "segments": [
                {"wire_id": "W_H", "start_node": "N_LEFT", "end_node": "N_RIGHT", "points": [{"x": 30, "y": 120}, {"x": 250, "y": 120}]},
                {"wire_id": "W_V", "start_node": "N_TOP", "end_node": "N_STEM", "points": [{"x": 140, "y": 45}, {"x": 140, "y": 112}]},
            ],
        }
        result = resolver_juncoes(image, wires)
        recovered = [
            node for node in result["nodes"]
            if node.get("decision_reason") == "endpoint_snapped_to_wire_forms_t"
        ]
        self.assertEqual(len(recovered), 1)
        self.assertEqual(result["synthesized_t_junction_count"], 1)
        self.assertTrue(recovered[0]["connected"])

    def test_user_can_add_missing_junction_by_click(self):
        stage9 = {
            "nodes": [],
            "segments": [
                {"wire_id": "W_H", "points": [{"x": 30, "y": 120}, {"x": 250, "y": 120}]},
                {"wire_id": "W_V", "points": [{"x": 140, "y": 45}, {"x": 140, "y": 115}]},
            ],
        }
        corrected = aplicar_correcoes_juncoes(stage9, [{
            "action": "add_junction", "x": 140, "y": 120,
        }])
        self.assertEqual(corrected["connected_junction_count"], 1)
        self.assertEqual(corrected["nodes"][0]["junction_type"], "manual_added_junction")

    def test_terminal_over_passing_wire_is_automatic_t(self):
        image = np.full((240, 280, 3), 255, np.uint8)
        cv2.line(image, (30, 120), (250, 120), (20, 20, 20), 4)
        cv2.line(image, (140, 70), (140, 120), (20, 20, 20), 4)
        wires = {
            "nodes": [
                {"node_id": "A", "position": {"x": 30, "y": 120}, "segment_degree": 1},
                {"node_id": "B", "position": {"x": 250, "y": 120}, "segment_degree": 1},
            ],
            "segments": [{
                "wire_id": "W_H", "start_node": "A", "end_node": "B",
                "points": [{"x": 30, "y": 120}, {"x": 250, "y": 120}],
            }],
        }
        terminals = {"components": [{
            "component_id": "D1",
            "terminals": [{"pin": "K", "position": {"x": 140, "y": 112}}],
        }]}
        result = resolver_juncoes(image, wires, terminals_info=terminals)
        found = [
            node for node in result["nodes"]
            if node.get("decision_reason") == "terminal_snapped_to_passing_wire_forms_t"
        ]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["attached_terminal_refs"], ["D1:K"])

    def test_manual_click_accepts_passing_wire_plus_terminal(self):
        stage9 = {
            "nodes": [],
            "segments": [{
                "wire_id": "W_H",
                "points": [{"x": 30, "y": 120}, {"x": 250, "y": 120}],
            }],
        }
        terminals = {"components": [{
            "component_id": "D1",
            "terminals": [{"pin": "K", "position": {"x": 140, "y": 112}}],
        }]}
        corrected = aplicar_correcoes_juncoes(
            stage9,
            [{"action": "add_junction", "x": 140, "y": 120}],
            terminals_info=terminals,
        )
        self.assertEqual(corrected["connected_junction_count"], 1)
        self.assertEqual(corrected["nodes"][0]["attached_terminal_refs"], ["D1:K"])

    def test_manual_click_accepts_two_segment_endpoints(self):
        stage9 = {
            "image_width": 800,
            "image_height": 600,
            "nodes": [],
            "segments": [
                {"wire_id": "W_LEFT", "points": [{"x": 40, "y": 120}, {"x": 138, "y": 120}]},
                {"wire_id": "W_STEM", "points": [{"x": 142, "y": 118}, {"x": 142, "y": 45}]},
            ],
        }
        corrected = aplicar_correcoes_juncoes(stage9, [{
            "action": "add_junction", "x": 140, "y": 120,
        }])
        node = corrected["nodes"][0]
        self.assertTrue(node["connected"])
        self.assertEqual(node["electrical_degree_candidate"], 3)
        self.assertEqual(set(node["incident_wire_ids"]), {"W_LEFT", "W_STEM"})

    def test_manual_click_accepts_endpoint_wire_plus_terminal(self):
        stage9 = {
            "nodes": [],
            "segments": [{
                "wire_id": "W_END",
                "points": [{"x": 30, "y": 120}, {"x": 140, "y": 120}],
            }],
        }
        terminals = {"components": [{
            "component_id": "Q1",
            "terminals": [{"pin": "B", "position": {"x": 145, "y": 120}}],
        }]}
        analysis = analisar_clique_juncao(stage9, 140, 120, terminals_info=terminals)
        self.assertTrue(analysis["can_confirm"])
        self.assertEqual(analysis["wire_count"], 1)
        self.assertEqual(analysis["terminal_count"], 1)
        corrected = aplicar_correcoes_juncoes(
            stage9,
            [{"action": "add_junction", "x": 140, "y": 120}],
            terminals_info=terminals,
        )
        self.assertEqual(corrected["nodes"][0]["attached_terminal_refs"], ["Q1:B"])

    def test_click_analysis_rejects_single_wire_without_terminal(self):
        stage9 = {
            "nodes": [],
            "segments": [{
                "wire_id": "W_ONLY",
                "points": [{"x": 30, "y": 120}, {"x": 140, "y": 120}],
            }],
        }
        analysis = analisar_clique_juncao(stage9, 140, 120)
        self.assertFalse(analysis["can_confirm"])
        with self.assertRaises(ValueError):
            aplicar_correcoes_juncoes(stage9, [{
                "action": "add_junction", "x": 140, "y": 120,
            }])

    def test_near_parallel_endpoints_do_not_create_false_t(self):
        image = np.full((220, 300, 3), 255, np.uint8)
        cv2.line(image, (25, 100), (275, 100), (20, 20, 20), 4)
        cv2.line(image, (40, 108), (150, 108), (20, 20, 20), 4)
        wires = {
            "nodes": [
                {"node_id": "A", "position": {"x": 25, "y": 100}, "segment_degree": 1},
                {"node_id": "B", "position": {"x": 275, "y": 100}, "segment_degree": 1},
                {"node_id": "C", "position": {"x": 40, "y": 108}, "segment_degree": 1},
                {"node_id": "D", "position": {"x": 150, "y": 108}, "segment_degree": 1},
            ],
            "segments": [
                {"wire_id": "W1", "start_node": "A", "end_node": "B", "points": [{"x": 25, "y": 100}, {"x": 275, "y": 100}]},
                {"wire_id": "W2", "start_node": "C", "end_node": "D", "points": [{"x": 40, "y": 108}, {"x": 150, "y": 108}]},
            ],
        }
        result = resolver_juncoes(image, wires)
        self.assertEqual(result["synthesized_t_junction_count"], 0)

    def test_manual_click_far_from_wires_is_rejected(self):
        stage9 = {
            "nodes": [],
            "segments": [
                {"wire_id": "W1", "points": [{"x": 20, "y": 20}, {"x": 80, "y": 20}]},
                {"wire_id": "W2", "points": [{"x": 20, "y": 80}, {"x": 80, "y": 80}]},
            ],
        }
        with self.assertRaises(ValueError):
            aplicar_correcoes_juncoes(stage9, [{
                "action": "add_junction", "x": 200, "y": 200,
            }])

    def test_stage8_to_stage9_end_to_end_crossing_without_dot(self):
        image = np.full((240, 320, 3), 255, np.uint8)
        cv2.line(image, (30, 120), (290, 120), (20, 20, 20), 4)
        cv2.line(image, (160, 25), (160, 215), (20, 20, 20), 4)
        wires, *_ = detectar_fios(
            image, components_info={}, ocr_info={}, min_segment_length=10
        )
        result = resolver_juncoes(image, wires)
        crossings = [
            node for node in result["nodes"]
            if node.get("electrical_status") == "crossing_unconnected"
        ]
        self.assertEqual(len(crossings), 1)
        self.assertEqual(result["review_count"], 0)

    def test_t_junction_is_connected(self):
        image = np.full((240, 240, 3), 255, np.uint8)
        branches = [
            [(120, 120), (30, 120)],
            [(120, 120), (210, 120)],
            [(120, 120), (120, 210)],
        ]
        for branch in branches:
            cv2.line(image, branch[0], branch[1], (20, 20, 20), 4)
        result = resolver_juncoes(image, _graph(branches))
        node = result["nodes"][0]
        self.assertEqual(node["electrical_status"], "junction_connected")
        self.assertEqual(node["junction_type"], "t_junction")
        self.assertTrue(node["connected"])
        self.assertTrue(result["ready_for_stage10"])

    def test_four_way_without_dot_is_two_separate_paths(self):
        image = np.full((240, 240, 3), 255, np.uint8)
        branches = [
            [(120, 120), (30, 120)],
            [(120, 120), (210, 120)],
            [(120, 120), (120, 30)],
            [(120, 120), (120, 210)],
        ]
        for branch in branches:
            cv2.line(image, branch[0], branch[1], (20, 20, 20), 4)
        result = resolver_juncoes(image, _graph(branches))
        node = result["nodes"][0]
        self.assertEqual(node["electrical_status"], "crossing_unconnected")
        self.assertFalse(node["dot_evidence"]["dot_present"])
        self.assertEqual(sorted(len(group) for group in node["connection_groups"]), [2, 2])

    def test_four_way_with_dot_is_connected(self):
        image = np.full((240, 240, 3), 255, np.uint8)
        branches = [
            [(120, 120), (30, 120)],
            [(120, 120), (210, 120)],
            [(120, 120), (120, 30)],
            [(120, 120), (120, 210)],
        ]
        for branch in branches:
            cv2.line(image, branch[0], branch[1], (20, 20, 20), 4)
        cv2.circle(image, (120, 120), 9, (20, 20, 20), -1, cv2.LINE_AA)
        result = resolver_juncoes(image, _graph(branches))
        node = result["nodes"][0]
        self.assertEqual(node["electrical_status"], "junction_connected")
        self.assertTrue(node["dot_evidence"]["dot_present"])
        self.assertEqual(len(node["connection_groups"]), 1)

    def test_manual_correction_overrides_automatic_decision(self):
        image = np.full((240, 240, 3), 255, np.uint8)
        branches = [
            [(120, 120), (30, 120)],
            [(120, 120), (210, 120)],
            [(120, 120), (120, 30)],
            [(120, 120), (120, 210)],
        ]
        result = resolver_juncoes(image, _graph(branches))
        corrected = aplicar_correcoes_juncoes(result, [{
            "node_id": "NODE_0001",
            "electrical_status": "junction_connected",
        }])
        node = corrected["nodes"][0]
        self.assertTrue(node["manual_override"])
        self.assertTrue(node["connected"])
        self.assertEqual(node["decision_reason"], "manual_user_correction")


if __name__ == "__main__":
    unittest.main()
