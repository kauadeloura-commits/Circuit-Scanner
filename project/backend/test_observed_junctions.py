"""Electrical branches must be visible in the original ink, not just the graph."""
import copy
import unittest
import cv2
import numpy as np
from backend.junction_detection import resolver_juncoes
from backend.test_junction_detection import _graph


class ObservedJunctionTests(unittest.TestCase):
    def test_skeleton_spur_on_a_bend_is_not_a_t(self):
        for width in (3, 8, 12):
            with self.subTest(width=width):
                image=np.full((240,240,3),255,np.uint8)
                branches=[[(120,120),(30,120)],[(120,120),(120,210)],[(120,120),(130,110)]]
                for branch in branches[:2]:cv2.line(image,*branch,(20,20,20),width)
                result=resolver_juncoes(image,_graph(branches))
                self.assertEqual(result['connected_junction_count'],0)
                self.assertEqual(result['nodes'][0]['electrical_status'],'wire_path')

    def test_nearby_terminal_on_a_straight_wire_does_not_invent_a_third_arm(self):
        image=np.full((240,280,3),255,np.uint8)
        cv2.line(image,(30,120),(250,120),(20,20,20),4)
        graph={'nodes':[],'segments':[{'wire_id':'W','points':[{'x':30,'y':120},{'x':250,'y':120}]}]}
        terminals={'components':[{'component_id':'R','terminals':[{'pin':'1','position':{'x':140,'y':118}}]}]}
        result=resolver_juncoes(image,graph,terminals)
        self.assertEqual(result['synthesized_terminal_t_junction_count'],0)

    def test_a_small_white_gap_is_not_an_ink_bridge(self):
        image=np.full((240,280,3),255,np.uint8)
        cv2.line(image,(30,120),(250,120),(20,20,20),1)
        cv2.line(image,(140,40),(140,117),(20,20,20),1)
        graph={'nodes':[{'node_id':'E','position':{'x':140,'y':117},'segment_degree':1}],
               'segments':[{'wire_id':'H','points':[{'x':30,'y':120},{'x':250,'y':120}]},
                           {'wire_id':'V','end_node':'E','points':[{'x':140,'y':40},{'x':140,'y':117}]}]}
        self.assertEqual(resolver_juncoes(image,graph)['synthesized_t_junction_count'],0)

    def test_a_four_arm_crossing_does_not_become_t_when_one_graph_arm_is_missing(self):
        image=np.full((240,240,3),255,np.uint8)
        branches=[[(120,120),(30,120)],[(120,120),(210,120)],[(120,120),(120,210)]]
        for branch in branches+[[ (120,120),(120,30)]]:cv2.line(image,*branch,(20,20,20),4)
        node=resolver_juncoes(image,_graph(branches))['nodes'][0]
        self.assertEqual(node['electrical_status'],'crossing_unconnected')
        self.assertEqual(sorted(map(len,node['connection_groups'])),[1,2])

    def test_dot_center_is_on_the_actual_dot_and_confidence_is_bounded(self):
        image=np.full((240,240,3),255,np.uint8)
        branches=[[(120,120),(30,120)],[(120,120),(210,120)],[(120,120),(120,30)],[(120,120),(120,210)]]
        for branch in branches:cv2.line(image,*branch,(20,20,20),4)
        cv2.circle(image,(124,122),9,(20,20,20),-1)
        graph=_graph(branches);original=copy.deepcopy(graph)
        node=resolver_juncoes(image,graph)['nodes'][0]
        self.assertTrue(node['dot_evidence']['dot_present'])
        self.assertLessEqual(node['confidence'],1)
        self.assertLessEqual(abs(node['position']['x']-124),1)
        self.assertLessEqual(abs(node['position']['y']-122),1)
        self.assertEqual(graph,original)

    def test_multiple_graph_candidates_at_one_dot_form_one_junction(self):
        image=np.full((240,240,3),255,np.uint8)
        branches=[[(120,120),(30,120)],[(120,120),(210,120)],[(120,120),(120,30)],[(120,120),(120,210)]]
        for branch in branches:cv2.line(image,*branch,(20,20,20),4)
        cv2.circle(image,(120,120),9,(20,20,20),-1)
        graph=_graph(branches)
        duplicate=copy.deepcopy(graph['nodes'][0]);duplicate['node_id']='SECOND';duplicate['position']['x']=123
        graph['nodes'].append(duplicate)
        extra=copy.deepcopy(graph['segments'])
        for i,wire in enumerate(extra):wire.update(wire_id=f'EXTRA_{i}',start_node='SECOND')
        graph['segments']+=extra
        result=resolver_juncoes(image,graph)
        self.assertEqual(result['connected_junction_count'],1)
        dot=next(n for n in result['nodes'] if n['connected'])
        self.assertEqual(len(dot['incident_wire_ids']),8)

    def test_two_nearby_t_junctions_are_not_merged_by_the_gap_snap_radius(self):
        image=np.full((240,280,3),255,np.uint8)
        cv2.line(image,(30,120),(250,120),(20,20,20),3)
        graph={'nodes':[],'segments':[{'wire_id':'H','points':[{'x':30,'y':120},{'x':250,'y':120}]}]}
        for i,x in enumerate((130,149)):
            cv2.line(image,(x,45),(x,120),(20,20,20),3)
            graph['nodes'].append({'node_id':f'E{i}','position':{'x':x,'y':112},'segment_degree':1})
            graph['segments'].append({'wire_id':f'V{i}','end_node':f'E{i}','points':[{'x':x,'y':45},{'x':x,'y':112}]})
        result=resolver_juncoes(image,graph)
        self.assertEqual(result['synthesized_t_junction_count'],2)


if __name__=='__main__':unittest.main()
