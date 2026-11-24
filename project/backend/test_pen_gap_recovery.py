import copy
import unittest
import cv2
import numpy as np
from backend.circuit_connectivity import construir_conectividade,source_signature
from backend.pen_gap_recovery import gap_policy
from backend.test_circuit_connectivity import component


def wire(wid,points):return dict(wire_id=wid,points=[dict(x=x,y=y) for x,y in points])


def broken_series(gap=10,scale=10,thickness=1):
    image=np.full((300,500,3),255,np.uint8)
    segments=[wire('LEFT',[(20,150),(120,150)]),wire('RIGHT',[(120+gap,150),(350,150)])]
    for s in segments:
        p,q=s['points'];cv2.line(image,(p['x'],p['y']),(q['x'],q['y']),(20,20,20),thickness)
    c=component('R','resistor',(349,120,420,180),{'1':(350,150),'2':(419,150)})
    graph=dict(segments=segments,nodes=[],stage9={'nodes':[]})
    if scale is not None:graph['pen_gap_settings']=dict(pixels_per_mm=scale)
    return image,graph,dict(components=[c])


def same_wire_net(result,a='LEFT',b='RIGHT'):
    return any(a in n['wire_ids'] and b in n['wire_ids'] for n in result['nets'])


class PenGapRecoveryTests(unittest.TestCase):
    def test_blue_pen_resistor_gap_works_without_paper_measurements(self):
        image=np.full((600,1000,3),235,np.uint8)
        cv2.line(image,(40,150),(130,150),(120,35,20),3)
        cv2.line(image,(139,150),(155,150),(120,35,20),3)
        terms={'components':[component('R','resistor',(138,120,230,180),{'1':(139,150),'2':(229,150)})]}
        graph=dict(segments=[wire('W',[(40,150),(130,150)])],nodes=[],stage9={'nodes':[]})
        result=construir_conectividade(image,graph,terms)
        self.assertEqual(next(r for r in result['terminal_connections'] if r['terminal_ref']=='R:1')['status'],'attached_pen_gap')
        self.assertEqual(result['pen_gap_policy']['scale_source'],'estimated_from_pen_stroke')

    def test_wire_gap_smaller_than_two_mm_connects(self):
        image,graph,terms=broken_series(19)
        result=construir_conectividade(image,graph,terms)
        self.assertTrue(same_wire_net(result))
        self.assertEqual(len(result['recovered_pen_gaps']),1)
        self.assertAlmostEqual(result['recovered_pen_gaps'][0]['estimated_gap_mm'],1.9)

    def test_exactly_two_mm_and_larger_gaps_stay_open(self):
        for gap in (20,21,35):
            with self.subTest(gap=gap):
                image,graph,terms=broken_series(gap)
                result=construir_conectividade(image,graph,terms)
                self.assertFalse(same_wire_net(result));self.assertFalse(result['recovered_pen_gaps'])

    def test_missing_resistor_to_wire_contact_is_attached(self):
        image=np.full((300,500,3),255,np.uint8)
        cv2.line(image,(40,150),(130,150),(20,20,20),1)
        cv2.line(image,(139,150),(155,150),(20,20,20),1)
        terms={'components':[component('R','resistor',(138,120,230,180),{'1':(139,150),'2':(229,150)})]}
        graph=dict(segments=[wire('W',[(40,150),(130,150)])],nodes=[],stage9={'nodes':[]},pen_gap_settings={'pixels_per_mm':10})
        result=construir_conectividade(image,graph,terms)
        record=next(r for r in result['terminal_connections'] if r['terminal_ref']=='R:1')
        self.assertEqual(record['status'],'attached_pen_gap');self.assertEqual(record['wire_id'],'W')
        self.assertNotIn('R:1',[w.get('terminal_ref') for w in result['warnings'] if w['code']=='unconnected_pin'])

    def test_gap_in_visible_lead_before_resistor_contact_is_recovered(self):
        image=np.full((300,500,3),255,np.uint8)
        cv2.line(image,(40,150),(100,150),(20,20,20),1)
        cv2.line(image,(109,150),(165,150),(20,20,20),1)
        terms={'components':[component('R','resistor',(154,120,230,180),{'1':(155,150),'2':(229,150)})]}
        graph=dict(segments=[wire('W',[(40,150),(100,150)])],nodes=[],stage9={'nodes':[]},pen_gap_settings={'pixels_per_mm':10})
        result=construir_conectividade(image,graph,terms)
        self.assertEqual(next(r for r in result['terminal_connections'] if r['terminal_ref']=='R:1')['status'],'attached_pen_gap')

    def test_near_parallel_wires_are_not_joined(self):
        image,graph,terms=broken_series()
        graph['segments']=[wire('LEFT',[(30,140),(250,140)]),wire('RIGHT',[(30,150),(250,150)])]
        image[:]=255
        cv2.line(image,(30,140),(250,140),(20,20,20),1);cv2.line(image,(30,150),(250,150),(20,20,20),1)
        result=construir_conectividade(image,graph,terms)
        self.assertFalse(same_wire_net(result));self.assertFalse(result['recovered_pen_gaps'])

    def test_a_short_gap_to_middle_of_bus_forms_t_connection(self):
        image,graph,terms=broken_series()
        image[:]=255;cv2.line(image,(30,150),(120,150),(20,20,20),1);cv2.line(image,(130,60),(130,230),(20,20,20),1)
        graph['segments']=[wire('LEFT',[(30,150),(120,150)]),wire('RIGHT',[(130,60),(130,230)])]
        result=construir_conectividade(image,graph,terms)
        self.assertTrue(same_wire_net(result))

    def test_manual_disconnected_crossing_is_not_overridden(self):
        image,graph,terms=broken_series()
        graph['stage9']['nodes']=[dict(node_id='REVIEWED',position=dict(x=125,y=150),electrical_status='crossing_unconnected',manual_override=True,connection_groups=[['LEFT'],['RIGHT']])]
        result=construir_conectividade(image,graph,terms)
        self.assertFalse(same_wire_net(result));self.assertFalse(result['recovered_pen_gaps'])

    def test_a_component_body_is_not_bridged(self):
        image,graph,terms=broken_series(10)
        terms['components'].append(component('SMALL','resistor',(117,140,133,160),{'1':(118,150),'2':(132,150)}))
        result=construir_conectividade(image,graph,terms)
        self.assertFalse(same_wire_net(result))

    def test_equally_close_different_candidates_are_not_joined(self):
        image,graph,terms=broken_series(10)
        image[:]=255;cv2.line(image,(20,150),(120,150),(20,20,20),1)
        graph['segments']=[wire('LEFT',[(20,150),(120,150)])]
        for wid,y in [('UP',146),('DOWN',154)]:
            cv2.line(image,(130,y),(230,y),(20,20,20),1);graph['segments'].append(wire(wid,[(130,y),(230,y)]))
        result=construir_conectividade(image,graph,terms)
        self.assertFalse(any('LEFT' in n['wire_ids'] and len(n['wire_ids'])>1 for n in result['nets']))

    def test_adaptive_tolerance_is_unchanged_by_photo_resize(self):
        image,graph,terms=broken_series(gap=7,scale=None,thickness=3)
        first=construir_conectividade(image,graph,terms)
        self.assertTrue(same_wire_net(first));self.assertFalse(first['pen_gap_policy']['physical_scale_known'])
        large=cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
        scaled=copy.deepcopy(graph);scaled_terms=copy.deepcopy(terms)
        for s in scaled['segments']:
            for p in s['points']:p['x']*=2;p['y']*=2
        for c in scaled_terms['components']:
            for key in c['bbox']:c['bbox'][key]*=2
            for t in c['terminals']:
                for key in t['position']:t['position'][key]*=2
        second=construir_conectividade(large,scaled,scaled_terms)
        self.assertEqual(same_wire_net(first),same_wire_net(second))

    def test_inputs_are_preserved_and_recovery_is_deterministic(self):
        image,graph,terms=broken_series();before=(image.copy(),copy.deepcopy(graph),copy.deepcopy(terms))
        a=construir_conectividade(image,graph,terms)
        graph['segments'].reverse();terms['components'].reverse()
        b=construir_conectividade(image,graph,terms)
        self.assertEqual(a['nets'],b['nets']);self.assertEqual(a['recovered_pen_gaps'],b['recovered_pen_gaps'])
        self.assertTrue(np.array_equal(image,before[0]));self.assertEqual(list(reversed(graph['segments'])),before[1]['segments']);self.assertEqual(terms,before[2])

    def test_turning_off_gap_recovery_preserves_strict_continuous_ink_mode(self):
        image,graph,terms=broken_series();graph['pen_gap_settings']['enabled']=False
        result=construir_conectividade(image,graph,terms)
        self.assertFalse(same_wire_net(result));self.assertFalse(result['recovered_pen_gaps'])

    def test_policy_settings_invalidate_old_connectivity_cache(self):
        image,graph,terms=broken_series();before=source_signature(graph,terms)
        graph['pen_gap_settings']['enabled']=False
        self.assertNotEqual(before,source_signature(graph,terms))

if __name__=='__main__':unittest.main()
