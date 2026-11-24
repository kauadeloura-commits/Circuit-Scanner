import copy
import json
import unittest
import cv2
import numpy as np
from backend.circuit_connectivity import construir_conectividade, source_signature
from backend.circuit_document import gerar_circuito, validar_circuito


def component(cid,kind,bbox,pins,value=None):
    return dict(component_id=cid,type=kind,bbox=dict(zip(('x1','y1','x2','y2'),bbox)),
                reference=dict(text=cid,source='manual'),value=value,
                terminals=[dict(pin=pin,position=dict(x=p[0],y=p[1]),search_radius_px=18,
                                wire_contact_confirmed=True) for pin,p in pins.items()])


def series_case(kind='resistor'):
    image=np.full((220,420,3),255,np.uint8)
    pins=['+','-'] if kind in {'battery','source_dc','capacitor_polarized'} else ['A','K'] if kind in {'diode','led'} else ['1','2']
    a=component('A',kind,(90,70,170,130),{pins[0]:(91,100),pins[1]:(169,100)},'1000')
    b=component('B','resistor',(250,70,330,130),{'1':(251,100),'2':(329,100)},'2200')
    g=component('G','gnd',(360,70,410,130),{'GND':(361,100)})
    routes={'LEFT':[(20,100),(90,100)],'MID':[(170,100),(250,100)],'RIGHT':[(330,100),(360,100)]}
    for p,q in [((20,100),(91,100)),((169,100),(251,100)),((329,100),(361,100))]:cv2.line(image,p,q,(20,20,20),3)
    # A visible component stroke connects its two pins, but it is not a wire.
    cv2.line(image,(91,100),(169,100),(20,20,20),3)
    cv2.line(image,(251,100),(329,100),(20,20,20),3)
    wires=dict(project_id='TEST',segments=[dict(wire_id=wid,points=[dict(x=x,y=y) for x,y in points]) for wid,points in routes.items()],
               nodes=[],stage9={'nodes':[]})
    return image,wires,{'components':[a,b,g]},pins


class ConnectivityTests(unittest.TestCase):
    def test_series_members_and_ground_are_correct_for_seven_two_pin_classes(self):
        for kind in ('resistor','capacitor','inductor','diode','led','source_dc','capacitor_polarized'):
            with self.subTest(kind=kind):
                image,wires,terms,pins=series_case(kind)
                result=construir_conectividade(image,wires,terms)
                member_sets={frozenset(n['terminal_refs']) for n in result['nets']}
                self.assertEqual(member_sets,{frozenset(['A:'+pins[0]]),frozenset(['A:'+pins[1],'B:1']),frozenset(['B:2','G:GND'])})
                ground=next(n for n in result['nets'] if n['name']=='GND')
                self.assertEqual(set(ground['terminal_refs']),{'B:2','G:GND'})
                self.assertEqual(result['attached_terminal_count'],5)

    def test_graph_connection_groups_are_respected_at_crossings(self):
        image,wires,terms,_=series_case()
        wires['stage9']['nodes']=[dict(node_id='X',electrical_status='crossing_unconnected',connection_groups=[['LEFT'],['MID','RIGHT']])]
        result=construir_conectividade(image,wires,terms)
        mapping={r['terminal_ref']:r['net'] for r in result['terminal_connections']}
        self.assertNotEqual(mapping['A:1'],mapping['A:2'])
        self.assertEqual(mapping['A:2'],mapping['B:2'])
        wires['stage9']['nodes'][0].update(electrical_status='junction_connected',connection_groups=[['LEFT','MID','RIGHT']])
        joined=construir_conectividade(image,wires,terms)
        self.assertEqual(len(joined['nets']),1)

    def test_pending_junction_never_unites_its_candidates(self):
        image,wires,terms,_=series_case()
        wires['stage9']['nodes']=[dict(node_id='P',electrical_status='review_required',review_required=True,connection_groups=[['LEFT','MID']])]
        result=construir_conectividade(image,wires,terms)
        self.assertEqual(result['net_count'],3)
        self.assertFalse(result['criterion_met'])
        self.assertIn('junction_pending',[w['code'] for w in result['warnings']])

    def test_short_white_gap_does_not_connect_pin_to_nearby_wire(self):
        image,wires,terms,_=series_case()
        image[95:106,80:90]=255
        wires['segments'][0]['points'][-1]['x']=78
        result=construir_conectividade(image,wires,terms)
        terminal=next(r for r in result['terminal_connections'] if r['terminal_ref']=='A:1')
        self.assertEqual(terminal['status'],'unconnected')

    def test_equal_distance_separate_networks_are_ambiguous(self):
        image,wires,terms,_=series_case()
        # Two disconnected outgoing conductors overlap at the pin region,
        # but Stage 9 has not declared a connection between them.
        wires['segments'].append(dict(wire_id='ALTERNATE',points=[dict(x=20,y=100),dict(x=90,y=100)]))
        result=construir_conectividade(image,wires,terms)
        terminal=next(r for r in result['terminal_connections'] if r['terminal_ref']=='A:1')
        self.assertEqual(terminal['status'],'ambiguous')
        nets=[n for n in result['nets'] if n['wire_ids']]
        self.assertTrue(any(n['wire_ids']==['LEFT'] for n in nets))
        self.assertTrue(any(n['wire_ids']==['ALTERNATE'] for n in nets))

    def test_all_ground_markers_share_the_electrical_reference(self):
        image,wires,terms,_=series_case()
        other=component('G2','gnd',(10,160,70,210),{'GND':(11,180)})
        terms['components'].append(other)
        result=construir_conectividade(image,wires,terms)
        ground=next(n for n in result['nets'] if n['name']=='GND')
        self.assertIn('G2:GND',ground['terminal_refs'])

    def test_vcc_classification_does_not_short_separate_outputs(self):
        image,wires,terms,_=series_case()
        terms['components']=[component('P1','vcc',(90,70,170,130),{'VCC':(91,100)}),
                             component('P2','vcc',(250,70,330,130),{'VCC':(251,100)})]
        result=construir_conectividade(image,wires,terms)
        self.assertNotEqual(result['terminal_connections'][0]['net'],result['terminal_connections'][1]['net'])
        for comp in terms['components']:comp['net_label']='SUPPLY'
        joined=construir_conectividade(image,wires,terms)
        self.assertEqual(joined['terminal_connections'][0]['net'],joined['terminal_connections'][1]['net'])

    def test_manual_terminal_junction_is_authoritative(self):
        image,wires,terms,_=series_case();image[95:106,20:90]=255
        wires['stage9']['nodes']=[dict(node_id='MANUAL',electrical_status='junction_connected',
                                     incident_wire_ids=['LEFT'],connection_groups=[['LEFT']],attached_terminal_refs=['A:1'],manual_override=True)]
        result=construir_conectividade(image,wires,terms)
        self.assertEqual(result['terminal_connections'][0]['status'],'attached_by_junction')

    def test_inputs_are_immutable_and_members_are_order_independent(self):
        image,wires,terms,_=series_case();before=(copy.deepcopy(wires),copy.deepcopy(terms),image.copy())
        first=construir_conectividade(image,wires,terms)
        wires['segments']=list(reversed(wires['segments']));terms['components']=list(reversed(terms['components']))
        second=construir_conectividade(image,wires,terms)
        self.assertEqual(first['nets'],second['nets'])
        self.assertTrue(np.array_equal(image,before[2]))
        self.assertEqual(list(reversed(wires['segments'])),before[0]['segments'])
        self.assertEqual(list(reversed(terms['components'])),before[1]['components'])

    def test_stale_signature_changes_after_pin_or_junction_edit(self):
        image,wires,terms,_=series_case();signature=source_signature(wires,terms)
        derived=copy.deepcopy(wires);derived['stage10']={'test':123};derived['stage11']={'test':456}
        self.assertEqual(signature,source_signature(derived,terms))
        terms['components'][0]['terminals'][0]['position']['x']+=1
        self.assertNotEqual(signature,source_signature(wires,terms))

    def test_missing_pins_are_reported_without_inventing_contacts(self):
        image,wires,terms,_=series_case();terms['components'][0]['terminals'].pop()
        result=construir_conectividade(image,wires,terms)
        self.assertEqual(result['terminal_count'],4)
        self.assertIn('pin_count_mismatch',[w['code'] for w in result['warnings']])

    def test_collinear_wire_fragments_join_only_over_visible_ink(self):
        image,wires,terms,_=series_case()
        mid=wires['segments'][1]
        mid['points'][-1]['x']=205
        wires['segments'].append(dict(wire_id='MID2',points=[dict(x=214,y=100),dict(x=250,y=100)]))
        result=construir_conectividade(image,wires,terms)
        self.assertEqual(len(result['recovered_wire_bridges']),1)
        mapping={r['terminal_ref']:r['net'] for r in result['terminal_connections']}
        self.assertEqual(mapping['A:2'],mapping['B:1'])
        image[96:105,208:212]=255
        disconnected=construir_conectividade(image,wires,terms)
        self.assertFalse(disconnected['recovered_wire_bridges'])

    def test_fragment_recovery_does_not_override_manual_crossing_or_bridge_a_component(self):
        image,wires,terms,_=series_case()
        wires['segments'][1]['points'][-1]['x']=205
        wires['segments'].append(dict(wire_id='MID2',points=[dict(x=214,y=100),dict(x=250,y=100)]))
        wires['stage9']['nodes']=[dict(node_id='X',position={'x':210,'y':100},electrical_status='crossing_unconnected',
                                     connection_groups=[['MID'],['MID2']],manual_override=True)]
        result=construir_conectividade(image,wires,terms)
        self.assertFalse(result['recovered_wire_bridges'])
        # Two short graph ends are inside opposite sides of a narrow resistor.
        image=np.full((160,260,3),255,np.uint8);cv2.line(image,(30,80),(230,80),(20,20,20),3)
        c=component('R','resistor',(110,60,130,100),{'1':(111,80),'2':(129,80)},'1000')
        graph=dict(nodes=[],segments=[dict(wire_id='L',points=[dict(x=30,y=80),dict(x=115,y=80)]),
                                     dict(wire_id='R',points=[dict(x=125,y=80),dict(x=230,y=80)])],stage9={'nodes':[]})
        self.assertFalse(construir_conectividade(image,graph,{'components':[c]})['recovered_wire_bridges'])

    def test_short_lead_erased_between_adjacent_boxes_is_recovered(self):
        image=np.full((180,300,3),255,np.uint8)
        cv2.line(image,(125,90),(145,90),(20,20,20),3)
        a=component('P','vcc',(80,60,126,120),{'VCC':(125,90)})
        b=component('R','resistor',(144,60,210,120),{'1':(145,90),'2':(209,90)},'1000')
        graph=dict(nodes=[],segments=[],stage9={'nodes':[]})
        result=construir_conectividade(image,graph,{'components':[a,b]})
        mapping={r['terminal_ref']:r['net'] for r in result['terminal_connections']}
        self.assertEqual(mapping['P:VCC'],mapping['R:1'])
        self.assertEqual(result['attached_terminal_count'],2)
        image[87:94,133:137]=255
        self.assertEqual(construir_conectividade(image,graph,{'components':[a,b]})['attached_terminal_count'],0)


class CircuitDocumentTests(unittest.TestCase):
    def test_ten_reference_cases_roundtrip_to_independent_json(self):
        for index in range(10):
            with self.subTest(index=index):
                kind=('resistor','capacitor','inductor','diode','led','source_dc','capacitor_polarized','battery','resistor','capacitor')[index]
                image,wires,terms,_=series_case(kind)
                terms['components'][0]['value']=dict(text='10k',normalized=10000.0,unit='ohm',source='manual')
                connectivity=construir_conectividade(image,wires,terms)
                del image,wires,terms
                circuit=json.loads(json.dumps(gerar_circuito(connectivity),allow_nan=False))
                self.assertTrue(validar_circuito(circuit)['valid'])
                self.assertEqual(circuit['components'][0]['value'],'10000')
                self.assertEqual(circuit['components'][0]['value_text'],'10k')
                self.assertFalse(circuit['confirmed_by_user'])
                serialized=json.dumps(circuit)
                for geometry in ('bbox','position','image_width','wire_id','original.jpeg'):self.assertNotIn(geometry,serialized)

    def test_provisional_names_are_unique_and_do_not_collide_with_ocr(self):
        image,wires,terms,_=series_case()
        terms['components'][0]['reference']=dict(text='R1',source='manual')
        terms['components'][1]['component_id']='DET_001';terms['components'][1]['reference']=dict(text='DET_001',source='provisional_detection_id')
        circuit=gerar_circuito(construir_conectividade(image,wires,terms))
        self.assertEqual([c['id'] for c in circuit['components']],['R1','R2','G'])
        self.assertIn('R2.1',[m for n in circuit['nets'] for m in n['members']])

    def test_missing_values_and_connections_remain_explicit(self):
        image,wires,terms,_=series_case();terms['components'][0]['value']=None;image[95:106,80:90]=255
        wires['segments'][0]['points'][-1]['x']=78
        circuit=gerar_circuito(construir_conectividade(image,wires,terms))
        self.assertIsNone(circuit['components'][0]['value'])
        self.assertTrue(circuit['validation']['review_required'])
        self.assertIn('value_missing',[w['code'] for w in circuit['warnings']])
        self.assertIn('unconnected_pin',[w['code'] for w in circuit['warnings']])

    def test_unknown_pin_and_duplicate_net_members_are_invalid(self):
        image,wires,terms,_=series_case();circuit=gerar_circuito(construir_conectividade(image,wires,terms))
        bad=copy.deepcopy(circuit);bad['nets'][0]['members'].append('DOES_NOT_EXIST.1')
        self.assertFalse(validar_circuito(bad)['valid'])
        bad=copy.deepcopy(circuit);bad['nets'][1]['members'].append(bad['nets'][0]['members'][0])
        self.assertFalse(validar_circuito(bad)['valid'])
        bad=copy.deepcopy(circuit);bad['nets'][0]['members'].clear()
        self.assertFalse(validar_circuito(bad)['valid'])


if __name__=='__main__':unittest.main()
