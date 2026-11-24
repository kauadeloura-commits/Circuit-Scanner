import copy
import json
from pathlib import Path
import unittest
import cv2
import numpy as np
from backend.observed_leads import recover_observed_leads


def part(cid,kind,box,points):
    return dict(component_id=cid,type=kind,bbox=dict(zip(('x1','y1','x2','y2'),box)),
                terminals=[dict(pin=pin,position=dict(x=p[0],y=p[1])) for pin,p in points.items()])


class ObservedLeadTests(unittest.TestCase):
    def test_shared_lead_inside_overlapping_transistor_boxes(self):
        mask=np.zeros((200,260),np.uint8);cv2.line(mask,(95,80),(125,80),1,3)
        a=part('Q1','transistor_pnp',(40,40,135,170),{'B':(50,110),'C':(95,80),'E':(95,160)})
        b=part('Q2','transistor_npn',(120,30,210,155),{'B':(125,80),'C':(180,40),'E':(180,145)})
        original=mask.copy();parts=copy.deepcopy([a,b])
        found=recover_observed_leads(mask,[a,b],[],[],{})
        self.assertEqual(len(found),1);self.assertEqual(set(found[0]['anchors']),{'T:Q1:C','T:Q2:B'})
        self.assertTrue(np.array_equal(mask,original));self.assertEqual([a,b],parts)
        mask[77:84,109:113]=0
        self.assertFalse(recover_observed_leads(mask,[a,b],[],[],{}))

    def test_bent_external_trace_hidden_inside_box_rejoins_fragments(self):
        mask=np.zeros((240,260),np.uint8)
        cv2.polylines(mask,[np.array([(90,200),(145,200),(150,140),(180,140)],np.int32)],False,1,3)
        a=part('Q','transistor_pnp',(40,50,160,210),{'B':(50,120),'C':(100,60),'E':(90,200)})
        b=part('R','resistor',(179,130,210,190),{'1':(180,140),'2':(209,180)})
        wires=[dict(wire_id='LOW',points=[dict(x=90,y=200),dict(x=120,y=200)]),
               dict(wire_id='HIGH',points=[dict(x=170,y=140),dict(x=180,y=140)])]
        found=recover_observed_leads(mask,[a,b],wires,[],{'T:Q:E':'W:LOW','W:LOW':'W:LOW','T:R:1':'W:HIGH','W:HIGH':'W:HIGH'})
        self.assertEqual(len(found),1);self.assertGreater(found[0]['length_px'],50)
        self.assertGreater(len(found[0]['points']),2)

    def test_native_fork_and_resistor_strokes_cannot_become_external_wires(self):
        for kind,pins in [('transistor_npn',{'B':(50,100),'C':(130,50),'E':(130,150)}),
                          ('resistor',{'1':(50,100),'2':(130,100)})]:
            mask=np.zeros((200,240),np.uint8)
            for p in pins.values():cv2.line(mask,p,(90,100),1,5)
            c=part('P',kind,(45,40,140,160),pins)
            marker=part('V','vcc',(125,20,175,55),{'VCC':(130,50)})
            found=recover_observed_leads(mask,[c,marker],[],[],{})
            for lead in found:self.assertFalse(len([k for k in lead['anchors'] if k.startswith('T:P:')])>1)

    def test_resolved_crossing_and_pending_junction_are_never_overridden(self):
        mask=np.zeros((140,220),np.uint8);cv2.line(mask,(50,70),(170,70),1,3)
        a=part('A','vcc',(20,40,52,100),{'VCC':(50,70)})
        b=part('B','vcc',(168,40,200,100),{'VCC':(170,70)})
        self.assertTrue(recover_observed_leads(mask,[a,b],[],[],{}))
        node=dict(position=dict(x=110,y=70))
        self.assertFalse(recover_observed_leads(mask,[a,b],[],[node],{}))

    def test_separate_parallel_wires_and_empty_images_stay_separate(self):
        mask=np.zeros((140,220),np.uint8)
        a=part('A','vcc',(20,40,52,100),{'VCC':(50,70)})
        b=part('B','vcc',(168,40,200,100),{'VCC':(170,80)})
        cv2.line(mask,(50,70),(160,70),1,2);cv2.line(mask,(60,80),(170,80),1,2)
        self.assertFalse(recover_observed_leads(mask,[a,b],[],[],{}))
        self.assertFalse(recover_observed_leads(np.zeros_like(mask),[a,b],[],[],{}))

    def test_ambiguous_anchors_and_invalid_coordinates_do_not_force_a_connection(self):
        mask=np.zeros((140,220),np.uint8);cv2.line(mask,(50,70),(170,70),1,3)
        a=part('A','vcc',(20,40,52,100),{'VCC':(50,70)})
        b=part('B','vcc',(168,40,200,100),{'VCC':(170,70)})
        self.assertFalse(recover_observed_leads(mask,[a,b],[],[],{}, {'T:A:VCC'}))
        a['terminals'][0]['position']['x']=float('nan')
        self.assertFalse(recover_observed_leads(mask,[a,b],[],[],{}))

    def test_prj445_every_photographed_branch_and_pin_survives_export_and_save(self):
        from backend.circuit_connectivity import construir_conectividade
        from backend.circuit_document import gerar_circuito
        from backend.digital_twin import make_diagram
        from backend.circuitjs_export import export_circuitjs
        from backend.falstad_edit import import_edits,check_native_connections
        folder=Path(__file__).parent/'test_fixtures/observed445'
        fixture=json.loads((folder/'inputs.json').read_text(encoding='utf8'))
        image=cv2.imread(str(folder/'photo.jpeg'))
        new=construir_conectividade(image,fixture['wires'],fixture['terminals'])
        expected={frozenset(m) for m in fixture['expected']}
        self.assertEqual({frozenset(n['terminal_refs']) for n in new['nets']},expected)
        self.assertEqual(new['unconnected_terminal_count'],0)
        self.assertEqual(new['attached_terminal_count'],14)
        self.assertEqual(len(new['recovered_observed_leads']),2)
        c=gerar_circuito(new);c.update(status='confirmed',confirmed_by_user=True)
        diagram=make_diagram(c,new,fixture['wires'])
        self.assertEqual(diagram['source_layout'],'image');self.assertFalse(diagram['warnings'])
        native=export_circuitjs(c,diagram);check_native_connections(c,native)
        saved,_,_=import_edits(c,native,native['circuit'])
        self.assertEqual({frozenset(n['members']) for n in saved['nets']},
                         {frozenset(n['members']) for n in c['nets']})
        # Removing photographed ink must remove its connection; references or
        # component types alone cannot supply it.
        image[500:516,334:345]=255
        broken=construir_conectividade(image,fixture['wires'],fixture['terminals'])
        self.assertNotEqual({frozenset(n['terminal_refs']) for n in broken['nets']},expected)

    def test_native_connection_check_catches_a_wire_one_grid_cell_short(self):
        from backend.falstad_edit import check_native_connections,describe,parse_deck
        text='r 0 0 64 0 0 1000\nr 128 0 192 0 0 2200\nw 64 0 120 0 0\n'
        bindings=[dict(component_id=cid,type='resistor',element_index=i,
                       pins={pin:dict(position=p) for pin,p in describe(parse_deck(text)[i])[2].items()})
                  for i,cid in enumerate(('R1','R2'))]
        c=dict(schema_version='circuit-1.0',components=[dict(id=cid,type='resistor',pins=['1','2'],metadata={}) for cid in ('R1','R2')],
               nets=[dict(name='A',members=['R1.1']),dict(name='B',members=['R1.2','R2.1']),dict(name='C',members=['R2.2'])])
        with self.assertRaisesRegex(ValueError,'alterou uma ligação'):
            check_native_connections(c,dict(circuit=text,components=bindings))
        check_native_connections(c,dict(circuit=text.replace('120 0','128 0'),components=bindings))
