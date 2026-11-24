import copy
import json
import math
import unittest
import xml.etree.ElementTree as ET
from backend.falstad_compact import compact_native,_graph
from backend.falstad_edit import describe,parse_deck,import_edits


def deck(lines):
    text='$ 3 0.000005 10.20027730826997 50 5 50 5e-11\n'+'\n'.join(lines)+'\n'
    components=[]
    for i,e in enumerate(parse_deck(text)):
        if e['code'] in {'w','x','207'}:continue
        kind,_,posts,_=describe(e)
        components.append(dict(component_id='C'+str(i),type=kind,element_index=i,
                               pins={pin:dict(position=list(p),post_index=k) for k,(pin,p) in enumerate(posts.items())}))
    return dict(circuit=text,components=components,source_revision='unchanged',edited=True)


class CompactNativeTests(unittest.TestCase):
    def test_sparse_wires_and_long_component_leads_shrink_without_changing_branch(self):
        old=deck(['v 0 0 0 512 0 0 40 5 0 0 .5',
                  'r 1024 0 1024 512 0 4700','r 1024 1024 1536 1024 0 22000',
                  'w 0 0 1024 0 0','w 0 512 512 512 0','w 512 512 1024 512 0',
                  'w 512 512 512 1024 0','w 512 1024 1024 1024 0','g 1536 1024 1536 1048 0'])
        untouched=copy.deepcopy(old);new=compact_native({},old)
        self.assertEqual(old,untouched)
        self.assertEqual(_graph(parse_deck(old['circuit'])),_graph(parse_deck(new['circuit'])))
        self.assertLess(new['compaction']['wire_length_after'],new['compaction']['wire_length_before']*.4)
        self.assertLess(new['compaction']['after'][1],new['compaction']['before'][1]*.4)
        self.assertEqual(len([e for e in parse_deck(old['circuit']) if e['code']=='w']),len([e for e in parse_deck(new['circuit']) if e['code']=='w']))
        for before,after in zip(parse_deck(old['circuit']),parse_deck(new['circuit'])):
            self.assertEqual((before['code'],before['flags'],before['extra']),(after['code'],after['flags'],after['extra']))
        self.assertEqual(compact_native({},new),new)

    def test_fixed_three_pin_offsets_polarity_and_parameters_survive_all_orientations(self):
        for code,flags,extras in [('t',0,'1 0 0 58 default'),('t',1,'-1 0 0 120 default'),
                                  ('f',32,'2 .05'),('f',41,'-2 .06'),('a',8,'15 -15 1000000 0 0 200000'),
                                  ('a',9,'15 -15 1000000 0 0 123456')]:
            for a,b in [((0,512),(512,512)),((512,512),(0,512)),((512,0),(512,512)),((512,512),(512,0))]:
                with self.subTest(code=code,flags=flags,a=a,b=b):
                    old=deck([f'{code} {a[0]} {a[1]} {b[0]} {b[1]} {flags} {extras}'])
                    new=compact_native({},old);e=parse_deck(new['circuit'])[0]
                    self.assertEqual(e['extra'],parse_deck(old['circuit'])[0]['extra'])
                    self.assertEqual(e['flags'],flags)
                    self.assertEqual(new['components'][0]['pins'].keys(),describe(e)[2].keys())
                    self.assertEqual({p:tuple(v['position']) for p,v in new['components'][0]['pins'].items()},describe(e)[2])
                    self.assertEqual(e['a'][0]==e['b'][0],a[0]==b[0])
                    self.assertGreaterEqual(math.dist(e['a'],e['b']),32)

    def test_crossing_without_junction_does_not_become_connected(self):
        old=deck(['r 0 512 128 512 0 1000','r 512 0 512 128 0 2000',
                  'w 128 512 1024 512 0','w 512 128 512 1024 0'])
        new=compact_native({},old)
        self.assertEqual(_graph(parse_deck(old['circuit'])),_graph(parse_deck(new['circuit'])))
        self.assertEqual(len(_graph(parse_deck(new['circuit']))),4)

    def test_diagonal_wire_keeps_middle_junction(self):
        old=deck(['r 512 512 512 576 0 1000','w 0 0 1024 1024 0','g 1024 1024 1024 1048 0'])
        new=compact_native({},old)
        self.assertEqual(_graph(parse_deck(old['circuit'])),_graph(parse_deck(new['circuit'])))

    def test_separate_ground_drawings_are_never_joined_by_new_wire(self):
        old=deck(['r 0 0 0 512 0 1000','g 0 1024 0 1048 0','w 0 512 0 1024 0',
                  'r 1024 0 1024 512 0 2000','g 1024 1024 1024 1048 0','w 1024 512 1024 1024 0'])
        new=compact_native({},old);entries=parse_deck(new['circuit'])
        self.assertEqual(sum(e['code']=='w' for e in entries),2)
        self.assertTrue(all(e['a'][0]==e['b'][0] for e in entries if e['code']=='w'))

    def test_native_xml_models_scopes_and_capacitance_remain_unchanged(self):
        old=deck(['c 0 0 0 1024 0 .000001 0 0','g 0 1536 0 1560 0','w 0 1024 0 1536 0'])
        old['circuit']='<cir f="3"><mm nm="custom" vt="2" be=".05"/><c x="0 0 0 1024" f="0" c=".000001" vd=".2" iv=".3"/><g x="0 1536 0 1560" f="0"/><w x="0 1024 0 1536" f="0"/><o data="scope-settings"/></cir>'
        new=compact_native({},old);before=ET.fromstring(old['circuit']);after=ET.fromstring(new['circuit'])
        for a,b in zip(before,after):
            self.assertEqual(a.tag,b.tag)
            self.assertEqual({k:v for k,v in a.attrib.items() if k!='x'},{k:v for k,v in b.attrib.items() if k!='x'})
        self.assertEqual(_graph(parse_deck(old['circuit'])),_graph(parse_deck(new['circuit'])))

    def test_loading_compacted_download_against_original_baseline_keeps_part_names(self):
        old=deck(['R 0 0 0 -32 0 0 40 5 0 0 .5','r 0 512 0 1024 0 1000',
                  'r 512 0 1024 0 0 2000','g 1536 0 1536 24 0',
                  'w 0 0 0 512 0','w 0 0 512 0 0','w 1024 0 1536 0 0'])
        current=dict(components=[dict(id=b['component_id'],type=b['type'],pins=list(b['pins']),metadata={})
                                 for b in old['components']],nets=[])
        original,_,_=import_edits(current,old,old['circuit'])
        compact=compact_native(original,old)
        restored,_,_=import_edits(original,old,compact['circuit'])
        self.assertEqual({frozenset(n['members']) for n in original['nets']},
                         {frozenset(n['members']) for n in restored['nets']})
        self.assertEqual({c['id']:c.get('value') for c in original['components']},
                         {c['id']:c.get('value') for c in restored['components']})

    def test_implicit_ground_stays_implicit_after_compacted_download(self):
        old=deck(['r 0 0 0 512 0 1000','g 0 1024 0 1048 0','w 0 512 0 1024 0'])
        old['components']=old['components'][:1]
        current=dict(components=[dict(id='C0',type='resistor',pins=['1','2'],metadata={})],nets=[])
        original,_,_=import_edits(current,old,old['circuit'])
        compact=compact_native(original,old)
        restored,_,_=import_edits(original,old,compact['circuit'])
        self.assertEqual([c['id'] for c in restored['components']],['C0'])
        self.assertEqual({frozenset(n['members']) for n in original['nets']},
                         {frozenset(n['members']) for n in restored['nets']})


if __name__=='__main__':unittest.main()
