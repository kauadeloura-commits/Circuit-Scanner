import unittest
from unittest.mock import patch
import cv2
import numpy as np
from backend.drawing_mode import using
from backend.physical_annotation_reader import word_consensus
from backend.word_extent import complete_span_priorities


def reading(text,confidence,minimum):
    return dict(text=text,confidence=confidence,min_character_confidence=minimum)


def token(text,end,readers=2,confidence=.96):
    return dict(text=text,confidence=confidence,bbox=dict(x1=10,y1=10,x2=end,y2=30),
                word_evidence=[dict(engine=e) for e in ('general_word','circuit_word')[:readers]])


class CompleteWordRecoveryTests(unittest.TestCase):
    def test_complete_decimal_prefix_beats_clipped_value(self):
        full,partial=token('4k7',65),token('4k',45,confidence=.99)
        with using('handwritten'):p=complete_span_priorities([partial,full])
        self.assertEqual(p[id(full)],1)
        self.assertEqual(p[id(partial)],-1)

    def test_extra_digit_without_fuller_photographic_extent_has_no_priority(self):
        full,partial=token('4k7',46),token('4k',45)
        with using('handwritten'):self.assertEqual(complete_span_priorities([partial,full]),{})

    def test_single_reader_cannot_extend_a_value(self):
        full,partial=token('4k7',65,readers=1),token('4k',45)
        with using('handwritten'):self.assertEqual(complete_span_priorities([partial,full]),{})

    def test_does_not_change_digital_selection(self):
        with using('digital'):self.assertEqual(complete_span_priorities([token('4k',45),token('4k7',65)]),{})

    def test_unrelated_leading_digit_is_not_a_complete_prefix_extension(self):
        with using('handwritten'):self.assertEqual(complete_span_priorities([token('6k',45),token('96k',65)]),{})

    def test_long_model_can_use_character_agreement_instead_of_short_word_probability(self):
        rows=[('general_word','original',reading('1N5819',.86,.52)),
              ('general_word','white_margin',reading('1n5819',.85,.51)),
              ('circuit_word','edge0.08',reading('1N5819',.75,.85)),
              ('circuit_word','edge0.16',reading('1N5819',.80,.88))]
        result=word_consensus(rows,'diode')
        self.assertIsNotNone(result)
        self.assertEqual(result['text'].upper(),'1N5819')
        self.assertTrue(result['needs_review'])

    def test_trimming_alone_cannot_invent_a_model(self):
        for confidence in (.85,.999):
            rows=[(e,'edge'+v,reading('1N5819',confidence,confidence))
                  for e in ('general_word','circuit_word') for v in ('.08','.16')]
            self.assertIsNone(word_consensus(rows,'diode'))

    def test_model_needs_both_readers_and_compatible_component(self):
        rows=[('circuit_word',v,reading('1N5819',.80,.88)) for v in ('a','b','c')]
        self.assertIsNone(word_consensus(rows,'diode'))
        rows += [('general_word',v,reading('1N5819',.86,.52)) for v in ('original','contrast')]
        self.assertIsNone(word_consensus(rows,'opamp'))

    def test_conflicting_model_readers_do_not_vote_a_catalogue_answer(self):
        rows=[('general_word',v,reading('BC548',.86,.52)) for v in ('original','contrast')]
        rows += [('circuit_word',v,reading('BC547',.80,.88)) for v in ('edge.08','edge.16')]
        self.assertIsNone(word_consensus(rows,'transistor_npn'))

    def test_empty_and_weak_words_stay_unresolved(self):
        self.assertIsNone(word_consensus([],'diode'))
        self.assertIsNone(word_consensus([('general_word','original',reading('1N5819',.40,.10))],'diode'))

    def test_typed_name_can_schedule_model_despite_wrong_initial_region_owner(self):
        from backend.physical_annotation_reader import recover
        image=np.full((160,240,3),255,np.uint8)
        cv2.putText(image,'1N5819',(62,110),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
        detection=dict(detection_id='diode',**{'class':'diode'},bbox=dict(x1=30,y1=60,x2=55,y2=90))
        name=dict(text='D3',confidence=.99,bbox=dict(x1=60,y1=60,x2=90,y2=80),component_context=dict(detection_id='wrong',kind='opamp'))
        model_box=dict(x1=60,y1=95,x2=140,y2=115,region_source='trained_whole_text_detector')
        general=lambda images:[reading('1N5819',.86,.52) for _ in images]
        domain=lambda images:[reading('1N5819',.75,.85) for _ in images]
        with using('handwritten'),patch('backend.annotation_lanes.regions',return_value=(image,[])),patch('backend.ocr_regions.text_line_regions',return_value=[]),patch('backend.local_word_ocr.read_words',side_effect=general),patch('backend.domain_word_ocr.read_words',side_effect=domain):
            tokens,_=recover(image,[detection],[name],[],[],word_regions=[model_box])
        code=[t for t in tokens if t['text'].upper()=='1N5819']
        self.assertTrue(code)
        self.assertEqual(code[0]['component_context']['detection_id'],'diode')

    def test_no_unbounded_model_retries_on_a_blank_unnamed_symbol(self):
        from backend.physical_annotation_reader import recover
        image=np.full((80,80,3),255,np.uint8)
        detection=dict(detection_id='diode',**{'class':'diode'},bbox=dict(x1=30,y1=30,x2=55,y2=60))
        with using('handwritten'),patch('backend.local_word_ocr.read_words') as reader:
            tokens,pending=recover(image,[detection],[],[],[])
        self.assertEqual((tokens,pending),([],[]))
        reader.assert_not_called()


if __name__=='__main__':unittest.main()
