"""Regression checks for the supervised text/component separation boundary."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import cv2
import numpy as np
from backend import text_component_classifier as guard
from backend.glyph_classifier import status
from backend.handwriting_reference import BANK


class TrainedTextGuardTests(unittest.TestCase):
    def test_both_supplemental_models_are_loaded(self):
        state=status(BANK)
        self.assertTrue(state['ready'])
        self.assertTrue(state['font_specialist_ready'])
        self.assertTrue(set('QDLUF')<=set(state['font_specialist_classes']))
        self.assertIsNotNone(guard.probability(np.zeros(1980,np.float32)))

    def test_empty_crop_and_absent_weights_are_nonfatal(self):
        image=np.full((50,50,3),255,np.uint8)
        self.assertEqual(guard.text_evidence(image,dict(x1=30,y1=30,x2=20,y2=20)),(False,{}))
        with TemporaryDirectory() as folder:
            self.assertIsNone(guard.probability(np.zeros(1980,np.float32),Path(folder)/'missing.npz'))

    def test_touching_photo_tokens_are_read_before_isolated_glyph_guesses(self):
        from backend.handwriting_reference import library,read_line
        _,refs=library(trained=True)
        for expected,kind in [('R1','resistor'),('R2','resistor'),('Q1','transistor_pnp')]:
            with self.subTest(token=expected):
                entry=next(e for e,_ in refs if e['id'].startswith('PHOTO_NEW_TOKEN_') and e['text']==expected)
                result=read_line(cv2.imread(str(BANK/entry['file']),0),refs,kind)
                self.assertFalse(result['unresolved']);self.assertEqual(result['text'],expected)

    def test_incompatible_weights_are_rejected_before_inference(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'bad.npz'
            np.savez(path,feature_version=guard.VERSION,mean=np.zeros(2),scale=np.ones(2),
                     w1=np.zeros((96,2)),b1=np.zeros(96),w2=np.zeros((2,96)),b2=np.zeros(2),threshold=.995)
            self.assertIsNone(guard.probability(np.zeros(2,np.float32),path))

    def test_observed_lead_bypasses_isolated_glyph_model(self):
        vector=np.zeros(1980,np.float32);vector[-2]=.25
        with patch.object(guard,'features',return_value=vector),patch.object(guard,'probability',return_value=None) as predict:
            self.assertEqual(guard.text_evidence(np.zeros((4,4,3),np.uint8),{}),(False,{}))
            predict.assert_called_once_with(vector,guard.CONTEXT_MODEL)

    def test_manual_component_is_preserved_before_any_text_filter(self):
        from backend.recognition import filtrar_falsos_positivos_geometricos
        detection=dict(id='MANUAL',**{'class':'capacitor'},manual_reviewed=True,
                       bbox=dict(x1=30,y1=30,x2=80,y2=90),confidence=.9)
        with patch.object(guard,'text_evidence',side_effect=AssertionError('manual box was filtered')):
            kept,rejected=filtrar_falsos_positivos_geometricos(np.full((120,120,3),255,np.uint8),[detection])
        self.assertEqual(len(kept),1);self.assertEqual(rejected,[])

    def test_printed_reference_is_not_a_capacitor(self):
        from backend.recognition import filtrar_falsos_positivos_geometricos
        image=np.full((160,300,3),255,np.uint8)
        cv2.putText(image,'R9',(45,110),cv2.FONT_HERSHEY_SIMPLEX,2,(110,35,25),4,cv2.LINE_AA)
        detection=dict(id='FALSE',**{'class':'capacitor'},
                       bbox=dict(x1=40,y1=60,x2=135,y2=118),confidence=.99)
        kept,rejected=filtrar_falsos_positivos_geometricos(image,[detection])
        self.assertEqual(kept,[]);self.assertEqual(len(rejected),1)
        self.assertTrue(rejected[0]['geometry_validation']['evidence']['hard_text_rejection'])


if __name__=='__main__':unittest.main()
