import copy
import unittest
from unittest.mock import patch,Mock
import sys
from tempfile import TemporaryDirectory
from pathlib import Path
import cv2
import numpy as np
from backend.symbol_semantics import resolve_marked_families
from backend.recognition import corrigir_diodo_led_por_setas
from backend.test_terminal_identity import fixture,plus,rotate_fixture
from backend import resistor_geometry as rg,cpu_inference,ocr
from backend import performance as perf

IDS={'diode':3,'led':5,'capacitor':7,'capacitor_polarized':8}


def diode_fixture(arrows=False):
    image=np.full((280,280,3),255,np.uint8);ink=(25,25,25)
    cv2.line(image,(35,140),(100,140),ink,3)
    cv2.polylines(image,[np.array([[100,100],[100,180],[150,140]])],True,ink,3)
    cv2.line(image,(155,97),(155,183),ink,3);cv2.line(image,(155,140),(230,140),ink,3)
    if arrows:
        cv2.arrowedLine(image,(169,90),(193,66),ink,2,tipLength=.35)
        cv2.arrowedLine(image,(188,109),(212,85),ink,2,tipLength=.35)
    return image,dict(x1=88.,y1=95.,x2=162.,y2=187.),{}


def detection(kind,bbox,**extra):
    return dict(detection_id='X',**{'class':kind},class_id=IDS[kind],bbox=bbox,confidence=.99,**extra)


class VisibleSymbolSemanticsTests(unittest.TestCase):
    def test_led_without_arrows_is_diode_in_four_rotations(self):
        for k in range(4):
            image,box,_=rotate_fixture(*diode_fixture(),k)
            result,_=resolve_marked_families(image,[detection('led',box)],IDS)
            with self.subTest(rotation=k):self.assertEqual(result[0]['class'],'diode')

    def test_diode_with_arrows_outside_box_is_led_and_includes_them(self):
        for k in range(4):
            image,box,_=rotate_fixture(*diode_fixture(True),k)
            result,_=resolve_marked_families(image,[detection('diode',box)],IDS)
            c=result[0]
            with self.subTest(rotation=k):
                self.assertEqual(c['class'],'led',c)
                self.assertGreaterEqual(len(c['led_emission_validation']['arrows']),2)
                for arrow in c['led_emission_validation']['arrows']:
                    x1,y1,x2,y2=arrow['bounds'];b=c['bbox']
                    self.assertTrue(b['x1']<=x1 and b['y1']<=y1 and b['x2']>=x2 and b['y2']>=y2)

    def test_nearby_letters_and_numbers_are_not_emission_arrows(self):
        for text in ('LED1','1N4007','77','FF','11'):
            image,box,_=diode_fixture()
            cv2.putText(image,text,(165,80),cv2.FONT_HERSHEY_SIMPLEX,.5,(25,25,25),2)
            result,_=resolve_marked_families(image,[detection('led',box)],IDS)
            with self.subTest(text=text):self.assertEqual(result[0]['class'],'diode')

    def test_without_plus_capacitor_is_nonpolar_in_four_rotations(self):
        for k in range(4):
            image,box,_=rotate_fixture(*fixture('capacitor_polarized',mark=False),k)
            result,_=resolve_marked_families(image,[detection('capacitor_polarized',box)],IDS)
            with self.subTest(rotation=k):self.assertEqual(result[0]['class'],'capacitor')

    def test_plus_outside_capacitor_box_is_required_and_included(self):
        image,box,_=fixture('capacitor_polarized',mark=False)
        box=dict(x1=85.,y1=115.,x2=180.,y2=178.)
        plus(image,177,103)
        for k in range(4):
            rotated,bbox,_=rotate_fixture(image,box,{},k)
            result,_=resolve_marked_families(rotated,[detection('capacitor',bbox)],IDS)
            with self.subTest(rotation=k):
                self.assertEqual(result[0]['class'],'capacitor_polarized')
                self.assertNotEqual(result[0]['bbox'],bbox)

    def test_a_t_or_small_separate_stroke_is_not_plus(self):
        image,box,_=fixture('capacitor_polarized',mark=False)
        cv2.line(image,(185,96),(185,105),(20,20,20),3)
        cv2.line(image,(177,96),(193,96),(20,20,20),3)
        result,_=resolve_marked_families(image,[detection('capacitor_polarized',box)],IDS)
        self.assertEqual(result[0]['class'],'capacitor')

    def test_neighbor_sign_and_arrows_are_not_borrowed(self):
        image,box,_=diode_fixture(True)
        other=dict(detection_id='Y',**{'class':'opamp'},bbox=dict(x1=162.,y1=58.,x2=220.,y2=115.))
        result,_=resolve_marked_families(image,[detection('led',box),other],IDS)
        self.assertEqual(result[0]['class'],'diode')

    def test_manual_class_is_preserved(self):
        image,box,_=diode_fixture()
        det=detection('led',box,manual_reviewed=True)
        result,_=resolve_marked_families(image,[det],IDS)
        self.assertEqual(result[0],det)

    def test_resolution_is_repeatable_and_does_not_mutate_input(self):
        image,box,_=diode_fixture(True);dets=[detection('diode',box)];before=copy.deepcopy(dets)
        a,_=resolve_marked_families(image,dets,IDS);b,_=resolve_marked_families(image,a,IDS)
        self.assertEqual(a[0]['class'],b[0]['class']);self.assertEqual(a[0]['bbox'],b[0]['bbox'])
        self.assertEqual(dets,before)


class CPUOptimizationContractsTests(unittest.TestCase):
    def test_nested_ocr_and_geometry_are_not_counted_twice(self):
        with TemporaryDirectory() as tmp,patch.object(perf,'perf_counter',side_effect=[0.,1.,3.,5.]):
            with perf.timed(tmp,'ocr'):
                with perf.timed(tmp,'geometry'):pass
            timings=perf.read_json(Path(tmp)/'performance.json')['stages']
            self.assertEqual(timings['ocr']['duration_seconds'],5.)
            self.assertEqual(timings['ocr']['work_seconds'],3.)
            self.assertEqual(timings['geometry']['work_seconds'],2.)

    def test_mask_cache_is_scoped_content_checked_and_copy_safe(self):
        image=np.full((100,120,3),255,np.uint8)
        @rg.symbol_analysis_session
        def run():
            a=rg.symbol_mask(image);a[0,0]=0
            b=rg.symbol_mask(image)
            image[10,10]=0;c=rg.symbol_mask(image)
            return b,c
        with patch.object(rg,'_symbol_mask',side_effect=lambda i,r:np.min(i,axis=2).copy()) as mask:
            b,c=run();self.assertEqual(mask.call_count,2)
            self.assertEqual(b[0,0],255);self.assertEqual(b[10,10],255);self.assertEqual(c[10,10],0)
            rg.symbol_mask(image);self.assertEqual(mask.call_count,3)

    def test_cpu_thread_budget_is_configured_once(self):
        fake=Mock()
        with patch.dict(sys.modules,{'torch':fake}),patch.object(cpu_inference,'_READY',False),patch.object(cpu_inference.os,'cpu_count',return_value=12):
            cpu_inference.configure_cpu_inference();cpu_inference.configure_cpu_inference()
            fake.set_num_threads.assert_called_once_with(4);fake.set_num_interop_threads.assert_called_once_with(1)

    def test_easyocr_global_and_local_calls_do_not_double_enlarge(self):
        reader=Mock();reader.readtext.return_value=[];reader.readtext_batched.return_value=[[]]
        image=np.full((300,400,3),255,np.uint8)
        with patch.object(ocr,'_carregar_easyocr',return_value=reader):
            ocr._executar_easyocr(image,1)
            mosaic,maps=ocr.preparar_mosaico_componentes(image,[dict(detection_id='X',bbox=dict(x1=100,y1=100,x2=150,y2=160))])
            ocr._executar_easyocr_mosaico(mosaic,maps)
        self.assertEqual(reader.readtext.call_args.kwargs['canvas_size'],1600)
        self.assertEqual(reader.readtext.call_args.kwargs['mag_ratio'],1)
        self.assertEqual(reader.readtext_batched.call_args.kwargs['canvas_size'],320)


if __name__=='__main__':unittest.main()
