import copy,json,unittest,time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch,Mock
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np
from backend import performance as perf,main,ocr,recognition,production_validation as pv
from backend.test_circuit_api import CircuitAPItests
from backend.test_unit_first_ocr import token,detection

class CacheContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.base=Path(self.tmp.name);self.folder=self.base/'TEST';self.folder.mkdir();(self.base/'backend').mkdir()
        (self.base/'backend/algorithm.py').write_text('v1');(self.folder/'original.png').write_bytes(b'photo')
        self.project={'project_id':'TEST','stored_filename':'original.png'};self.calls=[]
        self.p=patch.dict(globals(),BASE_DIR=self.base,PROJECTS_DIR=self.base,buscar_projeto=lambda pid:self.project,preparar_resposta_projeto=lambda p:dict(p));self.p.start()
        @perf.cached_project_stage('process')
        def process(project_id,force=False):
            self.calls.append(1);time.sleep(.02)
            (self.folder/'processed.png').write_bytes(b'processed');(self.folder/'processing.json').write_text('{}')
            return {'done':True}
        self.process=process
    def tearDown(self):self.p.stop();self.tmp.cleanup()
    def test_unchanged_input_reuses_but_force_modified_output_image_and_version_do_not(self):
        self.assertFalse(self.process('TEST')['processing_reused']);self.assertTrue(self.process('TEST')['processing_reused'])
        self.process('TEST',True);self.assertEqual(len(self.calls),2)
        (self.folder/'processed.png').write_bytes(b'modified');self.process('TEST');self.assertEqual(len(self.calls),3)
        (self.folder/'original.png').write_bytes(b'new photo');self.process('TEST');self.assertEqual(len(self.calls),4)
        (self.base/'backend/algorithm.py').write_text('v2');self.process('TEST');self.assertEqual(len(self.calls),5)
        (self.folder/'processing.json').unlink();self.process('TEST');self.assertEqual(len(self.calls),6)
        data=perf.read_json(self.folder/'performance.json');self.assertEqual(data['stages']['process']['success'],True)
    def test_parallel_identical_requests_compute_once(self):
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(self.process,['TEST','TEST']))
        self.assertEqual(len(self.calls),1);self.assertEqual(sum(r['processing_reused'] for r in results),1)
    def test_bad_cache_and_failed_work_do_not_create_success(self):
        (self.folder/'processing_cache.json').write_text('[]');self.assertFalse(self.process('TEST')['processing_reused'])
        (self.folder/'processing_cache.json').write_text('{"process":[]}');self.assertFalse(self.process('TEST')['processing_reused'])
        (self.folder/'performance.json').write_text('{"stages":[],"recent_runs":{}}');self.process('TEST',True)
        @perf.cached_project_stage('process')
        def fail(project_id,force=False):raise ValueError('test failure')
        with self.assertRaises(ValueError):fail('TEST',True)
        self.assertFalse(perf.read_json(self.folder/'performance.json')['stages']['process']['success'])
    def test_active_model_and_template_updates_invalidate_runtime(self):
        (self.base/'models').mkdir();model=self.base/'models/circuit_scanner_yolo.pt';model.write_bytes(b'v1')
        first=perf.runtime_signature(self.base);model.write_bytes(b'v2');self.assertNotEqual(first,perf.runtime_signature(self.base))
        reference=perf.runtime_signature(self.base,include_assets=False);model.write_bytes(b'v3');self.assertEqual(reference,perf.runtime_signature(self.base,include_assets=False))
        (self.base/'ocr_samples').mkdir();(self.base/'ocr_samples/index.json').write_text('{"updated":true}')
        self.assertNotEqual(first,perf.runtime_signature(self.base))

    def test_recent_equal_length_writes_do_not_trust_coarse_timestamp_cache(self):
        file=self.base/'backend/algorithm.py'
        with patch.object(perf,'time_ns',return_value=file.stat().st_mtime_ns),patch.object(perf,'_file_hash',return_value='stale'):
            first=perf.file_hash(file);file.write_text('v2')
            self.assertNotEqual(first,perf.file_hash(file))

class PipelineReuseTests(unittest.TestCase):
    setUp=CircuitAPItests.setUp
    tearDown=CircuitAPItests.tearDown
    update=CircuitAPItests.update
    def prepare_geometry(self):
        from backend.test_terminal_identity import fixture
        image,box,_=fixture('capacitor_polarized');cv2.imwrite(str(self.folder/'original.png'),image)
        info={'components':[dict(component_id='C1',detection_id='A',type='capacitor_polarized',bbox=box,reference={'text':'C1'},value={'text':'10uF'},class_confidence=.99)],'tokens':[]}
        (self.folder/'components.json').write_text(json.dumps(info));self.project['ocr_info']=json.dumps(info)
        return info
    def test_geometry_is_reused_and_value_edit_preserves_contacts(self):
        info=self.prepare_geometry()
        with patch.object(main,'atualizar_terminais',side_effect=self.update),patch.object(main,'atualizar_terminais_pos_fios',side_effect=self.update),patch.object(main,'detectar_fios',wraps=main.detectar_fios) as wires:
            first=main._gerar_terminais_automaticamente('TEST',self.project,info)
            before=(self.folder/'wires_mask.png').read_bytes();points=copy.deepcopy(first['components'][0]['terminals'])
            second=main._gerar_terminais_automaticamente('TEST',self.project,info)
            self.assertTrue(second.get('processing_reused'),second);self.assertEqual(wires.call_count,1)
            info['components'][0]['value']['text']='22uF'
            (self.folder/'components.json').write_text(json.dumps(info));self.project['ocr_info']=json.dumps(info)
            (self.folder/'circuit.json').write_text('{"stale":true}')
            third=main._gerar_terminais_automaticamente('TEST',self.project,info)
            self.assertTrue(third['processing_reused']);self.assertEqual(wires.call_count,1)
            self.assertEqual(third['components'][0]['terminals'],points);self.assertEqual(third['components'][0]['value']['text'],'22uF')
            self.assertEqual(before,(self.folder/'wires_mask.png').read_bytes());self.assertFalse((self.folder/'circuit.json').exists())
            info['components'][0]['bbox']['x1']+=2
            main._gerar_terminais_automaticamente('TEST',self.project,info);self.assertEqual(wires.call_count,2)
            (self.folder/'wires_skeleton.png').unlink();main._gerar_terminais_automaticamente('TEST',self.project,info);self.assertEqual(wires.call_count,3)
        self.assertTrue(self.client.get('/projects/TEST/performance').json()['stages']['geometry']['success'])
    def test_scalar_value_edit_does_not_call_geometry_but_updates_terminal_metadata(self):
        info=self.prepare_geometry()
        self.project['detections_info']=json.dumps({'detections':[{'detection_id':'A','class_name':'capacitor_polarized','bbox':info['components'][0]['bbox']}]})
        with patch.object(main,'atualizar_terminais',side_effect=self.update),patch.object(main,'atualizar_terminais_pos_fios',side_effect=self.update):main._gerar_terminais_automaticamente('TEST',self.project,info)
        with patch.object(main,'atualizar_ocr',side_effect=self.update),patch.object(main,'atualizar_terminais',side_effect=self.update),patch.object(main,'salvar_amostras_ocr',return_value=0),patch.object(main,'_gerar_terminais_automaticamente') as geometry:
            response=self.client.put('/projects/TEST/ocr/components',json={'components':[dict(detection_id='A',reference='C1',value='22uF')]})
        self.assertEqual(response.status_code,200,response.text);geometry.assert_not_called()
        saved=json.loads((self.folder/'terminals.json').read_text());self.assertEqual(saved['components'][0]['value']['text'],'22uF')
    def test_geometry_key_changes_for_manual_contacts_text_regions_and_corrections(self):
        info=self.prepare_geometry();key=perf.geometry_key(main.BASE_DIR,self.folder,self.project,info)
        current=perf.read_json(self.folder/'terminals.json');current['components'][0]['terminal_detection']={'status':'manual_ground_truth'}
        perf.write_json(self.folder/'terminals.json',current);self.assertNotEqual(key,perf.geometry_key(main.BASE_DIR,self.folder,self.project,info))
        key=perf.geometry_key(main.BASE_DIR,self.folder,self.project,info);(self.folder/'terminal_corrections.json').write_text('{"changed":true}')
        self.assertNotEqual(key,perf.geometry_key(main.BASE_DIR,self.folder,self.project,info))
        key=perf.geometry_key(main.BASE_DIR,self.folder,self.project,info);info['tokens']=[{'bbox':{'x1':1,'y1':1,'x2':5,'y2':7}}];perf.write_json(self.folder/'components.json',info)
        self.assertNotEqual(key,perf.geometry_key(main.BASE_DIR,self.folder,self.project,info))

    def test_guided_detection_defers_geometry_until_ocr_once_and_restores_context(self):
        from backend.test_guided_workflow import GuidedTests
        self.seed_terms=json.loads((self.folder/'terminals.json').read_text())
        detected={'count':1,'detections':[{'detection_id':'A','class':'capacitor','confidence':.99,'bbox':{'x1':10,'y1':10,'x2':90,'y2':90}}]}
        def read(pid):
            self.assertFalse(perf.DEFER_INITIAL_GEOMETRY.get())
            main._gerar_terminais_automaticamente(pid,self.project,{})
            GuidedTests.seed(self)
        with patch.object(main,'processar_projeto'),patch.object(main,'detectar_componentes',return_value=detected),patch.object(main,'atualizar_deteccoes',side_effect=self.update),patch.object(main,'ler_referencias_e_valores',side_effect=read),patch.object(main,'_gerar_terminais_automaticamente') as geometry:
            GuidedTests.step(self);GuidedTests.step(self)
            geometry.assert_not_called()
            self.assertEqual(GuidedTests.step(self)['phase'],'terminals');self.assertEqual(geometry.call_count,1)
        self.assertFalse(perf.DEFER_INITIAL_GEOMETRY.get())

    def status(self):
        from backend.test_guided_workflow import GuidedTests
        return GuidedTests.status(self)

    def test_ocr_endpoint_reuses_engine_after_final_detection_boxes_but_checks_geometry(self):
        info=self.prepare_geometry()
        det={'detection_id':'A','class':'capacitor_polarized','confidence':.99,'bbox':info['components'][0]['bbox']}
        self.project['detections_info']=json.dumps({'detections':[det],'count':1})
        perf.write_json(self.folder/'detections.json',{'detections':[det]})
        def read(**kw):
            final=copy.deepcopy(info);final.update(reference_ocr_count=1,value_ocr_count=1,retained_detections=[dict(det,bbox=dict(det['bbox'],x1=det['bbox']['x1']+1))])
            perf.write_json(kw['output_json_path'],final);cv2.imwrite(str(kw['output_image_path']),cv2.imread(str(kw['image_path'])))
            return final
        with patch.object(main,'executar_ocr_componentes',side_effect=read) as engine,patch.object(main,'atualizar_ocr',side_effect=self.update),patch.object(main,'atualizar_deteccoes',side_effect=self.update),patch.object(main,'_gerar_terminais_automaticamente',return_value={}) as geometry:
            a=self.client.post('/projects/TEST/ocr');b=self.client.post('/projects/TEST/ocr')
            self.assertEqual(a.status_code,200,a.text);self.assertEqual(b.status_code,200,b.text)
            self.assertTrue(b.json()['processing_reused']);self.assertEqual(engine.call_count,1);self.assertEqual(geometry.call_count,2)
            self.assertEqual(self.client.post('/projects/TEST/ocr?force=true').status_code,200);self.assertEqual(engine.call_count,2)

class FocusedOCRTests(unittest.TestCase):
    def test_numeric_ink_with_unit_cannot_become_an_alternate_component_reference(self):
        dets=[detection('A','inductor',20)]
        numeric=token('L00',22,30,confidence=.47);numeric['raw_text']='loo'
        value=token('loomH',22,30,confidence=.79);value['bbox']['x2']+=30
        result=ocr.criar_componentes_logicos([numeric,value],dets)['components'][0]
        self.assertEqual(result['reference']['source'],'generated');self.assertEqual(result['value']['text'],'100mH')
        # A real identifier elsewhere, or a different numeric reading, survives.
        genuine=token('L1',110,30)
        result=ocr.criar_componentes_logicos([numeric,value,genuine],dets)['components'][0]
        self.assertEqual(result['reference']['text'],'L1')
        genuine=token('L1',22,30)
        result=ocr.criar_componentes_logicos([value,genuine],dets)['components'][0]
        self.assertEqual(result['reference']['text'],'L1')
    def test_confident_value_and_reference_do_not_trigger_extra_reads_and_raw_text_survives(self):
        image=np.full((180,300,3),255,np.uint8);dets=[detection('A','capacitor',20)]
        readings=[token('C1',22),token('10uF',22,115),token('anotacao',200)]
        for t in readings:t['confidence']=.96
        with patch.object(ocr,'easyocr_disponivel',return_value=True),patch.object(ocr,'_executar_easyocr',return_value=readings),patch.object(ocr,'_executar_easyocr_mosaico') as mosaic,patch.object(ocr,'_executar_tesseract_tsv') as extra,patch.object(ocr,'_reread_unit_numbers',return_value=([],[])),patch.object(ocr,'_detectar_tokens_por_amostras',return_value=[]),patch.object(ocr,'_aplicar_biblioteca_ocr',side_effect=lambda _,tokens,d:tokens):
            result,_=ocr.extrair_tokens_ocr(image,'tesseract',dets)
        mosaic.assert_not_called();extra.assert_not_called();self.assertTrue(any(t['text']=='anotacao' for t in result.raw_tokens))
    def test_only_uncertain_component_is_selected_for_mosaic(self):
        dets=[detection('A','capacitor',20),detection('B','resistor',180)]
        readings=[token('C1',22),token('10uF',22,115),token('R1',182)]
        for t in readings:t['confidence']=.96
        self.assertEqual([d['detection_id'] for d in ocr._uncertain_ocr_detections(readings,dets)],['B'])
    def test_local_reads_restore_coordinates_and_merge_overlapping_regions(self):
        image=np.full((500,600,3),255,np.uint8);dets=[detection('A','capacitor',200),detection('B','capacitor',205)]
        with patch.object(ocr,'_executar_tesseract_tsv',return_value=[dict(text='10uF',bbox={'x1':1,'y1':2,'x2':10,'y2':20})]) as read:
            result=ocr._reread_uncertain_regions(image,'tesseract',dets,[])
        self.assertEqual(read.call_count,1);self.assertGreater(result[0]['bbox']['x1'],1)
    def test_large_general_image_stays_at_native_resolution(self):
        image=np.full((1400,2000,3),255,np.uint8);variants,scale=ocr._preparar_variantes(image)
        self.assertEqual(scale,1);self.assertEqual(variants[0][1].shape[:2],image.shape[:2])

class AdaptiveViewsTests(unittest.TestCase):
    def test_weak_or_missing_component_always_uses_extra_views(self):
        image=np.full((200,300,3),255,np.uint8)
        self.assertTrue(recognition._needs_additional_views(image,[]));self.assertTrue(recognition._needs_additional_views(image,[{'confidence':.5}]))
    def test_unexplained_thin_ink_requires_extra_views(self):
        image=np.full((200,300,3),255,np.uint8);box={'x1':100,'y1':60,'x2':160,'y2':140}
        cv2.rectangle(image,(100,60),(160,140),(0,0,0),2);cv2.line(image,(0,100),(100,100),(0,0,0),2);cv2.line(image,(160,100),(299,100),(0,0,0),2)
        d={'class':'resistor','confidence':.99,'bbox':box}
        with patch.object(recognition,'consolidar_multiview',side_effect=lambda d,**kw:d),patch.object(recognition,'filtrar_falsos_positivos_geometricos',side_effect=lambda i,d:(d,[])):
            self.assertFalse(recognition._needs_additional_views(image,[d]))
            cv2.line(image,(230,45),(230,62),(0,0,0),1)
            self.assertTrue(recognition._needs_additional_views(image,[d]))
    def test_identical_views_share_prediction_with_restored_source_labels(self):
        image=np.full((200,300,3),255,np.uint8);model=Mock();model.predict.return_value=[]
        with TemporaryDirectory() as tmp,patch.object(recognition,'_carregar_modelo',return_value=model):
            path=Path(tmp);cv2.imwrite(str(path/'original.png'),image)
            result=recognition.detectar_componentes(path/'original.png',path/'result.png',path/'result.json')
        self.assertTrue(result['performance']['identical_views_reused']);self.assertLess(model.predict.call_count,len(result['performance']['view_timings']))

class ReferenceApprovalTests(unittest.TestCase):
    def write_report(self,root,passed=10):
        root=Path(root);root.mkdir(exist_ok=True,parents=True);report=dict(passed=passed,total=10)
        perf.write_json(root/'validation_report.json',report)
        for i in range(1,11):
            p=root/f'T{i:02d}';p.mkdir(exist_ok=True);(p/'project.zip').write_bytes(b'fixture zip')
        return report
    def test_reuses_only_verified_passing_report_and_force_version_damage_rerun(self):
        with TemporaryDirectory() as tmp,patch.object(pv,'run_reference_suite',side_effect=self.write_report) as run,patch.object(perf,'runtime_signature',return_value='v1') as version:
            a=pv.run_reference_suite_cached(tmp);self.assertFalse(a['cached'])
            self.assertTrue(pv.run_reference_suite_cached(tmp)['cached']);self.assertEqual(run.call_count,1)
            pv.run_reference_suite_cached(tmp,force=True);self.assertEqual(run.call_count,2)
            version.return_value='v2';pv.run_reference_suite_cached(tmp);self.assertEqual(run.call_count,3)
            (Path(tmp)/'T03/project.zip').write_bytes(b'changed');pv.run_reference_suite_cached(tmp);self.assertEqual(run.call_count,4)
    def test_failure_is_never_reused(self):
        with TemporaryDirectory() as tmp,patch.object(pv,'run_reference_suite',side_effect=lambda p:self.write_report(p,9)) as run:
            self.assertFalse(pv.run_reference_suite_cached(tmp)['cached']);self.assertFalse(pv.run_reference_suite_cached(tmp)['cached']);self.assertEqual(run.call_count,2)
            self.assertFalse((Path(tmp)/'reference_cache.json').exists())
