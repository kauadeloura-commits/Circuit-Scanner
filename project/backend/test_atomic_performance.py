import json,unittest,os,ctypes,threading
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from backend import performance as perf

class AtomicPerformanceTests(unittest.TestCase):
    @unittest.skipUnless(os.name=='nt','Windows sharing mode regression')
    def test_real_windows_reader_without_delete_sharing(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'performance.json';perf.write_json(path,{'previous':True})
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
            kernel.CreateFileW.restype=ctypes.c_void_p
            kernel.CloseHandle.argtypes=[ctypes.c_void_p]
            handle=kernel.CreateFileW(str(path),0x80000000,3,None,3,0,None)
            self.assertNotEqual(handle,ctypes.c_void_p(-1).value)
            release=threading.Timer(.07,lambda:kernel.CloseHandle(handle));release.start()
            try:perf.write_json(path,{'completed':True})
            finally:release.join()
            self.assertEqual(json.loads(path.read_text()),{'completed':True})

    def test_transient_lock_retries_without_truncating_previous_document(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'data.json';perf.write_json(path,{'saved':'old'})
            original=Path.replace;attempts=[]
            def replace(source,target):
                attempts.append(1)
                self.assertEqual(json.loads(path.read_text()),{'saved':'old'})
                if len(attempts)<3:raise PermissionError(13,'held by reader')
                return original(source,target)
            with patch.object(Path,'replace',replace),patch.object(perf,'sleep') as sleep:
                perf.write_json(path,{'saved':'new'})
            self.assertEqual(len(attempts),3);self.assertEqual(sleep.call_count,2)
            self.assertEqual(json.loads(path.read_text()),{'saved':'new'});self.assertEqual(list(Path(folder).glob('*.tmp')),[])

    def test_permanent_lock_preserves_saved_file_and_optional_timing_does_not_fail_work(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'performance.json';perf.write_json(path,{'previous':True})
            with patch.object(Path,'replace',side_effect=PermissionError(13,'held')),patch.object(perf,'sleep'),self.assertLogs(perf._LOGGER,level='WARNING'):
                with perf.timed(folder,'recognition'):
                    result='completed'
            self.assertEqual(result,'completed');self.assertEqual(json.loads(path.read_text()),{'previous':True})
            self.assertEqual(list(Path(folder).glob('*.tmp')),[])

    def test_original_processing_error_is_not_hidden_by_timing_failure(self):
        with TemporaryDirectory() as folder:
            with patch.object(perf,'write_json',side_effect=PermissionError('held')),self.assertLogs(perf._LOGGER,level='WARNING'):
                with self.assertRaisesRegex(ValueError,'actual processing failure'):
                    with perf.timed(folder,'recognition'):raise ValueError('actual processing failure')

    def test_critical_save_still_reports_permanent_permission_error(self):
        with TemporaryDirectory() as folder:
            with patch.object(Path,'replace',side_effect=PermissionError('held')),patch.object(perf,'sleep'):
                with self.assertRaises(PermissionError):perf.write_json(Path(folder)/'components.json',{'reviewed':True})

if __name__=='__main__':unittest.main()
