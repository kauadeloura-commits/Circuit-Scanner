"""Bound CPU thread oversubscription for small vision models; no GPU required."""
from threading import Lock,Thread
import os,sys

_LOCK=Lock()
_READY=False
_IMAGE_READY=False
_WARMUP_THREAD=None
_BLAS_LIMITER=None


def configure_image_processing():
    global _IMAGE_READY,_BLAS_LIMITER
    with _LOCK:
        import cv2
        budget=max(1,min(2,os.cpu_count() or 1))
        # Detector libraries also change OpenCV's global setting. Restore our
        # budget on each analysis entry, including requests in worker threads.
        if not _IMAGE_READY or cv2.getNumThreads()!=budget:cv2.setNumThreads(budget)
        if not _IMAGE_READY:
            try:
                from threadpoolctl import threadpool_limits
                _BLAS_LIMITER=threadpool_limits(limits=budget,user_api='blas')
            except ImportError:pass
        _IMAGE_READY=True
        torch=sys.modules.get('torch')
        if torch is not None and _READY:
            budget=max(1,min(4,os.cpu_count() or 1))
            current=torch.get_num_threads()
            if isinstance(current,int) and current!=budget:torch.set_num_threads(budget)


def start_model_warmup():
    """Prepare the existing CPU detector once while the user opens the site."""
    global _WARMUP_THREAD
    if os.environ.get('SCANNER_DISABLE_AI_WARMUP')=='1':return
    configure_image_processing()
    with _LOCK:
        if _WARMUP_THREAD is not None:return
        def warm():
            try:
                import numpy as np
                from backend.recognition import _carregar_modelo
                model=_carregar_modelo()
                model.predict(np.zeros((64,64,3),np.uint8),imgsz=64,device='cpu',conf=.99,verbose=False,save=False,max_det=1)
            except Exception:
                # Normal recognition reports a missing/incompatible model; a
                # background preparation failure must not stop the website.
                pass
        _WARMUP_THREAD=Thread(target=warm,name='scanner-model-warmup',daemon=True)
        _WARMUP_THREAD.start()


def configure_cpu_inference():
    global _READY
    with _LOCK:
        import torch
        budget=max(1,min(4,os.cpu_count() or 1))
        if _READY:
            current=torch.get_num_threads()
            if isinstance(current,int) and current!=budget:torch.set_num_threads(budget)
            return
        torch.set_num_threads(budget)
        # Changing inter-op threads after an earlier inference is forbidden
        # by PyTorch. The intra-op budget remains valid in that case.
        try:torch.set_num_interop_threads(1)
        except RuntimeError:pass
        _READY=True
