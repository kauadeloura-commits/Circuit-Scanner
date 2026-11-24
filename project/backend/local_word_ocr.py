"""Optional CPU word locator/reader. No downloads, network calls, or photo lookup."""
from pathlib import Path
from functools import lru_cache
import math,sys,threading
import cv2
import numpy as np

ASSETS=Path(__file__).resolve().parent/'handwriting_references'
VENDOR=Path(__file__).resolve().parent/'vendor'
_LOCK=threading.RLock()

@lru_cache(maxsize=4)
def _session(path,stamp):
    try:
        try:import onnxruntime as ort
        except ImportError:
            if not (VENDOR/'onnxruntime').is_dir():return None
            if str(VENDOR) not in sys.path:sys.path.insert(0,str(VENDOR))
            import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.enable_cpu_mem_arena=False
        return ort.InferenceSession(path,sess_options=options,providers=['CPUExecutionProvider'])
    except Exception as exc:
        # Native ONNX errors do not all inherit RuntimeError. This reader is
        # optional: a missing/incompatible/corrupted asset must not abort OCR.
        import logging
        logging.getLogger(__name__).warning('Leitor complementar indisponível: %s', type(exc).__name__)
        return None

def _get(name):
    path=ASSETS/name
    try:return _session(str(path),path.stat().st_mtime_ns)
    except OSError:return None

def locate_words(image):
    if image is None or not image.size:return []
    session=_get('word_detection.onnx')
    if session is None:return []
    h,w=image.shape[:2];scale=min(1.,1280/max(h,w))
    nh,nw=max(32,round(h*scale/32)*32),max(32,round(w*scale/32)*32)
    bgr=image if image.ndim==3 else cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    tensor=(cv2.resize(bgr,(nw,nh)).astype(np.float32)/255.-.5)/.5
    with _LOCK:prob=session.run(None,{session.get_inputs()[0].name:tensor.transpose(2,0,1)[None]})[0][0,0]
    bitmap=cv2.dilate((prob>.3).astype(np.uint8),np.ones((2,2),np.uint8))
    contours,_=cv2.findContours(bitmap,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    output=[];ph,pw=prob.shape
    for contour in contours[:1000]:
        rect=cv2.minAreaRect(contour);(cx,cy),(rw,rh),angle=rect
        if min(rw,rh)<3:continue
        x,y,bw,bh=cv2.boundingRect(contour);mask=np.zeros((bh,bw),np.uint8)
        cv2.drawContours(mask,[contour-np.array([x,y])],-1,1,-1)
        score=float(cv2.mean(prob[y:y+bh,x:x+bw],mask)[0])
        if score<.60:continue
        perimeter=cv2.arcLength(contour,True);distance=cv2.contourArea(contour)*1.6/max(1,perimeter)
        points=cv2.boxPoints(((cx,cy),(rw+2*distance,rh+2*distance),angle))
        box=dict(x1=max(0,float(points[:,0].min()*w/pw)),y1=max(0,float(points[:,1].min()*h/ph)),
                 x2=min(w,float(points[:,0].max()*w/pw)),y2=min(h,float(points[:,1].max()*h/ph)))
        if min(box['x2']-box['x1'],box['y2']-box['y1'])<5:continue
        angle=(angle+(90 if rh>rw else 0)+90)%180-90
        output.append(dict(bbox=box,confidence=score,region_source='local_dbnet_whole_word',
                           text_angle_degrees=float(angle),text_aspect_ratio=float(max(rw,rh)/min(rw,rh)),
                           text_polygon=points.tolist()))
    return sorted(output,key=lambda r:(r['bbox']['y1'],r['bbox']['x1']))[:96]

def read_words(images):
    if not images:return []
    session=_get('word_recognition.onnx');result=[None]*len(images)
    if session is None:return result
    metadata=session.get_modelmeta().custom_metadata_map
    characters=['blank']+metadata.get('character','').splitlines()+[' ']
    if len(characters)<10:return result
    for start in range(0,len(images),4):
        batch=images[start:start+4];valid=[(i,img) for i,img in enumerate(batch) if img is not None and img.size]
        if not valid:continue
        width=max(320,min(1024,max(math.ceil(48*img.shape[1]/img.shape[0]) for i,img in valid)))
        tensors=[]
        for _,img in valid:
            if img.ndim==2:img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            rw=min(width,max(1,math.ceil(48*img.shape[1]/img.shape[0])))
            value=(cv2.resize(img,(rw,48)).astype(np.float32)/255.-.5)/.5
            tensor=np.zeros((3,48,width),np.float32);tensor[:,:,:rw]=value.transpose(2,0,1);tensors.append(tensor)
        with _LOCK:pred=session.run(None,{session.get_inputs()[0].name:np.stack(tensors)})[0]
        for (index,_),p in zip(valid,pred):
            indices=p.argmax(axis=1);scores=p.max(axis=1);keep=indices!=0;keep[1:]&=indices[1:]!=indices[:-1]
            selected=indices[keep];confidence=scores[keep]
            if np.any(selected>=len(characters)):continue
            text=''.join(characters[i] for i in selected)
            result[start+index]=dict(text=text,confidence=float(confidence.mean()) if len(confidence) else 0.,
                                     min_character_confidence=float(confidence.min()) if len(confidence) else 0.)
    return result

def accept_word(row,kind=None):
    if not row or row['confidence']<.99 or row['min_character_confidence']<.94:return None
    from backend.sequence_ocr import compatible
    if not compatible(row['text'],None):return None
    return dict(row,confidence=row['confidence']*100,unresolved=False,reference_ids=[],
                reading_method='local_neural_whole_inscription',input_variant='original_photograph',trained_glyph_classifier=True)

def trusted_word(image,kind=None):
    return accept_word(read_words([image])[0],kind)
