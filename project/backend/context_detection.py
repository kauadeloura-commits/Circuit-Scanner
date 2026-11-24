"""Resolve auxiliary page labels without publishing text as components."""
from pathlib import Path
from functools import lru_cache
import numpy as np

AUXILIARY_NAMES={15:'text_aux',16:'terminal_aux',17:'transistor_unspecified_aux',18:'other_symbol_aux'}
SUBTYPE_MODEL=Path(__file__).resolve().parent.parent/'models/subtype_detector.pt'
TRANSISTOR_TYPES={10:'transistor_npn',11:'transistor_pnp',12:'mosfet_n',13:'mosfet_p'}

@lru_cache(maxsize=1)
def _subtype_model(stamp):
    from ultralytics import YOLO
    return YOLO(str(SUBTYPE_MODEL))

def transistor_subtype(image,box):
    if not SUBTYPE_MODEL.exists():return None
    h,w=image.shape[:2];size=max(box['x2']-box['x1'],box['y2']-box['y1']);pad=max(8,round(size*.35))
    x1,y1=max(0,int(box['x1'])-pad),max(0,int(box['y1'])-pad)
    x2,y2=min(w,int(box['x2'])+pad),min(h,int(box['y2'])+pad)
    if x2<=x1 or y2<=y1:return None
    crop=image[y1:y2,x1:x2];model=_subtype_model(SUBTYPE_MODEL.stat().st_mtime_ns)
    outputs=model.predict(crop,imgsz=320,conf=.15,verbose=False,device='cpu')
    if not outputs or outputs[0].boxes is None:return None
    target=np.array([box['x1']-x1,box['y1']-y1,box['x2']-x1,box['y2']-y1]);choices=[]
    for predicted,score,cid in zip(outputs[0].boxes.xyxy.cpu().numpy(),outputs[0].boxes.conf.cpu().numpy(),outputs[0].boxes.cls.cpu().numpy()):
        # The public generic transistor label also contains MOSFET symbols.
        # Restricting this resolver to BJTs turned every such proposal into
        # NPN/PNP or discarded a real MOSFET before its geometry was checked.
        if int(cid) not in TRANSISTOR_TYPES:continue
        ix=max(0,min(target[2],predicted[2])-max(target[0],predicted[0]));iy=max(0,min(target[3],predicted[3])-max(target[1],predicted[1]))
        overlap=ix*iy/max(1,(target[2]-target[0])*(target[3]-target[1]))
        if overlap>=.45 and score>=.35:choices.append((float(score),int(cid)))
    if not choices:return None
    choices.sort(reverse=True)
    if len(choices)>1 and choices[0][1]!=choices[1][1] and choices[0][0]-choices[1][0]<.12:return None
    return choices[0][1],choices[0][0]

def separate_predictions(image,detections):
    components=[];text=[]
    for det in detections:
        name=det['class']
        if name=='text_aux':
            if det['confidence']>=.60:text.append(det)
            continue
        if name=='transistor_unspecified_aux':
            subtype=transistor_subtype(image,det['bbox'])
            if subtype is None:continue
            cid,confidence=subtype
            components.append(dict(det,class_id=cid,**{'class':TRANSISTOR_TYPES[cid]},confidence=min(det['confidence'],confidence),subtype_source='local_trained_subtype_detector'))
            continue
        if name=='terminal_aux':
            from backend.recognition import _validar_classe_geometrica
            valid,score,evidence=_validar_classe_geometrica(image,det['bbox'],'vcc')
            if valid:components.append(dict(det,class_id=9,**{'class':'vcc'},geometry_validation=dict(valid=True,score=score,evidence=evidence),class_resolution='terminal_with_one_physical_power_stem'))
            continue
        if name.endswith('_aux'):continue
        components.append(det)
    return components,text
