"""Small supervised text/background guard; physical leads remain authoritative."""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from backend.glyph_classifier import features as glyph_features

MODEL=Path(__file__).resolve().parent/'handwriting_references/text_component_classifier.npz'
CONTEXT_MODEL=MODEL.with_name('text_context_classifier.npz')
VERSION='body-and-halo-hog-1980-v1'

def features(image,bbox,page_mask=None):
    from backend.resistor_geometry import symbol_mask
    from backend.handwriting_reference import normalize
    mask=symbol_mask(image) if page_mask is None else page_mask
    h,w=mask.shape;x1,y1=max(0,int(bbox['x1'])),max(0,int(bbox['y1']))
    x2,y2=min(w,int(np.ceil(bbox['x2']))),min(h,int(np.ceil(bbox['y2'])))
    if x2<=x1 or y2<=y1:return None
    body=mask[y1:y2,x1:x2];pad=max(5,min(40,round(min(x2-x1,y2-y1)*.35)))
    halo=mask[max(0,y1-pad):min(h,y2+pad),max(0,x1-pad):min(w,x2+pad)]
    vectors=[]
    for crop in (body,halo):
        normalized=normalize(crop)
        if normalized is None:return None
        vectors.append(glyph_features(*normalized))
    from backend.vcc_topology import external_leads
    ports=external_leads(image,bbox,mask)
    return np.r_[*vectors,min(8,ports)/4,float(np.count_nonzero(body))/body.size].astype(np.float32)

@lru_cache(maxsize=4)
def _load(path,stamp):
    try:
        with np.load(path,allow_pickle=False) as d:
            if str(d['feature_version'])!=VERSION:return None
            model={k:d[k].astype(np.float32) for k in ['mean','scale','w1','b1','w2','b2']}
            shapes={'mean':(1980,),'scale':(1980,),'w1':(96,1980),'b1':(96,),
                    'w2':(2,96),'b2':(2,)}
            if any(model[k].shape!=shape or not np.all(np.isfinite(model[k]))
                   for k,shape in shapes.items()):return None
            if np.any(model['scale']<=0):return None
            model['threshold']=float(d['threshold'])
            if not .5<model['threshold']<1:return None
            return model
    except (OSError,ValueError,KeyError):return None

def probability(vector,path=MODEL):
    path=Path(path)
    try:s=path.stat()
    except OSError:return None
    model=_load(str(path),(s.st_mtime_ns,s.st_size))
    if model is None or vector is None or vector.shape!=model['mean'].shape:return None
    x=(vector-model['mean'])/model['scale']
    hidden=np.maximum(0,np.sum(model['w1']*x[None,:],axis=1)+model['b1'])
    logits=np.sum(model['w2']*hidden[None,:],axis=1)+model['b2'];logits-=logits.max()
    scores=np.exp(logits);scores/=scores.sum()
    return float(scores[1]),model['threshold']

def text_evidence(image,bbox,page_mask=None):
    if not MODEL.is_file() and not CONTEXT_MODEL.is_file():return False,{}
    vector=features(image,bbox,page_mask)
    if vector is None:return False,{}
    # The isolated-glyph model never rejects a photographed lead. Its separate
    # context specialist handles ruled paper and partially clipped words; its
    # much stricter threshold was calibrated against components WITH leads.
    models=[]
    if vector[-2]==0:models.append((MODEL,.995,'supervised_text_without_external_component_leads'))
    from backend.drawing_mode import digital
    if not digital():models.append((CONTEXT_MODEL,.9999,'supervised_text_in_notebook_context'))
    for path,minimum,reason in models:
        result=probability(vector,path)
        if result is None:continue
        p,threshold=result;threshold=max(minimum,threshold)
        if p>=threshold:
            return True,dict(reason=reason,text_probability=round(p,6),threshold=threshold,
                             metric=VERSION,hard_text_rejection=True)
    return False,{}
