"""Small supervised handwriting classifier; the supplied outlines remain evidence."""
from pathlib import Path
from functools import lru_cache
import json
import cv2
import numpy as np

MODEL='glyph_classifier.npz'
SPECIALIST='glyph_specialist.npz'
FEATURE_VERSION='hog48-numpy-projections-aspect-1'

@lru_cache(maxsize=1)
def _cells():
    y,x=np.indices((48,48));return (y//8*6+x//8).ravel()*9

def _hog(shape):
    gx=cv2.Sobel(shape,cv2.CV_32F,1,0,ksize=1);gy=cv2.Sobel(shape,cv2.CV_32F,0,1,ksize=1)
    magnitude,angle=cv2.cartToPolar(gx,gy)
    bins=np.floor(angle.ravel()*9/(2*np.pi)).astype(np.int32)%9
    histogram=np.bincount(_cells()+bins,weights=magnitude.ravel(),minlength=324).reshape(6,6,9)
    blocks=np.concatenate((histogram[:-1,:-1],histogram[:-1,1:],histogram[1:,:-1],histogram[1:,1:]),axis=2).reshape(25,36)
    blocks/=np.sqrt(np.sum(blocks*blocks,axis=1,keepdims=True)+1e-6)
    blocks=np.minimum(blocks,.2);blocks/=np.sqrt(np.sum(blocks*blocks,axis=1,keepdims=True)+1e-6)
    return blocks.ravel()

def features(shape,aspect=None):
    if shape is None or not shape.size or not np.any(shape):return None
    points=cv2.findNonZero((shape>0).astype(np.uint8));x,y,w,h=cv2.boundingRect(points)
    original_aspect=float(aspect if aspect is not None else w/h)
    if shape.shape!=(48,48):
        crop=shape[y:y+h,x:x+w];scale=40/max(w,h);nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
        canvas=np.zeros((48,48),np.uint8);patch=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_AREA)
        canvas[(48-nh)//2:(48-nh)//2+nh,(48-nw)//2:(48-nw)//2+nw]=(patch>100)*255;shape=canvas
    shape=(shape>0).astype(np.uint8)*255
    hog=_hog(shape)
    horizontal=np.mean(shape.reshape(12,4,48)>0,axis=(1,2));vertical=np.mean(shape.reshape(48,12,4)>0,axis=(0,2))
    pooled=cv2.resize(shape,(8,8),interpolation=cv2.INTER_AREA).ravel()/255
    return np.r_[hog,horizontal,vertical,pooled,np.clip(np.log(original_aspect),-2,2)].astype(np.float32)

def model_stamp(folder):
    path=Path(folder)/MODEL
    try:
        s=path.stat()
        try:
            extra=(Path(folder)/SPECIALIST).stat();specialist=(extra.st_mtime_ns,extra.st_size)
        except OSError:specialist=None
        return (s.st_mtime_ns,s.st_ctime_ns,s.st_size,specialist)
    except OSError:return None

@lru_cache(maxsize=4)
def _load(folder,stamp):
    try:
        with np.load(Path(folder)/MODEL,allow_pickle=False) as data:
            if str(data['feature_version'])!=FEATURE_VERSION:return None
            weights=data['weights'].astype(np.float32);bias=data['bias'].astype(np.float32);labels=data['labels'].tolist()
            if weights.shape!=(len(labels),989) or bias.shape!=(len(labels),) or not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):return None
            return weights,bias,labels
    except (OSError,ValueError,KeyError):return None

@lru_cache(maxsize=2048)
def _vector(blob,aspect):return features(np.frombuffer(blob,np.uint8).reshape(48,48),aspect)

@lru_cache(maxsize=2048)
def _scores(blob,aspect,folder,stamp):
    model=_load(folder,stamp);vector=_vector(blob,aspect)
    if model is None or vector is None:return None
    weights,bias,labels=model
    # The small matrix is reduced directly; no extra BLAS thread pool.
    scores=np.sum(weights*vector[None,:],axis=1)+bias
    return scores,labels

@lru_cache(maxsize=4)
def _load_specialist(folder,stamp):
    try:
        with np.load(Path(folder)/SPECIALIST,allow_pickle=False) as data:
            if str(data['feature_version'])!=FEATURE_VERSION:return None
            names=['mean','scale','a_weight','a_bias','b_weight','b_bias','c_weight','c_bias','d_weight','d_bias']
            model={k:data[k].astype(np.float32) for k in names};model['labels']=data['labels'].tolist()
            shapes={'mean':(989,),'scale':(989,),'a_weight':(192,989),'a_bias':(192,),
                    'b_weight':(96,192),'b_bias':(96,),'c_weight':(len(model['labels']),96),
                    'c_bias':(len(model['labels']),),'d_weight':(2,96),'d_bias':(2,)}
            if any(model[k].shape!=shape for k,shape in shapes.items()):return None
            if any(not np.all(np.isfinite(model[k])) for k in names) or np.any(model['scale']<=0):return None
            return model
    except (OSError,ValueError,KeyError):return None

@lru_cache(maxsize=2048)
def _specialist_scores(blob,aspect,folder,stamp):
    model=_load_specialist(folder,stamp)
    if model is None:return None
    x=(_vector(blob,aspect)-model['mean'])/model['scale']
    hidden=np.maximum(0,np.sum(model['a_weight']*x[None,:],axis=1)+model['a_bias'])
    hidden=np.maximum(0,np.sum(model['b_weight']*hidden[None,:],axis=1)+model['b_bias'])
    logits=np.sum(model['c_weight']*hidden[None,:],axis=1)+model['c_bias']
    domain=np.sum(model['d_weight']*hidden[None,:],axis=1)+model['d_bias']
    p=np.exp(logits-logits.max());p/=p.sum();d=np.exp(domain-domain.max());d/=d.sum()
    return p,logits,model['labels'],float(d[1])

@lru_cache(maxsize=32)
def _absolute_folder(folder):return str(Path(folder).resolve())

def support(shape,aspect,folder,allowed=None):
    stamp=model_stamp(folder)
    if not stamp:return None
    result=_scores(shape.tobytes(),float(aspect),_absolute_folder(str(folder)),stamp)
    if result is None:return None
    scores,labels=result
    expert=_specialist_scores(shape.tobytes(),float(aspect),_absolute_folder(str(folder)),stamp)
    if expert is not None:
        probabilities,logits,expert_labels,font_probability=expert;order=np.argsort(probabilities)[::-1]
        selected=int(order[0]);label=expert_labels[selected];probability=float(probabilities[selected])
        numeric=allowed is not None and set(allowed)<=set('0123456789')
        if numeric:
            # The expected numeric suffix resolves identical printed outlines
            # such as I/l/1 and O/o/0. Letters remain letters in model names.
            aliases={'I':'1','l':'1','O':'0','o':'0'}
            grouped={digit:sum(float(p) for p,c in zip(probabilities,expert_labels) if aliases.get(c,c)==digit) for digit in allowed}
            label=max(grouped,key=grouped.get);probability=grouped[label]
        # Familiar handwriting keeps the previously validated classifier.
        # The specialist supplies a new alphabet or a very clear font glyph.
        novel=label not in labels
        if allowed is not None and label not in allowed:
            alias={'I':'1','l':'1','O':'0','o':'0'}.get(label)
            if font_probability>=.995 and alias in allowed:label=alias
        if probability>=.995 and (novel or font_probability>=(.98 if numeric else .997)) and (allowed is None or label in allowed):
            margin=float(logits[selected]-logits[int(order[1])])
            if numeric:margin=max(margin,float(np.log(max(probability,1e-8)/max(1e-8,1-probability))))
            return dict(text=label,margin=margin,score=float(logits[selected]),boost=.045,
                        font_specialist=True,probability=probability,font_probability=font_probability)
    indices=[i for i,label in enumerate(labels) if allowed is None or label in allowed]
    if not indices:return None
    order=np.array(indices)[np.argsort(scores[indices])[::-1]]
    top=int(order[0]);margin=float(scores[top]-scores[order[1]]) if len(order)>1 else 0.
    if float(scores[top])<.15 or margin<.18:return None
    return dict(text=labels[top],margin=margin,score=float(scores[top]),boost=min(.045,.018+.025*margin))

def status(folder):
    stamp=model_stamp(folder)
    loaded=_load(str(Path(folder).resolve()),stamp) if stamp else None
    specialist=_load_specialist(str(Path(folder).resolve()),stamp) if stamp else None
    return dict(ready=loaded is not None,feature_version=FEATURE_VERSION,classes=loaded[2] if loaded else [],method='supervised_linear_softmax',
                font_specialist_ready=specialist is not None,font_specialist_classes=specialist['labels'] if specialist else [])
