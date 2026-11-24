"""Read complete electrical inscriptions with a compact CTC sequence model.

No character splitting and no lookup by project/image identity are used.
The optional checkpoint is accepted only after independent validation.
"""
from pathlib import Path
from functools import lru_cache
import re
import cv2
import numpy as np

MODEL=Path(__file__).resolve().parent/'handwriting_references/sequence_ocr.npz'
HEIGHT=32
WIDTH=192

def prepare(mask):
    if mask is None or mask.size==0:return None
    points=cv2.findNonZero((mask>0).astype(np.uint8))
    if points is None:return None
    x,y,w,h=cv2.boundingRect(points)
    scale=min((HEIGHT-6)/h,(WIDTH-12)/w)
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    result=np.zeros((HEIGHT,WIDTH),np.uint8)
    resized=cv2.resize(mask[y:y+h,x:x+w],(nw,nh),interpolation=cv2.INTER_AREA)
    result[(HEIGHT-nh)//2:(HEIGHT-nh)//2+nh,4:4+nw]=resized
    return result

def architecture(classes):
    import torch
    from torch import nn
    class Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn=nn.Sequential(nn.Conv2d(1,24,3,padding=1),nn.BatchNorm2d(24),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(24,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(48,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d((2,1)))
            self.rnn=nn.GRU(256,96,batch_first=True,bidirectional=True)
            self.output=nn.Linear(192,classes)
        def forward(self,x):
            features=self.cnn(x).permute(0,3,1,2).flatten(2)
            sequence,_=self.rnn(features)
            return self.output(sequence).transpose(0,1)
    return Reader()

@lru_cache(maxsize=1)
def _load(stamp):
    import torch,json
    with np.load(MODEL,allow_pickle=False) as data:
        alphabet=str(data['alphabet']);model=architecture(len(alphabet)+1)
        model.load_state_dict({k:torch.from_numpy(data[k]) for k in model.state_dict()})
        calibration=json.loads(str(data['calibration']))
    model.eval()
    return model,alphabet,calibration

def decode(logits,alphabet):
    import torch
    probabilities=torch.softmax(logits,dim=-1)
    confidence,indices=probabilities.max(-1)
    rows=[]
    for b in range(indices.shape[1]):
        last=0;text=[];scores=[]
        for i,p in zip(indices[:,b].tolist(),confidence[:,b].tolist()):
            if i and i!=last:text.append(alphabet[i-1]);scores.append(p)
            last=i
        rows.append(dict(text=''.join(text),confidence=float(np.exp(np.mean(np.log(np.maximum(scores,1e-8))))) if scores else 0.,min_character_confidence=min(scores,default=0.)))
    return rows

def compatible(text,kind=None):
    from backend.reference_labels import reference_classes
    refs=reference_classes(text)
    if refs:return kind is None or kind in refs
    if re.fullmatch(r'[RCLDQT]\d{1,4}',text):
        allowed={'R':{'resistor'},'C':{'capacitor','capacitor_polarized'},'L':{'inductor'},'D':{'diode','led'},'Q':{'transistor_npn','transistor_pnp'}}
        return kind is None or kind in allowed[text[0]]
    if re.fullmatch(r'(LED\d{1,3}|[+-]?V(?:CC|EE|DD|SS)\d*|[βB]=\d{1,3}|hFE=\d{1,3})',text):return True
    if re.fullmatch(r'(BC\d{3}[A-C]?|1N\d{4}|LM\d{3}|IRF\d{3,4})',text):return True
    from backend.measurement_text import parse_measurement
    parsed=parse_measurement(text)
    if not parsed:return False
    expected={'resistor':'ohm','capacitor':'farad','capacitor_polarized':'farad','inductor':'henry','vcc':'volt','source_dc':'volt','battery':'volt'}
    return not kind or kind not in expected or parsed.get('unit')==expected[kind]

def read_lines(masks,kinds=None):
    if not MODEL.is_file():return [None]*len(masks)
    import torch
    model,alphabet,calibration=_load(MODEL.stat().st_mtime_ns)
    prepared=[prepare(m) for m in masks];valid=[i for i,p in enumerate(prepared) if p is not None];result=[None]*len(masks)
    if not valid:return result
    with torch.inference_mode():
        values=torch.from_numpy(np.stack([prepared[i] for i in valid]).astype(np.float32)/255.).unsqueeze(1)
        decoded=decode(model(values),alphabet)
    for i,row in zip(valid,decoded):
        threshold=calibration.get('minimum_confidence',.995)
        kind=(kinds or [None]*len(masks))[i]
        if row['confidence']>=threshold and row['min_character_confidence']>=.85 and compatible(row['text'],kind):
            result[i]=dict(row,confidence=row['confidence']*100,reference_ids=[],unresolved=False,reading_method='trained_whole_inscription_ctc',trained_glyph_classifier=True)
    return result

