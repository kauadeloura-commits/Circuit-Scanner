"""Locally compiled circuit-specific word reader; fixed-size inference only."""
from pathlib import Path
from functools import lru_cache
import json,math,threading
import cv2,numpy as np
from PIL import Image
ASSETS=Path(__file__).resolve().parent/'handwriting_references'
_LOCK=threading.RLock()

@lru_cache(maxsize=1)
def _load(stamp):
    import torch
    model=torch.jit.load(str(ASSETS/'circuit_word_reader.ts'),map_location='cpu').eval()
    metadata=json.loads((ASSETS/'circuit_word_reader.json').read_text(encoding='utf8'))
    return model,metadata['charset']

def read_words(images):
    output=[None]*len(images);path=ASSETS/'circuit_word_reader.ts'
    if not images:return output
    if not path.is_file():return output
    import torch
    try:model,chars=_load(path.stat().st_mtime_ns)
    except (OSError,ValueError,RuntimeError):return output
    for start in range(0,len(images),4):
        selected=[(i,img) for i,img in enumerate(images[start:start+4]) if img is not None and img.size]
        if not selected:continue
        values=[]
        for i,img in selected:
            if img.ndim==2:img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            resized=Image.fromarray(img[:,:,::-1]).resize((128,32),Image.Resampling.BILINEAR)
            values.append(torch.from_numpy(np.asarray(resized).copy().transpose(2,0,1)).float()/127.5-1.)
        while len(values)<4:values.append(torch.zeros_like(values[0]))
        with _LOCK,torch.inference_mode():probs=model(torch.stack(values)).softmax(-1)
        scores,ids=probs.max(-1)
        for (index,_),row,confidence in zip(selected,ids.tolist(),scores.tolist()):
            if 0 not in row:continue
            end=row.index(0)
            letters=row[:end]
            if any(x<1 or x>len(chars) for x in letters):continue
            score=np.array(confidence[:min(len(row),end+1)])
            if not len(score) or not np.all(np.isfinite(score)):continue
            output[start+index]=dict(text=''.join(chars[x-1] for x in letters),confidence=float(np.prod(score)),
                                     min_character_confidence=float(score.min()),source_model='circuit_parseq')
    return output

def same_reading(a,b):
    import re
    x,y=(str(r['text']).strip().replace('µ','u').replace('μ','u').replace('Ω','Ω') for r in (a,b))
    if x==y:return True
    if re.fullmatch(r'(?:[RCLDQ]|LED)\d+',x,re.I) and x.upper()==y.upper():return True
    from backend.measurement_text import parse_measurement
    p,q=parse_measurement(x),parse_measurement(y)
    try:return bool(p and q and p['unit']==q['unit'] and math.isclose(float(p['normalized']),float(q['normalized']),rel_tol=1e-10,abs_tol=0.))
    except (KeyError,TypeError,ValueError):return False

def choose_word(general,specialized):
    from backend.local_word_ocr import accept_word
    a,b=accept_word(general),accept_word(specialized)
    if a and b:return b if same_reading(a,b) else None
    return b or a
