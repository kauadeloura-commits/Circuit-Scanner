"""Project-owned, explicit input domain; never infer it from pen color."""
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
from inspect import unwrap
from pathlib import Path
import json

CURRENT=ContextVar('drawing_type',default='legacy')
LABELS={'handwritten':'Desenho manuscrito','digital':'Desenho digital/impresso','legacy':'Tipo não informado'}
VERSION='drawing-domain-1'

def validate(value,*,allow_legacy=False):
    if value in {'handwritten','digital'} or (allow_legacy and value=='legacy'):return value
    raise ValueError('Selecione Desenho manuscrito ou Desenho digital/impresso.')

def read(folder):
    p=Path(folder)/'drawing_settings.json'
    if not p.exists():return {'drawing_type':'legacy','policy_version':VERSION}
    value=json.loads(p.read_text(encoding='utf8'))
    validate(value.get('drawing_type'),allow_legacy=True)
    return value

def save(folder,value):
    from backend.performance import write_json
    value=validate(value,allow_legacy=True)
    write_json(Path(folder)/'drawing_settings.json',{'drawing_type':value,'policy_version':VERSION})

@contextmanager
def using(value):
    token=CURRENT.set(validate(value,allow_legacy=True))
    try:yield
    finally:CURRENT.reset(token)

def project_context(fn):
    @wraps(fn)
    def wrapped(project_id,*args,**kwargs):
        folder=unwrap(fn).__globals__['PROJECTS_DIR']/project_id
        with using(read(folder)['drawing_type']):return fn(project_id,*args,**kwargs)
    return wrapped

def digital():return CURRENT.get()=='digital'

def printed_ink(image):
    """Preserve original thin achromatic and colored strokes without dilation."""
    import cv2,numpy as np
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    background=cv2.GaussianBlur(gray,(0,0),max(5,min(gray.shape)/70))
    normalized=cv2.divide(gray,np.maximum(background,1),scale=255)
    weak=((gray<245)&(normalized<235)).astype(np.uint8)
    count,labels=cv2.connectedComponents(weak,connectivity=8)
    strong=np.unique(labels[(gray<220)&(normalized<210)])
    keep=np.zeros(count,np.uint8);keep[strong]=255;keep[0]=0
    return keep[labels]
