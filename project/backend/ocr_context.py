"""Component neighborhoods guide reading; raw model letters remain untouched."""
import copy
import math
import re
import cv2
import numpy as np

BJT={'transistor_npn','transistor_pnp'}


def region_context(box,detections):
    pairs=[]
    for d in detections:
        if d.get('class')=='gnd':continue
        b=d['bbox'];x=(box['x1']+box['x2'])/2;y=(box['y1']+box['y2'])/2
        if d.get('class')=='vcc' and b['x1']<=x<=b['x2'] and b['y1']<=y<=b['y2']:continue
        dx=max(b['x1']-x,0,x-b['x2']);dy=max(b['y1']-y,0,y-b['y2'])
        radius=max(45,min(180,max(b['x2']-b['x1'],b['y2']-b['y1'])*.65))
        distance=math.hypot(dx,dy)
        # A large transistor box must not take the value beside a small R/C.
        # Rank true pixel proximity; normalized box size favored large symbols.
        score=0 if distance==0 else distance/80+math.hypot(x-(b['x1']+b['x2'])/2,y-(b['y1']+b['y2'])/2)/640
        if distance<=radius*1.8:pairs.append((score,d))
    if not pairs:return None
    pairs.sort(key=lambda p:p[0]);score,d=pairs[0]
    # An overlap of two different symbols cannot justify a label correction.
    ambiguous=any((other_score==0 if score==0 else abs(other_score-score)<.08)
                  and other['class']!=d['class'] for other_score,other in pairs[1:])
    return dict(detection_id=d.get('detection_id'),kind=d['class'],distance=round(score,4),ambiguous=ambiguous)


def prioritized_regions(page,regions,detections):
    output=[];skipped=[]
    for box in regions:
        crop=page[box['y1']:box['y2'],box['x1']:box['x2']]
        n,_,stats,_=cv2.connectedComponentsWithStats((crop<128).astype(np.uint8),8)
        substantial=[s for s in stats[1:] if s[4]>=max(8,crop.shape[0]*.20)]
        context=region_context(box,detections)
        # Solitary units/numbers near a component still receive a first pass.
        if len(substantial)<2 and context is None:
            skipped.append(dict(bbox=dict(box),reason='isolated_symbol_or_paper_fragment'));continue
        output.append(dict(box,component_context=context))
    return sorted(output,key=lambda b:(b['component_context'] is None,
        (b['component_context'] or {}).get('distance',99),b['y1'],b['x1'])),skipped


def soft_text_page(image,binary):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    foreground=binary<128
    if not np.any(foreground):return binary.copy()
    low=float(np.percentile(gray[foreground],5));high=max(low+30,float(np.percentile(gray,80)))
    contrasted=np.clip((gray.astype(np.float32)-low)*255/(high-low),0,255).astype(np.uint8)
    vicinity=cv2.dilate(foreground.astype(np.uint8),np.ones((3,3),np.uint8))>0
    return np.where(vicinity,contrasted,255).astype(np.uint8)


def gain_context(box,detections):
    nearest=box.get('component_context') or region_context(box,detections)
    if not nearest:return None
    if nearest['kind'] in BJT:return nearest if not nearest.get('ambiguous') else None
    if nearest.get('distance')==0:return None
    return region_context(box,[d for d in detections if d.get('class') in BJT])


def repair_gain_label(token,detections):
    match=re.fullmatch(r'\s*\(?([βBPpb])\s*([=:Ee])\s*([0-9OoIiLlSsZzB.,]+)\s*\)?\s*',token['text'])
    if not match:return token
    context=gain_context(dict(token['bbox'],component_context=token.get('component_context')),detections)
    if not context or context.get('ambiguous'):return token
    digits=''.join({'O':'0','o':'0','I':'1','i':'1','l':'1','L':'1','S':'5','s':'5','Z':'2','z':'2','B':'8'}.get(c,c) for c in match[3]).replace(',','.')
    try:number=float(digits)
    except ValueError:return token
    if not 0<number<=100000:return token
    corrected='β='+digits
    if corrected==token['text']:return dict(token,component_context=context)
    result=copy.deepcopy(token);result['text']=corrected;result['raw_text']=token.get('raw_text',token['text'])
    result['component_context']=context
    result['contextual_corrections']=[dict(from_text=token['text'],to_text=corrected,reason='bjt_gain_label_and_numeric_grammar')]
    result['needs_review']=True
    return result


def equals_number_box(page,box):
    crop=page[box['y1']:box['y2'],box['x1']:box['x2']]
    _,labels,stats,_=cv2.connectedComponentsWithStats((crop<128).astype(np.uint8),8)
    bars=[]
    for index,(x,y,w,h,area) in enumerate(stats[1:],1):
        if w<max(5,crop.shape[0]*.10) or not .08*crop.shape[1]<x<.65*crop.shape[1]:continue
        yy,xx=np.nonzero(labels[y:y+h,x:x+w]==index)
        _,(rw,rh),angle=cv2.minAreaRect(np.column_stack((xx,yy)).astype(np.float32))
        direction=(angle+(90 if rw<rh else 0)+90)%180-90
        if max(rw,rh)<1.6*max(1,min(rw,rh)) or abs(direction)>35:continue
        bars.append((x,y,w,h))
    for x,y,w,h in bars:
        for xx,yy,ww,hh in bars:
            overlap=min(x+w,xx+ww)-max(x,xx)
            if (yy>y+h and yy-y<max(8,crop.shape[0]*.55) and overlap>.6*min(w,ww)
                    and min(w,ww)>.6*max(w,ww)):
                end=max(x+w,xx+ww)
                if np.any(crop[:,:max(0,min(x,xx)-2)]<128) and np.any(crop[:,end+3:]<128):
                    return dict(box,x1=int(box['x1']+end+2))
    return None


def retry_allowlist(kind):
    if kind in BJT or kind in {'mosfet_n','mosfet_p','opamp','diode'}:
        return '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz=:.+-()'
    if kind=='led':return 'LEDled0123456789'
    if kind in {'resistor','capacitor','capacitor_polarized','inductor','battery','source_dc'}:
        return '0123456789.,+-=RrCcLlVvHhFfΩµμuUnNpPmMkKGgoOhHsS'
    return None
