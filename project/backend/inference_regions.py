"""Enlarge a whole drawing inside a photograph without replacing full-page views."""
import numpy as np

def circuit_context_roi(shape,detections,text_regions=()):
    h,w=shape[:2]
    if max(h,w)<700:return None
    anchors=[d['bbox'] for d in detections if float(d.get('confidence',0))>=.4 and not d.get('class','').endswith('_aux')]
    if len(anchors)<2:return None
    x1=min(b['x1'] for b in anchors);y1=min(b['y1'] for b in anchors)
    x2=max(b['x2'] for b in anchors);y2=max(b['y2'] for b in anchors)
    sizes=[max(b['x2']-b['x1'],b['y2']-b['y1']) for b in anchors]
    margin=max(60,min(180,round(float(np.median(sizes))*1.5)))
    for item in text_regions:
        b=item.get('bbox',item)
        dx=max(x1-b['x2'],b['x1']-x2,0);dy=max(y1-b['y2'],b['y1']-y2,0)
        if np.hypot(dx,dy)>margin:continue
        x1=min(x1,b['x1']);y1=min(y1,b['y1']);x2=max(x2,b['x2']);y2=max(y2,b['y2'])
    x1,y1=max(0,int(x1)-margin),max(0,int(y1)-margin)
    x2,y2=min(w,int(np.ceil(x2))+margin),min(h,int(np.ceil(y2))+margin)
    fraction=(x2-x1)*(y2-y1)/max(1,w*h)
    if not .08<=fraction<=.72 or min(x2-x1,y2-y1)<96:return None
    return dict(x1=x1,y1=y1,x2=x2,y2=y2,source_area_fraction=round(fraction,4),policy='additional_circuit_context_view_full_page_preserved')
