"""Choose handwriting orientation from actual complete exterior words."""
import numpy as np

def choose(image,detections):
    from backend.local_word_ocr import locate_words,read_words
    from backend.domain_word_ocr import read_words as domain_words
    from backend.handwriting_orientation import transform_box
    from backend.annotation_word_reader import _key
    h,w=image.shape[:2];jobs=[];info=[]
    for axis in (0,1):
        view=np.rot90(image,axis).copy()
        boxes=locate_words(view)
        candidates=[]
        for item in boxes:
            b=item['bbox'];original=transform_box(b,(-axis)%4,*view.shape[:2]) if axis else b
            area=max(1,(original['x2']-original['x1'])*(original['y2']-original['y1']))
            overlap=max((max(0,min(original['x2'],d['bbox']['x2'])-max(original['x1'],d['bbox']['x1']))*
                max(0,min(original['y2'],d['bbox']['y2'])-max(original['y1'],d['bbox']['y1']))/area for d in detections),default=0)
            if overlap>.45:continue
            if b['x2']-b['x1']<1.2*(b['y2']-b['y1']):continue
            candidates.append(item)
        for item in sorted(candidates,key=lambda p:-p['confidence'])[:14]:
            b=item['bbox'];crop=view[int(b['y1']):int(b['y2']),int(b['x1']):int(b['x2'])]
            if not crop.size:continue
            for half in (0,2):
                jobs.append(np.rot90(crop,half).copy());info.append((axis+half)%4)
    if not jobs:return None
    general=read_words(jobs);domain=domain_words(jobs);scores=[0.]*4;counts=[0]*4
    for k,a,b in zip(info,general,domain):
        choices=[r for r in (a,b) if r and r.get('confidence',0)>=.94 and r.get('min_character_confidence',0)>=.80 and _key(r.get('text',''))]
        if not choices:continue
        best=max(choices,key=lambda r:r['confidence']);key=_key(best['text'])
        weight=1.3 if key[0]=='reference' else 1.
        scores[k]+=weight*best['confidence'];counts[k]+=1
    order=sorted(range(4),key=lambda k:scores[k],reverse=True)
    if counts[order[0]]<3 or scores[order[0]]<scores[order[1]]+1.:return None
    return order[0]
