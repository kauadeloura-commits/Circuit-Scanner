"""Recover complete annotations joined to a component outline or lead."""
import math
import cv2
import numpy as np
from backend.measurement_text import parse_measurement,DIMENSIONS,SYMBOLS

def recover_values(image,detections,tokens,refs):
    from backend import handwriting_reference as hw
    from backend.ocr_regions import text_line_regions
    candidates=[d for d in detections if d.get('class') in {'resistor','capacitor','capacitor_polarized','inductor'}]
    present=set()
    def distance(a,b):
        x,y=(a['x1']+a['x2'])/2,(a['y1']+a['y2'])/2
        return math.hypot(max(b['x1']-x,x-b['x2'],0),max(b['y1']-y,y-b['y2'],0))+.05*math.hypot(x-(b['x1']+b['x2'])/2,y-(b['y1']+b['y2'])/2)
    for t in tokens:
        compatible=[d for d in candidates if (v:=parse_measurement(t.get('text',''),d['class'])) and v['unit']==DIMENSIONS[d['class']]]
        if compatible:
            d=min(compatible,key=lambda d:distance(t['bbox'],d['bbox']))
            if distance(t['bbox'],d['bbox'])<=85:present.add(d.get('detection_id'))
    recovered=[]
    for d in [d for d in candidates if d.get('detection_id') not in present][:12]:
        b=d['bbox'];size=max(b['x2']-b['x1'],b['y2']-b['y1']);margin=max(26,min(58,size*.7))
        ax,ay=max(0,int(b['x1']-margin)),max(0,int(b['y1']-margin));bx,by=min(image.shape[1],int(b['x2']+margin)),min(image.shape[0],int(b['y2']+margin))
        patch=image[ay:by,ax:bx];gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
        mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        for other in detections:
            ob=other['bbox']
            if other.get('class')=='resistor':
                from backend.resistor_contacts import body_for_box
                body=body_for_box(image,other['bbox'])
                if body:
                    polygon=np.rint(np.asarray(body['evidence']['body_corners'])-[ax,ay]).astype(np.int32)
                    cv2.fillPoly(mask,[polygon],0)
                    continue
                # A detector box also contains leads and may overlap a value.
                # When the body contour is joined to writing, erase only its
                # central axis band, retaining adjacent letter tops.
                ob=dict(ob);w,h=ob['x2']-ob['x1'],ob['y2']-ob['y1']
                if w>=h:
                    center=(ob['y1']+ob['y2'])/2;ob.update(y1=center-h*.32,y2=center+h*.32)
                else:
                    center=(ob['x1']+ob['x2'])/2;ob.update(x1=center-w*.32,x2=center+w*.32)
            x1,y1=max(0,int(ob['x1'])-ax),max(0,int(ob['y1'])-ay);x2,y2=min(mask.shape[1],int(ob['x2'])-ax),min(mask.shape[0],int(ob['y2'])-ay)
            if x1<x2 and y1<y2:mask[y1:y2,x1:x2]=0
        found=[]
        for region in text_line_regions(255-mask):
            box={k:region[k]+(ax if k in {'x1','x2'} else ay) for k in ('x1','y1','x2','y2')}
            if any(max(0,min(box['x2'],t['bbox']['x2'])-max(box['x1'],t['bbox']['x1']))*max(0,min(box['y2'],t['bbox']['y2'])-max(box['y1'],t['bbox']['y1']))>.3*min((box['x2']-box['x1'])*(box['y2']-box['y1']),(t['bbox']['x2']-t['bbox']['x1'])*(t['bbox']['y2']-t['bbox']['y1'])) for t in tokens+recovered):continue
            closest=min(candidates,key=lambda c:distance(box,c['bbox']))
            if closest.get('detection_id')!=d.get('detection_id'):continue
            reading=hw.read_line(mask[region['y1']:region['y2'],region['x1']:region['x2']],refs,d['class'])
            value=parse_measurement(reading.get('text',''),d['class'])
            if not value or reading.get('unresolved'):
                rh=region['y2']-region['y1']
                if region['x2']-region['x1']<rh*2:
                    region=dict(region,x1=max(0,region['x1']-round(rh*.9)),x2=min(patch.shape[1],region['x2']+round(rh*.9)))
                    box={k:region[k]+(ax if k in {'x1','x2'} else ay) for k in ('x1','y1','x2','y2')}
                original=patch[region['y1']:region['y2'],region['x1']:region['x2']]
                retry=hw.read_line(hw.ink(original),refs,d['class'])
                candidate=parse_measurement(retry.get('text',''),d['class'])
                if candidate and not retry.get('unresolved'):reading,value=retry,candidate
                # A value can touch the long horizontal/vertical edge of the
                # symbol. Remove only that long edge in the annotation crop;
                # shorter unit strokes and numeric glyphs remain intact.
                gray=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
                isolated=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
                length=max(20,round(isolated.shape[0]*.7))
                line=cv2.morphologyEx(isolated,cv2.MORPH_OPEN,np.ones((1,length),np.uint8))
                line[round(isolated.shape[0]*.32):]=0
                yy,xx=np.indices(line.shape)
                owned=(xx+box['x1']>=b['x1'])&(xx+box['x1']<=b['x2'])&(yy+box['y1']>=b['y1'])&(yy+box['y1']<=b['y2'])
                line[~owned]=0
                isolated[line>0]=0
                complete=hw.read_line(isolated,refs,d['class'])
                from backend.handwriting_values import read_value
                numeric=read_value(isolated,refs,d['class'])
                observed=parse_measurement(complete.get('text',''),d['class'])
                replacement=parse_measurement(numeric.get('text',''),d['class']) if numeric else None
                if numeric and (not observed or not observed.get('unit_explicit') or replacement and replacement.get('unit_explicit')):complete=numeric
                measured=parse_measurement(complete.get('text',''),d['class'])
                if measured and not complete.get('unresolved') and complete.get('confidence',0)>=65 and (np.any(line) or complete.get('confidence',0)>=reading.get('confidence',0)):reading,value=complete,measured
                if value and value.get('unit_explicit') and value.get('numeric_reading')=='7':
                    clean,height=hw._clean_line(isolated);_,trained=hw.library(refs[0][0].get('_bank_directory',hw.BANK),trained=True)
                    lattice=hw._Segments(clean,trained,height)
                    for cut in lattice.cuts:
                        if not lattice.start<cut<lattice.start+height*.9:continue
                        digit=lattice.crop(lattice.start,cut);points=cv2.findNonZero(digit)
                        if points is None or cv2.boundingRect(points)[3]<height*.6:continue
                        rank=lattice.rank(lattice.start,cut)
                        if rank and rank[0]['text'] in {'1','7'} and rank[0]['score']>=.60 and hw._numeric_stem_reference(digit,trained)=='1':
                            text='1'+value['prefix']+SYMBOLS[value['unit']]
                            reading=dict(reading,text=text,reading_method='unit_then_supplied_lower_stem_reference',needs_review=True,alternatives=[text,reading['text']])
                            value=parse_measurement(text,d['class']);break
            if not value or value['unit']!=DIMENSIONS[d['class']] or reading.get('unresolved') or reading.get('confidence',0)<65:continue
            if reading.get('confidence',0)<70:reading=dict(reading,needs_review=True)
            if not value.get('unit_explicit') and not value.get('prefix'):continue
            found.append(dict(reading,bbox=box,source='handwriting_reference',mask_variant='component_local_body_erased',
                              component_context=dict(detection_id=d.get('detection_id'),kind=d['class'],distance=round(distance(box,b)/80,3),ambiguous=False)))
        if found:
            best=min(found,key=lambda t:distance(t['bbox'],b));recovered.append(best)
    return recovered
