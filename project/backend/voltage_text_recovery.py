"""Recover a proposed power bar only with explicit voltage text and real ink."""
import re,math
import cv2,numpy as np
from backend.measurement_text import parse_measurement
from backend.resistor_geometry import symbol_mask
from backend.vcc_topology import external_leads

def _distance(a,b):
    return math.hypot(max(a['x1']-b['x2'],b['x1']-a['x2'],0),max(a['y1']-b['y2'],b['y1']-a['y2'],0))

def _bar_and_stem(mask,bbox):
    h,w=mask.shape;x1,y1=max(0,int(bbox['x1'])-2),max(0,int(bbox['y1'])-2)
    x2,y2=min(w,int(np.ceil(bbox['x2']))+3),min(h,int(np.ceil(bbox['y2']))+3)
    crop=mask[y1:y2,x1:x2]
    if not crop.size:return False
    lines=cv2.HoughLinesP(crop,1,np.pi/180,threshold=8,minLineLength=max(8,round(min(crop.shape)*.35)),maxLineGap=3)
    if lines is None:return False
    segments=[np.asarray(row,float).reshape(2,2) for row in np.asarray(lines).reshape(-1,4)]
    for bar in segments:
        u=bar[1]-bar[0];length=np.linalg.norm(u)
        if length<12:continue
        center=bar.mean(axis=0)
        for stem in segments:
            v=stem[1]-stem[0];size=np.linalg.norm(v)
            if size<8 or abs(float(u@v))/max(1,length*size)>.25:continue
            if min(np.linalg.norm(p-center) for p in stem)>max(5,length*.15):continue
            if max(np.linalg.norm(p-center) for p in stem)<max(8,length*.16):continue
            return dict(bar=[(p+[x1,y1]).tolist() for p in bar],stem=[(p+[x1,y1]).tolist() for p in stem])
    return False

def recover_voltage_ports(image,detections,proposals,tokens):
    labels=[]
    for token in tokens:
        if float(token.get('confidence',0))<.65 or not token.get('bbox'):continue
        text=str(token.get('text','')).strip();measurement=parse_measurement(text)
        reference=bool(re.fullmatch(r'[+-]?V(?:CC|EE|DD|SS)\d*',text,re.I))
        if reference or measurement and measurement.get('unit')=='volt' and measurement.get('unit_explicit'):labels.append(token)
    if not labels:return []
    mask=symbol_mask(image);found=[]
    from backend.drawing_mode import digital
    if digital():
        mask=mask.copy()
        for token in labels:
            b=token['bbox'];x,y=max(0,int(b['x1'])),max(0,int(b['y1']));z,t=min(mask.shape[1],int(np.ceil(b['x2']))+1),min(mask.shape[0],int(np.ceil(b['y2']))+1)
            mask[y:t,x:z]=0
    for proposal in proposals:
        from backend.drawing_mode import digital
        if proposal.get('class') not in ({'vcc','gnd'} if digital() else {'vcc'}):continue
        if not digital() and float(proposal.get('best_view_confidence',proposal.get('confidence',0)))<.60:continue
        b=proposal['bbox'];cx,cy=(b['x1']+b['x2'])/2,(b['y1']+b['y2'])/2
        if any(d['bbox']['x1']<=cx<=d['bbox']['x2'] and d['bbox']['y1']<=cy<=d['bbox']['y2'] for d in detections+found):continue
        from backend.printed_power import box_exit_count
        body=_bar_and_stem(mask,b)
        exits=box_exit_count(mask,b) if digital() else external_leads(image,b,mask)
        if exits!=1 or not body:continue
        nearby=[]
        for token in labels:
            t=token['bbox'];limit=max(65,min(160,max(t['x2']-t['x1'],t['y2']-t['y1'])*1.6));distance=_distance(t,b)
            if distance>limit:continue
            if any(d.get('class') in {'vcc','source_dc','battery'} and _distance(t,d['bbox'])+5<distance for d in detections+found):continue
            nearby.append((distance,token))
        if not nearby:continue
        distance,token=min(nearby,key=lambda p:p[0])
        if digital() and isinstance(body,dict):
            bar=np.asarray(body['bar']);stem=np.asarray(body['stem']);center=bar.mean(axis=0);stem_u=stem[1]-stem[0];bar_u=bar[1]-bar[0];normal=np.array([-bar_u[1],bar_u[0]])
            denominator=float(stem_u@normal)
            if abs(denominator)>1e-6:center=stem[0]+stem_u*float((center-stem[0])@normal)/denominator
            far=max(stem,key=lambda p:np.linalg.norm(p-center));unit=(far-center)/max(1,np.linalg.norm(far-center));length=np.linalg.norm(bar_u)
            end=center.copy()
            for step in np.arange(1,min(65,length*1.1),.5):
                p=np.rint(center+unit*step).astype(int)
                if not(1<=p[0]<mask.shape[1]-1 and 1<=p[1]<mask.shape[0]-1) or not mask[p[1]-1:p[1]+2,p[0]-1:p[0]+2].any():break
                end=center+unit*step
            points=np.vstack((bar,end));pad=max(4,length*.18)
            b=dict(x1=max(0,float(points[:,0].min()-pad)),y1=max(0,float(points[:,1].min()-pad)),x2=min(mask.shape[1]-1,float(points[:,0].max()+pad)),y2=min(mask.shape[0]-1,float(points[:,1].max()+pad)))
            b.update(width=b['x2']-b['x1'],height=b['y2']-b['y1'],center_x=(b['x1']+b['x2'])/2,center_y=(b['y1']+b['y2'])/2)
        found.append(dict(detection_id=f'PWR_TXT_{len(found)+1:03d}',class_id=9,**{'class':'vcc'},bbox=dict(b),confidence=min(.94,float(proposal.get('best_view_confidence',proposal.get('confidence',.7)))),
            inference_sources=['explicit_voltage_text_and_power_bar'],review_required=False,review_reasons=[],class_resolution='voltage_label_and_one_lead_power_bar',
            geometry_validation=dict(valid=True,score=.91,evidence=dict(power_bar_with_one_external_stem=True,external_lead_count=1,voltage_text=token['text'],text_distance_px=round(distance,2)))))
    return found
