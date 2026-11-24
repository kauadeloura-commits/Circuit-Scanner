"""Match resistor leads beyond the body; keep contact pixels inside detection."""
import numpy as np
from backend.resistor_geometry import iec_bodies,cached_vision

@cached_vision
def body_for_box(image,box):
    center=np.array([(box['x1']+box['x2'])/2,(box['y1']+box['y2'])/2])
    size=max(box['x2']-box['x1'],box['y2']-box['y1'])
    # Search the local symbol, not the entire photo once for each resistor.
    # Include its leads and paper border so body evidence stays complete.
    x1=max(0,int(box['x1']-size*.55));y1=max(0,int(box['y1']-size*.55))
    x2=min(image.shape[1],int(box['x2']+size*.55)+1);y2=min(image.shape[0],int(box['y2']+size*.55)+1)
    candidates=iec_bodies(image[y1:y2,x1:x2])
    for b in candidates:
        b['center']=[b['center'][0]+x1,b['center'][1]+y1]
        b['evidence']['body_center']=list(b['center'])
        b['evidence']['body_corners']=[[x+x1,y+y1] for x,y in b['evidence']['body_corners']]
        for key in ('x1','x2'):b['bbox'][key]+=x1
        for key in ('y1','y2'):b['bbox'][key]+=y1
    candidates=[b for b in candidates if np.linalg.norm(np.array(b['center'])-center)<size*.35]
    return min(candidates,key=lambda b:np.linalg.norm(np.array(b['center'])-center)) if candidates else None

def lead_contacts(image,component,contacts):
    if component.get('type')!='resistor':return contacts,None
    body=body_for_box(image,component['bbox'])
    if not body:return contacts,None
    evidence=body['evidence'];center=np.asarray(body['center']);axis=np.asarray(evidence['body_axis']);normal=np.array([-axis[1],axis[0]])
    length,width=evidence['body_size'];sides=[[],[]]
    for c in contacts:
        delta=c['point']-center;along=float(delta@axis);across=abs(float(delta@normal))
        if abs(along)<length/2+1 or across>max(4,width*.35):continue
        sides[int(along>0)].append(c)
    selected=[min(s,key=lambda c:abs(float((c['point']-center)@normal))+.25*c['stage8_distance_px']) for s in sides if s]
    return selected,dict(method='two_opposed_leads_outside_resistor_body',body=evidence,selected=len(selected),rejected=len(contacts)-len(selected))

def extend_boxes(image,detections):
    """Include observed lead pixels when the model box encloses only the body."""
    output=[]
    for d in detections:
        if d.get('class')!='resistor' or d.get('manual_reviewed'):output.append(d);continue
        body=body_for_box(image,d['bbox'])
        if not body:output.append(d);continue
        b=d['bbox'];observed=body['bbox'];merged=dict(b)
        evidence=body['evidence'];axis=np.asarray(evidence['body_axis']);normal=np.array([-axis[1],axis[0]])
        center=np.asarray(body['center']);length,width=evidence['body_size']
        corners=np.array([[x,y] for x in (b['x1'],b['x2']) for y in (b['y1'],b['y2'])])
        along=(corners-center)@axis;across=(corners-center)@normal
        if np.ptp(along)>1.8*length:
            # A detector may follow a long attached wire to the image edge.
            # Bound the symbol by its verified body and short observed leads;
            # this also keeps wire erasure from removing an entire real net.
            lead=min(length*.25,min(evidence.get('lead_support_px',[length*.25]*2)))
            limit=length/2+max(6,lead)
            points=np.array([center+axis*a+normal*n for a in (-limit,limit) for n in (float(across.min()),float(across.max()))])
            merged.update(x1=max(0,float(points[:,0].min())),y1=max(0,float(points[:,1].min())),x2=min(image.shape[1],float(points[:,0].max())),y2=min(image.shape[0],float(points[:,1].max())))
        for key in ('x1','y1'):merged[key]=min(merged[key],observed[key])
        for key in ('x2','y2'):merged[key]=max(merged[key],observed[key])
        # Never expand across a different symbol.
        if any(o is not d and merged['x1']<o['bbox']['x1']<o['bbox']['x2']<merged['x2'] and merged['y1']<o['bbox']['y1']<o['bbox']['y2']<merged['y2'] for o in detections):output.append(d);continue
        merged.update(width=merged['x2']-merged['x1'],height=merged['y2']-merged['y1'],center_x=(merged['x1']+merged['x2'])/2,center_y=(merged['y1']+merged['y2'])/2)
        output.append(dict(d,bbox=merged,resistor_body=evidence if (evidence:=body['evidence']) else {}))
    return output
