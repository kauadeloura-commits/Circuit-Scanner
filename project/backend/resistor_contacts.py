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
    if not body:
        # The learned axis can propose an orientation when a photographed
        # rectangle has a real pen gap. Both opposed lead pixels must still
        # have been observed independently by the contact detector.
        learned=component.get('learned_terminal_geometry') or {}
        points=learned.get('points') or {};box=component['bbox']
        if not learned.get('accepted') or learned.get('keypoint_confidence',0)<.75 or not all(pin in points for pin in ('1','2')):return contacts,None
        a,b=[np.array([points[pin]['x'],points[pin]['y']],float) for pin in ('1','2')]
        axis=b-a;length=float(np.linalg.norm(axis));scale=min(box['x2']-box['x1'],box['y2']-box['y1'])
        if length<scale*.6:return contacts,None
        axis/=length;normal=np.array([-axis[1],axis[0]])
        center=np.array([(box['x1']+box['x2'])/2,(box['y1']+box['y2'])/2]);sides=[[],[]]
        for contact in contacts:
            delta=contact['point']-center;along=float(delta@axis);across=abs(float(delta@normal))
            if abs(along)<scale*.22 or across>max(6,scale*.22):continue
            sides[int(along>0)].append(contact)
        if not all(sides):return contacts,None
        selected=[min(side,key=lambda p:abs(float((p['point']-center)@normal))+.10*p['stage8_distance_px']) for side in sides]
        return selected,dict(method='two_opposed_observed_leads_along_trained_axis',axis=axis.tolist(),selected=2,rejected=len(contacts)-2)
    evidence=body['evidence'];center=np.asarray(body['center']);axis=np.asarray(evidence['body_axis']);normal=np.array([-axis[1],axis[0]])
    length,width=evidence['body_size'];sides=[[],[]]
    for c in contacts:
        delta=c['point']-center;along=float(delta@axis);across=abs(float(delta@normal))
        # A hand-drawn lead can leave the end wall off-center, then bend.
        # Keep the full end-wall width (plus ink rounding), while the axial
        # test still excludes corners and side-wall/ruling intersections.
        from backend.drawing_mode import CURRENT
        allowance=width*.5+2 if CURRENT.get()=='handwritten' else width*.35
        if abs(along)<length/2+1 or across>max(4,allowance):continue
        sides[int(along>0)].append(c)
    selected=[min(s,key=lambda c:abs(float((c['point']-center)@normal))+.25*c['stage8_distance_px']) for s in sides if s]
    return selected,dict(method='two_opposed_leads_outside_resistor_body',body=evidence,selected=len(selected),rejected=len(contacts)-len(selected))

def extend_boxes(image,detections):
    """Include observed lead pixels when the model box encloses only the body."""
    output=[]
    for d in detections:
        if d.get('class')!='resistor' or d.get('manual_reviewed'):output.append(d);continue
        if d.get('symbol_box_refinement')=='verified_resistor_body_and_short_leads_without_reprojected_halo':
            # OCR may trim a supposed word through an actual resistor edge.
            # Keep the stable frame, but restore verified body/lead pixels if
            # text separation removed them after the first geometric pass.
            b=d['bbox'];corners=d.get('resistor_body',{}).get('body_corners',[])
            lead=d.get('verified_resistor_lead_bbox')
            if lead and any(not b['x1']<=x<=b['x2'] or not b['y1']<=y<=b['y2'] for x,y in corners):
                restored={k:(min(b[k],lead[k]) if k.endswith('1') else max(b[k],lead[k])) for k in ('x1','y1','x2','y2')}
                restored.update(width=restored['x2']-restored['x1'],height=restored['y2']-restored['y1'],
                                center_x=(restored['x1']+restored['x2'])/2,center_y=(restored['y1']+restored['y2'])/2)
                output.append(dict(d,bbox=restored))
            else:output.append(d)
            continue
        body=body_for_box(image,d['bbox'])
        if not body:output.append(d);continue
        b=d['bbox'];observed=body['bbox'];merged=dict(b)
        evidence=body['evidence'];axis=np.asarray(evidence['body_axis']);normal=np.array([-axis[1],axis[0]])
        center=np.asarray(body['center']);length,width=evidence['body_size']
        corners=np.array([[x,y] for x in (b['x1'],b['x2']) for y in (b['y1'],b['y2'])])
        along=(corners-center)@axis;across=(corners-center)@normal
        from backend.drawing_mode import CURRENT
        observed_area=(observed['x2']-observed['x1'])*(observed['y2']-observed['y1'])
        if (CURRENT.get()=='handwritten' and evidence.get('opposed_leads') and length>=60 and width>0
                and np.ptp(across)>2.6*width
                and (b['x2']-b['x1'])*(b['y2']-b['y1'])>1.8*observed_area):
            # Reprojecting an axis-aligned frame onto a tilted body repeatedly
            # inflated the halo at each stage. Use the independently verified
            # body and its short observed leads as the stable local frame.
            extent=(max(float(along.min()),-length*.90),min(float(along.max()),length*.90))
            points=np.array([center+axis*a+normal*n for a in extent for n in (-width,width)])
            merged=dict(x1=max(b['x1'],float(points[:,0].min())),y1=max(b['y1'],float(points[:,1].min())),
                        x2=min(b['x2'],float(points[:,0].max())),y2=min(b['y2'],float(points[:,1].max())))
            merged.update(width=merged['x2']-merged['x1'],height=merged['y2']-merged['y1'],
                          center_x=(merged['x1']+merged['x2'])/2,center_y=(merged['y1']+merged['y2'])/2)
            output.append(dict(d,bbox=merged,resistor_body=evidence,
                               verified_resistor_lead_bbox=dict(observed),
                               symbol_box_refinement='verified_resistor_body_and_short_leads_without_reprojected_halo'))
            continue
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
        # A rotated proposal may surround several body widths of empty paper.
        # Retain its observed lead extent, limiting only excessive lateral halo.
        corners=np.array([[x,y] for x in (merged['x1'],merged['x2']) for y in (merged['y1'],merged['y2'])])
        across=(corners-center)@normal;along=(corners-center)@axis
        if width>0 and np.ptp(across)>3*width:
            points=np.array([center+axis*a+normal*n for a in (float(along.min()),float(along.max())) for n in (-1.5*width,1.5*width)])
            merged.update(x1=max(0.,float(points[:,0].min())),y1=max(0.,float(points[:,1].min())),x2=min(float(image.shape[1]-1),float(points[:,0].max())),y2=min(float(image.shape[0]-1),float(points[:,1].max())))
        merged.update(width=merged['x2']-merged['x1'],height=merged['y2']-merged['y1'],center_x=(merged['x1']+merged['x2'])/2,center_y=(merged['y1']+merged['y2'])/2)
        output.append(dict(d,bbox=merged,resistor_body=evidence if (evidence:=body['evidence']) else {}))
    return output
