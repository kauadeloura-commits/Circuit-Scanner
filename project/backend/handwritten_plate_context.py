"""Two capacitor plates need real, opposed conductors beyond the symbol box."""
from itertools import combinations
import cv2,numpy as np
from backend.resistor_geometry import cached_vision


@cached_vision
def connected_plate_evidence(image,bbox):
    from backend.drawing_mode import CURRENT
    if CURRENT.get()!='handwritten' or min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])<12:return None
    from backend.terminal_identity import _crop
    from backend.terminal_wire_contacts import _contacts
    mask,origin=_crop(image,bbox);contacts=_contacts(image,bbox,None)
    if len(contacts)!=2 or contacts[0].get('face_axis')!=contacts[1].get('face_axis'):return None
    p,q=(np.asarray(c['point'],float) for c in contacts);axis=q-p;distance=float(np.linalg.norm(axis))
    bw,bh=bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'];major=0 if abs(axis[0])>abs(axis[1]) else 1
    if distance<max(12,.75*(bw if major==0 else bh)):return None
    axis/=distance;normal=np.array([-axis[1],axis[0]])
    raw=cv2.HoughLinesP(mask,1,np.pi/180,8,minLineLength=12,maxLineGap=5)
    if raw is None:return None
    strokes=[]
    for line in raw.reshape(-1,4):
        ends=line.reshape(2,2)+origin;v=ends[1]-ends[0];length=float(np.linalg.norm(v))
        if abs(float(v@axis))>length*.25:continue
        center=ends.mean(axis=0);t=float((center-p)@axis)
        if not 0<t<distance or abs(float((center-p)@normal))>length*.25:continue
        if length>=max(14,min(bw,bh)*.30):strokes.append((t,length,center))
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    choices=[]
    for first,second in combinations(strokes,2):
        a,b=sorted((first,second),key=lambda s:s[0]);gap=b[0]-a[0];length=max(a[1],b[1])
        if not max(6,length*.10)<gap<length*.95 or min(a[1],b[1])<length*.60:continue
        if abs(float((a[2]-b[2])@normal))>length*.20:continue
        support=[]
        for at,u in ((p,-axis),(q,axis)):
            hits=[]
            for x,y in np.rint(at+np.arange(3,max(18,length*.55),2)[:,None]*u).astype(int):
                if not 2<=x<gray.shape[1]-2 or not 2<=y<gray.shape[0]-2:hits.append(False);continue
                paper=np.median(gray[max(0,y-8):y+9,max(0,x-8):x+9])
                hits.append(float(gray[y-2:y+3,x-2:x+3].min())<paper-12)
            support.append(float(np.mean(hits)))
        if min(support)<.80:continue
        proof=dict(capacitor_body_with_opposed_leads=True,outer_plate_centers=[a[2].tolist(),b[2].tolist()],
                   outer_plate_lengths=[a[1],b[1]],plate_gap_px=gap,electrical_axis=axis.tolist(),
                   outside_symbol_lead_support=support,internal_ink_bridge_not_used_as_wire=True,
                   metric='two_transverse_plates_with_two_observed_outward_external_leads')
        choices.append((min(a[1],b[1])/length+min(support),proof))
    return max(choices,key=lambda c:c[0])[1] if choices else None
