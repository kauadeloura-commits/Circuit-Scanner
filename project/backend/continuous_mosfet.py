"""Verify an isolated gate, continuous channel and a visible source arrow.

The body-arrow and source-lead-arrow conventions have opposite N/P meanings.
Never infer either the class or D/S roles from the power supply location.
"""
import cv2,numpy as np,math
from backend.resistor_geometry import cached_vision

@cached_vision
def read(image,box):
    from backend.terminal_identity import _crop,_marks,_mosfet,_lines
    mask,origin=_crop(image,box);_,labels=_marks(mask,origin,box)
    candidate=_mosfet(mask,origin,box,labels)
    if candidate is None:return None
    points=candidate[0];g=points['G'];others=[points['D'],points['S']]
    p=np.rint(g-origin).astype(int);p=np.clip(p,[0,0],np.array(mask.shape[::-1])-1)
    patch=labels[max(0,p[1]-2):p[1]+3,max(0,p[0]-2):p[0]+3];values=patch[patch>0]
    if not len(values):return None
    gid=int(np.bincount(values).argmax());lines=_lines(mask,origin);minimum=min(box['x2']-box['x1'],box['y2']-box['y1'])
    def label_at(p):
        x,y=p;patch=labels[max(0,y-3):y+4,max(0,x-3):x+4];values=patch[patch>0]
        return int(np.bincount(values).argmax()) if len(values) else 0
    # The gate's T junction splits a single photographed bar into two LSD
    # segments. Reunite only collinear segments on that same isolated ink path;
    # an external wire or nearby model lettering cannot extend this bar.
    joined=[]
    for line in lines:
        if label_at(np.rint(line['center']-origin).astype(int))!=gid:continue
        v=np.array([math.cos(line['angle']),math.sin(line['angle'])]);u=np.array([-v[1],v[0]])
        spans=[]
        for other in lines:
            if label_at(np.rint(other['center']-origin).astype(int))!=gid:continue
            w=np.array([math.cos(other['angle']),math.sin(other['angle'])]);delta=other['center']-line['center']
            if abs(v@w)<.99 or abs(delta@u)>max(5,minimum*.04):continue
            projection=float(delta@v);half=other['length']*.5
            if abs(projection)>(line['length']+other['length'])*.5+minimum*.20:continue
            spans.extend([projection-half,projection+half])
        if spans:
            low,high=min(spans),max(spans)
            joined.append(dict(line,center=line['center']+v*(low+high)*.5,length=high-low))
    gates=[]
    for line in lines+joined:
        p=np.rint(line['center']-origin).astype(int)
        if not(0<=p[0]<mask.shape[1] and 0<=p[1]<mask.shape[0]) or label_at(p)!=gid:continue
        if line['length']<minimum*.45:continue
        v=np.array([math.cos(line['angle']),math.sin(line['angle'])]);u=np.array([-v[1],v[0]])
        if np.mean([(q-line['center'])@u for q in others])<0:u=-u
        offsets=np.array([(q-line['center'])@u for q in others])
        if offsets.min()<minimum*.08:continue
        channels=[]
        for channel in lines:
            w=np.array([math.cos(channel['angle']),math.sin(channel['angle'])]);delta=channel['center']-line['center']
            if abs(v@w)<.97 or channel['length']<line['length']*.50:continue
            if not max(3,minimum*.05)<delta@u<minimum*.60 or abs(delta@v)>line['length']*.35:continue
            q=np.rint(channel['center']-origin).astype(int)
            if 0<=q[0]<mask.shape[1] and 0<=q[1]<mask.shape[0] and label_at(q) not in (0,gid):channels.append(channel)
        if channels:gates.append((line['length'],line,max(channels,key=lambda c:c['length']),v,u))
    if not gates:return None
    _,gate,channel,v,u=max(gates,key=lambda x:x[0]);depth=float((channel['center']-gate['center'])@u)
    distance=cv2.distanceTransform(mask,cv2.DIST_L2,5);values=distance[distance>0]
    if not len(values):return None
    radius=max(1,int(np.percentile(values,75)*1.25))
    contours=[]
    for r in range(max(1,radius-1),min(10,radius*2+2)):
        eroded=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1)))
        contours.extend(cv2.findContours(eroded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0])
    arrows=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if not max(2,minimum**2*.0007)<area<minimum**2*.10:continue
        hull=cv2.convexHull(contour);tri_area,tri=cv2.minEnclosingTriangle(hull)
        if tri is None or area/max(1,tri_area)<.60:continue
        tri=tri.reshape(3,2)+origin;c=tri.mean(axis=0)
        a=float((c-gate['center'])@u);along=float((c-channel['center'])@v)
        # Source arrows lie outside the channel, attached to one current
        # branch. Central body arrows are left to the separate body reader.
        if not depth-1<a<depth+minimum*.65 or abs(along)>gate['length']*.65:continue
        q=np.sort(tri@u);span=q[2]-q[0]
        if span<max(3,minimum*.04) or min(q[1]-q[0],q[2]-q[1])>span*.28:continue
        outward=q[2]-q[1]>q[1]-q[0]
        source=int(np.argmin([abs(float((q-c)@v)) for q in others]))
        if abs(float((others[0]-c)@v))-abs(float((others[1]-c)@v))==0:continue
        arrows.append((area,'mosfet_n' if outward else 'mosfet_p',source,c,tri))
    if not arrows or len({a[1] for a in arrows})!=1:return None
    _,kind,source,center,triangle=max(arrows,key=lambda a:a[0])
    return dict(kind=kind,points=dict(G=g.tolist(),S=others[source].tolist(),D=others[1-source].tolist()),
                method='isolated_gate_parallel_continuous_channel_and_source_lead_arrow',
                arrow_center=center.tolist(),triangle=triangle.tolist(),channel_depth=depth)
