"""Verify faint or rule-interrupted power rings in the original photograph."""
import cv2,numpy as np
from backend.resistor_geometry import symbol_mask,cached_vision
from backend.vcc_topology import external_leads

@cached_vision
def faint_ring_with_one_lead(image,bbox):
    h,w=image.shape[:2];bw,bh=bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'];pad=max(6,round(min(bw,bh)*.25))
    x1,y1=max(0,int(bbox['x1'])-pad),max(0,int(bbox['y1'])-pad)
    x2,y2=min(w,int(np.ceil(bbox['x2']))+pad),min(h,int(np.ceil(bbox['y2']))+pad)
    crop=image[y1:y2,x1:x2]
    if not crop.size or min(crop.shape[:2])<14:return None
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
    paper=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
    normalized=cv2.divide(gray,np.maximum(paper,1),scale=255)
    normalized=cv2.GaussianBlur(normalized,(3,3),.7)
    max_radius=max(6,min(70,round(min(bw,bh)*.65)))
    circles=cv2.HoughCircles(normalized,cv2.HOUGH_GRADIENT,1,8,param1=60,param2=10,minRadius=5,maxRadius=max_radius)
    if circles is None:return None
    raw=symbol_mask(crop,remove_rules=False);clean=symbol_mask(image)
    # A fitted loop in R/P is not a power head. Its own ink must actually
    # leave the complete detector proposal, rather than borrowing a nearby
    # digit's stroke as the supposed wire.
    # Use original strokes here: rule removal can cut the faint head/stem
    # intersection that this fallback exists to recover. External conductor
    # counting below still uses the rule-cleaned full page.
    _,connected=cv2.connectedComponents((raw>0).astype(np.uint8),8)
    gy,gx=np.indices(raw.shape);gx=gx+x1;gy=gy+y1
    outside_distance=np.maximum.reduce([bbox['x1']-gx,gx-bbox['x2'],bbox['y1']-gy,gy-bbox['y2']])
    exiting=set(np.unique(connected[outside_distance>=min(8,max(3,pad-2))]));exiting.discard(0)
    if not exiting:return None
    contours,hierarchy=cv2.findContours(clean[y1:y2,x1:x2],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        holes=[cv2.contourArea(c) for i,c in enumerate(contours) if hierarchy[0,i,3]>=0]
        # A BJT's outer ring is partitioned by the base and two branches.
        # A small circle fitted inside one partition is not a supply head.
        if holes and sum(area>=max(8,.15*max(holes)) for area in holes)>1:return None
    points=np.linspace(0,2*np.pi,128,endpoint=False);yy,xx=np.indices(raw.shape);candidates=[]
    for circle in circles[0]:
        cx,cy,radius=map(float,circle)
        if radius<min(bw,bh)*.18:continue
        gx,gy=float(cx+x1),float(cy+y1)
        if not bbox['x1']<=gx<=bbox['x2'] or not bbox['y1']<=gy<=bbox['y2']:continue
        support=np.zeros(128,bool)
        for offset in np.linspace(-max(2,radius*.15),max(2,radius*.15),5):
            xs=np.rint(cx+(radius+offset)*np.cos(points)).astype(int);ys=np.rint(cy+(radius+offset)*np.sin(points)).astype(int)
            valid=(xs>=0)&(xs<raw.shape[1])&(ys>=0)&(ys<raw.shape[0])
            support[valid]|=raw[ys[valid],xs[valid]]>0
        disk=(xx-cx)**2+(yy-cy)**2<=(radius*.5)**2
        if support.mean()<.78 or not disk.any() or np.mean(raw[disk]>0)>.18:continue
        annulus=np.abs(np.sqrt((xx-cx)**2+(yy-cy)**2)-radius)<=max(3,radius*.2)
        head_labels=set(np.unique(connected[annulus]));head_labels.discard(0)
        if not head_labels&exiting:continue
        body=dict(x1=max(0,gx-radius-3),y1=max(0,gy-radius-3),x2=min(w-1,gx+radius+3),y2=min(h-1,gy+radius+3))
        if external_leads(image,body,clean)!=1:continue
        candidates.append((float(support.mean()),body))
    if not candidates:return None
    score,body=max(candidates,key=lambda p:p[0])
    return dict(validation_mode='photographic_ring_arc_and_one_external_lead',ring_arc_coverage=round(score,3),power_head_bbox=body,external_lead_count=1,junction_rejected=False)
