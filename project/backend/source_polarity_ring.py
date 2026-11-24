"""A source rim may enclose signs or touch their pen strokes.

Group cavities of the same connected rim, verify its photographed elliptic
outline and two opposed external wires. Neither a text zero nor a circular
transistor enclosure can be established by roundness alone.
"""
import cv2
import numpy as np
from backend.resistor_geometry import symbol_mask
from backend.vcc_topology import external_leads


def polarity_ring_evidence(image,bbox,page_mask=None):
    mask=symbol_mask(image) if page_mask is None else page_mask
    h,w=mask.shape;bw,bh=bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1']
    if min(bw,bh)<=0:return None
    pad=max(8,.25*max(bw,bh));x1,y1=max(0,int(bbox['x1']-pad)),max(0,int(bbox['y1']-pad))
    x2,y2=min(w,int(bbox['x2']+pad+1)),min(h,int(bbox['y2']+pad+1))
    local=mask[y1:y2,x1:x2]
    if not local.size:return None
    local=cv2.morphologyEx(local,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,hierarchy=cv2.findContours(local,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:return None
    parents={}
    for i,c in enumerate(contours):
        parent=int(hierarchy[0,i,3])
        if parent<0 or cv2.contourArea(c)<3:continue
        x,y,cw,ch=cv2.boundingRect(c);cx,cy=x+x1+cw/2,y+y1+ch/2
        if not bbox['x1']-.1*bw<=cx<=bbox['x2']+.1*bw or not bbox['y1']-.1*bh<=cy<=bbox['y2']+.1*bh:continue
        parents.setdefault(parent,[]).append(c)
    for cavities in parents.values():
        hull=cv2.convexHull(np.concatenate(cavities));x,y,cw,ch=cv2.boundingRect(hull)
        area=cv2.contourArea(hull);perimeter=cv2.arcLength(hull,True)
        # Rounded photographic corners still leave four long straight sides
        # in an IEC resistor. Thick dilation alone makes that rectangle look
        # elliptic; require a curved cavity before checking the rim pixels.
        polygon=cv2.approxPolyDP(hull,.025*perimeter,True)
        if len(polygon)<5 or area/max(1,cw*ch)>.87:continue
        if min(cw,ch)<max(8,.28*min(bw,bh)) or max(cw,ch)>2.1*min(cw,ch):continue
        if cw>bw*1.35 or ch>bh*1.35 or area<.16*bw*bh or 4*np.pi*area/max(1,perimeter**2)<.68:continue
        cx,cy=x+x1+cw/2,y+y1+ch/2
        if abs(cx-(bbox['x1']+bbox['x2'])/2)>.35*bw or abs(cy-(bbox['y1']+bbox['y2'])/2)>.35*bh:continue
        # Sample around an ellipse in the actual ring. A triangular fork
        # cannot supply a nearly complete curved boundary.
        stroke=max(3,round(.06*min(cw,ch)))
        angles=np.linspace(0,2*np.pi,96,endpoint=False)
        xx=np.rint(x+cw/2+(cw/2-.5)*np.cos(angles)).astype(int)
        yy=np.rint(y+ch/2+(ch/2-.5)*np.sin(angles)).astype(int)
        near=cv2.dilate(local,np.ones((stroke*2+1,stroke*2+1),np.uint8))
        valid=(xx>=0)&(xx<near.shape[1])&(yy>=0)&(yy<near.shape[0])
        coverage=float(np.mean(near[yy[valid],xx[valid]]>0)) if np.any(valid) else 0
        if coverage<.92:continue
        body=dict(x1=float(max(0,x+x1-stroke)),y1=float(max(0,y+y1-stroke)),
                  x2=float(min(w-1,x+x1+cw+stroke)),y2=float(min(h-1,y+y1+ch+stroke)))
        leads=external_leads(image,body,mask,directions=True)
        if len(leads)!=2 or float(np.dot(*leads))>-.65:continue
        interior=np.zeros_like(local);cv2.drawContours(interior,[hull],-1,255,-1)
        interior=cv2.erode(interior,np.ones((max(3,stroke),max(3,stroke)),np.uint8))
        signs=cv2.bitwise_and(local,interior)
        count,sign_labels,sign_stats,_=cv2.connectedComponentsWithStats((signs>0).astype(np.uint8),8)
        from backend.terminal_identity import _plus_score
        plus_score=0.
        for j,(sx,sy,sw,sh,ink) in enumerate(sign_stats[1:],1):
            if ink<4 or min(sw,sh)<4:continue
            plus_score=max(plus_score,_plus_score((sign_labels[sy:sy+sh,sx:sx+sw]==j).astype(np.uint8)*255))
        return dict(source_body_bbox=body,ring_with_two_connected_leads=True,ring_external_leads=2,
                    opposed_connected_leads=2,polarity_partitioned_cavities=len(cavities),
                    elliptical_rim_coverage=round(coverage,3),internal_plus_score=round(plus_score,3),metric='observed_ring_with_signs_two_opposed_wires')
    return None
