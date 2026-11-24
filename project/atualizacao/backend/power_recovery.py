"""Recover hollow one-lead power markers missed by the model; never junctions."""
import cv2,numpy as np
from backend.resistor_geometry import symbol_mask
from backend.vcc_topology import external_leads,ring_bounds,symbol_ink_labels,context_symbol_mask

def recover_power(image,detections,class_id):
    mask=symbol_mask(image)
    trace_mask=context_symbol_mask(image,detections)
    # Keep the photographed hole outline; recovered pale stroke fringes are
    # used to trace the lead, without shrinking or splitting that outline.
    contours,hierarchy=cv2.findContours(symbol_mask(image),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:return []
    found=[];h,w=mask.shape
    _,page_labels=cv2.connectedComponents(trace_mask,8)
    known=symbol_ink_labels(page_labels,detections)
    for i,c in enumerate(contours):
        if hierarchy[0,i,3]<0:continue
        x,y,bw,bh=cv2.boundingRect(c);area=cv2.contourArea(c);per=cv2.arcLength(c,True)
        if min(bw,bh)<6 or max(bw,bh)>max(64,min(h,w)*.18) or max(bw,bh)>1.8*min(bw,bh) or not per or 4*np.pi*area/per**2<.55:continue
        box=ring_bounds(x,y,bw,bh,mask.shape,mask)
        cx,cy=x+bw/2,y+bh/2
        if any(d['bbox']['x1']<=cx<=d['bbox']['x2'] and d['bbox']['y1']<=cy<=d['bbox']['y2'] for d in detections+found):continue
        if external_leads(image,box,mask)!=1:continue
        if known:
            region=page_labels[int(box['y1']):int(box['y2'])+1,int(box['x1']):int(box['x2'])+1]
            if not set(int(v) for v in np.unique(region) if v)&known:
                # Blur can interrupt the narrow stem at the rim. Require a
                # nearby continuation that belongs to a recognized symbol.
                gap=max(6,min(20,int(max(bw,bh)*.35)))
                region=page_labels[max(0,int(box['y1'])-gap):min(h,int(box['y2'])+gap+1),max(0,int(box['x1'])-gap):min(w,int(box['x2'])+gap+1)]
                if not set(int(v) for v in np.unique(region) if v)&known:continue
        # Include a short piece of the real stem, allowing a terminal on ink
        # rather than on the circle. The connected ring identifies ownership.
        margin=2*max(bw,bh)
        ax,ay=max(0,x-margin),max(0,y-margin);bx,by=min(w,x+bw+margin),min(h,y+bh+margin)
        n,labels,_,_=cv2.connectedComponentsWithStats(mask[ay:by,ax:bx],8)
        ring=labels[max(0,y-ay-2):y-ay+bh+3,max(0,x-ax-2):x-ax+bw+3]
        counts=np.bincount(ring.ravel(),minlength=n);counts[0]=0
        if not counts.max():continue
        yy,xx=np.nonzero(labels==counts.argmax());xx=xx+ax;yy=yy+ay
        outside=(xx<box['x1'])|(xx>box['x2'])|(yy<box['y1'])|(yy>box['y2'])
        if not np.any(outside):continue
        # The lead exits radially through the middle. A letter R has a stem
        # tangent to its loop and must never become a supply marker.
        radial=(np.abs(xx-cx)<max(3,bw*.22))|(np.abs(yy-cy)<max(3,bh*.22))
        if np.count_nonzero(outside&radial)<max(5,max(bw,bh)*.3):continue
        distances=np.hypot(xx-cx,yy-cy)
        if distances.max()<max(bw,bh)*1.25:continue
        shaft=outside&(distances>max(bw,bh)*.95)&(distances<max(bw,bh)*2.8)
        if np.count_nonzero(shaft)<6:continue
        coords=np.column_stack((xx[shaft],yy[shaft])).astype(np.float32)
        _,(major,minor),_=cv2.minAreaRect(coords)
        if max(major,minor)<2*max(1,min(major,minor)):continue
        centerline=coords.mean(axis=0);_,_,axes=np.linalg.svd(coords-centerline,full_matrices=False)
        normal=np.array([-axes[0,1],axes[0,0]])
        if abs(float((np.array([cx,cy])-centerline)@normal))>max(3,max(bw,bh)*.25):continue
        near=shaft&(distances<max(bw,bh)*1.30)
        if not np.any(near):continue
        box.update(x1=float(min(box['x1'],xx[near].min()-2)),y1=float(min(box['y1'],yy[near].min()-2)),x2=float(max(box['x2'],xx[near].max()+2)),y2=float(max(box['y2'],yy[near].max()+2)))
        box.update(width=box['x2']-box['x1'],height=box['y2']-box['y1'],center_x=(box['x1']+box['x2'])/2,center_y=(box['y1']+box['y2'])/2)
        found.append(dict(detection_id='PWR_RECOVER_'+str(len(found)+1),**{'class':'vcc'},class_id=class_id,bbox=box,confidence=.94,best_view_confidence=.94,inference_sources=['one_lead_hollow_power_geometry'],geometry_validation=dict(valid=True,score=.94,external_lead_count=1,hollow_ring=True)))
    return found
