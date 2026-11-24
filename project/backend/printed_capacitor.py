"""Printed capacitor polarity from observed plate shapes and opposed leads.

This policy is explicitly digital-only. It never changes conductor pixels or
uses component names, values, dataset identity, or expected net connections.
"""
import cv2
import numpy as np
from backend.resistor_geometry import cached_vision


def plate_ink(image):
    """Keep actual solid interiors; local contrast alone makes them look hollow."""
    from backend.drawing_mode import printed_ink
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    # Do not fill a pale antialiased gap between two dark plates. At small
    # print sizes that gap can be gray rather than pure white.
    threshold=min(180,float(np.percentile(gray,85))-35)
    return np.maximum(printed_ink(image),np.where(gray<threshold,255,0).astype(np.uint8))


@cached_vision
def plate_polarity(image,bbox):
    from backend.drawing_mode import digital
    empty=dict(valid=False,method='printed_plate_shape',reason='no_unambiguous_asymmetric_plates')
    if not digital():return {**empty,'reason':'printed_conventions_disabled_for_this_domain'}
    if image is None or not image.size:return {**empty,'reason':'empty_image'}
    from backend.capacitor_geometry import capacitor_body_evidence
    ok,body=capacitor_body_evidence(image,bbox)
    if not ok:return {**empty,'body':body,'reason':'two_opposed_lead_bearing_plates_not_confirmed'}
    centers=np.asarray(body['outer_plate_centers'],float)
    lengths=np.asarray(body['outer_plate_lengths'],float)
    u=np.asarray(body['electrical_axis'],float);v=np.array([-u[1],u[0]])
    gap=abs(float((centers[1]-centers[0])@u))
    if gap<2:return empty
    pad=3
    x=max(0,int(bbox['x1'])-pad);y=max(0,int(bbox['y1'])-pad)
    endx=min(image.shape[1],int(np.ceil(bbox['x2']))+pad+1)
    endy=min(image.shape[0],int(np.ceil(bbox['y2']))+pad+1)
    roi=image[y:endy,x:endx]
    if not roi.size:return empty
    scale=max(2,min(5,int(np.ceil(100/max(1,min(lengths))))))
    enlarged=cv2.resize(roi,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(enlarged,cv2.COLOR_BGR2GRAY) if enlarged.ndim==3 else enlarged
    mask=plate_ink(enlarged)
    # An outline contains a narrow rectangular white cell. An oval digit does
    # not suffice; the body proof above also demands separate opposed leads.
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    holes=[[] for _ in centers]
    if hierarchy is not None:
        for j,contour in enumerate(contours):
            if hierarchy[0,j,3]<0 or cv2.contourArea(contour)<4*scale*scale:continue
            rect=cv2.minAreaRect(contour);a,b=rect[1]
            if min(a,b)<1 or min(a,b)/max(a,b)>.42 or cv2.contourArea(contour)<.65*a*b:continue
            cell=np.zeros(mask.shape,np.uint8);cv2.drawContours(cell,[contour],-1,255,-1)
            values=gray[cell>0]
            if not len(values) or np.median(values)<max(225,float(np.percentile(gray,85))-20):continue
            p=np.asarray(rect[0])/scale+[x,y]
            i=int(np.argmin([abs((p-c)@u) for c in centers]))
            if abs((p-centers[i])@u)>.85*gap or abs((p-centers[i])@v)>.22*lengths[i]:continue
            if not .35*lengths[i]<=max(a,b)/scale<=1.1*lengths[i]:continue
            holes[i].append(p.tolist())
    # Sample away from the midpoint wire. A curved plate has a consistent
    # quadratic centerline, while an outlined or filled straight plate does not.
    profiles=[]
    for center,length in zip(centers,lengths):
        radius=max(2,min(.62*gap,.30*length))
        offsets=np.linspace(-radius,radius,max(17,int(2*radius*scale)+1))
        ts=np.r_[np.linspace(-.40,-.16,9),np.linspace(.16,.40,9)]
        samples=[]
        widths=[]
        for t in ts:
            points=center+v*(t*length)+offsets[:,None]*u
            xx=np.rint((points[:,0]-x)*scale).astype(int);yy=np.rint((points[:,1]-y)*scale).astype(int)
            valid=(xx>=0)&(yy>=0)&(xx<mask.shape[1])&(yy<mask.shape[0])
            on=np.zeros(len(xx),bool);on[valid]=mask[yy[valid],xx[valid]]>0
            indices=np.flatnonzero(on)
            if not len(indices):continue
            # Keep only the nearest stroke group; a neighbor across the gap
            # must not become apparent thickness or curvature of this plate.
            groups=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
            nearest=min(groups,key=lambda g:abs(float(np.mean(offsets[g]))))
            samples.append((t,float(np.mean(offsets[nearest]))))
            widths.append(float(offsets[nearest[-1]]-offsets[nearest[0]]+1/scale))
        detail=dict(samples=len(samples),hollow=bool(holes[len(profiles)]))
        if len(samples)>=14:
            t,a=np.asarray(samples).T
            coef=np.polyfit(t,a,2);fitted=np.polyval(coef,t)
            residual=float(np.sqrt(np.mean((a-fitted)**2)))
            sag=float(coef[0]*.16)
            detail.update(sag_px=round(sag,3),residual_px=round(residual,3),stroke_width_px=round(float(np.median(widths)),3),
                          curved=bool(abs(sag)>=max(.7,.035*length) and residual<=max(.45,.015*length) and abs(sag)>=2*residual),
                          straight=bool(abs(sag)<=max(.55,.027*length) and residual<=max(.45,.02*length)))
        profiles.append(detail)
    evidence={**empty,'body':body,'plates':profiles}
    positive=None;method=None
    if sum(bool(h) for h in holes)==1:
        i=next(i for i,h in enumerate(holes) if h)
        other=profiles[1-i]
        if other.get('straight') and other.get('stroke_width_px',0)>=max(1.2,.035*lengths[1-i]):
            positive=i;method='printed_hollow_positive_and_filled_negative_plate'
    if positive is None and sum(bool(p.get('curved')) for p in profiles)==1:
        negative=next(i for i,p in enumerate(profiles) if p.get('curved'))
        if profiles[1-negative].get('straight') and not any(holes):
            positive=1-negative;method='printed_straight_positive_and_curved_negative_plate'
    if positive is None:return evidence
    return {**evidence,'valid':True,'method':method,'reason':'asymmetric_plate_convention_with_opposed_leads',
            'positive_plate':centers[positive].tolist(),'negative_plate':centers[1-positive].tolist()}


def positive_contact(evidence,pair):
    """Assign signs on existing lead positions; never move or synthesize a lead."""
    if not evidence.get('valid') or len(pair)!=2:return None
    p,n=np.asarray(evidence['positive_plate']),np.asarray(evidence['negative_plate'])
    u=p-n;length=float(np.linalg.norm(u))
    if length<2:return None
    offsets=(np.asarray(pair)-(p+n)/2)@(u/length)
    if min(offsets)>=-1 or max(offsets)<=1:return None
    return int(np.argmax(offsets))
