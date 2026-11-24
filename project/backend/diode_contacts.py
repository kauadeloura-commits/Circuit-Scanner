"""Select diode/LED leads from the triangular body and its cathode bar."""
import cv2
import numpy as np
from backend.resistor_geometry import cached_vision


@cached_vision
def body_axes(image,bbox,_faint_retry=True):
    from backend.terminal_identity import _crop
    mask,origin=_crop(image,bbox)
    area=(bbox['x2']-bbox['x1'])*(bbox['y2']-bbox['y1'])
    proposals=[];contours=[]
    for size in (1,2,3):
        view=mask if size==1 else cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((size,size),np.uint8))
        contours.extend(cv2.findContours(view,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0])
    from backend.drawing_mode import digital
    from backend.drawing_mode import CURRENT
    minimum_fraction=.015 if CURRENT.get()=='handwritten' else .035
    if digital():
        # Separate a solid triangular body from its much thinner shafts.
        # The original, unmodified mask must still confirm the cathode bar.
        for size in (3,5):
            core=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size)))
            contours.extend(cv2.findContours(core,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0])
    for contour in contours:
        size=abs(cv2.contourArea(contour))
        if not max(14,area*minimum_fraction)<=size<=area*.65:continue
        hull=cv2.convexHull(contour)
        points=cv2.approxPolyDP(hull,.045*cv2.arcLength(hull,True),True).reshape(-1,2).astype(float)+origin
        if len(points)!=3:continue
        for k,apex in enumerate(points):
            base=np.delete(points,k,axis=0);center=base.mean(axis=0);side=base[1]-base[0]
            width=float(np.linalg.norm(side));delta=apex-center;height=float(np.linalg.norm(delta))
            if width<8 or height<max(7,width*.3):continue
            axis=delta/height;normal=np.array([-axis[1],axis[0]])
            if abs(float(side@axis))/width>.35:continue
            support=0.
            for shift in np.linspace(-2,max(3,height*.60),17):
                p=apex+axis*shift-origin;scores=[]
                for sign in (-1,1):
                    seen=[]
                    for offset in np.linspace(width*.22,width*.52,14):
                        x,y=np.rint(p+sign*normal*offset).astype(int)
                        seen.append(0<=x<mask.shape[1] and 0<=y<mask.shape[0] and bool(np.any(mask[max(0,y-2):y+3,max(0,x-2):x+3])))
                    scores.append(float(np.mean(seen)))
                support=max(support,min(scores))
            if support>=.85:proposals.append(dict(axis=axis,normal=normal,base=center,apex=apex,height=height,width=width,area=size,bar_support=support))
    if not proposals and _faint_retry and CURRENT.get()=='handwritten':
        from backend.notebook_rulings import _paper_colors
        _,_,normalized=_paper_colors(image)
        for threshold in (225,235):
            enhanced=np.full_like(image,255);enhanced[normalized.mean(axis=2)<threshold]=0
            for body in body_axes(enhanced,bbox,_faint_retry=False):
                body['local_contrast_retry']=True;proposals.append(body)
    return proposals


def select_diode_contacts(image,component,contacts):
    if component.get('type') not in {'diode','led'} or len(contacts)<2:return contacts,None
    ranked=[]
    for body in body_axes(image,component['bbox']):
        axis=body['axis'];normal=body['normal'];base=body['base'];height=body['height'];width=body['width']
        for i,a in enumerate(contacts):
            pa=np.asarray(a['point']);along_a=float((pa-base)@axis);side_a=abs(float((pa-base)@normal))
            if along_a>height*.22 or side_a>max(6,width*.50):continue
            for j,k in enumerate(contacts):
                if j==i:continue
                pk=np.asarray(k['point']);along_k=float((pk-base)@axis);side_k=abs(float((pk-body['apex'])@normal))
                if along_k<height*.78 or side_k>max(6,width*.50):continue
                if along_k-along_a<height*.70:continue
                cost=(side_a+side_k)/width+.1*(abs(along_a)+abs(along_k-height))/height
                cost+=.35*(bool(a.get('near_edge_endpoint'))+bool(k.get('near_edge_endpoint')))
                ranked.append((cost,i,j,body))
    if not ranked:return contacts,None
    ranked.sort(key=lambda x:x[0]);cost,i,j,body=ranked[0]
    # Several contours can describe the same body. Competing contact pairs,
    # rather than duplicate contours, determine whether the result is clear.
    alternate=next((r for r in ranked[1:] if (r[1],r[2])!=(i,j)),None)
    if alternate is not None and alternate[0]-cost<.08:return contacts,None
    meta=dict(source='observed_triangle_bar_and_opposed_external_leads',confirmed=True,bar_support=round(body['bar_support'],3),
              selected_contacts={'A':np.asarray(contacts[i]['point']).tolist(),'K':np.asarray(contacts[j]['point']).tolist()},
              rejected_contacts=len(contacts)-2)
    return [contacts[i],contacts[j]],meta
