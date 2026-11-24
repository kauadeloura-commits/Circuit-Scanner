"""Alternate ground symbol: a closed triangle with one external conductor."""
import cv2
import numpy as np
from backend.resistor_geometry import symbol_mask
from backend.vcc_topology import external_leads

def triangle_ground_evidence(image,bbox):
    mask=symbol_mask(image);h,w=mask.shape
    x1,y1=max(0,int(bbox['x1'])-5),max(0,int(bbox['y1'])-5)
    x2,y2=min(w,int(np.ceil(bbox['x2']))+6),min(h,int(np.ceil(bbox['y2']))+6)
    crop=mask[y1:y2,x1:x2]
    if not crop.size:return None
    crop=cv2.morphologyEx(crop,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,hierarchy=cv2.findContours(crop,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:return None
    for i,contour in enumerate(contours):
        if hierarchy[0,i,3]<0:continue
        perimeter=cv2.arcLength(contour,True);area=cv2.contourArea(contour)
        polygon=cv2.approxPolyDP(contour,.06*perimeter,True)
        if area<12 or len(polygon)!=3:continue
        xx,yy,bw,bh=cv2.boundingRect(contour)
        if min(bw,bh)<4 or max(bw,bh)>3*min(bw,bh):continue
        box=dict(x1=float(max(0,x1+xx-4)),y1=float(max(0,y1+yy-4)),x2=float(min(w-1,x1+xx+bw+4)),y2=float(min(h-1,y1+yy+bh+4)))
        if external_leads(image,box,mask)==1:
            return dict(closed_ground_triangle_with_one_lead=True,ground_body_bbox=box,external_lead_count=1)
    return None
