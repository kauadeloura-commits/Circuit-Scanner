"""Preserve strong learned symbols when pen gaps defeat rigid templates."""
import cv2
import numpy as np
from backend.vcc_topology import external_leads

def opposed_resistor_leads(image,bbox,mask):
    vectors=external_leads(image,bbox,mask,directions=True)
    if len(vectors)!=2 or float(np.dot(*vectors))>-.65:return None
    h,w=mask.shape;x1,y1=max(0,int(bbox['x1'])),max(0,int(bbox['y1']))
    x2,y2=min(w,int(np.ceil(bbox['x2']))+1),min(h,int(np.ceil(bbox['y2']))+1)
    crop=mask[y1:y2,x1:x2]
    if not crop.size or np.count_nonzero(crop)<20:return None
    axis=np.asarray(vectors[0])-vectors[1];axis/=max(1e-8,np.linalg.norm(axis));normal=np.array([-axis[1],axis[0]])
    yy,xx=np.nonzero(crop);points=np.column_stack((xx,yy));spread=float(np.diff(np.percentile(points@normal,[5,95]))[0])
    thickness=cv2.distanceTransform(crop,cv2.DIST_L2,3);stroke=max(1.,2*float(np.percentile(thickness[crop>0],65)))
    if spread<max(5,2.4*stroke):return None
    return dict(two_opposed_original_ink_leads=True,external_lead_count=2,lead_direction_dot=round(float(np.dot(*vectors)),3),transverse_body_spread=round(spread,2),stroke_width=round(stroke,2))

def partial_ground_with_one_lead(image,bbox,evidence,mask):
    if int(evidence.get('horizontal_bar_candidates',0))<2 or int(evidence.get('vertical_stem_candidates',0))<1:return None
    if external_leads(image,bbox,mask)!=1:return None
    return dict(two_visible_ground_bars_and_one_external_stem=True,external_lead_count=1)
