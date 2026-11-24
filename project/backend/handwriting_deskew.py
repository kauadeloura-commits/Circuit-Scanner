"""Deskew a coherently tilted page using photographed word outlines only."""
from copy import deepcopy
import math
import cv2
import numpy as np


def estimate_angle(regions):
    rows=[(float(r['text_angle_degrees']),float(r['confidence'])) for r in regions
          if r.get('text_aspect_ratio',0)>=1.6 and r.get('confidence',0)>=.60
          and 'text_angle_degrees' in r]
    if len(rows)<4:return None
    def difference(a,b):return abs((a-b+45)%90-45)
    candidate=max(rows,key=lambda p:sum(w for a,w in rows if difference(a,p[0])<=12))[0]
    inliers=[(a,w) for a,w in rows if difference(a,candidate)<=12]
    if len(inliers)<4 or sum(w for a,w in inliers)<.55*sum(w for a,w in rows):return None
    vector=sum(w*np.exp(4j*math.radians(a)) for a,w in inliers)
    angle=math.degrees(np.angle(vector))/4
    return float(angle) if 10<=abs(angle)<=45 else None


def transform_box(box,matrix,polygon=None):
    points=np.asarray(polygon if polygon is not None else
        [[box[x],box[y]] for x,y in [('x1','y1'),('x1','y2'),('x2','y2'),('x2','y1')]],float)
    mapped=np.c_[points,np.ones(len(points))]@matrix.T
    result=dict(box,x1=float(mapped[:,0].min()),y1=float(mapped[:,1].min()),
                x2=float(mapped[:,0].max()),y2=float(mapped[:,1].max()))
    result.update(width=result['x2']-result['x1'],height=result['y2']-result['y1'],
                  center_x=(result['x1']+result['x2'])/2,center_y=(result['y1']+result['y2'])/2)
    return result,mapped.tolist()


def prepare(image,detections,regions):
    angle=estimate_angle(regions)
    if angle is None:return None
    h,w=image.shape[:2];matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1.)
    cosine,sine=abs(matrix[0,0]),abs(matrix[0,1]);width=math.ceil(w*cosine+h*sine);height=math.ceil(h*cosine+w*sine)
    matrix[:,2]+=np.array([width/2-w/2,height/2-h/2])
    view=cv2.warpAffine(image,matrix,(width,height),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    mapped=deepcopy(detections)
    for d in mapped:d['bbox']=transform_box(d['bbox'],matrix)[0]
    proposals=[]
    for r in regions:
        item=deepcopy(r);item['bbox'],item['text_polygon']=transform_box(r['bbox'],matrix,r.get('text_polygon'))
        proposals.append(item)
    return view,mapped,proposals,cv2.invertAffineTransform(matrix),angle


def restore(items,inverse,angle):
    output=[]
    for t in items:
        item=deepcopy(t);item['bbox'],item['text_polygon']=transform_box(t['bbox'],inverse,t.get('text_polygon'))
        item['page_deskew_degrees']=angle
        item['text_rotation_degrees']=float(item.get('text_rotation_degrees',0))+angle
        if item.get('replaced_readings'):
            item['replaced_readings']=restore(item['replaced_readings'],inverse,angle)
        output.append(item)
    return output
