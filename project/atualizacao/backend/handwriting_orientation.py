"""Orient complete inscriptions, then map all evidence to the source photograph."""
import copy
import cv2
import numpy as np
from backend.ocr_regions import text_line_regions

def transform_box(box,k,height,width):
    corners=np.array([[box['x1'],box['y1']],[box['x2'],box['y2']]],float)
    x,y=corners[:,0],corners[:,1]
    if k==1:x,y=y,width-x
    elif k==2:x,y=width-x,height-y
    elif k==3:x,y=height-y,x
    result=dict(box,x1=float(min(x)),y1=float(min(y)),x2=float(max(x)),y2=float(max(y)))
    result.update(width=result['x2']-result['x1'],height=result['y2']-result['y1'],center_x=(result['x1']+result['x2'])/2,center_y=(result['y1']+result['y2'])/2)
    return result

def orient(image,page,detections,refs):
    from backend.handwriting_reference import candidates
    layouts=[]
    for k in (0,1):
        rotated=np.rot90(page,k).copy();lines=[]
        for box in text_line_regions(rotated):
            crop=(rotated[box['y1']:box['y2'],box['x1']:box['x2']]<128).astype(np.uint8)*255
            if not crop.size:continue
            _,_,stats,_=cv2.connectedComponentsWithStats(crop,8)
            glyphs=[s for s in stats[1:] if s[4]>=10 and s[3]>=crop.shape[0]*.3]
            aspect=crop.shape[1]/max(1,crop.shape[0])
            if len(glyphs)>=2 and aspect>1.1:lines.append((box,min(8,len(glyphs))*min(4,aspect)))
        layouts.append((sum(v for _,v in lines),k,lines))
    axis=1 if layouts[1][0]>layouts[0][0]*1.3 else 0
    lines=layouts[axis][2];rotated=np.rot90(page,axis).copy()
    singles=[(e,s) for e,s in refs if len(e['text'])==1]
    scores=[0.,0.];count=0
    for box,_ in sorted(lines,key=lambda p:p[1],reverse=True)[:12]:
        crop=(rotated[box['y1']:box['y2'],box['x1']:box['x2']]<128).astype(np.uint8)*255
        if not crop.size:continue
        _,labels,stats,_=cv2.connectedComponentsWithStats(crop,8)
        glyphs=sorted([(i,s) for i,s in enumerate(stats[1:],1) if s[4]>=10 and s[3]>=crop.shape[0]*.4 and s[2]<1.3*s[3]],key=lambda p:p[1][0])
        for index,(x,y,w,h,_) in glyphs[:3]:
            mask=(labels[y:y+h,x:x+w]==index).astype(np.uint8)*255
            for direction in (0,1):
                ranking=candidates(np.rot90(mask,2*direction).copy(),singles)
                if ranking:scores[direction]+=max(0,float(ranking[0]['score'])-.55)**2
            count+=1
    k=(axis+2)%4 if count>=4 and scores[1]>scores[0]*1.12 else axis
    height,width=image.shape[:2];rotated_detections=copy.deepcopy(detections)
    for d in rotated_detections:d['bbox']=transform_box(d['bbox'],k,height,width)
    return np.rot90(image,k).copy(),np.rot90(page,k).copy(),rotated_detections,k
