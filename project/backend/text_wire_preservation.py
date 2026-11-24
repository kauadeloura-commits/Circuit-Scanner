"""Preserve observed conductors crossing OCR rectangles; never draw a new wire."""
import cv2
import numpy as np

def crossing_ink(ink,bbox):
    h,w=ink.shape;x1,y1=max(0,int(bbox['x1'])-2),max(0,int(bbox['y1'])-2)
    x2,y2=min(w,int(np.ceil(bbox['x2']))+3),min(h,int(np.ceil(bbox['y2']))+3)
    bw,bh=x2-x1,y2-y1
    if min(bw,bh)<3:return None
    margin=max(8,min(40,round(min(bw,bh)*.5)))
    ax,ay=max(0,x1-margin),max(0,y1-margin);bx,by=min(w,x2+margin),min(h,y2+margin)
    crop=ink[ay:by,ax:bx];minimum=max(12,min(bw,bh)+2*max(5,margin*.5))
    lines=cv2.HoughLinesP(crop,1,np.pi/180,threshold=max(10,int(minimum*.6)),minLineLength=minimum,maxLineGap=2)
    if lines is None:return None
    support=np.zeros_like(crop);count=0
    for xa,ya,xb,yb in np.asarray(lines).reshape(-1,4):
        a=np.array([xa+ax,ya+ay],float);b=np.array([xb+ax,yb+ay],float)
        horizontal=(a[0]<x1-5 and b[0]>x2+5) or (b[0]<x1-5 and a[0]>x2+5)
        vertical=(a[1]<y1-5 and b[1]>y2+5) or (b[1]<y1-5 and a[1]>y2+5)
        if not(horizontal or vertical):continue
        points=np.rint(np.linspace(a,b,max(2,int(np.linalg.norm(b-a))+1))).astype(int)
        inside=(points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)
        points=points[inside]
        if not len(points) or np.mean(ink[points[:,1],points[:,0]]>0)<.85:continue
        cv2.line(support,(int(xa),int(ya)),(int(xb),int(yb)),255,3);count+=1
    if not count:return None
    keep=np.zeros_like(crop);keep[y1-ay:y2-ay,x1-ax:x2-ax]=255
    keep=cv2.bitwise_and(cv2.bitwise_and(keep,support),crop)
    return (ax,ay,bx,by),keep,count
