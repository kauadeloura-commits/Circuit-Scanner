"""Only a visible filled body arrow may resolve MOSFET N/P conflicts."""
import cv2,numpy as np,math
from backend.terminal_identity import _crop,_marks,_mosfet,_lines
from backend.local_symbol_roles import local_view

def visible_channel_type(image,box):
    view,b,_,origin=local_view(image,box,4)
    mask,offset=_crop(view,b);_,labels=_marks(mask,offset,b)
    roles=_mosfet(mask,offset,b,labels)
    if roles is None or not roles[1]:return None,dict(reason='gate_and_source_not_confirmed')
    points=roles[0];middle=(points['D']+points['S'])/2
    lines=_lines(mask,offset);choices=[]
    for l in lines:
        v=np.array([math.cos(l['angle']),math.sin(l['angle'])]);normal=np.array([-v[1],v[0]])
        if l['length']<min(mask.shape)*.25:continue
        p=np.rint(l['center']-offset).astype(int)
        if not(0<=p[0]<mask.shape[1] and 0<=p[1]<mask.shape[0]):continue
        label=labels[p[1],p[0]]
        gp=np.rint(points['G']-offset).astype(int);gp=np.clip(gp,[0,0],np.array(mask.shape[::-1])-1)
        if not label or label!=labels[gp[1],gp[0]]:continue
        if normal@(middle-l['center'])<0:normal=-normal
        choices.append((l['length'],l['center'],v,normal))
    if not choices:return None,dict(reason='isolated_gate_axis_missing')
    _,center,v,u=max(choices,key=lambda r:r[0])
    distance=cv2.distanceTransform(mask,cv2.DIST_L2,5);values=distance[distance>0]
    radius=max(2,int(np.percentile(values,55)*1.25))
    eroded=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    contours,_=cv2.findContours(eroded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    arrows=[];minimum=min(b['x2']-b['x1'],b['y2']-b['y1'])
    for contour in contours:
        area=cv2.contourArea(contour)
        if not minimum**2*.001<area<minimum**2*.10:continue
        hull=cv2.convexHull(contour);solid=area/max(1,cv2.contourArea(hull))
        if solid<.72:continue
        _,tri=cv2.minEnclosingTriangle(hull)
        if tri is None:continue
        tri=tri.reshape(3,2)+offset;c=tri.mean(axis=0)
        across=float((c-center)@u);along=float((c-center)@v)
        if not minimum*.08<across<minimum*.70 or abs(along)>minimum*.24:continue
        q=np.sort(tri@u);spread=q[-1]-q[0]
        if spread<max(6,minimum*.06):continue
        left=q[1]-q[0];right=q[2]-q[1]
        if min(left,right)>spread*.25:continue
        target='mosfet_p' if right>left else 'mosfet_n'
        arrows.append((area*solid,target,tri))
    if not arrows:return None,dict(reason='filled_body_arrow_not_separated')
    if len({a[1] for a in arrows})!=1:return None,dict(reason='conflicting_arrow_directions')
    _,kind,triangle=max(arrows,key=lambda a:a[0])
    return kind,dict(method='isolated_gate_and_visible_filled_body_arrow',triangle=((triangle/4)+origin).tolist(),erosion_radius=radius/4)
