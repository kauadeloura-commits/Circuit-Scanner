"""Read photographed opamp signs inside an observed triangle, including touching ink."""
import cv2
import numpy as np


def _triangles(mask, origin, bbox):
    area_box=(bbox['x2']-bbox['x1'])*(bbox['y2']-bbox['y1'])
    seen=set()
    # Closing is used only to propose a body outline, never to connect wires.
    for size in (1,2,3,5,7,11,15):
        closed=mask if size==1 else cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((size,size),np.uint8))
        for contour in cv2.findContours(closed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]:
            hull=cv2.convexHull(contour)
            if not .06*area_box<cv2.contourArea(hull)<.85*area_box:continue
            vertices=cv2.approxPolyDP(hull,.045*cv2.arcLength(hull,True),True).reshape(-1,2).astype(float)+origin
            if len(vertices)>3:
                vertices=cv2.approxPolyDP(hull,.08*cv2.arcLength(hull,True),True).reshape(-1,2).astype(float)+origin
            if len(vertices)!=3:continue
            if any(not(bbox['x1']-3<=p[0]<=bbox['x2']+3 and bbox['y1']-3<=p[1]<=bbox['y2']+3) for p in vertices):continue
            key=tuple(sorted(tuple(np.rint(p/3).astype(int)) for p in vertices))
            if key in seen:continue
            seen.add(key)
            yield vertices
    # Handwritten corners may have a real pen gap. Three observed body sides
    # can establish a triangle without closing or extending electrical wires.
    from itertools import combinations
    lines=cv2.HoughLinesP(mask,1,np.pi/180,threshold=15,
        minLineLength=max(15,int(min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])*.28)),maxLineGap=8)
    if lines is None:return
    selected=[]
    for raw in sorted(np.asarray(lines).reshape(-1,4),key=lambda l:-np.linalg.norm(l[2:]-l[:2])):
        a,b=raw[:2].astype(float)+origin,raw[2:].astype(float)+origin
        unit=(b-a)/np.linalg.norm(b-a)
        if any(abs(float(unit@u))>.98 and abs(float(unit[0]*(c-a)[1]-unit[1]*(c-a)[0]))<6 for c,d,u in selected):continue
        selected.append((a,b,unit))
        if len(selected)>=14:break
    def intersect(a,u,b,v):
        m=np.c_[u,-v]
        if abs(np.linalg.det(m))<.25:return None
        return a+u*np.linalg.solve(m,b-a)[0]
    for triple in combinations(selected,3):
        vertices=[intersect(triple[i][0],triple[i][2],triple[(i+1)%3][0],triple[(i+1)%3][2]) for i in range(3)]
        if any(v is None for v in vertices):continue
        vertices=np.array(vertices)
        if any(not(bbox['x1']-3<=p[0]<=bbox['x2']+3 and bbox['y1']-3<=p[1]<=bbox['y2']+3) for p in vertices):continue
        if not .08*area_box<cv2.contourArea(vertices.astype(np.float32))<.75*area_box:continue
        supported=True
        for i in range(3):
            points=np.linspace(vertices[i],vertices[(i+1)%3],30)[2:-2]-origin
            hits=[np.any(mask[max(0,round(p[1])-2):round(p[1])+3,max(0,round(p[0])-2):round(p[0])+3]) for p in points]
            if np.mean(hits)<.85:supported=False;break
        key=tuple(sorted(tuple(np.rint(p/3).astype(int)) for p in vertices))
        if supported and key not in seen:seen.add(key);yield vertices


def _interior_marks(mask,origin,vertices,unit,base_unit):
    inside=np.zeros_like(mask)
    cv2.fillPoly(inside,[np.rint(vertices-origin).astype(np.int32)],255)
    # The hole's convex hull restores the triangle boundary when a '+' touches
    # it. Removing that boundary separates the sign from the amplifier body.
    distance=cv2.distanceTransform(inside,cv2.DIST_L2,5)
    interior=(mask>0)&(distance>2.0)
    n,labels,stats,centers=cv2.connectedComponentsWithStats(interior.astype(np.uint8),8)
    length=float(np.linalg.norm(vertices[1]-vertices[0]))
    result=[]
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]<5:continue
        y,x=np.nonzero(labels==i);points=np.c_[x,y].astype(float)+origin
        center=points.mean(axis=0)
        across=points@base_unit;along=points@unit
        a=float(np.ptp(along)+1);b=float(np.ptp(across)+1)
        if min(a,b)<2 or max(a,b)>length*.45:continue
        covariance=np.cov(points.T)
        eigen=np.linalg.eigvalsh(covariance)
        linearity=float(eigen[-1]/max(.25,eigen[0]))
        xx,yy,w,h=stats[i,:4]
        blob=(labels[yy:yy+h,xx:xx+w]==i).astype(np.uint8)*255
        contours=cv2.findContours(blob,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        contour=max(contours,key=cv2.contourArea)
        solidity=cv2.contourArea(contour)/max(1,cv2.contourArea(cv2.convexHull(contour)))
        if (solidity>.86 and (a/b>=1.15 or linearity>=1.5)) or a/b>=1.65 or linearity>=3.5:
            kind='-'
        elif solidity<=.86 and .45<=a/b<=1.9 and min(a,b)>=4:
            # Require a cross rather than certifying an arbitrary ink dot.
            from backend.terminal_identity import _plus_score
            if _plus_score(blob)<.38:continue
            kind='+'
        else:continue
        result.append(dict(kind=kind,point=center,score=.85,source='triangle_interior_local_symbol_ocr'))
    return result


def signed_triangle(mask,origin,bbox,existing_marks):
    from backend.terminal_identity import _support,_opamp_lead_exit
    candidates=[]
    for vertices in _triangles(mask,origin,bbox):
        for apex_index in range(3):
            apex=vertices[apex_index];base=np.array([vertices[(apex_index+1)%3],vertices[(apex_index+2)%3]])
            midpoint=base.mean(axis=0);direction=apex-midpoint
            length=float(np.linalg.norm(direction));base_length=float(np.linalg.norm(base[1]-base[0]))
            if min(length,base_length)<10:continue
            unit=direction/length;base_unit=(base[1]-base[0])/base_length
            if abs(float(unit@base_unit))>.45:continue
            out_support=_support(mask,apex,unit,np.arange(3,max(10,length*.24),1),origin)
            if out_support<.70:continue
            clusters=[]
            for t in np.linspace(.10,.90,81):
                p=base[0]+(base[1]-base[0])*t
                if _support(mask,p,-unit,np.arange(4,max(12,length*.28),1),origin)<.75:continue
                if not clusters or t-clusters[-1][-1][0]>.025:clusters.append([])
                clusters[-1].append((t,p))
            ports=[np.mean([p for _,p in group],axis=0) for group in clusters]
            if len(ports)!=2 or np.linalg.norm(ports[0]-ports[1])<base_length*.20:continue
            marks=_interior_marks(mask,origin,vertices,unit,base_unit)
            # A minus may look like a plus to an axis-aligned glyph reader.
            # Prefer the stroke measured in this triangle's own coordinates.
            marks += [m for m in existing_marks if cv2.pointPolygonTest(vertices.astype(np.float32),tuple(map(float,m['point'])),True)>0
                      and not any(np.linalg.norm(m['point']-local['point'])<max(8,base_length*.08) for local in marks)]
            choices=[]
            for plus in [m for m in marks if m['kind']=='+']:
                for minus in [m for m in marks if m['kind']=='-']:
                    mapping=[];cost=0
                    for mark in (plus,minus):
                        offset=[abs(float((mark['point']-p)@base_unit)) for p in ports]
                        j=int(np.argmin(offset))
                        depth=float((mark['point']-ports[j])@unit)
                        if offset[j]>max(5,base_length*.16) or not(0<depth<length*.60):break
                        mapping.append(j);cost+=offset[j]
                    if len(mapping)==2 and len(set(mapping))==2:choices.append((cost,mapping,plus,minus))
            if not choices:continue
            cost,mapping,plus,minus=min(choices,key=lambda x:x[0])
            points={'+':_opamp_lead_exit(mask,ports[mapping[0]],-unit,bbox,origin,vertices),
                    '-':_opamp_lead_exit(mask,ports[mapping[1]],-unit,bbox,origin,vertices),
                    'OUT':_opamp_lead_exit(mask,apex,unit,bbox,origin,vertices)}
            candidates.append((cost/base_length,points,apex,[plus,minus]))
    if not candidates:return None
    _,points,apex,marks=min(candidates,key=lambda x:x[0])
    return points,apex,marks
