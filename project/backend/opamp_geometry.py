"""Read photographed opamp signs inside an observed triangle, including touching ink."""
import cv2
import numpy as np


def _triangles(mask, origin, bbox):
    area_box=(bbox['x2']-bbox['x1'])*(bbox['y2']-bbox['y1'])
    seen=set()
    # Closing is used only to propose a body outline, never to connect wires.
    for size in (1,2,3):
        closed=mask if size==1 else cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((size,size),np.uint8))
        for contour in cv2.findContours(closed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]:
            hull=cv2.convexHull(contour)
            if not .06*area_box<cv2.contourArea(hull)<.85*area_box:continue
            vertices=cv2.approxPolyDP(hull,.045*cv2.arcLength(hull,True),True).reshape(-1,2).astype(float)+origin
            if len(vertices)!=3:continue
            if any(not(bbox['x1']-3<=p[0]<=bbox['x2']+3 and bbox['y1']-3<=p[1]<=bbox['y2']+3) for p in vertices):continue
            key=tuple(sorted(tuple(np.rint(p/3).astype(int)) for p in vertices))
            if key in seen:continue
            seen.add(key)
            yield vertices


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
        if a/b>=1.65:
            kind='-'
        elif .45<=a/b<=1.9 and min(a,b)>=4:
            # Require a cross rather than certifying an arbitrary ink dot.
            from backend.terminal_identity import _plus_score
            xx,yy,w,h=stats[i,:4]
            blob=(labels[yy:yy+h,xx:xx+w]==i).astype(np.uint8)*255
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
            marks += [m for m in existing_marks if cv2.pointPolygonTest(vertices.astype(np.float32),tuple(map(float,m['point'])),True)>0]
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


def separate_adjacent_symbols(image,detections):
    """Keep nearby ground glyphs outside an amplifier's photographed body box."""
    from copy import deepcopy
    from backend.terminal_identity import _crop,_marks
    from backend.paper_ink import paper_relative_view
    result=deepcopy(detections)
    for d in result:
        if d.get('class')!='opamp' or d.get('manual_reviewed'):continue
        b=d['bbox']
        neighbors=[o for o in result if o is not d and o.get('class')=='gnd' and
            b['x1']<(o['bbox']['x1']+o['bbox']['x2'])/2<b['x2'] and
            b['y1']<(o['bbox']['y1']+o['bbox']['y2'])/2<b['y2'] and
            max(b['x1'],o['bbox']['x1'])<min(b['x2'],o['bbox']['x2']) and
            max(b['y1'],o['bbox']['y1'])<min(b['y2'],o['bbox']['y2'])]
        if not neighbors:continue
        # The contrast view proposes an outline only, never a new conductor.
        mask,origin=_crop(paper_relative_view(image),b);marks,_=_marks(mask,origin,b,include_attached=True)
        signed=signed_triangle(mask,origin,b,marks)
        if signed is None:continue
        _,apex,_=signed
        triangles=[v for v in _triangles(mask,origin,b) if np.min(np.linalg.norm(v-apex,axis=1))<4]
        if not triangles:continue
        vertices=min(triangles,key=lambda v:cv2.contourArea(v.astype(np.float32)))
        margin=max(8.,min(12.,float(np.ptp(vertices,axis=0).min())*.18))
        tight=dict(x1=max(b['x1'],float(vertices[:,0].min()-margin)),y1=max(b['y1'],float(vertices[:,1].min()-margin)),
                   x2=min(b['x2'],float(vertices[:,0].max()+margin)),y2=min(b['y2'],float(vertices[:,1].max()+margin)))
        if tight['x1']>=tight['x2'] or tight['y1']>=tight['y2']:continue
        tight.update(width=tight['x2']-tight['x1'],height=tight['y2']-tight['y1'],center_x=(tight['x1']+tight['x2'])/2,center_y=(tight['y1']+tight['y2'])/2)
        d.update(bbox=tight,opamp_body_box=dict(method='two_visible_signs_triangle_excludes_adjacent_ground',vertices=vertices.tolist()))
    return result
