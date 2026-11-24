"""Recover conductors hidden by detection boxes using continuous source ink.

The rectangle of a three-pin detection includes external leads. Its pin hull
is a tighter barrier: an external trace can pass beside it, never through the
symbol. No dilation or distance-only joining is used in this second pass.
"""
from collections import defaultdict
import heapq
import math
import cv2
import numpy as np

VERSION = 'observed-external-leads-1'


def _trace(mask, start, end, limit):
    """Bounded A* on actual foreground pixels, retaining photographed bends."""
    h,w=mask.shape;queue=[(math.dist(start,end),0.,start)];cost={start:0.};previous={}
    while queue and len(cost)<16000:
        _,distance,p=heapq.heappop(queue)
        if distance!=cost[p]:continue
        if p==end:
            path=[p]
            while p in previous:p=previous[p];path.append(p)
            return list(reversed(path)),distance
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            q=(p[0]+dx,p[1]+dy)
            if not (0<=q[0]<w and 0<=q[1]<h and mask[q[1],q[0]]):continue
            new=distance+(math.sqrt(2) if dx and dy else 1)
            if new+math.dist(q,end)>limit or new>=cost.get(q,math.inf):continue
            cost[q]=new;previous[q]=p;heapq.heappush(queue,(new+math.dist(q,end),new,q))
    return None


def recover_observed_leads(ink,components,segments,protected_nodes,roots,blocked_anchors=()):
    mask=(ink>0).astype(np.uint8);h,w=mask.shape;anchors=[]
    for c in components:
        terms=c.get('terminals') or [];points=[]
        for t in terms:
            p=t.get('position') or {}
            if 'x' not in p or 'y' not in p:continue
            if not all(isinstance(p[k],(int,float)) and math.isfinite(p[k]) for k in ('x','y')):continue
            if not (0<=p['x']<w and 0<=p['y']<h):continue
            xy=(round(p['x']),round(p['y']));points.append(xy)
            anchors.append(('T:'+c['component_id']+':'+t['pin'],xy,c['component_id']))
        box=c.get('bbox') or {}
        if len(points)==3 and c.get('type') in {'transistor_npn','transistor_pnp','mosfet_n','mosfet_p','opamp'}:
            cv2.fillConvexPoly(mask,cv2.convexHull(np.array(points,np.int32)),0)
        elif c.get('type')!='gnd' and all(k in box for k in ('x1','y1','x2','y2')):
            margin=max(3,min(12,min(box['x2']-box['x1'],box['y2']-box['y1'])*.15))
            a=(max(0,math.ceil(box['x1']+margin)),max(0,math.ceil(box['y1']+margin)))
            b=(min(w-1,math.floor(box['x2']-margin)),min(h-1,math.floor(box['y2']-margin)))
            if a[0]<=b[0] and a[1]<=b[1]:cv2.rectangle(mask,a,b,0,-1)
    # A three-pixel portal allows ink at the reviewed pin; it does not restore
    # the internal fork, plates or resistor stroke removed above.
    for _,p,_ in anchors:
        x,y=p;xa,xb=max(0,x-2),min(w,x+3);ya,yb=max(0,y-2),min(h,y+3)
        mask[ya:yb,xa:xb]=(ink[ya:yb,xa:xb]>0)
    for n in protected_nodes:
        p=n.get('position') or {}
        if 'x' in p and 'y' in p:cv2.circle(mask,(round(p['x']),round(p['y'])),8,0,-1)
    for s in segments:
        points=s.get('points') or []
        if len(points)<2:continue
        for p in (points[0],points[-1]):anchors.append(('W:'+s['wire_id'],(round(p['x']),round(p['y'])),None))
    _,labels=cv2.connectedComponents(mask,connectivity=8);groups=defaultdict(list)
    for key,p,cid in anchors:
        x,y=p;xa,xb=max(0,x-3),min(w,x+4);ya,yb=max(0,y-3),min(h,y+4)
        yy,xx=np.nonzero(mask[ya:yb,xa:xb])
        if not len(xx):continue
        choices=sorted((math.hypot(a+xa-x,b+ya-y),(a+xa,b+ya),int(labels[b+ya,a+xa])) for a,b in zip(xx,yy))
        # An anchor beside two distinct raw traces cannot choose one by luck.
        near=[v for v in choices if v[0]<=choices[0][0]+.5]
        if len({v[2] for v in near})!=1:continue
        _,point,label=choices[0];groups[label].append((key,point,cid))
    parent=dict(roots)
    def find(key):
        parent.setdefault(key,key)
        while parent[key]!=key:parent[key]=parent[parent[key]];key=parent[key]
        return key
    recovered=[]
    for members in groups.values():
        if any(key in blocked_anchors for key,_,_ in members):continue
        # A surviving internal stroke must not short two pins of one symbol.
        pin_components=[cid for _,_,cid in members if cid is not None]
        if len(pin_components)!=len(set(pin_components)):continue
        candidates=[]
        for i,(a,p,ca) in enumerate(members):
            for b,q,cb in members[i+1:]:
                distance=math.dist(p,q)
                if a==b or find(a)==find(b) or distance>180:continue
                candidates.append((distance,a,b,p,q))
        for distance,a,b,p,q in sorted(candidates):
            if find(a)==find(b):continue
            traced=_trace(mask,p,q,min(240,max(24,distance*2.5+12)))
            if not traced:continue
            path,length=traced
            simplified=cv2.approxPolyDP(np.array(path,np.float32),1.2,False).reshape(-1,2)
            parent[find(a)]=find(b)
            recovered.append(dict(anchors=[a,b],length_px=round(length,3),
                                  points=[dict(x=int(x),y=int(y)) for x,y in simplified],
                                  evidence='continuous_source_ink_outside_pin_hulls'))
    return recovered
