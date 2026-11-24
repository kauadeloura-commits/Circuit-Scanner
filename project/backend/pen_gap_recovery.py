"""Join very short pen interruptions between compatible external conductors.

The photo is inspected at its original resolution. Without a measured scale,
2 mm is a size hint, estimated from a 0.5 mm pen stroke, never an exact ruler.
Parallel tracks, symbol bodies and reviewed disconnected crossings remain
barriers. The original image and upstream wire graph are never modified.
"""
from collections import defaultdict
import math
import cv2
import numpy as np
from backend.junction_detection import _nearest_on_polyline

VERSION='short-pen-gaps-1'
LIMIT_MM=2.0


def _point(position):
    try:
        p=np.array([float(position['x']),float(position['y'])])
        return p if np.all(np.isfinite(p)) else None
    except (KeyError,TypeError,ValueError):return None


def _normal(p,box):
    return np.array(min([(p[0]-box['x1'],(-1.,0.)),(box['x2']-p[0],(1.,0.)),
                         (p[1]-box['y1'],(0.,-1.)),(box['y2']-p[1],(0.,1.))],key=lambda v:v[0])[1])


def gap_policy(ink,wires):
    settings=wires.get('pen_gap_settings') or {}
    report=dict(version=VERSION,enabled=settings.get('enabled',True) is not False,
                maximum_gap_mm=LIMIT_MM,physical_scale_known=False,
                inspection='original_resolution_local_strokes')
    if not report['enabled']:return report
    measured=settings.get('pixels_per_mm')
    if isinstance(measured,(int,float)) and math.isfinite(measured) and measured>0:
        return dict(report,physical_scale_known=True,scale_source='measured',
                    pixels_per_mm=float(measured),maximum_gap_px=float(measured)*LIMIT_MM)
    distance=cv2.distanceTransform((ink>0).astype(np.uint8),cv2.DIST_L2,5)
    samples=[];h,w=ink.shape
    for segment in wires.get('segments') or []:
        points=segment.get('points') or []
        for a,b in zip(points,points[1:]):
            p=_point(a);q=_point(b)
            if p is None or q is None:continue
            for t in (.25,.5,.75):
                x,y=np.rint(p*(1-t)+q*t).astype(int)
                if 0<=x<w and 0<=y<h:
                    value=float(distance[max(0,y-1):min(h,y+2),max(0,x-1):min(w,x+2)].max())
                    if 0<value<17:samples.append(max(1.,2*value-1))
    if not samples:
        peaks=(distance>0)&(distance<17)&(distance>=cv2.dilate(distance,np.ones((3,3),np.uint8))-.01)
        samples=(2*distance[peaks]-1).clip(1).tolist()
    width=float(np.median(samples)) if samples else 1.
    # A crop must not turn the allowed interruption into a large open space.
    # Both limits scale with a resized photo; display zoom changes neither.
    maximum=min(4*width,max(2.,min(h,w)*.03))
    return dict(report,scale_source='estimated_from_pen_stroke',assumed_pen_width_mm=.5,
                pen_width_px=round(width,3),pixels_per_mm=width/.5,
                maximum_gap_px=round(maximum,3))


def _barriers(shape,components,protected_nodes):
    blocked=np.zeros(shape,np.uint8)
    for c in components:
        box=c.get('bbox') or {}
        if not all(k in box for k in ('x1','y1','x2','y2')):continue
        terminals=[p for t in c.get('terminals',[]) if (p:=_point(t.get('position') or {})) is not None]
        if len(terminals)==3 and c.get('type') in {'transistor_npn','transistor_pnp','mosfet_n','mosfet_p','opamp'}:
            cv2.fillConvexPoly(blocked,cv2.convexHull(np.rint(terminals).astype(np.int32)),1)
        elif c.get('type')!='gnd':
            margin=max(3,min(12,min(box['x2']-box['x1'],box['y2']-box['y1'])*.15))
            a=(math.ceil(box['x1']+margin),math.ceil(box['y1']+margin))
            b=(math.floor(box['x2']-margin),math.floor(box['y2']-margin))
            if a[0]<=b[0] and a[1]<=b[1]:cv2.rectangle(blocked,a,b,1,-1)
        for p in terminals:cv2.circle(blocked,tuple(np.rint(p).astype(int)),2,0,-1)
    for node in protected_nodes:
        p=_point(node.get('position') or {})
        if p is not None:cv2.circle(blocked,tuple(np.rint(p).astype(int)),8,1,-1)
    return blocked


def _hits(mask,points):
    xy=np.rint(points).astype(int);h,w=mask.shape
    valid=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<w)&(xy[:,1]<h)
    result=np.zeros(len(xy),bool)
    result[valid]=mask[xy[valid,1],xy[valid,0]]>0
    return result


def _nearest_ink(mask,p):
    x,y=np.rint(p).astype(int);h,w=mask.shape
    if not(0<=x<w and 0<=y<h):return None
    ya,xa=max(0,y-2),max(0,x-2)
    yy,xx=np.nonzero(mask[ya:min(h,y+3),xa:min(w,x+3)])
    if not len(xx):return None
    points=np.column_stack((xx+xa,yy+ya)).astype(float)
    return min(points,key=lambda q:float(np.linalg.norm(q-p)))


def _support(mask,p,inward,length):
    samples=p+np.linspace(0,length,max(4,math.ceil(length)+1))[:,None]*inward
    return float(np.mean(_hits(mask,samples)))>=.65


def _interruption(mask,blocked,p,q,maximum):
    distance=float(np.linalg.norm(q-p))
    if distance<1:return None
    points=np.linspace(p,q,max(3,math.ceil(distance)+1))
    if np.any(_hits(blocked,points)):return None
    direction=(q-p)/distance;side=np.array([-direction[1],direction[0]])
    hits=_hits(mask,points)|_hits(mask,points+side)|_hits(mask,points-side)
    if not hits[0] or not hits[-1]:return None
    runs=[];start=None
    for i,value in enumerate(hits):
        if not value and start is None:start=i
        if value and start is not None:
            left,right=start-1,i
            length=float(np.linalg.norm(points[right]-points[left]))
            runs.append((length,points[left],points[right]));start=None
    if not runs:return None
    # Several small missing pixels must not collectively cross a wide opening.
    if sum(r[0] for r in runs)>=maximum-1e-6:return None
    return max(runs,key=lambda r:r[0])


def recover_pen_gaps(ink,components,wires,connections,protected_nodes,roots):
    mask=(ink>0).astype(np.uint8);policy=gap_policy(mask,wires)
    if not policy['enabled']:return [],policy
    maximum=policy['maximum_gap_px'];blocked=_barriers(mask.shape,components,protected_nodes)
    mask[blocked>0]=0
    by_component={str(c.get('component_id') or c.get('detection_id')):c for c in components}
    records={r['terminal_ref']:r for r in connections}
    parent=dict(roots)
    def find(key):
        parent.setdefault(key,key)
        while parent[key]!=key:parent[key]=parent[parent[key]];key=parent[key]
        return key
    anchors=[];incident=defaultdict(int)
    segments=sorted(wires.get('segments') or [],key=lambda s:str(s['wire_id']))
    for s in segments:
        for name in ('start_node','end_node'):
            if s.get(name):incident[str(s[name])]+=1
    support=max(3.,min(7.,policy.get('pen_width_px',3)*1.3))
    def add(key,p,normal,cid=None):
        actual=_nearest_ink(mask,p)
        if actual is None or not _support(mask,actual,-normal,support):return
        anchors.append(dict(key=key,p=actual,normal=normal,cid=cid,index=len(anchors)))
    for s in segments:
        points=s.get('points') or []
        if len(points)<2 or s.get('orientation')=='closed_loop_or_curve':continue
        for side,name in ((0,'start_node'),(-1,'end_node')):
            if s.get(name) and incident[str(s[name])]!=1:continue
            p=_point(points[side]);normal=None
            if p is None:continue
            following=points[1:] if side==0 else list(reversed(points[:-1]))
            for point in following:
                neighbor=_point(point)
                if neighbor is None:continue
                delta=p-neighbor;length=float(np.linalg.norm(delta))
                if length>=max(2,support):normal=delta/length;break
            if normal is not None:add('W:'+str(s['wire_id']),p,normal)
    for ref,record in sorted(records.items()):
        if record['status'] in {'ambiguous','invalid_position'}:continue
        p=_point(record.get('position') or {});c=by_component.get(record['component_id']);box=(c or {}).get('bbox') or {}
        if p is not None and all(k in box for k in ('x1','y1','x2','y2')):add('T:'+ref,p,_normal(p,box),record['component_id'])
    candidates={}
    def candidate(a,key,q,normal=None,cid=None,target_index=None):
        if a['key']==key or find(a['key'])==find(key) or (cid is not None and cid==a['cid']):return
        p=a['p'];q=_nearest_ink(mask,q)
        if q is None:return
        delta=q-p;distance=float(np.linalg.norm(delta))
        # Masked leads can contain visible ink before the actual pen break.
        search=max(maximum,80.) if a['cid'] is not None or cid is not None else maximum
        if not 1<=distance<=search:return
        direction=delta/distance
        if float(direction@a['normal'])<.65:return
        if normal is not None and float(-direction@normal)<.65:return
        if a['cid'] is not None and cid is not None:
            ra=records[a['key'][2:]];rb=records[key[2:]]
            if ra['status']!='unconnected' and rb['status']!='unconnected':return
        gap=_interruption(mask,blocked,p,q,maximum)
        if gap is None:return
        length,left,right=gap
        identity=tuple(sorted((a['key'],key)))+(tuple(np.rint(left).astype(int)),tuple(np.rint(right).astype(int)))
        item=dict(anchors=[a['key'],key],length_px=round(length,3),estimated_gap_mm=round(length/policy['pixels_per_mm'],3),
                  points=[dict(x=float(left[0]),y=float(left[1])),dict(x=float(right[0]),y=float(right[1]))],
                  evidence='short_pen_interruption_between_facing_external_strokes',_score=length,
                  _endpoints=[a['index']]+([target_index] if target_index is not None else []))
        if identity not in candidates or item['_score']<candidates[identity]['_score']:candidates[identity]=item
    for i,a in enumerate(anchors):
        for b in anchors[i+1:]:candidate(a,b['key'],b['p'],b['normal'],b['cid'],b['index'])
        # A broken branch may stop immediately before the middle of a bus.
        for s in segments:
            key='W:'+str(s['wire_id'])
            if a['key']==key or find(a['key'])==find(key):continue
            near=_nearest_on_polyline(a['p'],s)
            if not near or near['distance']>max(maximum,80. if a['cid'] is not None else maximum):continue
            q=np.array(near['point']);points=s['points']
            if min(np.linalg.norm(q-_point(points[0])),np.linalg.norm(q-_point(points[-1])))<support:continue
            angle=math.radians(near['angle']);tangent=np.array([math.cos(angle),math.sin(angle)])
            delta=q-a['p'];distance=float(np.linalg.norm(delta))
            if distance<1 or abs(float((delta/distance)@tangent))>.55:continue
            if not(_support(mask,q,tangent,support) and _support(mask,q,-tangent,support)):continue
            candidate(a,key,q)
    choices=defaultdict(dict)
    for item in candidates.values():
        for index in item['_endpoints']:
            other=item['anchors'][1] if anchors[index]['key']==item['anchors'][0] else item['anchors'][0]
            root=find(other);previous=choices[index].get(root)
            if previous is None or item['_score']<previous:choices[index][root]=item['_score']
    ambiguous={index for index,scores in choices.items() if len(scores)>1 and sorted(scores.values())[1]-min(scores.values())<=max(1.,maximum*.12)}
    recovered=[]
    for item in sorted(candidates.values(),key=lambda c:(c['_score'],sorted(c['anchors']))):
        if any(index in ambiguous for index in item['_endpoints']):continue
        if any(item['_score']>min(choices[index].values())+1e-6 for index in item['_endpoints']):continue
        a,b=item['anchors']
        if find(a)==find(b):continue
        parent[find(a)]=find(b)
        item.pop('_score');item.pop('_endpoints');recovered.append(item)
    policy.update(recovered_count=len(recovered),ambiguous_endpoint_count=len(ambiguous))
    return recovered,policy
