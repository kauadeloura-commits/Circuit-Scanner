"""Locate NPN leads on original ink, independent of screen orientation.

The global wire skeleton can join a notebook rule to a nearby lead. Its
cluster centroid is therefore context, never proof that a pin lies on ink.
"""
from itertools import combinations, permutations
from collections import deque
import math

import cv2
import numpy as np
from backend.resistor_geometry import cached_vision

from backend.terminal_identity import _crop, _lines, _sample, _support
from backend.terminal_detection import _line_segment_intersection_params, _angle_distance_180
from backend.wire_detection import _skeletonize


def _arrow_wings(mask, origin, near, far, extension):
    vector = far-near
    shaft_length = float(np.linalg.norm(vector))
    if shaft_length < 10:
        return {"score": 0.0, "direction": None}
    direction = vector/shaft_length
    normal = np.array([-direction[1], direction[0]])
    # LSD often stops at the first wing of a filled arrow. Extend the search
    # along the same shaft to include the tip, rather than changing branches.
    length = shaft_length + max(12.0, shaft_length*extension)
    start = near-origin
    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
    delta = np.stack((xx-start[0], yy-start[1]), axis=2)
    along, across = delta@direction, delta@normal
    zone = (along>length*.06) & (along<length*.92) & (np.abs(across)<max(15, length*.28))
    gray = cv2.GaussianBlur(255-np.where(zone, mask, 0).astype(np.uint8), (3,3), 0)
    edges = cv2.Canny(gray, 20, 85)
    raw = cv2.HoughLinesP(edges, 1, np.pi/180, 4, minLineLength=4, maxLineGap=5)
    if raw is None:
        return {"score": 0.0, "direction": None}
    wings = []
    for x1,y1,x2,y2 in np.asarray(raw).reshape(-1,4):
        a,b = np.array([x1,y1], float), np.array([x2,y2], float)
        v = b-a
        size = float(np.linalg.norm(v))
        if size < 3 or size > length*.5:
            continue
        angle = math.degrees(math.acos(float(np.clip(abs(np.dot(v/size, direction)), 0, 1))))
        if 15 < angle < 88:
            wings.append((a,b,size))
    best = {"score": 0.0, "direction": None}
    for a,b in combinations(wings,2):
        va,vb = a[1]-a[0], b[1]-b[0]
        if abs(va[0]*vb[1]-va[1]*vb[0])/(a[2]*b[2]) < .3:
            continue
        gap,ia,ib = min((float(np.linalg.norm(a[i]-b[j])),i,j) for i in (0,1) for j in (0,1))
        if gap > max(4, length*.07):
            continue
        tip = (a[ia]+b[ib])/2
        ta,tb = a[1-ia]-tip, b[1-ib]-tip
        lateral_a,lateral_b = float(np.dot(ta,normal)), float(np.dot(tb,normal))
        # A bend in a wire has one arm. An arrow has two wings on opposed
        # sides of the shaft, both pointing back from the same tip.
        if lateral_a*lateral_b >= 0 or min(abs(lateral_a),abs(lateral_b)) < 2.5:
            continue
        if abs(np.dot(tip-start,normal)) > max(5, length*.06):
            continue
        if not length*.06 < np.dot(tip-start,direction) < length*1.03:
            continue
        axial_a,axial_b = float(np.dot(ta,direction)), float(np.dot(tb,direction))
        if axial_a*axial_b <= 0 or min(abs(axial_a),abs(axial_b)) < length*.025:
            continue
        score = min(abs(lateral_a),abs(lateral_b))+min(abs(axial_a),abs(axial_b))-gap
        if score > best["score"]:
            best = {"score": float(score), "direction": "outward" if axial_a<0 else "inward",
                    "tip": [round(float(v),2) for v in tip+origin]}
    return best


_HEAD_TEMPLATES = None


def _head_templates():
    global _HEAD_TEMPLATES
    if _HEAD_TEMPLATES is not None:
        return _HEAD_TEMPLATES
    groups = {}
    for size in (19,25,31,37,43):
        templates = []
        for thickness in (3,5,7):
            stem = np.zeros((size,size),np.uint8)
            middle = size//2
            cv2.line(stem,(0,middle),(size-1,middle),1,thickness)
            circle = stem.copy()
            cv2.circle(circle,(middle,middle),int(size*.31),1,-1)
            templates.append(('dot',circle.astype(np.float32)))
            for kind in ('outward','inward'):
                tip,base = (int(size*.73),int(size*.27)) if kind=='outward' else (int(size*.27),int(size*.73))
                for tip_shift in (-6,0,6):
                    for base_shift in (-4,0,4):
                        for spread in (.36,.44):
                            points = [(tip,middle+tip_shift),(base,middle+base_shift-int(size*spread)),
                                      (base,middle+base_shift+int(size*spread))]
                            filled = stem.copy()
                            cv2.fillPoly(filled,[np.array(points,np.int32)],1)
                            templates.append((kind,filled.astype(np.float32)))
                            opened = stem.copy()
                            cv2.line(opened,points[0],points[1],1,thickness)
                            cv2.line(opened,points[0],points[2],1,thickness)
                            templates.append((kind,opened.astype(np.float32)))
        groups[size] = templates
    _HEAD_TEMPLATES = groups
    return groups


def _shape_scores(mask,origin,near,far):
    vector = far-near
    length = float(np.linalg.norm(vector))
    unit = vector/length
    normal = np.array([-unit[1],unit[0]])
    # A head belongs to the symbol branch. Searching beyond the external
    # contact was matching distant wire elbows and junction dots as emitters.
    yy,xx = np.mgrid[-42:43,0:int(length)+1]
    coordinates = near-origin+xx[:,:,None]*unit+yy[:,:,None]*normal
    strip = (cv2.remap(mask,coordinates[:,:,0].astype(np.float32),coordinates[:,:,1].astype(np.float32),
                       cv2.INTER_NEAREST)>0).astype(np.float32)
    result = {'outward':0.,'inward':0.,'dot':0.}
    for size,templates in _head_templates().items():
        if size>strip.shape[1]:
            continue
        # The old scores excluded these rows/columns after matching. Restrict
        # matching to the same candidate positions, retaining every template.
        x0=max(0,int(length*.25)-size//2)
        y0=max(0,42-size//2-16)
        y1=min(strip.shape[0]-size+1,42-size//2+17)
        if y1<=y0 or strip.shape[1]-x0<size:continue
        searched=strip[y0:y1+size-1,x0:]
        count = cv2.matchTemplate(searched,np.ones((size,size),np.float32),cv2.TM_CCORR)
        bounds={}
        for kind,template in templates:
            ink=float(template.sum())
            if ink not in bounds:
                bounds[ink]=float(np.max(np.minimum(count,ink)/np.maximum(1,np.maximum(count,ink))))+2e-6
            if bounds[ink]<=result[kind]:continue
            overlap = cv2.matchTemplate(searched,template,cv2.TM_CCORR)
            score = overlap/np.maximum(1,count+ink-overlap)
            result[kind] = max(result[kind],float(cv2.minMaxLoc(score)[1]))
    return result


def _arrow_evidence(mask,origin,near,far):
    shapes = _shape_scores(mask,origin,near,far)
    direction = max(('outward','inward'),key=lambda key:shapes[key])
    other = 'inward' if direction=='outward' else 'outward'
    # A round dot can produce two short tangents resembling arrow wings.
    # Compare its full silhouette before trusting those local tangents.
    if shapes['dot']>=.86 and shapes['dot']>shapes[direction]+.02:
        return {'score':0.,'direction':None,'shape_scores':shapes,'rejected_round_dot':True}
    if shapes[direction]>=.84 and shapes[direction]-shapes[other]>=.055:
        return {'score':shapes[direction]*12.,'direction':direction,'shape_scores':shapes,
                'source':'arrow_head_shape'}
    # Short and longer searches accommodate both outlined handwritten heads
    # and filled heads that interrupt the detected shaft segment.
    wings = [_arrow_wings(mask,origin,near,far,extension) for extension in (.25,.50)]
    best = max(wings,key=lambda item:item['score'])
    if shapes[direction]>=.87 and shapes[direction]-shapes['dot']>=.10 and best['score']<shapes[direction]*10:
        # A broad handwritten head can identify the emitter branch even
        # when its skew prevents a reliable inward/outward direction. Keep
        # those two decisions separate instead of choosing a distant wire.
        return {'score':shapes[direction]*10.,'direction':None,'shape_scores':shapes,
                'source':'arrow_head_branch_with_ambiguous_direction'}
    return {**best,'shape_scores':shapes,'source':'two_opposed_arrow_wings'}


def _body_geometries(mask, origin, bbox):
    scale = min(bbox['x2']-bbox['x1'], bbox['y2']-bbox['y1'])
    lines = [dict(p1=l['a'], p2=l['b'], length=l['length'], angle=math.degrees(l['angle']),
                  u=(l['b']-l['a'])/l['length']) for l in _lines(mask,origin)
             if l['length'] > max(12, scale*.13)]
    # Handwritten bars and shafts are split by their attached branches and
    # arrowheads. Join collinear pieces only when photographed ink bridges
    # the gap; a missing LSD segment does not imply a missing terminal.
    merged=[]
    for first,second in combinations(lines,2):
        if _angle_distance_180(first['angle'],second['angle'])>15:continue
        endpoints=np.array([first['p1'],first['p2'],second['p1'],second['p2']])
        center=endpoints.mean(axis=0)
        _,_,axes=np.linalg.svd(endpoints-center,full_matrices=False)
        u=axes[0];normal=np.array([-u[1],u[0]])
        if np.max(np.abs((endpoints-center)@normal))>max(5,scale*.055):continue
        projected=(endpoints-center)@u
        intervals=[sorted(projected[:2]),sorted(projected[2:])]
        gap=max(intervals[0][0],intervals[1][0])-min(intervals[0][1],intervals[1][1])
        if gap>scale*.35:continue
        a,b=center+u*projected.min(),center+u*projected.max()
        length=float(np.linalg.norm(b-a))
        if length<=max(first['length'],second['length'])*1.15:continue
        hits=[_sample(mask,p,origin,2) for p in np.linspace(a,b,max(2,int(length)))]
        if np.mean(hits)<.65:continue
        if any(np.linalg.norm(a-l['p1'])+np.linalg.norm(b-l['p2'])<8 for l in merged):continue
        merged.append(dict(p1=a,p2=b,length=length,u=u,angle=math.degrees(math.atan2(u[1],u[0]))))
    lines+=sorted(merged,key=lambda l:l['length'],reverse=True)[:24]
    found = []
    for bar in lines:
        if bar['length'] < scale*.22:
            continue
        normal = np.array([-bar['u'][1],bar['u'][0]])
        branches,leads = [],[]
        for other in lines:
            if other is bar:
                continue
            angle = _angle_distance_180(bar['angle'], other['angle'])
            inter = _line_segment_intersection_params(bar['p1'],bar['p2'],other['p1'],other['p2'])
            if inter is None:
                continue
            p,tbar,tother = inter
            p = np.array(p)
            if not (-.25<tbar<1.25 and -.25<tother<1.25):
                continue
            far = max((other['p1'],other['p2']), key=lambda q:np.linalg.norm(q-p))
            side = float(np.dot(far-p,normal))
            item = dict(p=p, far=far, side=side, attach=float(np.dot(p-bar['p1'],bar['u'])), line=other)
            if 20<angle<75 and abs(side)>scale*.12:
                branches.append(item)
            if 70<angle<110 and abs(side)>scale*.12:
                leads.append(item)
        for a,b in combinations(branches,2):
            if a['side']*b['side']<=0 or abs(a['attach']-b['attach'])<scale*.12:
                continue
            for lead in leads:
                if lead['side']*a['side']>=0:
                    continue
                lo,hi = sorted((a['attach'],b['attach']))
                if not lo-scale*.15<=lead['attach']<=hi+scale*.15:
                    continue
                if np.dot(a['far']-lead['p'],bar['u'])*np.dot(b['far']-lead['p'],bar['u'])>=0:
                    continue
                score = bar['length']*.2+a['line']['length']+b['line']['length']+lead['line']['length']*.5
                found.append((float(score),bar,a,b,lead))
    return sorted(found,key=lambda g:g[0],reverse=True)[:6]


def _external_contacts(mask, origin, bbox, skeleton, allowed_ink=None):
    local_skeleton = _skeletonize(mask)
    ports = []
    for side,start,end in (('top',(bbox['x1'],bbox['y1']),(bbox['x2'],bbox['y1'])),
                           ('bottom',(bbox['x1'],bbox['y2']),(bbox['x2'],bbox['y2'])),
                           ('left',(bbox['x1'],bbox['y1']),(bbox['x1'],bbox['y2'])),
                           ('right',(bbox['x2'],bbox['y1']),(bbox['x2'],bbox['y2']))):
        a,b = np.array(start,float),np.array(end,float)
        runs = []
        for point in np.linspace(a,b,max(2,int(np.linalg.norm(b-a)))):
            if _sample(mask,point,origin,1):
                if not runs or np.linalg.norm(point-runs[-1][-1])>3:
                    runs.append([])
                runs[-1].append(point)
        ports.extend({'point':np.mean(run,axis=0),'side':side} for run in runs)
    contacts = []
    outward = {'top':np.array([0.,-1.]),'bottom':np.array([0.,1.]),
               'left':np.array([-1.,0.]),'right':np.array([1.,0.])}
    scale = min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])
    reach = max(7,min(18,scale*.13))
    for port in ports:
        x,y = np.rint(port['point']-origin).astype(int)
        xa,ya = max(0,x-12),max(0,y-12)
        xb,yb = min(mask.shape[1],x+13),min(mask.shape[0],y+13)
        yy,xx = np.nonzero(local_skeleton[ya:yb,xa:xb])
        if len(xx)<4:
            continue
        points = np.stack((xx+xa,yy+ya),axis=1).astype(float)
        points = points[np.linalg.norm(points-(port['point']-origin),axis=1)<=12]
        if len(points)<4:
            continue
        values,vectors = np.linalg.eigh(np.cov(points.T))
        if values[-1]/max(1e-6,float(values.sum()))<.88:
            # A ruling crossing a thin lead can destroy a global PCA fit.
            # Test the through-going conductor locally instead of fitting both
            # arms of that crossing as if they belonged to the terminal.
            candidates=[]
            normal=outward[port['side']]
            for degrees in range(-70,71,10):
                angle=math.radians(degrees)
                rotation=np.array([[math.cos(angle),-math.sin(angle)],
                                   [math.sin(angle),math.cos(angle)]])
                unit=rotation@normal
                score=min(_support(mask,port['point'],unit,np.arange(2,10,2),origin),
                          _support(mask,port['point'],-unit,np.arange(2,10,2),origin))
                candidates.append((score,unit))
            score,tangent=max(candidates,key=lambda v:v[0])
            if score<.8:
                continue
        else:
            tangent = vectors[:,-1]
        if np.dot(tangent,outward[port['side']])<0:
            tangent = -tangent
        if np.dot(tangent,outward[port['side']])<.25:
            continue
        # Reject a plate/arrow touching the bbox: a lead must continue OUTSIDE
        # the body, rather than stop at the glyph itself.
        outside_reach = min(9,max(7,scale*.06))
        if (_support(mask,port['point'],tangent,np.arange(2,outside_reach,2),origin)<.70
                or _support(mask,port['point'],-tangent,np.arange(2,reach,2),origin)<.70):
            continue
        xa,ya = max(0,x-5),max(0,y-5)
        xb,yb = min(mask.shape[1],x+6),min(mask.shape[0],y+6)
        yy,xx = np.nonzero(local_skeleton[ya:yb,xa:xb])
        if not len(xx):
            continue
        pixels = np.stack((xx+xa,yy+ya),axis=1)+origin
        if allowed_ink is not None:
            local_pixels = np.rint(pixels-origin).astype(int)
            pixels = pixels[allowed_ink[local_pixels[:,1],local_pixels[:,0]]>0]
            if not len(pixels):
                continue
        point = pixels[np.argmin(np.linalg.norm(pixels-port['point'],axis=1))]
        gx,gy = np.rint(point).astype(int)
        radius = int(max(12,min(20,scale*.13)))
        context_distance=0.
        if skeleton is not None:
            sx,sy = max(0,gx-radius),max(0,gy-radius)
            yy,xx = np.nonzero(skeleton[sy:min(skeleton.shape[0],gy+radius+1),sx:min(skeleton.shape[1],gx+radius+1)])
            if not len(xx):
                continue
            context_distance = float(np.min(np.hypot(xx+sx-gx,yy+sy-gy)))
            if context_distance > radius:
                continue
        if any(np.linalg.norm(point-c['point'])<8 for c in contacts):
            continue
        contacts.append(dict(point=(float(point[0]),float(point[1])), x=float(point[0]), y=float(point[1]),
                             pixels=20, linearity=1., boundary_distance=0.,
                             original_ink=True, stage8_distance_px=round(context_distance,2)))
    return contacts


def _traced_body_geometries(mask,origin,bbox,contacts,limit=6,closing_size=5):
    """Find the bar reached by the three observed leads, including bends."""
    scale=min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])
    connected=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((closing_size,closing_size),np.uint8))>0
    height,width=mask.shape
    yy,xx=np.nonzero(connected)
    ink=np.stack((xx,yy),axis=1)
    maps=[]
    for contact in contacts:
        p=np.array(contact['point'])-origin
        if not len(ink):continue
        nearest=ink[int(np.argmin(np.linalg.norm(ink-p,axis=1)))]
        if np.linalg.norm(nearest-p)>7:continue
        distances=np.full(mask.shape,-1,np.int32)
        sx,sy=nearest.astype(int);distances[sy,sx]=0
        queue=deque([(sx,sy)])
        while queue:
            x,y=queue.popleft()
            for dx,dy in ((-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)):
                nx,ny=x+dx,y+dy
                if 0<=nx<width and 0<=ny<height and connected[ny,nx] and distances[ny,nx]<0:
                    distances[ny,nx]=distances[y,x]+1;queue.append((nx,ny))
        maps.append((contact,distances))
    found=[]
    bar_lines=_lines(mask,origin)
    for first,second in combinations(bar_lines.copy(),2):
        if _angle_delta_radians(first['angle'],second['angle'])>math.radians(12):continue
        points=np.array([first['a'],first['b'],second['a'],second['b']]);center=points.mean(axis=0)
        _,_,axes=np.linalg.svd(points-center,full_matrices=False);unit=axes[0];normal=np.array([-unit[1],unit[0]])
        if np.max(np.abs((points-center)@normal))>max(4,scale*.04):continue
        projection=(points-center)@unit
        start,end=center+unit*projection.min(),center+unit*projection.max()
        length=float(np.linalg.norm(end-start))
        if length<max(first['length'],second['length'])*1.15:continue
        if np.mean([_sample(mask,p,origin,2) for p in np.linspace(start,end,max(2,int(length)))])<.85:continue
        bar_lines.append(dict(a=start,b=end,length=length,angle=math.atan2(unit[1],unit[0])%math.pi))
    for line in bar_lines:
        if line['length']<scale*.22:continue
        a,b=line['a'],line['b'];u=(b-a)/line['length'];normal=np.array([-u[1],u[0]])
        bar=dict(p1=a,p2=b,u=u,length=line['length'],angle=math.degrees(math.atan2(u[1],u[0])))
        targets=[]
        for p in np.linspace(a,b,max(2,int(line['length']))):
            x,y=np.rint(p-origin).astype(int)
            for dx in (-2,0,2):
                for dy in (-2,0,2):
                    if 0<=x+dx<width and 0<=y+dy<height and connected[y+dy,x+dx]:targets.append((x+dx,y+dy))
        if not targets:continue
        targets=np.array(targets)
        branches=[];leads=[]
        for contact,distances in maps:
            values=distances[targets[:,1],targets[:,0]]
            valid=values>=0
            if not np.any(valid):continue
            options=targets[valid];values=values[valid]
            local=options[int(np.argmin(values))]
            p=local+origin;far=np.array(contact['point'])
            vector=far-p;length=float(np.linalg.norm(vector))
            if length<scale*.16:continue
            side=float(vector@normal)
            angle=_angle_distance_180(bar['angle'],math.degrees(math.atan2(vector[1],vector[0])))
            item=dict(p=p,far=far,side=side,attach=float((p-a)@u),
                      line=dict(length=length),path_length=int(values.min()))
            if 15<angle<78 and abs(side)>scale*.12:branches.append(item)
            if 65<angle<=90 and abs(side)>scale*.12:leads.append(item)
        for first,second in combinations(branches,2):
            if first['side']*second['side']<=0 or abs(first['attach']-second['attach'])<scale*.10:continue
            for lead in leads:
                if lead['side']*first['side']>=0:continue
                lo,hi=sorted((first['attach'],second['attach']))
                if not lo-scale*.15<=lead['attach']<=hi+scale*.15:continue
                if (first['far']-lead['p'])@u * ((second['far']-lead['p'])@u)>=0:continue
                lengths=[v['path_length'] for v in (first,second,lead)]
                direct=[v['line']['length'] for v in (first,second,lead)]
                # Reject a path that reaches the bar through another wire or
                # via a notebook rule instead of its own short symbol branch.
                if any(path>distance*1.65+8 for path,distance in zip(lengths,direct)):continue
                score=sum(direct)+bar['length']*.2-sum(lengths)*.2
                found.append((score,bar,first,second,lead))
    return sorted(found,key=lambda g:g[0],reverse=True)[:limit]


def _angle_delta_radians(a,b):
    return abs((a-b+math.pi/2)%math.pi-math.pi/2)


def _branch_arrow_evidence(mask,origin,bar,branch,other,bbox):
    """Score the head on its own branch, excluding the base and other lead."""
    yy,xx=np.indices(mask.shape);points=np.stack((xx+origin[0],yy+origin[1]),axis=2)
    normal=np.array([-bar['u'][1],bar['u'][0]])
    side=1 if np.dot(branch['far']-bar['p1'],normal)>0 else -1
    above=(points-bar['p1'])@normal*side
    def distance(item):
        vector=item['far']-item['p'];length=np.linalg.norm(vector);unit=vector/max(1,length)
        along=np.clip((points-item['p'])@unit,0,length)
        return np.linalg.norm(points-(item['p']+along[:,:,None]*unit),axis=2)
    own,neighbor=distance(branch),distance(other)
    isolated=mask.copy();isolated[(above<=3)|(own>neighbor+2)]=0
    inside=(points[:,:,0]>=bbox['x1'])&(points[:,:,0]<=bbox['x2'])&(points[:,:,1]>=bbox['y1'])&(points[:,:,1]<=bbox['y2'])
    isolated[~inside]=0
    evidence=_arrow_evidence(isolated,origin,branch['p'],branch['far'])
    shapes=evidence.get('shape_scores',{})
    direction=max(('outward','inward'),key=lambda key:shapes.get(key,0))
    head=shapes.get(direction,0);contrast=head-shapes.get('dot',0)
    if head>=.80 and contrast>=.12 and (abs(shapes.get('outward',0)-shapes.get('inward',0))>=.035 or evidence.get('tip') is not None and evidence['score']>=5):
        # The head identifies the emitter even when the handwritten tip is
        # asymmetric. Direction is a separate measurement, never a reason
        # to move the emitter onto the branch without the head.
        evidence['score']=max(evidence['score'],head*12)
        other_direction='inward' if direction=='outward' else 'outward'
        if abs(head-shapes.get(other_direction,0))<.055:evidence['direction']=None
        evidence['source']='branch_head_silhouette_and_wings'
    elif contrast<.075 or evidence.get('source')=='arrow_head_branch_with_ambiguous_direction' and contrast<.18:
        evidence.update(score=0.,direction=None,rejected_base_or_plain_branch=True)
    if abs(shapes.get('outward',0)-shapes.get('inward',0))<.055:
        evidence['direction']=None
    # A filled head is wider than its own shaft. Measure it away from the
    # base bar and external wire elbow; their corners are not arrow wings.
    vector=branch['far']-branch['p'];length=float(np.linalg.norm(vector));unit=vector/max(1,length)
    along=(points-branch['p'])@unit
    across=np.abs((points-branch['p'])@np.array([-unit[1],unit[0]]))
    scale=min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])
    head_mask=np.where((along>length*.13)&(along<length*.90)&(across<max(20,scale*.18))&(above>8),isolated,0).astype(np.uint8)
    widths=cv2.distanceTransform(head_mask,cv2.DIST_L2,5)
    shaft=widths[head_mask>0]
    baseline=float(np.percentile(shaft,85)) if len(shaft) else 0.
    peak=float(widths.max()) if widths.size else 0.
    ratio=peak/max(1,baseline)
    if peak>=4.8 and ratio>=1.60 and head-shapes.get('dot',0)>=.015:
        py,px=np.unravel_index(int(widths.argmax()),widths.shape)
        center=np.array([px,py])+origin;relative=points-center
        axial=relative@unit;lateral=relative@np.array([-unit[1],unit[0]])
        wings=(head_mask>0)&(np.abs(axial)<peak*1.8)&(np.abs(lateral)<peak*1.8)&(np.abs(lateral)>peak*.5)
        samples=axial[wings];skew=float(np.mean(((samples-samples.mean())/max(1,samples.std()))**3)) if len(samples)>=12 else 0.
        # A circular junction dot is symmetric along the shaft. A triangular
        # head tapers; thickness alone must not certify a dot as an emitter.
        if abs(skew)>=.045:
            evidence.update(score=max(evidence['score'],peak*1.25),source='filled_head_wider_than_its_own_shaft',head_center=center.tolist(),head_width_ratio=round(ratio,3),head_axial_skew=round(skew,4))
    if evidence.get('tip') is not None:
        x,y=evidence['tip'];margin=max(3,min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])*.015)
        bar_distance=abs(float((np.array([x,y])-bar['p1'])@normal))
        if bar_distance<max(6,scale*.09) and evidence.get('source')!='filled_head_wider_than_its_own_shaft':
            evidence.update(score=0.,direction=None,rejected_base_bar_corner=True)
        if min(x-bbox['x1'],bbox['x2']-x,y-bbox['y1'],bbox['y2']-y)<margin:
            # A lead clipped at the detection boundary can look like two
            # wings. A complete arrowhead must lie inside the symbol box.
            evidence.update(score=0.,direction=None,rejected_clipped_wire_corner=True)
    evidence['metric']='exclusive_branch_head_without_base_bar'
    return evidence


def detect_npn_wire_contacts(image, bbox, skeleton, color_reference=None):
    return _detect_colored_bjt_contacts(image,bbox,skeleton,color_reference)


def _detect_colored_bjt_contacts(image, bbox, skeleton, color_reference=None):
    return _cached_colored_bjt_contacts(image,bbox,skeleton,color_reference,color_reference is image)


@cached_vision
def _cached_colored_bjt_contacts(image,bbox,skeleton,color_reference,same_reference):
    """Prefer the approved pen color, retaining raw edges if it fragments ink."""
    if same_reference:color_reference=image
    if color_reference is not None and color_reference is not image:
        found,meta=_detect_npn_wire_contacts(color_reference,bbox,skeleton)
        if found is not None and meta.get('arrow_confirmed'):
            # Color filtering can break a light emitter stem and substitute a
            # nearby letter. If a proposed lead misses the wire skeleton,
            # compare the original photograph's complete three-lead solution.
            # Prefer it only with a verified arrow and three closer conductors.
            distances=meta.get('stage8_distances_px') or []
            if skeleton is not None and distances and max(distances)>5:
                raw,proof=_detect_npn_wire_contacts(image,bbox,skeleton)
                native=proof.get('stage8_distances_px') or []
                if raw is not None and proof.get('arrow_confirmed') and native and max(native)<=5:
                    proof['image_pass']='original_ink_complete_leads_over_fragmented_color'
                    return raw,proof
            meta['image_pass']='component_ink';return found,meta
    found,meta=_detect_npn_wire_contacts(image,bbox,skeleton,color_reference)
    if found is not None:
        meta['image_pass']='original_ink_paths'
        return found,meta
    if color_reference is not None and color_reference is not image:
        found, meta = _detect_npn_wire_contacts(color_reference, bbox, skeleton)
        if found is not None:
            meta['image_pass'] = 'component_ink'
            return found, meta
    meta['image_pass'] = 'original_edges_with_component_ink_contacts'
    return found, meta


def detect_bjt_wire_contacts(image,bbox,skeleton,component_type,color_reference=None):
    """Use the same orientation-independent bar/arrow evidence for NPN and PNP."""
    found,meta=_detect_colored_bjt_contacts(image,bbox,skeleton,color_reference)
    meta['source']='bjt_original_ink_bar_branches_and_external_leads'
    if found is not None:
        contacts,evidence=found
        expected='outward' if component_type=='transistor_npn' else 'inward'
        conflict=bool(meta.get('detected_arrow_direction') and meta['detected_arrow_direction']!=expected)
        meta['direction_conflict']=conflict
        evidence['detector_class']=component_type
        evidence['source']=meta['source']
        for h in evidence.get('hypotheses') or []:
            h['direction_conflict']=conflict
    return found,meta


def _detect_npn_wire_contacts(image, bbox, skeleton, color_reference=None,*,arrow_mask_image=None,contact_image=None):
    """Return three ink-centered contacts and independent B/E evidence.

    Empty output preserves the established fallback for unreadable symbols.
    This detector is invoked for NPN only; the validated PNP flow is retained.
    """
    meta = {"source":"npn_original_ink_bar_branches_and_external_leads", "body_found":False}
    mask,origin = _crop(image,bbox)
    # Remove thin broken residual notebook rules before clustering contacts.
    # The original mask remains the coordinate reference; no image is resized.
    clean = cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    arrow_clean=_crop(arrow_mask_image,bbox)[0] if arrow_mask_image is not None else clean
    geometries = _body_geometries(clean,origin,bbox)
    if not geometries:
        geometries = _body_geometries(mask,origin,bbox)
    allowed_ink = None
    if color_reference is not None and color_reference is not image:
        allowed_ink,_ = _crop(color_reference,bbox)
    from backend.terminal_wire_contacts import _contacts
    observed=_contacts(contact_image if contact_image is not None else image,bbox,skeleton)
    if allowed_ink is not None:
        observed=[p for p in observed if _sample(allowed_ink,np.array(p['point']),origin,3)]
    contacts=[dict(p,point=tuple(p['point']),x=float(p['point'][0]),y=float(p['point'][1]),
                   pixels=20,linearity=1.,boundary_distance=0.,original_ink=True)
              for p in observed]
    if len(contacts)>=3:
        geometries=geometries+_traced_body_geometries(mask,origin,bbox,contacts)
    if not geometries and len(contacts)==3:
        from backend.drawing_mode import CURRENT
        if CURRENT.get()=='handwritten':
            # This is the internal fork drawing only. The three external
            # contacts still come from unmodified photographed conductor ink.
            geometries=_traced_body_geometries(mask,origin,bbox,contacts,closing_size=7)
    meta['external_contact_count'] = len(contacts)
    if not geometries or len(contacts)<3:
        meta['reason']='npn_body_or_three_original_ink_leads_not_proven'
        return None,meta
    scale = min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])
    solutions = []
    arrow_cache = {}
    candidates=[]
    for order,(score,bar,a,b,lead) in enumerate(geometries):
        anchors = (lead['far'],a['far'],b['far'])
        ranked = []
        for indexes in permutations(range(len(contacts)),3):
            distances = [float(np.linalg.norm(np.array(contacts[i]['point'])-anchor)) for i,anchor in zip(indexes,anchors)]
            if max(distances) > max(22,scale*.48):
                continue
            context=sum(float(contacts[i]['stage8_distance_px']) for i in indexes)
            ranked.append((sum(distances)+context*1.5,indexes))
        if not ranked:
            continue
        total,indexes = min(ranked)
        # An arrow contributes at most 25 to this hypothesis's final quality.
        # Once its best possible score cannot beat an evaluated hypothesis,
        # thousands of head-template comparisons cannot change the winner.
        upper_quality=score-total*.65+25
        candidates.append((upper_quality,order,score,total,bar,a,b,lead,indexes))
    # Evaluate the strongest physical fit first. Every remaining hypothesis
    # retains the same score bound and templates; only provably losing work
    # is omitted. Original order breaks ties exactly as before.
    for upper_quality,order,score,total,bar,a,b,lead,indexes in sorted(candidates,key=lambda c:(-c[0],c[1])):
        if solutions and upper_quality<max(s[0] for s in solutions):
            continue
        arrows = []
        for branch,other in ((a,b),(b,a)):
            key = tuple(np.rint(np.r_[bar['p1'],bar['p2'],branch['p'],branch['far'],other['p'],other['far']]).astype(int))
            if key not in arrow_cache:
                arrow_cache[key] = _branch_arrow_evidence(arrow_clean,origin,bar,branch,other,bbox)
            arrows.append(arrow_cache[key])
        strong = int(arrows[1]['score']>arrows[0]['score'])
        maximum,minimum = arrows[strong]['score'],arrows[1-strong]['score']
        arrow_confirmed = maximum>=5.0 and maximum>=max(1.,minimum)*1.5
        quality = score-total*.65+(25 if arrow_confirmed else 0)
        solutions.append((quality,bar,a,b,lead,indexes,arrows,strong,arrow_confirmed,order))
    if not solutions:
        meta['reason']='npn_external_leads_do_not_match_body'
        return None,meta
    _,bar,a,b,lead,indexes,arrows,strong,confirmed,_ = max(solutions,key=lambda s:(s[0],-s[-1]))
    chosen = [contacts[i] for i in indexes]
    from backend.handwritten_local_bjt import reconcile_weak_elbow
    strong,pose_role_proof=reconcile_weak_elbow(image,bbox,(a,b),chosen,arrows,strong)
    bpoint = chosen[0]['point']
    epoint = chosen[strong+1]['point'] if confirmed else None
    direction = arrows[strong]['direction'] if confirmed else None
    conflict = bool(direction and direction!='outward')
    meta.update(body_found=True, arrow_confirmed=confirmed, reason='npn_three_original_ink_leads',
                base_lead_near=lead['p'].tolist(),base_lead_far=lead['far'].tolist(),
                base_bar_points=[bar['p1'].tolist(),bar['p2'].tolist()],
                branch_anchors=[dict(near=c['p'].tolist(),far=c['far'].tolist()) for c in (a,b)],
                base_bar_angle_degrees=round(float(bar['angle']),2), arrow_evidence=arrows,
                branch_head_margin_normalized=round(float((arrows[strong]['score']-arrows[1-strong]['score'])/max(1,arrows[strong]['score']+arrows[1-strong]['score'])),4),
                identity_metric='exclusive_branch_head_then_base_then_remaining_collector',
                detected_arrow_direction=direction, direction_conflict=conflict,
                stage8_distances_px=[p['stage8_distance_px'] for p in chosen])
    if pose_role_proof:
        meta.update(pose_role_reconciliation=pose_role_proof,identity_metric='pose_consensus_and_observed_three_lead_geometry')
    evidence = {"detector_class":"transistor_npn", "source":meta['source'],"accepted":confirmed,
        "symbol_geometry":meta, "hypotheses":[{"source":"physical_arrow_then_base_bar", "physical":True,
            "points":{"B":{"x":bpoint[0],"y":bpoint[1]}},"keypoint_confidences":{"B":.94},
            "direction_conflict":conflict}]}
    if epoint is not None:
        evidence['hypotheses'][0]['points']['E'] = {'x':epoint[0],'y':epoint[1]}
        evidence['hypotheses'][0]['keypoint_confidences']['E'] = .94
    return (chosen,evidence),meta
