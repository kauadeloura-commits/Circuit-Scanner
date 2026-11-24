"""Resistor evidence from a closed body or a continuous alternating lead path."""
import cv2
import numpy as np
from contextvars import ContextVar
from functools import wraps
import hashlib
import copy
from collections import OrderedDict
from threading import RLock

_SYMBOL_CACHE=ContextVar('symbol_analysis_cache',default=None)
_VISION_RESULTS=dict(entries=OrderedDict(),bytes=0)
_VISION_LOCK=RLock()
_VISION_CACHE_LIMIT=128*1024*1024

def session_result(key,compute):
    """Reuse a result only inside the current immutable analysis session."""
    scope=_SYMBOL_CACHE.get()
    if scope is None:return compute()
    if key not in scope:scope[key]=compute()
    return copy.deepcopy(scope[key])


def cached_vision(fn):
    """Reuse content-identical pure analysis across stages, bounded to 128 MiB."""
    def freeze(value):
        if isinstance(value,np.ndarray):
            buffer=memoryview(value) if value.flags.c_contiguous else value.tobytes()
            return ('array',value.shape,value.dtype.str,hashlib.sha256(buffer).digest())
        if isinstance(value,dict):return tuple((k,freeze(v)) for k,v in sorted(value.items()))
        if isinstance(value,(list,tuple)):return tuple(freeze(v) for v in value)
        if isinstance(value,(str,int,float,bool,type(None),np.number)):return value
        raise TypeError('uncacheable analysis argument')
    def size(value):
        if isinstance(value,np.ndarray):return value.nbytes
        if isinstance(value,dict):return sum(size(v) for v in value.values())
        if isinstance(value,(list,tuple)):return sum(size(v) for v in value)
        return 64
    @wraps(fn)
    def wrapped(*args,**kwargs):
        scope=_SYMBOL_CACHE.get()
        if scope is None:return fn(*args,**kwargs)
        from backend.drawing_mode import CURRENT
        try:key=(fn,CURRENT.get(),freeze(args),freeze(kwargs))
        except TypeError:return fn(*args,**kwargs)
        state=_VISION_RESULTS
        entries=state['entries']
        with _VISION_LOCK:
            if key in entries:
                value,weight=entries.pop(key);entries[key]=(value,weight)
                return copy.deepcopy(value)
        value=fn(*args,**kwargs);weight=size(value)
        if weight<=_VISION_CACHE_LIMIT:
            with _VISION_LOCK:
                if key in entries:state['bytes']-=entries.pop(key)[1]
                while entries and (state['bytes']+weight>_VISION_CACHE_LIMIT or len(entries)>=256):
                    _,(_,old_weight)=entries.popitem(last=False);state['bytes']-=old_weight
                entries[key]=(copy.deepcopy(value),weight);state['bytes']+=weight
        return value
    return wrapped


def symbol_analysis_session(fn):
    @wraps(fn)
    def wrapped(*args,**kwargs):
        from backend.cpu_inference import configure_image_processing
        configure_image_processing()
        if _SYMBOL_CACHE.get() is not None:return fn(*args,**kwargs)
        token=_SYMBOL_CACHE.set({})
        try:return fn(*args,**kwargs)
        finally:_SYMBOL_CACHE.reset(token)
    return wrapped


def symbol_mask(image,remove_rules=True):
    cache=_SYMBOL_CACHE.get()
    if cache is None or image is None or image.size==0:return _symbol_mask(image,remove_rules)
    from backend.drawing_mode import CURRENT
    key=(CURRENT.get(),image.shape,image.dtype.str,hashlib.sha256(image.tobytes()).digest(),bool(remove_rules))
    if key not in cache:cache[key]=_symbol_mask(image,remove_rules)
    # Callers may remove marks locally. Cache ownership must stay isolated.
    return cache[key].copy()


@cached_vision
def _symbol_mask(image,remove_rules=True):
    if image is None or image.size==0:return np.zeros((0,0),np.uint8)
    from backend.drawing_mode import digital,printed_ink
    if digital():return printed_ink(image)
    bgr=image if image.ndim==3 else cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    low,mid,high=np.percentile(gray,[5,50,99])
    if mid<60 and high>180:bgr=255-bgr
    from backend.notebook_rulings import _paper_colors,detect_notebook_rule_profile,apply_notebook_rule_profile
    _,_,normalized=_paper_colors(bgr)
    normal=np.mean(normalized,axis=2).astype(np.uint8)
    threshold,_=cv2.threshold(normal,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    mask=(normal<max(140,min(235,threshold+1))).astype(np.uint8)*255
    if remove_rules and min(gray.shape)>=200:
        profile=detect_notebook_rule_profile(bgr)
        _,erase=apply_notebook_rule_profile(bgr,profile)
        hsv=cv2.cvtColor(normalized.astype(np.uint8),cv2.COLOR_BGR2HSV)
        protected=(normal<145)|((hsv[:,:,1]>45)&(normal<205))
        mask[(erase>0)&~protected]=0
        for family in profile.get('families',[]):
            for line in family['lines']:
                band=np.zeros(mask.shape,np.uint8)
                width=max(3,int(round(2*line.get('half_width_px',2)+3)))
                if line.get('polyline'):cv2.polylines(band,[np.asarray(line['polyline'],np.int32)],False,255,width)
                else:cv2.line(band,tuple(line['start']),tuple(line['end']),255,width)
                yy,xx=np.nonzero((band>0)&~protected)
                ref=np.asarray(line['normalized_rgb'][::-1])
                close=np.max(np.abs(normalized[yy,xx]-ref),axis=1)<=max(24,line['color_tolerance'])
                mask[yy[close],xx[close]]=0
    return mask


def _run(profile,reverse=False):
    values=profile[::-1] if reverse else profile
    start=next((i for i,v in enumerate(values[:5]) if v),None)
    if start is None:return 0
    total=0;gap=0
    for val in values[start:]:
        if val:total+=1;gap=0
        else:
            gap+=1
            if gap>2:break
    return total


@cached_vision
def iec_bodies(image,mask=None):
    colored_bodies=[]
    if mask is None and image.ndim==3:
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        colored=((hsv[:,:,1]>35)&(hsv[:,:,2]<235)).astype(np.uint8)*255
        if np.count_nonzero(colored)>30:colored_bodies=iec_bodies(image,colored)
    mask=symbol_mask(image) if mask is None else mask
    if not mask.size:return []
    h,w=mask.shape;mind=min(h,w)
    closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,hierarchy=cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:return []
    holes=[c for i,c in enumerate(contours) if hierarchy[0,i,3]>=0]
    merged=[]
    # A notebook rule through an otherwise closed body can split its hole.
    # Reunite only adjoining cells with matching spans; the outer walls and
    # the two actual leads still have to pass the same independent checks.
    for i,a in enumerate(holes):
        ax,ay,aw,ah=cv2.boundingRect(a)
        if min(aw,ah)<4 or max(aw,ah)<12:continue
        for b in holes[i+1:]:
            bx,by,bw,bh=cv2.boundingRect(b)
            horizontal=(min(ah,bh)>.75*max(ah,bh) and abs(ay-by)<max(4,.15*max(ah,bh)) and
                        -3<=max(ax,bx)-min(ax+aw,bx+bw)<=max(9,.20*min(ah,bh)))
            vertical=(min(aw,bw)>.75*max(aw,bw) and abs(ax-bx)<max(4,.15*max(aw,bw)) and
                      -3<=max(ay,by)-min(ay+ah,by+bh)<=max(9,.20*min(aw,bw)))
            if horizontal or vertical:merged.append(cv2.convexHull(np.concatenate([a,b])))
    contours=merged+holes
    results=list(colored_bodies)
    wall_distance=cv2.distanceTransform(255-symbol_mask(image),cv2.DIST_L2,5)
    for i,contour in enumerate(contours):
        perimeter=cv2.arcLength(contour,True)
        if perimeter<=0:continue
        hull=cv2.convexHull(contour)
        vertices=cv2.approxPolyDP(hull,.035*cv2.arcLength(hull,True),True).reshape(-1,2)
        if len(vertices)>4:
            for tolerance in (.045,.055):
                proposal=cv2.approxPolyDP(hull,tolerance*cv2.arcLength(hull,True),True).reshape(-1,2)
                if len(proposal)==4:vertices=proposal;break
        if len(vertices)!=4 or not cv2.isContourConvex(vertices):
            # Pen joints can add a fifth corner to a rectangular hole. Use
            # its enclosing rectangle only for a nearly filled elongated
            # hull; the four ink walls and opposed leads are still mandatory.
            trial_center,trial_dimensions,trial_angle=cv2.minAreaRect(contour)
            if (max(trial_dimensions)/max(1,min(trial_dimensions))>=1.7
                and cv2.contourArea(hull)/max(1,np.prod(trial_dimensions))>=.86):
                vertices=cv2.boxPoints((trial_center,trial_dimensions,trial_angle))
            else:continue
        # A four-point approximation of an oval leaves curved arcs outside
        # the quadrilateral. A rectangular body's hull closely matches it.
        center,dimensions,angle=cv2.minAreaRect(contour)
        major,minor=max(dimensions),min(dimensions)
        hull_limit=1.36 if major/max(1,minor)>=2.5 else 1.24
        hull_ratio=cv2.contourArea(hull)/max(1,cv2.contourArea(vertices))
        if hull_ratio>1.24 and cv2.contourArea(hull)/max(1,major*minor)>.86:
            # Small inward notches may shift the polygon's corner positions.
            # Use the enclosing rectangle only when the hull fills it; ovals
            # cannot pass this fill check. All four ink walls still must exist.
            vertices=cv2.boxPoints((center,dimensions,angle))
            hull_ratio=cv2.contourArea(hull)/max(1,cv2.contourArea(vertices))
        if hull_ratio>hull_limit:continue
        if minor<4 or major<12 or major>max(180,.22*mind) or major/max(1,minor)>8:continue
        if cv2.contourArea(contour)/max(1,major*minor)<.73:continue
        contour_points=cv2.convexHull(contour).reshape(-1,2).astype(float)
        errors=[]
        for point in contour_points:
            distances=[]
            for j in range(4):
                a=vertices[j].astype(float);b=vertices[(j+1)%4].astype(float);edge=b-a
                t=np.clip(np.dot(point-a,edge)/max(1,np.dot(edge,edge)),0,1)
                distances.append(np.linalg.norm(point-a-t*edge))
            errors.append(min(distances))
        if np.percentile(errors,85)>max(2.5,.20*minor):continue
        # Four almost right angles distinguish a body from a diamond or triangle.
        corners=vertices.astype(float)
        # Convex hulls can turn a wire with two emission arrows into an
        # elongated rectangle. Every proposed wall must exist in the ink,
        # rather than only in the convex enclosure of a concave contour.
        walls=[]
        for a,b in zip(corners,np.roll(corners,-1,axis=0)):
            samples=np.rint(a+(b-a)*np.linspace(.15,.85,31)[:,None]).astype(int)
            samples=np.clip(samples,[0,0],np.array(mask.shape[::-1])-1)
            walls.append(float(np.mean(wall_distance[samples[:,1],samples[:,0]]<=max(2,minor*.10))))
        if min(walls)<.60:continue
        if any(abs(np.dot(corners[(j-1)%4]-corners[j],corners[(j+1)%4]-corners[j]))/max(1,np.linalg.norm(corners[(j-1)%4]-corners[j])*np.linalg.norm(corners[(j+1)%4]-corners[j]))>.48 for j in range(4)):continue
        side_lengths=np.linalg.norm(np.roll(corners,-1,axis=0)-corners,axis=1)
        if any(min(side_lengths[j],side_lengths[j+2])<.55*max(side_lengths[j],side_lengths[j+2]) for j in (0,1)):continue
        rect=cv2.boxPoints((center,dimensions,angle));edges=np.roll(rect,-1,axis=0)-rect
        if image.ndim==3:
            wall_saturation=[]
            for a,b in zip(rect,np.roll(rect,-1,axis=0)):
                colors=[]
                for t in np.linspace(.20,.80,12):
                    px,py=np.rint(a+(b-a)*t).astype(int)
                    patch=image[max(0,py-2):min(h,py+3),max(0,px-2):min(w,px+3)]
                    ink_patch=mask[max(0,py-2):min(h,py+3),max(0,px-2):min(w,px+3)]
                    if patch.size and np.any(ink_patch):colors.extend(cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)[:,:,1][ink_patch>0])
                wall_saturation.append(float(np.median(colors)) if colors else 0)
            if min(wall_saturation)<25 and sum(s>45 for s in wall_saturation)>=3:continue
        u=edges[np.argmax(np.linalg.norm(edges,axis=1))];u=u/np.linalg.norm(u)
        wall_vectors=(np.roll(corners,-1,axis=0)-corners)/side_lengths[:,None]
        v=np.array([-u[1],u[0]])
        if np.any(np.maximum(np.abs(wall_vectors@u),np.abs(wall_vectors@v))<.86):continue
        for axis in (u,np.array([-u[1],u[0]])):
            normal=np.array([-axis[1],axis[0]])
            projected=(rect-np.array(center))@axis;transverse=(rect-np.array(center))@normal
            length=float(np.ptp(projected));width=float(np.ptp(transverse))
            if length<.85*width:continue
            probe=max(12,min(70,int(.55*length)));margin=max(5,int(width*.25))
            cw=int(np.ceil(length))+2*probe+6;ch=int(np.ceil(width))+2*margin+6
            matrix=np.array([np.r_[axis,cw/2-np.dot(axis,center)],np.r_[normal,ch/2-np.dot(normal,center)]],np.float32)
            crop=cv2.warpAffine(mask,matrix,(cw,ch),flags=cv2.INTER_NEAREST)
            xa=int(round(cw/2-length/2));xb=int(round(cw/2+length/2));ya=int(round(ch/2-width/2));yb=int(round(ch/2+width/2))
            if np.mean(crop[max(0,ya-4):yb+5,max(0,xa-4):xb+5]>0)>.55:continue
            band=max(2,min(8,int(width*.24)))
            top=np.mean(np.any(crop[max(0,ya-band):ya+2,xa:xb]>0,axis=0))
            bottom=np.mean(np.any(crop[yb-1:yb+band+1,xa:xb]>0,axis=0))
            left=np.mean(np.any(crop[ya:yb,max(0,xa-band):xa+2]>0,axis=1))
            right=np.mean(np.any(crop[ya:yb,xb-1:xb+band+1]>0,axis=1))
            if min(top,bottom,left,right)<.62:continue
            strip=max(2,int(width*.32));cy=ch//2
            before=np.any(crop[cy-strip:cy+strip+1,:xa+1]>0,axis=0)
            after=np.any(crop[cy-strip:cy+strip+1,xb-1:]>0,axis=0)
            lead_min=max(8,int(length*.09))
            leads=[_run(before,True),_run(after)]
            if min(leads)<lead_min:continue
            # A tiny cell made by the crossing of long buses has walls continuing
            # far beyond its corners, rather than two centered leads.
            extensions=[]
            for y in (ya,yb):
                extensions.extend([_run(np.any(crop[max(0,y-2):y+3,:max(0,xa-band)]>0,axis=0),True),
                                   _run(np.any(crop[max(0,y-2):y+3,xb+band:]>0,axis=0))])
            if all(value>=max(lead_min,int(length*.45)) for value in extensions):continue
            pad_axis=max(5,min(12,length*.16));pad_normal=max(4,min(10,width*.22))
            boundary=np.array([np.array(center)+axis*a+normal*b for a in (-length/2-pad_axis,length/2+pad_axis) for b in (-width/2-pad_normal,width/2+pad_normal)])
            x1,y1=boundary.min(axis=0);x2,y2=boundary.max(axis=0)
            if x1<1 or y1<1 or x2>w-2 or y2>h-2:continue
            bbox=dict(x1=float(x1),y1=float(y1),x2=float(x2),y2=float(y2))
            if any(np.linalg.norm(np.array(center)-np.array(old['center']))<max(5,minor*.6) for old in results):break
            results.append(dict(bbox=bbox,center=list(map(float,center)),score=.92,
                                evidence=dict(iec_rectangular_body=True,iec_line_body=True,opposed_leads=True,lead_support_px=leads,body_size=[length,width],body_center=list(map(float,center)),body_axis=list(map(float,axis)),body_corners=rect.tolist())))
            break
    return results


def _thin_symbol(mask):
    """Preserve continuous diagonal strokes when reducing them to one pixel."""
    yy,xx=np.nonzero(mask)
    result=np.zeros_like(mask)
    if not len(xx):return result
    x0,x1=max(0,int(xx.min())-1),min(mask.shape[1],int(xx.max())+2)
    y0,y1=max(0,int(yy.min())-1),min(mask.shape[0],int(yy.max())+2)
    from backend.thinning import thin_binary
    pixels=thin_binary(np.pad(mask[y0:y1,x0:x1]>0,1),max_iter=None)
    result[y0:y1,x0:x1]=pixels[1:-1,1:-1].astype(np.uint8)*255
    return result


def zigzag_paths(image,mask=None):
    mask=symbol_mask(image) if mask is None else mask
    if not mask.size:return []
    from backend.wire_detection import _extract_graph
    nodes,segments,_=_extract_graph(_thin_symbol(mask),min_segment_length=.1)
    # Thinning a thick acute corner can produce a tiny three-edge cycle.
    # Collapse only these local artifacts; retain real branching strokes.
    parent={n['node_id']:n['node_id'] for n in nodes}
    # Closed contours and clipped skeleton fragments have no graph endpoint.
    # They are not open zigzag paths and must not enter union-find with None.
    segments=[s for s in segments if s.get('start_node') in parent and s.get('end_node') in parent]
    def root(n):
        while parent[n]!=n:n=parent[n]
        return n
    short={}
    for s in segments:
        a,b=s.get('start_node'),s.get('end_node')
        if a!=b and s['length_px']<10:
            short.setdefault(a,{})[b]=s['length_px']
            short.setdefault(b,{})[a]=s['length_px']
    for a,neighbors in short.items():
        for b,ab in neighbors.items():
            for c in set(neighbors)&set(short.get(b,{})):
                if ab+neighbors[c]+short[b][c]<=22:
                    parent[root(b)]=root(a);parent[root(c)]=root(a)
    segments=[dict(s,start_node=root(s['start_node']),end_node=root(s['end_node'])) for s in segments]
    segments=[s for s in segments if s.get('start_node')!=s.get('end_node')]
    # Remove only very short dead-end artifacts from thinning at tooth tips.
    # Keeping every short intermediate edge preserves the actual zigzag path.
    for _ in range(3):
        degrees={}
        for segment in segments:
            for key in ('start_node','end_node'):
                nid=segment.get(key)
                if nid:degrees[nid]=degrees.get(nid,0)+1
        segments=[s for s in segments if not (s['length_px']<6 and any(degrees.get(s.get(key))==1 for key in ('start_node','end_node')))]
    incident={n['node_id']:[] for n in nodes}
    for i,s in enumerate(segments):
        for key in ('start_node','end_node'):
            if s.get(key) in incident:incident[s[key]].append(i)
    seen=set();paths=[]
    half_width=cv2.distanceTransform(mask,cv2.DIST_L2,3)
    for i,s in enumerate(segments):
        if i in seen:continue
        points=np.array([[p['x'],p['y']] for p in s['points']],float);seen.add(i)
        ends=[s.get('start_node'),s.get('end_node')]
        for side in (0,1):
            node=ends[side]
            while node in incident and len(incident[node])==2:
                candidates=[j for j in incident[node] if j not in seen]
                if not candidates:break
                j=candidates[0];seen.add(j);other=segments[j]
                extension=np.array([[p['x'],p['y']] for p in other['points']],float)
                if other.get('start_node')!=node:extension=extension[::-1]
                node=other.get('end_node') if other.get('start_node')==node else other.get('start_node')
                points=np.vstack([extension[::-1],points]) if side==0 else np.vstack([points,extension])
        if len(points)<8:continue
        px=np.clip(np.rint(points[:,0]).astype(int),0,mask.shape[1]-1);py=np.clip(np.rint(points[:,1]).astype(int),0,mask.shape[0]-1)
        epsilon=max(1.5,min(4.5,float(np.median(half_width[py,px]))*1.5))
        simplified=cv2.approxPolyDP(points.astype(np.float32),epsilon,False).reshape(-1,2)
        if len(simplified)<8:continue
        delta=simplified[-1]-simplified[0];span=np.linalg.norm(delta)
        if span<35:continue
        axis=delta/span;normal=np.array([-axis[1],axis[0]])
        vectors=np.diff(simplified,axis=0);lengths=np.linalg.norm(vectors,axis=1)
        along=vectors@axis;cross=vectors@normal
        directions=cross/np.maximum(lengths,1)
        teeth=np.where((np.abs(directions)>.30)&(along>.12*lengths)&(lengths>=4))[0]
        # A true resistor has at least five alternating tooth edges,
        # with collinear straight leads at both ends. Letters W/M fail this test.
        for start in range(len(teeth)):
            run=[int(teeth[start])]
            for idx in teeth[start+1:]:
                previous=run[-1]
                # Rasterized thick strokes can leave a short flat at a tooth
                # tip. It belongs to the same continuous path, not a lead.
                flat_tip=(idx==previous+2 and abs(directions[previous+1])<.25 and
                          lengths[previous+1]<.35*min(lengths[previous],lengths[idx]))
                if (idx!=previous+1 and not flat_tip) or directions[idx]*directions[previous]>=0:break
                run.append(int(idx))
            a,b=run[0],run[-1]
            if len(run)<5 or a==0 or b>=len(vectors)-1:continue
            if abs(directions[a-1])>.24 or abs(directions[b+1])>.24 or min(lengths[a-1],lengths[b+1])<6:continue
            body=simplified[a:b+2];width=np.ptp(body@normal);length=np.ptp(body@axis)
            if width<7 or length<20 or length<width*1.2:continue
            pad=max(5,min(14,width*.35));lo=body.min(axis=0)-pad;hi=body.max(axis=0)+pad
            bbox=dict(x1=float(max(0,lo[0])),y1=float(max(0,lo[1])),x2=float(min(mask.shape[1]-1,hi[0])),y2=float(min(mask.shape[0]-1,hi[1])))
            paths.append(dict(bbox=bbox,center=list(map(float,(lo+hi)/2)),score=.90,evidence=dict(distinct_zigzag_segments=len(run),opposed_leads=True,alternating_connected_teeth=True)))
            break
    return paths
