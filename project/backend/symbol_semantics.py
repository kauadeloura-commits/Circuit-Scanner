"""Visible emission arrows and polarity signs determine related symbol classes."""
import math
from itertools import combinations
import cv2
import numpy as np
from backend.terminal_identity import _crop, _marks, _four_arm_junction
from backend.resistor_geometry import _thin_symbol


def _search_box(image,bbox):
    margin=max(16,min(240,min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])*.55))
    return dict(x1=max(0,bbox['x1']-margin),y1=max(0,bbox['y1']-margin),
                x2=min(image.shape[1]-1,bbox['x2']+margin),y2=min(image.shape[0]-1,bbox['y2']+margin))


def _owned(point,bbox,neighbors):
    def distance(b):
        return math.hypot(max(b['x1']-point[0],0,point[0]-b['x2']),max(b['y1']-point[1],0,point[1]-b['y2']))
    own=distance(bbox)
    return not any(distance(b)+3<own for b in neighbors)


def _mark_crop(image,box,ink_hue=False):
    mask,origin=_crop(image,box)
    if image.ndim==3:
        x,y=origin.astype(int)
        hsv=cv2.cvtColor(image[y:y+mask.shape[0],x:x+mask.shape[1]],cv2.COLOR_BGR2HSV)
        # Pale blue arrow wings may have been erased with a neutral rule by
        # the grayscale crop. Recover chromatic ink directly from the photo.
        gray=cv2.cvtColor(image[y:y+mask.shape[0],x:x+mask.shape[1]],cv2.COLOR_BGR2GRAY)
        colored=(hsv[:,:,1]>35)&(gray<210)
        reliable_hue=False
        values=gray[mask>0]
        # Restrict hue only when a substantial colored background can be
        # mistaken for ink. On plain paper retain pale/shifted pen fringes.
        colored_background=np.count_nonzero(colored & (mask==0))>=mask.size*.20
        if ink_hue and colored_background and len(values):
            core=(mask>0)&(gray<=float(np.percentile(values,25))+8)
            chromatic=core&(hsv[:,:,1]>35)
            if np.count_nonzero(chromatic)>=max(12,.60*np.count_nonzero(core)):
                counts=np.bincount((hsv[:,:,0][chromatic]//10).astype(int),minlength=18)
                hue=int(np.argmax(counts+np.roll(counts,1)+np.roll(counts,-1)))*10+5
                difference=np.abs(hsv[:,:,0].astype(float)-hue)
                colored&=np.minimum(difference,180-difference)<=18
                reliable_hue=True
        if np.count_nonzero(colored)>=20 and (reliable_hue or np.count_nonzero(colored)>=.65*np.count_nonzero(mask)):
            mask=cv2.morphologyEx(colored.astype(np.uint8)*255,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    return mask,origin


def visible_plus(image,bbox,neighbors=(),*,_small_retry=True):
    search=_search_box(image,bbox);mask,origin=_mark_crop(image,search,ink_hue=True)
    marks,labels=_marks(mask,origin,search,include_attached=True,maximum_fraction=.50)
    found=[]
    for mark in marks:
        if mark['kind']!='+' or not _owned(mark['point'],bbox,neighbors):continue
        if not search['x1']<=mark['point'][0]<=search['x2'] or not search['y1']<=mark['point'][1]<=search['y2']:continue
        # A sign at the center cannot distinguish the two sides of a capacitor.
        center=np.array([(bbox['x1']+bbox['x2'])/2,(bbox['y1']+bbox['y2'])/2])
        if np.linalg.norm(mark['point']-center)<3:continue
        if mark.get('attached'):
            # Reject a crossing on an actual long conductor. A short spur
            # ending in a plus, or a plus touching a plate, remains a sign.
            from backend.terminal_identity import _support
            if any(min(_support(mask,mark['point'],sign*np.array([math.cos(a),math.sin(a)]),np.arange(15,27,3),origin)
                       for sign in (-1,1))>=.8 for a in np.arange(0,math.pi,math.pi/6)):
                continue
        label=int(mark['label']);ys,xs=np.nonzero(labels==label)
        if not mark.get('attached'):
            glyph=(labels==label).astype(np.uint8)*255
            sk=(_thin_symbol(glyph)>0).astype(np.uint8)
            degree=cv2.filter2D(sk,cv2.CV_16S,np.ones((3,3),np.int16))-sk
            _,_,_,junctions=cv2.connectedComponentsWithStats(((sk>0)&(degree>=3)).astype(np.uint8),8)
            radius=max(7,min(32,min(np.ptp(xs)+1,np.ptp(ys)+1)*.5))
            centers=list(junctions[1:])
            # Thick handwriting can split one intersection into adjacent
            # skeleton junctions. Check their common center at glyph scale.
            for p in junctions[1:]:
                nearby=[q for q in junctions[1:] if np.linalg.norm(p-q)<=radius*.45]
                if len(nearby)>1:centers.append(np.mean(nearby,axis=0))
            confirmed=any(_four_arm_junction(sk,p,radius) for p in centers)
            if not confirmed and len(junctions)>1:
                intersections=junctions[1:];center=intersections.mean(axis=0)
                # Uneven crosses can leave two nearby junctions. Four real
                # endpoints must still make two opposed, perpendicular arms.
                if np.max(np.linalg.norm(intersections-center,axis=1))<=radius*.35:
                    ends=np.column_stack(np.where((sk>0)&(degree==1)))[:,::-1].astype(float)
                    if len(ends)==4:
                        vectors=ends-center;lengths=np.linalg.norm(vectors,axis=1)
                        unit=vectors/np.maximum(1,lengths[:,None])
                        if min(lengths)>=max(3,max(lengths)*.15):
                            for a,b in combinations(range(4),2):
                                c,d=[i for i in range(4) if i not in (a,b)]
                                if float(unit[a]@unit[b])<-.65 and float(unit[c]@unit[d])<-.65 and abs(float(unit[a]@unit[c]))<.65:
                                    confirmed=True;break
                    elif len(ends)==3 and mark['score']>=.58:
                        # Reducing a thick cross may erase a short skeleton
                        # arm. Verify four directions in the actual ink past
                        # the stroke width; a T has no ink on its fourth side.
                        distance=cv2.distanceTransform(glyph,cv2.DIST_L2,5)
                        px,py=np.rint(center).astype(int)
                        start=max(3,float(distance[py,px])+1)
                        radii=np.arange(start,start+3,.75)
                        supported=cv2.dilate(glyph,np.ones((3,3),np.uint8))
                        for angle in np.arange(0,math.pi/2,math.pi/12):
                            directions=[np.array([math.cos(angle),math.sin(angle)]),np.array([-math.sin(angle),math.cos(angle)])]
                            scores=[]
                            for axis in directions:
                                for sign in (-1,1):
                                    points=np.rint(center+sign*radii[:,None]*axis).astype(int)
                                    valid=(points[:,0]>=0)&(points[:,0]<sk.shape[1])&(points[:,1]>=0)&(points[:,1]<sk.shape[0])
                                    hits=np.zeros(len(points),bool);hits[valid]=supported[points[valid,1],points[valid,0]]>0
                                    scores.append(np.mean(hits))
                            if min(scores)>=.75:confirmed=True;break
            if not confirmed:continue
        if mark.get('attached'):
            radius=max(5,min(14,min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])*.12))
            bound=[mark['point'][0]-radius,mark['point'][1]-radius,mark['point'][0]+radius,mark['point'][1]+radius]
        else:bound=[xs.min()+origin[0],ys.min()+origin[1],xs.max()+origin[0],ys.max()+origin[1]]
        found.append(dict(point=mark['point'].tolist(),score=float(mark['score']),bounds=[float(v) for v in bound],attached=bool(mark.get('attached'))))
    if not found and _small_retry and min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])<=85:
        from backend.local_symbol_roles import local_view
        from backend.drawing_mode import CURRENT
        for scale in ((2,3) if CURRENT.get()=='handwritten' else (2,)):
            view,box,_,offset=local_view(image,bbox,scale)
            adjacent=[{k:(b[k]-offset[0 if k.startswith('x') else 1])*scale for k in ('x1','y1','x2','y2')} for b in neighbors]
            for mark in visible_plus(view,box,adjacent,_small_retry=False):
                found.append(dict(mark,point=(np.array(mark['point'])/scale+offset).tolist(),
                                  bounds=[v/scale+offset[i%2] for i,v in enumerate(mark['bounds'])],reading_scale=scale))
            if found:break
    from backend.drawing_mode import digital
    if found and digital():
        from backend.capacitor_geometry import capacitor_body_evidence
        ok,body=capacitor_body_evidence(image,bbox)
        if ok:
            unit=np.asarray(body['electrical_axis']);normal=np.array([-unit[1],unit[0]])
            found=[m for m in found if max(m['bounds'][2]-m['bounds'][0],m['bounds'][3]-m['bounds'][1])<=max(10,.7*max(body['outer_plate_lengths'])) and not any(abs(float((np.asarray(m['point'])-p)@normal))<=.6*length and abs(float((np.asarray(m['point'])-p)@unit))<max(3,.15*length) for p,length in zip(np.asarray(body['outer_plate_centers']),body['outer_plate_lengths']))]
    return found


def _head_from_skeleton_arms(ends,junctions,origin,x,y,center,bbox,neighbors,scale,bounds):
    # Pencil retracing leaves short extra skeleton spurs. A real head still
    # has one long shaft and two shorter wings on opposite sides of it.
    if not 3<=len(ends)<=8:return None
    from backend.drawing_mode import CURRENT
    if len(ends)!=3 and CURRENT.get()!='handwritten':return None
    for junction in junctions[1:]:
        vectors=ends-junction;lengths=np.linalg.norm(vectors,axis=1)
        for indexes in combinations(range(len(ends)),3):
            if min(lengths[list(indexes)])<2:continue
            for tail in indexes:
                wings=[i for i in indexes if i!=tail]
                if any(lengths[i]>.70*lengths[tail] for i in range(len(ends)) if i not in indexes):continue
                direction=-vectors[tail]/lengths[tail]
                u,v=(vectors[i]/lengths[i] for i in wings)
                if lengths[tail]<max(lengths[wings])*.65 or not .35<lengths[wings[0]]/lengths[wings[1]]<2.85:continue
                normal=np.array([-direction[1],direction[0]])
                if not -.997<float(u@v)<.88 or float(u@normal)*float(v@normal)>-.08:continue
                if float(u@direction)*float(v@direction)<-.20:continue
                point=junction+origin+[x,y];away=point-center
                if np.linalg.norm(away)<scale*.25 or float(direction@away)<np.linalg.norm(away)*.15:continue
                if not _owned(point,bbox,neighbors):continue
                return dict(point=point,direction=direction,head_confirmed=True,bounds=bounds,
                            skeleton_arms=len(ends),discarded_short_spurs=len(ends)-3)
    return None


def emission_arrows(image,bbox,neighbors=(),*,_extra_context=0,_retraced_retry=False,_page_ink=False):
    def crop(box):
        mask,origin=_mark_crop(image,box)
        if _page_ink:
            from backend.resistor_geometry import symbol_mask
            x,y=origin.astype(int);full=symbol_mask(image)
            mask=full[y:y+mask.shape[0],x:x+mask.shape[1]]
        return mask,origin
    search=_search_box(image,bbox);mask,origin=crop(search)
    if _extra_context:
        search=dict(x1=max(0,search['x1']-_extra_context),y1=max(0,search['y1']-_extra_context),
                    x2=min(image.shape[1]-1,search['x2']+_extra_context),y2=min(image.shape[0]-1,search['y2']+_extra_context))
        mask,origin=crop(search)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    center=np.array([(bbox['x1']+bbox['x2'])/2,(bbox['y1']+bbox['y2'])/2])
    scale=min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1']);arrows=[]
    from backend.drawing_mode import CURRENT
    if _retraced_retry and CURRENT.get()=='handwritten':
        from backend.diode_contacts import body_axes
        bodies=body_axes(image,bbox)
        if bodies:
            body=max(bodies,key=lambda b:b['area'])
            center=(body['base']+body['apex'])/2
    for label in range(1,count):
        x,y,w,h,area=stats[label]
        if area<9 or min(w,h)<3 or max(w,h)>max(18,scale*.7):continue
        patch=(labels[y:y+h,x:x+w]==label).astype(np.uint8)*255
        sk=(_thin_symbol(patch)>0).astype(np.uint8)
        degree=cv2.filter2D(sk,cv2.CV_16S,np.ones((3,3),np.int16),borderType=cv2.BORDER_CONSTANT)-sk
        ends=np.column_stack(np.nonzero((sk>0)&(degree==1)))[:,::-1].astype(float)
        _,_,_,junctions=cv2.connectedComponentsWithStats(((sk>0)&(degree>=3)).astype(np.uint8),8)
        strong=(_head_from_skeleton_arms(ends,junctions,origin,x,y,center,bbox,neighbors,scale,
                                      [x+origin[0],y+origin[1],x+w-1+origin[0],y+h-1+origin[1]])
                if len(ends)==3 or _retraced_retry else None)
        if strong:
            arrows.append(strong);continue
        # Filled/blurred heads may become a tiny loop instead of three
        # skeleton ends. They can accompany a fully verified neighboring
        # arrow when the head is wider than the shaft and directions agree.
        yy,xx=np.nonzero(patch);points=np.column_stack((xx,yy)).astype(float)
        _,vectors=np.linalg.eigh(np.cov(points.T));direction=vectors[:,-1]
        mean=points.mean(axis=0);global_mean=mean+origin+[x,y]
        if direction@(global_mean-center)<0:direction=-direction
        across=np.array([-direction[1],direction[0]])
        along=(points-mean)@direction;normal=(points-mean)@across
        low,high=np.percentile(along,[2,98]);length=high-low
        stem=normal[(along>low+.2*length)&(along<low+.55*length)]
        head=normal[along>low+.6*length]
        if len(stem)<3 or len(head)<3:continue
        stem_width=np.ptp(stem);head_width=np.ptp(head)
        away=global_mean-center
        if length<1.1*head_width or head_width<max(5,1.35*stem_width) or direction@away<.15*np.linalg.norm(away):continue
        if not _owned(global_mean,bbox,neighbors):continue
        arrows.append(dict(point=global_mean,direction=direction,head_confirmed=False,bounds=[x+origin[0],y+origin[1],x+w-1+origin[0],y+h-1+origin[1]]))
    pairs=[]
    for a,b in combinations(arrows,2):
        separation=float(np.linalg.norm(a['point']-b['point']))
        if not (a['head_confirmed'] or b['head_confirmed']):continue
        if float(a['direction']@b['direction'])<.82 or not max(4,scale*.05)<=separation<=max(16,scale*.6):continue
        pairs.append([a,b])
    if not pairs and not _extra_context:
        from backend.drawing_mode import digital
        if digital():
            from backend.printed_arrows import arrow_pairs
            pairs=arrow_pairs(image,bbox)
            pairs=[pair for pair in pairs if all(_owned(a['point'],bbox,neighbors) for a in pair)]
            if pairs:return pairs
        # Small model-box shifts change the local threshold at faint arrow
        # wings. Recheck a four-pixel context band, with the SAME body center,
        # scale and strict two-outward-arrowhead test; never infer LED from
        # the model label alone. The measured component box is unchanged.
        pairs=emission_arrows(image,bbox,neighbors,_extra_context=4,_retraced_retry=_retraced_retry,_page_ink=_page_ink)
        if not pairs and not _retraced_retry and CURRENT.get()=='handwritten':
            pairs=emission_arrows(image,bbox,neighbors,_retraced_retry=True,_page_ink=_page_ink)
        if not pairs and not _page_ink and CURRENT.get()=='handwritten':
            from backend.diode_contacts import body_axes
            if body_axes(image,bbox):pairs=emission_arrows(image,bbox,neighbors,_page_ink=True)
    return pairs


def shared_capacitor_plates(image,detections):
    """Unify two proposals only when their common region contains both plates."""
    items=[dict(d) for d in detections];removed=set()
    for i,a in enumerate(items):
      if i in removed or a.get('manual_reviewed') or a.get('class') not in {'capacitor','capacitor_polarized'}:continue
      for j in range(i+1,len(items)):
        b=items[j]
        if j in removed or b.get('manual_reviewed') or b.get('class') not in {'capacitor','capacitor_polarized'}:continue
        aa,bb=a['bbox'],b['bbox'];over=dict(x1=max(aa['x1'],bb['x1']),y1=max(aa['y1'],bb['y1']),x2=min(aa['x2'],bb['x2']),y2=min(aa['y2'],bb['y2']))
        w,h=over['x2']-over['x1'],over['y2']-over['y1']
        if min(w,h)<10:continue
        mask,_=_mark_crop(image,over);minlength=max(12,.45*min(w,h))
        lines=cv2.HoughLinesP(mask,1,np.pi/180,threshold=10,minLineLength=minlength,maxLineGap=2)
        if lines is None:continue
        segments=[np.array(l,float).reshape(2,2) for l in np.asarray(lines).reshape(-1,4)]
        pair=False
        for p,q in combinations(segments,2):
            u=p[1]-p[0];v=q[1]-q[0];lu,lv=np.linalg.norm(u),np.linalg.norm(v)
            if min(lu,lv)<minlength or abs(float(u@v))/lu/lv<.98:continue
            normal=np.array([-u[1],u[0]])/lu;distance=abs(float((q.mean(axis=0)-p.mean(axis=0))@normal))
            if 6<=distance<=.85*max(w,h):pair=True;break
        if not pair:continue
        merged=dict(x1=min(aa['x1'],bb['x1']),y1=min(aa['y1'],bb['y1']),x2=max(aa['x2'],bb['x2']),y2=max(aa['y2'],bb['y2']))
        merged.update(width=merged['x2']-merged['x1'],height=merged['y2']-merged['y1'],center_x=(merged['x1']+merged['x2'])/2,center_y=(merged['y1']+merged['y2'])/2)
        a.update(bbox=merged,shared_plate_proposals=True);removed.add(j)
    return [d for i,d in enumerate(items) if i not in removed]

def resolve_marked_families(image,detections,class_ids,*,expand_boxes=True):
    photograph=image
    from backend.ink_color import estimate_component_ink
    from backend.notebook_rulings import detect_notebook_rule_profile
    from backend.paper_ink import fragmentation_policy,wire_analysis_image
    profile=estimate_component_ink(image,{'detections':detections})
    rules=detect_notebook_rule_profile(image,profile)
    policy=fragmentation_policy(image,{'detections':detections},profile,rules)
    image=wire_analysis_image(image,{'conductor_ink_policy':policy})
    detections=shared_capacitor_plates(image,detections)
    result=[];counts=dict(led=0,capacitor=0)
    for det in detections:
        new=dict(det);kind=str(det.get('class'));bbox=det['bbox']
        if kind in {'mosfet_n','mosfet_p'} and not det.get('manual_reviewed'):
            from backend.mosfet_semantics import visible_channel_type
            channel,evidence=visible_channel_type(photograph,bbox)
            new['channel_arrow_validation']=evidence
            if channel and channel!=kind:
                new.update(resolved_from_class=kind,**{'class':channel,'class_id':class_ids[channel]},class_resolution='visible_body_arrow_direction_relative_to_isolated_gate')
        if kind not in {'diode','led','capacitor','capacitor_polarized'} or det.get('manual_reviewed'):
            result.append(new);continue
        neighbors=[d['bbox'] for d in detections if d is not det]
        if kind in {'diode','led'}:
            pairs=emission_arrows(image,bbox,neighbors);marks=[a for pair in pairs for a in pair]
            from backend.drawing_mode import CURRENT
            if not pairs and CURRENT.get()=='handwritten' and photograph is not image:
                pairs=emission_arrows(photograph,bbox,neighbors);marks=[a for pair in pairs for a in pair]
            target='led' if pairs else 'diode';family='led'
            new['led_emission_validation']=dict(valid=bool(pairs),method='two_aligned_outward_arrows_with_heads',
                arrows=[dict(point=a['point'].tolist(),bounds=[float(v) for v in a['bounds']]) for a in marks])
            if not pairs and kind=='led' and CURRENT.get()=='handwritten':
                # Joined/retraced arrowheads can defeat the strict head reader.
                # Lack of that confirmation must not negate a strong, repeated
                # learned LED proposal when paired peripheral strokes remain.
                from backend.diode_contacts import body_axes
                learned=float(det.get('best_learned_confidence',det.get('confidence',0)))
                views=int(det.get('consensus_source_count',0))
                if learned>=.80 and views>=2 and body_axes(photograph,bbox):
                    from backend.recognition import _evidencia_setas_emissao_led
                    partial,evidence=_evidencia_setas_emissao_led(photograph,bbox)
                    if partial:
                        target='led'
                        new['led_emission_validation'].update(partial_strokes=evidence,
                            method='learned_led_with_paired_peripheral_strokes_heads_unresolved')
                        new['review_required']=True
                        new['review_reasons']=list(dict.fromkeys([*new.get('review_reasons',[]),'LED: conferir as pontas das setas de emissão.']))
        else:
            marks=visible_plus(image,bbox,neighbors);family='capacitor'
            from backend.capacitor_sign_ownership import verified_plate_signs
            marks=verified_plate_signs(image,bbox,marks)
            from backend.drawing_mode import CURRENT
            if not marks and CURRENT.get()=='handwritten' and photograph is not image:
                marks=verified_plate_signs(photograph,bbox,visible_plus(photograph,bbox,neighbors))
            from backend.printed_capacitor import plate_polarity
            shape=plate_polarity(photograph,bbox) if not marks else None
            target='capacitor_polarized' if marks or (shape and shape['valid']) else 'capacitor'
            new['capacitor_polarity_validation']=dict(valid=target=='capacitor_polarized',
                method='visible_four_arm_plus' if marks else shape['method'] if shape and shape['valid'] else 'visible_four_arm_plus',
                marks=marks,printed_plate_evidence=shape)
        if target!=kind:
            new.update(resolved_from_class=kind,**{'class':target,'class_id':class_ids.get(target,new.get('class_id'))})
            counts[family]+=1
        new['class_resolution']='visible_emission_arrows' if family=='led' else new['capacitor_polarity_validation']['method']
        if expand_boxes and marks:
            expanded=dict(bbox)
            for mark in marks:
                x1,y1,x2,y2=mark['bounds']
                expanded.update(x1=min(expanded['x1'],x1-2),y1=min(expanded['y1'],y1-2),
                                x2=max(expanded['x2'],x2+2),y2=max(expanded['y2'],y2+2))
            expanded.update(x1=max(0,expanded['x1']),y1=max(0,expanded['y1']),
                            x2=min(image.shape[1]-1,expanded['x2']),y2=min(image.shape[0]-1,expanded['y2']))
            expanded.update(width=expanded['x2']-expanded['x1'],height=expanded['y2']-expanded['y1'],
                            center_x=(expanded['x1']+expanded['x2'])/2,center_y=(expanded['y1']+expanded['y2'])/2)
            new['symbol_body_bbox']=dict(bbox);new['bbox']=expanded
        from backend.printed_bounds import capacitor_bounds
        result.append(capacitor_bounds(photograph,new))
    return result,counts
