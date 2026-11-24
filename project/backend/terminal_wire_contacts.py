"""Finalize every component's pins after wires, strictly inside its box."""
from copy import deepcopy
from itertools import combinations, permutations
import math

import cv2
import numpy as np

from backend.ink_color import estimate_component_ink, image_with_component_ink
from backend.terminal_identity import _crop, _marks, _support, POLARIZED_TWO_PIN_CLASSES
from backend.wire_detection import _skeletonize
from backend.notebook_rulings import apply_notebook_rule_profile
from backend.circuit_types import PIN_SCHEMAS


def _bounds(bbox, shape):
    h,w=shape[:2]
    values=[float(bbox[k]) for k in ('x1','y1','x2','y2')]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Caixa de componente invalida.')
    x1,y1,x2,y2=values
    # Integer pixel centers must be strictly inside a possibly fractional box.
    bounds=(max(0,math.floor(x1)+1),max(0,math.floor(y1)+1),
            min(w-1,math.ceil(x2)-1),min(h-1,math.ceil(y2)-1))
    if bounds[0]>bounds[2] or bounds[1]>bounds[3]:
        raise ValueError('Caixa sem area interna para terminais.')
    return bounds


def _manual(component):
    td=component.get('terminal_detection') or {}
    terms=component.get('terminals') or []
    return td.get('status')=='manual_ground_truth' or (bool(terms) and
        all(str(t.get('source') or '').startswith('manual_ground_truth') for t in terms))


def _review(component,reason):
    component['review_required']=True
    component['review_reasons']=list(dict.fromkeys((component.get('review_reasons') or [])+[reason]))
    td=component.setdefault('terminal_detection',{})
    td['review_required']=True
    td['review_reasons']=list(dict.fromkeys((td.get('review_reasons') or [])+[reason]))


def constrain_terminal_positions(result,image,*,require_ink=False):
    """Enforce the box invariant even for model predictions and saved edits."""
    for component in result.get('components') or []:
        try:
            x1,y1,x2,y2=_bounds(component.get('bbox') or {},image.shape)
        except (KeyError,TypeError,ValueError,OverflowError):
            component['terminals']=[]
            _review(component,'caixa_sem_area_interna_para_terminais')
            continue
        mask,origin=_crop(image,component['bbox'])
        yy,xx=np.nonzero(mask)
        pixels=np.stack((xx,yy),axis=1)+origin
        pixels=pixels[(pixels[:,0]>=x1)&(pixels[:,0]<=x2)&(pixels[:,1]>=y1)&(pixels[:,1]<=y2)]
        verified=[]
        manual=_manual(component)
        for terminal in component.get('terminals') or []:
            try:
                p=np.array([float(terminal['position'][k]) for k in ('x','y')])
                if not np.all(np.isfinite(p)):
                    raise ValueError('nonfinite')
            except (ValueError,TypeError,KeyError):
                p=np.array([(x1+x2)/2,(y1+y2)/2])
                _review(component,'posicao_terminal_invalida')
            target=np.clip(p,[x1,y1],[x2,y2])
            moved=not np.array_equal(p,target)
            # Being inside a box is insufficient: a model point can still be
            # in empty paper. Validate automatic points on original ink too.
            rounded=np.rint(target).astype(int)-origin.astype(int)
            on_ink=bool(mask[rounded[1],rounded[0]]) if (0<=rounded[1]<mask.shape[0] and 0<=rounded[0]<mask.shape[1]) else False
            if terminal.get('wire_contact_confirmed'):
                # Contacts may restore photographed wire ink removed by the
                # notebook-rule mask. Validate against the photograph itself.
                gx,gy=np.rint(target).astype(int)
                pixel=image[gy,gx]
                on_ink=float(np.min(pixel))<210
            if moved or (require_ink and not manual and not on_ink):
                if len(pixels):
                    distances=np.linalg.norm(pixels-target,axis=1)
                    nearest=int(np.argmin(distances))
                    if distances[nearest]<=max(6,min(18,min(x2-x1+1,y2-y1+1)*.15)):
                        target=pixels[nearest]
                    else:
                        _review(component,'contato_interno_sem_tinta_confirmada')
                        if require_ink and not manual:continue
                else:
                    _review(component,'contato_interno_sem_tinta_confirmada')
                    if require_ink and not manual:continue
                terminal['bbox_constrained']=True
                if _manual(component):
                    terminal['identity_uncertain']=True
                    _review(component,'correcao_manual_antiga_fora_da_caixa')
            terminal['position']={'x':round(float(target[0]),2),'y':round(float(target[1]),2)}
            verified.append(terminal)
        component['terminals']=verified
        points=[tuple(t['position'].values()) for t in component.get('terminals') or []]
        if len(set(points))!=len(points):
            _review(component,'terminais_internos_sobrepostos')
        component.setdefault('terminal_detection',{})['pin_count']=len(points)
    result['terminal_count']=sum(len(c.get('terminals') or []) for c in result.get('components') or [])
    result['review_count']=sum(bool(c.get('review_required') or
        (c.get('terminal_detection') or {}).get('review_required')) for c in result.get('components') or [])
    result['terminals_inside_component_bbox']=True
    return result


def _contacts(image,bbox,skeleton):
    """Trace observed ink across each box face; tolerate curved/thin leads.

    The wire graph is read-only context. Its component erasure leaves gaps at
    the boundary, so contacts are traced in the source ink, not at graph
    centroids or extrapolated straight-line intersections.
    """
    mask,origin=_crop(image,bbox)
    ox,oy=origin.astype(int)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    gray=gray[oy:oy+mask.shape[0],ox:ox+mask.shape[1]]
    threshold,_=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    observed=(gray<=min(210,threshold+5)).astype(np.uint8)*255
    # Rule removal can erase the lighter center of a wire crossing a crease.
    # Restore only photographed ink that bridges two retained stroke pieces;
    # closing by itself must never manufacture pixels in blank paper.
    bridge=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    mask=cv2.bitwise_or(mask,cv2.bitwise_and(bridge,observed))
    values=gray[mask>0]
    from backend.drawing_mode import digital
    is_digital=digital()
    dark_limit=245 if is_digital else (min(210,float(np.percentile(values,25))+60) if len(values) else 0)
    x1,y1,x2,y2=_bounds(bbox,image.shape)
    local=_skeletonize(mask)
    thickness=cv2.distanceTransform(mask,cv2.DIST_L2,3)
    yy,xx=np.nonzero(local)
    pixels=np.stack((xx,yy),axis=1)+origin
    contacts=[]
    scale=min(x2-x1+1,y2-y1+1)
    radius=max(10,min(24,int(scale*.16)))
    depth=max(3,min(8,int(scale*.05)))
    for axis,edge,inward in ((0,x1,1),(0,x2,-1),(1,y1,1),(1,y2,-1)):
        other=1-axis
        lo,hi=(y1,y2) if axis==0 else (x1,x2)
        # Group crossings along a face without merging the distant leads.
        selected=pixels[(np.abs(pixels[:,axis]-edge)<=3)&(pixels[:,other]>=lo)&(pixels[:,other]<=hi)]
        crossing=[]
        strengths={}
        for p in selected:
            neighbors=pixels[np.linalg.norm(pixels-p,axis=1)<=5]-p
            variance=np.mean(neighbors**2,axis=0)
            if variance[axis]>=max(.15,variance[other]*.035):
                crossing.append(p)
                px,py=np.rint(p-origin).astype(int)
                strengths[tuple(p)]=variance[axis]/(variance[other]+1)*thickness[py,px]
        selected=np.array(crossing).reshape(-1,2)
        if not len(selected):continue
        groups=[]
        for p in sorted(selected,key=lambda p:p[other]):
            if not groups or p[other]-groups[-1][-1][other]>3:
                groups.append([])
            groups[-1].append(p)
        separated=[]
        for group in groups:
            if group[-1][other]-group[0][other]<=9:
                separated.append(group)
            else:
                peaks=[]
                for p in sorted(group,key=lambda p:strengths[tuple(p)],reverse=True):
                    if not any(abs(p[other]-q[other])<8 for q in peaks):peaks.append(p)
                separated.extend([[p] for p in peaks])
        for group in separated:
            center=np.mean(group,axis=0)
            # Choose a real ink pixel on the first interior row/column. This
            # also covers a thinning endpoint one pixel outside the box.
            target=center.copy();target[axis]=edge
            mx,my=np.rint(target-origin).astype(int)
            ax,ay=max(0,mx-radius),max(0,my-radius)
            bx,by=min(mask.shape[1],mx+radius+1),min(mask.shape[0],my+radius+1)
            patch=mask[ay:by,ax:bx]
            py,px=np.nonzero(patch)
            points=np.stack((px+ax,py+ay),axis=1)+origin
            inside=(points[:,0]>=x1)&(points[:,0]<=x2)&(points[:,1]>=y1)&(points[:,1]<=y2)
            near=inside&(np.abs(points[:,axis]-edge)<=3)
            if not np.any(near):continue
            available=points[near]
            distance=np.linalg.norm(available-target,axis=1)
            local_pixels=np.rint(available-origin).astype(int)
            cost=distance+gray[local_pixels[:,1],local_pixels[:,0]]*.02
            p=available[int(np.argmin(cost))]
            # Connected ink must continue both into and out of the box. A
            # polarity mark or a nearby, detached line cannot create a port.
            connected=patch if is_digital else cv2.morphologyEx(patch,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
            _,labels=cv2.connectedComponents(connected,8)
            ix,iy=np.rint(p-origin-[ax,ay]).astype(int)
            label=labels[iy,ix]
            if not label:continue
            cy,cx=np.nonzero(labels==label)
            chain=np.stack((cx+ax,cy+ay),axis=1)+origin
            signed=(chain[:,axis]-edge)*inward
            if signed.max()<depth or signed.min()>-min(3,depth):continue
            # Page rules may also cross a face. Compare their local stroke
            # darkness with the component ink, rather than accepting every
            # connected black pixel from an adaptive threshold.
            ix0,iy0=np.rint(p-origin).astype(int)
            nearby=gray[max(0,iy0-2):iy0+3,max(0,ix0-2):ix0+3]
            nearby_mask=mask[max(0,iy0-2):iy0+3,max(0,ix0-2):ix0+3]>0
            if np.median(nearby[nearby_mask])>dark_limit:continue
            wire_distance=float(radius)
            context_confirmed=False
            if skeleton is not None:
                sy,sx=np.nonzero(skeleton[max(0,int(p[1])-radius):min(skeleton.shape[0],int(p[1])+radius+1),
                    max(0,int(p[0])-radius):min(skeleton.shape[1],int(p[0])+radius+1)])
                if len(sx):
                    wire_pixels=np.stack((sx+max(0,int(p[0])-radius),sy+max(0,int(p[1])-radius)),axis=1)
                    wire_distance=float(np.linalg.norm(wire_pixels-p,axis=1).min())
                    context_confirmed=True
            candidate={'point':p,'stage8_distance_px':wire_distance,
                       'stage8_context_confirmed':context_confirmed,'source_ink_continuity':True,'face_axis':axis}
            existing=next((q for q in contacts if np.linalg.norm(p-q['point'])<7),None)
            if existing is None:contacts.append(candidate)
            elif wire_distance<existing['stage8_distance_px']:existing.update(candidate)
    # Expanded or rotated boxes can contain a whole short lead. Recover its
    # endpoint near the perimeter when the wire skeleton directly confirms
    # it, rather than forcing an intersection into empty paper at the face.
    binary=(local>0).astype(np.uint8)
    degree=cv2.filter2D(binary,cv2.CV_16S,np.ones((3,3),np.int16))-binary
    ey,ex=np.nonzero((binary>0)&(degree==1))
    for p in np.stack((ex,ey),axis=1)+origin:
        if not (x1<=p[0]<=x2 and y1<=p[1]<=y2):continue
        edge_distance=min(p[0]-x1,x2-p[0],p[1]-y1,y2-p[1])
        if edge_distance>max(8,scale*.18) or any(np.linalg.norm(p-q['point'])<12 for q in contacts):continue
        if skeleton is None:continue
        px,py=np.rint(p).astype(int)
        sy,sx=np.nonzero(skeleton[max(0,py-4):py+5,max(0,px-4):px+5])
        if not len(sx):continue
        contacts.append(dict(point=p,stage8_distance_px=float(np.linalg.norm(np.stack((sx+max(0,px-4),sy+max(0,py-4)),axis=1)-p,axis=1).min()),
                             stage8_context_confirmed=True,source_ink_continuity=True,near_edge_endpoint=True))
    return contacts


def _exclude_polarity_glyph_contacts(image,component,contacts):
    """A polarity glyph is evidence for a name, never a detached conductor.

    The saved wire skeleton can include glyphs, so proximity to it alone is
    insufficient. Reject both border crossings and thinning endpoints of a
    detached '+' blob. For an attached mark, require a conductor continuing
    beyond BOTH sides of its arms before accepting a nearby contact.
    """
    if component.get('type') not in POLARIZED_TWO_PIN_CLASSES:return contacts,0
    bbox=component['bbox'];mask,origin=_crop(image,bbox)
    scale=min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])
    marks,labels=_marks(mask,origin,bbox,include_attached=True,attached_margin=28)
    excluded=[]
    radius=max(6,min(14,int(max(12,scale*.34)*.4)))
    for mark in marks:
        if mark['kind']!='+':continue
        if not mark.get('attached'):
            yy,xx=np.nonzero(labels==mark['label'])
            if len(xx):
                excluded.append(('blob',np.array([xx.min(),yy.min()])+origin-2,
                                 np.array([xx.max(),yy.max()])+origin+2))
            continue
        # A '+' may be drawn on a wire: preserve that wire if it continues
        # beyond the glyph. A branch ending in a '+' has no such continuation.
        through=False
        for angle in np.arange(0,math.pi,math.pi/12):
            direction=np.array([math.cos(angle),math.sin(angle)])
            if min(_support(mask,mark['point'],sign*direction,np.arange(radius+3,radius+12,2),origin)
                   for sign in (-1,1))>=.7:
                through=True;break
        if not through:excluded.append(('attached',mark['point'],radius+3))
    def allowed(contact):
        point=contact['point']
        for kind,a,b in excluded:
            if kind=='blob' and np.all(point>=a) and np.all(point<=b):return False
            if kind=='attached' and np.linalg.norm(point-a)<=b:return False
        return True
    kept=[contact for contact in contacts if allowed(contact)]
    return kept,len(contacts)-len(kept)


def _select_polarized_contacts(image,component,contacts,*,plate_body=None):
    """Select opposite lead contacts instead of plate tips nearest a bad anchor."""
    if component.get('type') not in {'battery','capacitor_polarized','capacitor'} or not contacts:return contacts,None,None
    from backend.terminal_identity import _polarized_plate_body
    old=component.get('terminals') or []
    if len(old)!=2:return contacts,None,None
    pair=np.asarray([[t['position']['x'],t['position']['y']] for t in old])
    body=plate_body or _polarized_plate_body(image,component['bbox'],pair,component['type'])
    if not body or body.get('source')!='outer_plates_with_opposed_leads':return contacts,None,body
    unit=body['axis'];normal=np.array([-unit[1],unit[0]]);center=body['center']
    plate_offsets=sorted(float((p['center']-center)@unit) for p in body['plates']);gap=plate_offsets[-1]-plate_offsets[0]
    sides=[[],[]]
    for c in contacts:
        if c.get('face_axis') is not None and abs(unit[c['face_axis']])<.4:continue
        along=float((c['point']-center)@unit)
        side=0 if along<=plate_offsets[0]-max(1,.04*gap) else 1 if along>=plate_offsets[-1]+max(1,.04*gap) else None
        if side is not None:sides[side].append(c)
    chosen=[]
    for side in sides:
        if side:
            chosen.append(min(side,key=lambda c:abs(float((c['point']-center)@normal))+.3*c['stage8_distance_px']))
    # Preserve only observed contacts; never create a point at an empty face.
    meta=dict(method='opposed_outer_plate_lead_contacts',candidates=len(contacts),selected=len(chosen),
              rejected_plate_or_nonopposed_contacts=len(contacts)-len(chosen),axis=[round(float(v),4) for v in unit])
    return chosen,meta,body

def _source_lead_contacts(image,component,contacts):
    if component.get('type')!='source_dc' or len(contacts)<2:return contacts,None
    # The circle is a recovery cue for an incomplete proposal. A complete
    # existing pair can follow a bent lead that leaves the circle's axis.
    if len(component.get('terminals') or [])==2:return contacts,None
    from backend.terminal_identity import _circle_axis
    mask,origin=_crop(image,component['bbox'])
    leads=_circle_axis(mask,origin,component['bbox'])
    if leads is None:return contacts,None
    anchors=np.asarray(leads);scale=min(component['bbox']['x2']-component['bbox']['x1'],component['bbox']['y2']-component['bbox']['y1'])
    ranked=[]
    for indexes in permutations(range(len(contacts)),2):
        distance=np.linalg.norm(np.array([contacts[j]['point'] for j in indexes])-anchors,axis=1)
        if distance.max()>max(6,scale*.22):continue
        ranked.append((float(distance.sum()),indexes))
    if not ranked:return contacts,None
    ranked.sort();cost,indexes=ranked[0]
    if len(ranked)>1 and ranked[1][0]-cost<max(3,scale*.06):return contacts,None
    return [contacts[i] for i in indexes],dict(method='source_circle_and_two_opposed_observed_leads',rejected=len(contacts)-2)


def _power_lead_contacts(image,component,contacts):
    if component.get('type') not in {'vcc','gnd'} or not contacts:return contacts,None
    if len(contacts)<2 and component.get('type')!='vcc':return contacts,None
    box=component['bbox'];mask,origin=_crop(image,box)
    _,labels=cv2.connectedComponents(mask,8)
    if component.get('type')=='vcc':
        from backend.vcc_topology import hollow_power_body
        body=hollow_power_body(image,box)
        if body:
            # The ring can be locally lighter than its stem and split in the
            # symbol-only mask. A unique observed contact beyond its measured
            # outline is the lead; rim contacts remain inside that outline.
            external=[contact for contact in contacts if contact.get('source_ink_continuity') and
                (contact['point'][0]<body['x1']-1 or contact['point'][0]>body['x2']+1 or
                 contact['point'][1]<body['y1']-1 or contact['point'][1]>body['y2']+1)]
            if len(external)==1:
                return external,dict(method='unique_observed_stem_beyond_measured_hollow_ring',rejected=len(contacts)-1)
            x,y,z,t=[int(body[k]-origin[0 if k.startswith('x') else 1]) for k in ('x1','y1','x2','y2')]
            owned=set(labels[max(0,y):min(labels.shape[0],t+1),max(0,x):min(labels.shape[1],z+1)].ravel())-{0}
            selected=[]
            for contact in contacts:
                x,y=np.rint(contact['point']-origin).astype(int)
                paths=set(labels[max(0,y-2):y+3,max(0,x-2):x+3].ravel())-{0}
                if paths.intersection(owned):selected.append(contact)
            external=[contact for contact in selected if
                contact['point'][0]<body['x1']-1 or contact['point'][0]>body['x2']+1 or
                contact['point'][1]<body['y1']-1 or contact['point'][1]>body['y2']+1]
            if len(external)==1:
                return external,dict(method='actual_power_stem_outside_hollow_ring',rejected=len(contacts)-1)
            if len(selected)==1:
                return selected,dict(method='actual_power_ring_connected_to_one_observed_lead',rejected=len(contacts)-1)
    center=np.array([(box['x1']+box['x2'])/2,(box['y1']+box['y2'])/2])
    ranked=[]
    for contact in contacts:
        delta=contact['point']-center;distance=float(np.linalg.norm(delta))
        if distance<3:continue
        unit=delta/distance
        x,y=np.rint(contact['point']-origin).astype(int)
        owned=set(labels[max(0,y-2):y+3,max(0,x-2):x+3].ravel())-{0}
        votes=[]
        for step in np.arange(3,15,2):
            x,y=np.rint(contact['point']+unit*step-origin).astype(int)
            zone=labels[max(0,y-1):y+2,max(0,x-1):x+2] if 0<=x<labels.shape[1] and 0<=y<labels.shape[0] else np.zeros(0)
            votes.append(bool(owned & set(zone.ravel())))
        support=float(np.mean(votes))
        if support>=.65:ranked.append((support,contact))
    if not ranked:return contacts,None
    ranked.sort(key=lambda p:p[0],reverse=True)
    if len(ranked)>1 and ranked[0][0]-ranked[1][0]<.2:return contacts,None
    return [ranked[0][1]],dict(method='single_outward_power_lead_not_symbol_rim',rejected=len(contacts)-1)


def refine_all_terminal_contacts(image,result,wires_info,skeleton):
    """Match electrical identities to observed leads, with a distinct contact per pin."""
    result=deepcopy(result)
    from backend.paper_ink import wire_analysis_image
    photograph=image
    image=wire_analysis_image(image,wires_info)
    local_paper=bool((wires_info or {}).get('conductor_ink_policy',{}).get('active'))
    profile=(wires_info or {}).get('ink_color_profile') or estimate_component_ink(image,result)
    _,ruling_mask=apply_notebook_rule_profile(image,(wires_info or {}).get('notebook_rule_profile'),profile)
    reference=image_with_component_ink(image,profile)
    from backend.external_conductor_ink import local_color_mask
    local_pen=local_color_mask(image,profile,ruling_mask)
    reference[local_pen>0]=image[local_pen>0]
    contact_view=cv2.cvtColor(255-local_pen*255,cv2.COLOR_GRAY2BGR) if np.any(local_pen) else image
    summary={'method':'all_classes_original_ink_inside_bbox_after_wires','components':[],
             'refined_count':0,'manual_preserved_count':0}
    for component in result.get('components') or []:
        terms=component.get('terminals') or []
        item={'component_id':component.get('component_id'),'type':component.get('type')}
        if _manual(component):
            summary['manual_preserved_count']+=1
            item['status']='manual_preserved'
            summary['components'].append(item)
            continue
        try:
            contacts=_contacts(contact_view,component['bbox'],skeleton)
            from backend.faint_capacitor_contacts import recover as recover_faint_capacitor_contacts
            contacts,faint_capacitor=recover_faint_capacitor_contacts(photograph,component,contacts,skeleton)
            if faint_capacitor:item['faint_capacitor_contact_recovery']=faint_capacitor
            # Shadows can make one photographed lead much lighter than the
            # global ink sample. Only a lead validated against both outer
            # plates may use the same hue at a different brightness.
            observed_polarized,_,original_plate_body=_select_polarized_contacts(image,component,contacts)
            plate_leads={id(p) for p in observed_polarized} if original_plate_body and original_plate_body.get('source')=='outer_plates_with_opposed_leads' else set()
            if profile.get('active') and profile.get('chromatic'):
                # Neutral page rules cannot become terminals of a colored
                # circuit. Restore a nearby matching ink pixel if necessary.
                box=component['bbox'];lo_x,lo_y,hi_x,hi_y=_bounds(box,image.shape)
                colored=[]
                for port in contacts:
                    px,py=np.rint(port['point']).astype(int)
                    ax,ay=max(lo_x,px-3),max(lo_y,py-3)
                    bx,by=min(hi_x+1,px+4),min(hi_y+1,py+4)
                    zone=reference[ay:by,ax:bx]
                    if zone.size==0:continue
                    yy,xx=np.nonzero(np.min(zone,axis=2)<210)
                    if not len(xx):
                        # Paper tint and shadows change raw pen hue across a
                        # photograph. A continuous boundary lead, confirmed by
                        # the wire graph, may use a nearby shade of the same ink.
                        # Internal bar/arrow endpoints cannot use this fallback.
                        hue,saturation,value=cv2.cvtColor(image[py:py+1,px:px+1],cv2.COLOR_BGR2HSV)[0,0]
                        delta=abs(float(hue)-float(profile['reference_hsv'][0]))
                        family=min(delta,180-delta)<=min(30,float(profile['hue_tolerance'])+18)
                        if (not port.get('near_edge_endpoint') and port.get('source_ink_continuity')
                                and port.get('stage8_context_confirmed') and port['stage8_distance_px']<=3
                                and family and saturation>=25 and value<=180):
                            port['local_wire_shade_confirmed']=True
                            colored.append(port)
                            continue
                        if id(port) in plate_leads:
                            hue,saturation,value=cv2.cvtColor(image[py:py+1,px:px+1],cv2.COLOR_BGR2HSV)[0,0]
                            delta=abs(float(hue)-float(profile['reference_hsv'][0]))
                            if min(delta,180-delta)<=profile['hue_tolerance'] and saturation>=15 and value<=210:
                                port['local_plate_ink_brightness_confirmed']=True
                                colored.append(port)
                        continue
                    candidates=np.stack((xx+ax,yy+ay),axis=1)
                    port['point']=candidates[int(np.argmin(np.linalg.norm(candidates-port['point'],axis=1)))]
                    colored.append(port)
                contacts=colored
        except (ValueError,KeyError,TypeError):
            contacts=[]
            original_plate_body=None
        kept=[]
        for contact in contacts:
            x,y=np.rint(contact['point']).astype(int)
            # A real lead can follow or cross a ruling at the box boundary.
            # Keep the observed contact if the already computed wire skeleton
            # reaches it within five pixels. The ruling color alone must not
            # remove a confirmed conductor; distant page lines stay excluded.
            retained_wire=bool(contact.get('stage8_context_confirmed')) and contact['stage8_distance_px']<=5
            if not ruling_mask[y,x] or retained_wire:kept.append(contact)
        item['rejected_notebook_rule_contacts']=len(contacts)-len(kept)
        contacts=kept
        contacts,rejected_glyphs=_exclude_polarity_glyph_contacts(reference,component,contacts)
        item['rejected_polarity_glyph_contacts']=rejected_glyphs
        contacts,source_selection=_source_lead_contacts(image,component,contacts)
        if source_selection:item['source_contact_selection']=source_selection
        if component.get('type')=='source_dc' and len(terms)==2:
            # A curved lead can end the local skeleton inside the proposal.
            # If both actual wires cross the box, those crossings establish
            # the source ports. An internal fragment must not beat a real
            # exit merely because it is closer to an old straight-line pin.
            boundary=[c for c in contacts if c.get('source_ink_continuity') and not c.get('near_edge_endpoint')]
            if len(boundary)==2 and len(contacts)>2:
                item['rejected_source_internal_endpoints']=len(contacts)-2
                contacts=boundary
        from backend.resistor_contacts import lead_contacts
        from backend.drawing_mode import CURRENT
        resistor_image=photograph if CURRENT.get()=='handwritten' else reference
        contacts,resistor_selection=lead_contacts(resistor_image,component,contacts)
        if resistor_selection:item['resistor_contact_selection']=resistor_selection
        contacts,power_selection=_power_lead_contacts(photograph if component.get('type')=='vcc' else image,component,contacts)
        if power_selection:item['power_contact_selection']=power_selection
        contacts,plate_selection,plate_body=_select_polarized_contacts(reference,component,contacts,plate_body=original_plate_body)
        if plate_selection:item['polarized_contact_selection']=plate_selection
        from backend.diode_contacts import select_diode_contacts
        from backend.drawing_mode import CURRENT
        diode_image=photograph if CURRENT.get()=='handwritten' else reference
        contacts,diode_selection=select_diode_contacts(diode_image,component,contacts)
        if diode_selection:item['diode_contact_selection']=diode_selection
        # A second pass over the same photograph must retain a previously
        # observed boundary contact if its original ink is still present.
        # Model proposals and points on newly blank paper cannot use this.
        for terminal in terms:
            if component.get('type') in {'mosfet_n','mosfet_p'} and terminal.get('identity_uncertain'):continue
            if not (terminal.get('wire_contact_confirmed') and terminal.get('original_ink_confirmed')):continue
            position=terminal.get('position') or {}
            if not all(isinstance(position.get(k),(int,float)) and math.isfinite(position[k]) for k in ('x','y')):continue
            point=np.array([position['x'],position['y']],float)
            bx=component['bbox'];px,py=np.rint(point).astype(int)
            if not (bx['x1']<px<bx['x2'] and bx['y1']<py<bx['y2'] and 0<=py<image.shape[0] and 0<=px<image.shape[1]):continue
            if float(np.min(photograph[py,px]))>=210 or ruling_mask[py,px] and not terminal.get('stage8_context_confirmed'):continue
            tolerance=12 if component.get('type')=='resistor' and resistor_selection else 6
            near=[p for p in contacts if np.linalg.norm(p['point']-point)<=tolerance]
            if near:
                min(near,key=lambda p:np.linalg.norm(p['point']-point))['point']=point
            elif len(contacts)<len(PIN_SCHEMAS.get(component.get('type'),[])):
                contacts.append(dict(point=point,stage8_distance_px=float(terminal.get('stage8_context_distance_px',0)),
                    stage8_context_confirmed=bool(terminal.get('stage8_context_confirmed')),source_ink_continuity=True,
                    local_wire_shade_confirmed=bool(terminal.get('local_wire_shade_confirmed')),retained_observed_contact=True))
        item['contact_count']=len(contacts)
        # A failed proposal must not discard a clearly observed power/passive
        # lead. Rebuild only a complete, unambiguous set of physical contacts;
        # the class specifies pin count, never a missing conductor's position.
        recoverable={'gnd':('GND',),'vcc':('VCC',),'resistor':('1','2'),
                     'capacitor':('1','2'),'inductor':('1','2')}
        pins=recoverable.get(component.get('type'))
        kind=component.get('type')
        if len(terms)<3 and len(contacts)>=3 and kind in {'transistor_npn','transistor_pnp','opamp','mosfet_n','mosfet_p'}:
            # A missing learned proposal is not a missing physical lead. Only
            # rebuild the full set when the photographed body establishes all
            # three roles; the final contact assignment still checks real ink.
            from backend.local_symbol_roles import small_bjt,small_mosfet
            if kind.startswith('transistor'):
                points,proof=small_bjt(photograph,component,skeleton)
                proven=proof.get('arrow_confirmed')
            elif kind.startswith('mosfet'):
                points,proof=small_mosfet(photograph,component);proven=proof.get('confirmed')
            else:
                from backend.terminal_identity import resolve_terminal_identity
                proposed,proof=resolve_terminal_identity(photograph,component,[])
                points={t['pin']:np.array([t['position']['x'],t['position']['y']]) for t in proposed}
                proven=proof.get('confirmed')
            if proven and points and len(points)==3:
                order=('B','C','E') if kind.startswith('transistor') else ('G','D','S') if kind.startswith('mosfet') else ('+','-','OUT')
                terms=[dict(pin=pin,pin_id=f"{component.get('component_id','COMP')}.{pin}",index=i+1,
                            position=dict(x=float(points[pin][0]),y=float(points[pin][1])),confidence=.90,
                            source='complete_three_observed_leads_and_body_identity') for i,pin in enumerate(order)]
                item['recovered_missing_proposals']=3
        if len(terms)<2 and len(contacts)==2 and component.get('type') in {'source_dc','battery'}:
            from backend.terminal_identity import resolve_terminal_identity
            recovered,identity=resolve_terminal_identity(photograph,component,[])
            if identity.get('confirmed') and len(recovered)==2:
                terms=recovered
                item['recovered_missing_proposals']=2
        if pins and len(terms)<len(pins) and len(contacts)==len(pins):
            recovered_count=len(pins)-len(terms)
            axis=int(np.argmax(np.ptp(np.array([p['point'] for p in contacts]),axis=0))) if len(contacts)>1 else 0
            contacts=sorted(contacts,key=lambda p:p['point'][axis])
            # Keep the identity of every surviving proposal. The remaining
            # name belongs to the independently observed remaining contact.
            ordered_pins=list(pins)
            if terms:
                assigned={}
                for terminal in terms:
                    if terminal.get('pin') not in pins:continue
                    point=np.array([terminal['position']['x'],terminal['position']['y']])
                    index=int(np.argmin([np.linalg.norm(p['point']-point) for p in contacts]))
                    if index in assigned:break
                    assigned[index]=terminal['pin']
                else:
                    missing=iter(pin for pin in pins if pin not in assigned.values())
                    ordered_pins=[assigned[i] if i in assigned else next(missing) for i in range(len(contacts))]
            terms=[dict(pin=pin,pin_id=f"{component.get('component_id','COMP')}.{pin}",index=i+1,
                        electrical_role='ground' if pin=='GND' else 'power' if pin=='VCC' else 'passive',
                        position=dict(x=float(p['point'][0]),y=float(p['point'][1])),confidence=.85,
                        source='complete_observed_external_contact_set') for i,(pin,p) in enumerate(zip(ordered_pins,contacts))]
            component.setdefault('terminal_detection',{})
            item['recovered_missing_proposals']=recovered_count
        if not terms or not contacts or len(terms)>4:
            component['terminals']=[]
            item['status']='no_visible_contact'
            summary['components'].append(item)
            continue
        anchors=np.array([[t['position']['x'],t['position']['y']] for t in terms])
        if component.get('type') in {'mosfet_n','mosfet_p'}:
            from backend.local_symbol_roles import small_mosfet
            proven_roles,role_proof=small_mosfet(photograph,component)
            if proven_roles and role_proof.get('confirmed') and all(t['pin'] in proven_roles for t in terms):
                anchors=np.array([proven_roles[t['pin']] for t in terms])
                item['mosfet_observed_role_anchors']=role_proof
        if diode_selection and {t.get('pin') for t in terms}=={'A','K'}:
            anchors=np.array([diode_selection['selected_contacts'][t['pin']] for t in terms])
        # Bound work in crowded boxes without discarding any pin's nearest lead.
        indexes=set()
        for anchor in anchors:
            indexes.update(sorted(range(len(contacts)),key=lambda i:np.linalg.norm(contacts[i]['point']-anchor))[:3])
        ranked=[]
        count=min(len(terms),len(contacts))
        for pin_indexes in combinations(range(len(terms)),count):
            for chosen in permutations(sorted(indexes),count):
                distances=[float(np.linalg.norm(contacts[i]['point']-anchors[j])) for i,j in zip(chosen,pin_indexes)]
                ranked.append((sum(distances),chosen,distances,pin_indexes))
        if not ranked:
            continue
        ranked.sort(key=lambda v:v[0])
        _,chosen,distances,pin_indexes=ranked[0]
        bbox=component['bbox'];scale=min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])
        ambiguous=len(ranked)>1 and ranked[1][0]-ranked[0][0]<max(3,scale*.04)
        confirmed=[]
        for j,index in zip(pin_indexes,chosen):
            terminal=terms[j]
            p=contacts[index]['point']
            terminal['position']={'x':float(p[0]),'y':float(p[1])}
            terminal['source']='wire_contact_inside_component_bbox'
            terminal['original_ink_confirmed']=True
            terminal['wire_contact_confirmed']=True
            terminal['stage8_context_distance_px']=contacts[index]['stage8_distance_px']
            terminal['stage8_context_confirmed']=contacts[index].get('stage8_context_confirmed',False)
            terminal['local_wire_shade_confirmed']=bool(contacts[index].get('local_wire_shade_confirmed'))
            terminal['search_radius_px']=max(8,int(math.ceil(contacts[index]['stage8_distance_px']))+6)
            if ambiguous:
                terminal['identity_uncertain']=True
            confirmed.append(terminal)
        component['terminals']=confirmed
        component['terminal_detection']['unobserved_pins']=[terms[j]['pin'] for j in range(len(terms)) if j not in pin_indexes]
        component['terminal_detection']['pin_count']=len(confirmed)
        if ambiguous:
            _review(component,'atribuicao_de_pinos_aos_fios_ambigua')
        component.setdefault('terminal_detection',{})['contact_method']=summary['method']
        # Electrical labels are resolved against final observed contacts; the
        # approved wire processing and the contact coordinates remain intact.
        from backend.terminal_identity import finalize_polarity_on_contacts
        # Blue notebook rules crossing a neutral battery lead were read as
        # a plus sign. The learned component ink view removes that unrelated
        # color while retaining the measured battery plates and contacts.
        polarity_image=photograph if local_paper else reference if component.get('type')=='battery' and profile.get('active') else image
        polarity=finalize_polarity_on_contacts(polarity_image,component,components=result.get('components'),plate_body=plate_body)
        if polarity is not None:
            item['polarity']=polarity
        from backend.circuit_context_review import finalize_diode_on_contacts
        diode_identity=finalize_diode_on_contacts(diode_image,component)
        if diode_identity is not None:item['diode_identity']=diode_identity
        from backend.terminal_identity import finalize_opamp_on_contacts
        opamp_identity=finalize_opamp_on_contacts(photograph,component)
        if opamp_identity is not None:item['opamp_identity']=opamp_identity
        from backend.terminal_contact_identity import finalize_bjt_on_contacts
        bjt_identity=finalize_bjt_on_contacts(photograph,component,skeleton,photograph if local_paper else reference)
        if bjt_identity is not None:item['bjt_identity']=bjt_identity
        if component.get('type') in {'mosfet_n','mosfet_p'}:
            from backend.local_symbol_roles import small_mosfet,apply_roles
            local,local_meta=small_mosfet(photograph,component)
            if local_meta.get('confirmed') and apply_roles(component,local,local_meta):
                item['mosfet_local_identity']=local_meta
        component.pop('terminal_candidates_for_review',None)
        item['status']='wire_contacts_inside_bbox'
        summary['refined_count']+=1
        summary['components'].append(item)
    result['wire_terminal_contacts']=summary
    result['pipeline_order']=['components','wires','terminals','junctions']
    from backend.drawing_mode import digital
    if digital():
        from backend.printed_bjt import finalize_circle_contacts
        for component in result.get('components',[]):finalize_circle_contacts(photograph,component)
    return constrain_terminal_positions(result,image,require_ink=True)
