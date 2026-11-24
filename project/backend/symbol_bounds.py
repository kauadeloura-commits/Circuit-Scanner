"""Keep symbol boxes from absorbing a neighboring body or clipping its stem."""
from copy import deepcopy
import math


def _core(d):
    b=d['bbox'];m=max(3,min(12,min(b['x2']-b['x1'],b['y2']-b['y1'])*.15))
    return dict(x1=b['x1']+m,y1=b['y1']+m,x2=b['x2']-m,y2=b['y2']-m)


def _overlap(a,b):
    return max(a['x1'],b['x1'])<min(a['x2'],b['x2']) and max(a['y1'],b['y1'])<min(a['y2'],b['y2'])


def _box(b):
    return dict(b,width=b['x2']-b['x1'],height=b['y2']-b['y1'],center_x=(b['x1']+b['x2'])/2,center_y=(b['y1']+b['y2'])/2)


def refine_capacitor_after_text(image,detections):
    """Bound a text-trimmed capacitor by two observed plates and short leads.

    Text removal can leave an asymmetric remnant of the original broad box.
    Only verified separate plate paths with opposed connected leads justify
    moving its edges. Manual boxes and unverified symbols stay as supplied.
    """
    return _refine_capacitor_bounds(image,detections,include_untrimmed=False)


def refine_broad_ordinary_capacitors(image,detections):
    """Separate a broad native proposal only when plates and both leads agree."""
    return _refine_capacitor_bounds(image,detections,include_untrimmed=True)


def refine_verified_polarized_frames(image,detections):
    """Keep detached polarity writing outside a well resolved native body frame.

    Two separate plates, their own sign and two original opposed conductors
    must all be observed. Small or uncertain plate crops keep their old frame.
    """
    from backend.drawing_mode import CURRENT
    if CURRENT.get()!='handwritten':return detections
    from backend.capacitor_geometry import capacitor_body_evidence
    from backend.terminal_wire_contacts import _contacts
    import numpy as np
    result=deepcopy(detections)
    for d in result:
        if d.get('manual_reviewed') or d.get('class')!='capacitor_polarized':continue
        if d.get('symbol_box_refinement')=='verified_polarized_body_and_two_native_leads_detached_sign_retained':continue
        signs=(d.get('capacitor_polarity_validation') or {}).get('marks') or []
        if not any(m.get('plate_owner_verified') and not m.get('attached') for m in signs):continue
        ok,e=capacitor_body_evidence(image,d['bbox'])
        if not ok:continue
        length=max(e['outer_plate_lengths'])
        if length<32:continue
        centers=np.asarray(e['outer_plate_centers']);normal=np.asarray(e['electrical_axis']);axis=np.array([-normal[1],normal[0]])
        points=np.array([c+normal*a*length*.40+axis*n*length*.65 for c in centers for a in (-1,1) for n in (-1,1)])
        b=d['bbox'];tight=dict(x1=max(b['x1'],float(points[:,0].min())),y1=max(b['y1'],float(points[:,1].min())),
                              x2=min(b['x2'],float(points[:,0].max())),y2=min(b['y2'],float(points[:,1].max())))
        if tight['x2']<=tight['x1'] or tight['y2']<=tight['y1']:continue
        ports=_contacts(image,tight,None);center=centers.mean(axis=0);limits=sorted((centers-center)@normal);chosen=[]
        for sign,limit in ((-1,limits[0]),(1,limits[-1])):
            valid=[p for p in ports if p.get('face_axis') is not None and abs(normal[p['face_axis']])>.8
                   and sign*((p['point']-center)@normal-limit)>2 and abs((p['point']-center)@axis)<length*.30]
            if not valid:break
            chosen.append(min(valid,key=lambda p:abs((p['point']-center)@axis)))
        if len(chosen)!=2:continue
        if any(o is not d and _overlap(tight,_core(o)) and not _overlap(b,_core(o)) for o in result):continue
        # Preserve half of the learned lead context. The plate-only frame is
        # evidence for a correction, rather than a new tight detection box.
        tight={k:(tight[k]+b[k])*.5 for k in ('x1','y1','x2','y2')}
        d.update(bbox=_box(tight),symbol_box_refinement='verified_polarized_body_and_two_native_leads_detached_sign_retained',
                 native_plate_frame_evidence=dict(plates=e,contacts=[p['point'].tolist() for p in chosen],polarity_marks=signs))
    return result


def _refine_capacitor_bounds(image,detections,*,include_untrimmed):
    from backend.drawing_mode import CURRENT
    if CURRENT.get()!='handwritten':return detections
    from backend.capacitor_geometry import capacitor_body_evidence
    import numpy as np
    output=deepcopy(detections)
    for d in output:
        if d.get('manual_reviewed'):continue
        if d.get('class') not in {'capacitor','capacitor_polarized'}:continue
        bounded=d.get('symbol_box_refinement') in {'verified_plates_after_text_separation','verified_plates_and_two_native_lead_contacts'}
        if bounded and d.get('plate_lead_context_checked'):continue
        untrimmed=not d.get('bbox_adjusted_for_text') and not bounded
        if include_untrimmed:
            if not untrimmed or d.get('class')!='capacitor':continue
        elif untrimmed:continue
        b=d['bbox'];ok,e=capacitor_body_evidence(image,b)
        if not ok:continue
        centers=np.asarray(e['outer_plate_centers'],float)
        normal=np.asarray(e['electrical_axis'],float);axis=np.array([-normal[1],normal[0]])
        length=max(e['outer_plate_lengths']);points=[]
        for center,size in zip(centers,e['outer_plate_lengths']):
            for side in (-1,1):
                for end in (-1,1):points.append(center+axis*side*(size*.5+length*.12)+normal*end*length*.25)
        points=np.asarray(points)
        tight=({k:float(b[k]) for k in ('x1','y1','x2','y2')} if bounded else
               dict(x1=max(b['x1'],float(points[:,0].min())),y1=max(b['y1'],float(points[:,1].min())),
                    x2=min(b['x2'],float(points[:,0].max())),y2=min(b['y2'],float(points[:,1].max()))))
        if tight['x2']<=tight['x1'] or tight['y2']<=tight['y1']:continue
        if untrimmed:
            if d.get('class')!='capacitor':continue
            old_area=(b['x2']-b['x1'])*(b['y2']-b['y1']);body_area=(tight['x2']-tight['x1'])*(tight['y2']-tight['y1'])
            if old_area<=3*body_area:continue
        from backend.terminal_wire_contacts import _contacts
        center=centers.mean(axis=0);offsets=sorted((centers-center)@normal)
        def observed_pair(box):
            ports=_contacts(image,box,None);chosen=[]
            for sign,limit in ((-1,offsets[0]),(1,offsets[-1])):
                valid=[p for p in ports if p.get('face_axis') is not None and abs(normal[p['face_axis']])>.8
                       and sign*((p['point']-center)@normal-limit)>2
                       and abs((p['point']-center)@axis)<length*.30]
                if not valid:return None
                chosen.append(min(valid,key=lambda p:abs((p['point']-center)@axis)))
            return chosen
        extension=None
        if untrimmed and not observed_pair(tight):continue
        if not observed_pair(tight):
            # A text crop can end on the plate itself. Extend only along the
            # electrical axis, only as far as the short-lead context above,
            # and only when real ink crosses both proposed box faces.
            coordinate=int(np.argmax(np.abs(normal)));name='xy'[coordinate]
            candidate=dict(tight)
            candidate[name+'1']=min(tight[name+'1'],max(0,float(points[:,coordinate].min())))
            candidate[name+'2']=max(tight[name+'2'],min(image.shape[1-coordinate]-1,float(points[:,coordinate].max())))
            enlarged=max(abs(candidate[name+k]-tight[name+k]) for k in ('1','2'))>=2
            intrudes=any(o is not d and _overlap(candidate,_core(o)) and not _overlap(tight,_core(o)) for o in output)
            ports=observed_pair(candidate) if enlarged and not intrudes else None
            if ports:
                tight=candidate
                extension=dict(method='short_axis_context_with_two_observed_outward_ink_contacts',
                               contacts=[p['point'].tolist() for p in ports])
        d['plate_lead_context_checked']=True
        if bounded and not extension:continue
        d.update(bbox=_box(tight),symbol_box_refinement='verified_plates_and_two_native_lead_contacts' if untrimmed else 'verified_plates_after_text_separation',plate_box_evidence=e)
        if extension:d['plate_lead_context_extension']=extension
    return output


def separate_symbol_bounds(image,detections):
    from backend.resistor_geometry import symbol_mask
    from backend.vcc_topology import external_leads
    output=deepcopy(detections)
    for d in output:
        if d.get('manual_reviewed'):continue
        kind=d.get('class');b=d['bbox']
        if kind=='gnd':
            evidence=(d.get('geometry_validation') or {}).get('evidence') or {}
            body=evidence.get('ground_body_bbox') or (d.get('terminal_bar_ground') or {}).get('ground_body_bbox')
            if not body:continue
            intruding=[o for o in output if o is not d and o.get('class') not in {'gnd','vcc'} and _overlap(b,_core(o)) and not _overlap(body,_core(o))]
            if not intruding:continue
            tight={k:(max(0,body[k]-2) if k.endswith('1') else min(image.shape[1 if k.startswith('x') else 0]-1,body[k]+2)) for k in ('x1','y1','x2','y2')}
            if any(_overlap(tight,_core(o)) for o in intruding):continue
            d.update(bbox=_box(tight),symbol_box_refinement='visible_ground_bars_exclude_neighbor_core')
        elif kind=='vcc':
            if d.get('symbol_box_refinement')=='one_verified_stem_context_not_neighbor_body':continue
            head=((d.get('geometry_validation') or {}).get('evidence') or {}).get('power_head_bbox')
            if not head:continue
            diameter=max(head['x2']-head['x1'],head['y2']-head['y1'])
            if diameter<20 or min(b['x2']-b['x1'],b['y2']-b['y1'])>2.5*diameter:continue
            pad=max(2.,min(10.,min(b['x2']-b['x1'],b['y2']-b['y1'])*.10))
            padded=dict(x1=max(0,b['x1']-pad),y1=max(0,b['y1']-pad),x2=min(image.shape[1]-1,b['x2']+pad),y2=min(image.shape[0]-1,b['y2']+pad))
            from backend.drawing_mode import CURRENT
            if CURRENT.get()=='handwritten':
                directions=external_leads(image,b,symbol_mask(image),directions=True)
                if len(directions)==1:
                    # A short stem needs context along its own axis. Adding
                    # paper on the transverse axis makes the head box vague.
                    across='y' if abs(directions[0][0])>abs(directions[0][1]) else 'x'
                    padded[across+'1']=b[across+'1'];padded[across+'2']=b[across+'2']
            if any(o is not d and _overlap(padded,_core(o)) for o in output):continue
            if external_leads(image,padded,symbol_mask(image))!=1:continue
            d.update(bbox=_box(padded),symbol_box_refinement='one_verified_stem_context_not_neighbor_body')
    return output


def clip_coil_box(image,box,detections,validator,mask):
    axis='x' if box['x2']-box['x1']>=box['y2']-box['y1'] else 'y';other='y' if axis=='x' else 'x'
    center=(box[axis+'1']+box[axis+'2'])/2;across=(box[other+'1']+box[other+'2'])/2;clipped=dict(box)
    for d in detections:
        b=d['bbox']
        if not b[other+'1']<across<b[other+'2']:continue
        c=(b[axis+'1']+b[axis+'2'])/2
        if c<center and b[axis+'2']>clipped[axis+'1'] and b[axis+'2']<center:
            clipped[axis+'1']=b[axis+'2']+1
        elif c>center and b[axis+'1']<clipped[axis+'2'] and b[axis+'1']>center:
            clipped[axis+'2']=b[axis+'1']-1
    length=clipped[axis+'2']-clipped[axis+'1']
    if length<max(24,(box[axis+'2']-box[axis+'1'])*.5):return box,None
    valid,score,proof=validator(image,clipped,mask)
    if valid and proof.get('curve_extrema',0)>=6 and proof.get('straight_tip_pair'):return clipped,(score,proof)
    return box,None
