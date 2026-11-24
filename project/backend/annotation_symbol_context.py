"""Pair photographed names/models before assigning to a verified physical body.

This module never supplies label digits, values, or image-specific coordinates.
Strong text is a hypothesis; a separate observed symbol must confirm its type.
"""
from copy import deepcopy
import math
import numpy as np

IDS={'resistor':0,'capacitor':1,'capacitor_polarized':2,'diode':3,'led':4,'inductor':5,
     'source_dc':6,'battery':7,'gnd':8,'vcc':9,'transistor_npn':10,'transistor_pnp':11,
     'mosfet_n':12,'mosfet_p':13,'opamp':14}
BJT={'transistor_npn','transistor_pnp'}
MOS={'mosfet_n','mosfet_p'}
CAP={'capacitor','capacitor_polarized'}

def confidence(t):
    v=float(t.get('confidence',0));return v/100 if v>1 else v

def frame(b):
    return dict(b,width=b['x2']-b['x1'],height=b['y2']-b['y1'],
                center_x=(b['x1']+b['x2'])/2,center_y=(b['y1']+b['y2'])/2)

def retained_context(before,observed):
    # The body evidence is a correction, not a replacement for real lead
    # context. Small moves retain native contacts and avoid snapping every
    # photographed component to an artificially tight drawing convention.
    return {k:.75*float(before[k])+.25*float(observed[k]) for k in ('x1','y1','x2','y2')}

def annotation_free(image,tokens):
    from backend.reference_labels import reference_classes
    from backend.component_models import model_classes
    from backend.measurement_text import parse_measurement
    clean=image.copy()
    for t in tokens:
        if confidence(t)<.94:continue
        text=t.get('text','');quantity=parse_measurement(text) or parse_measurement(text,'resistor')
        if not(reference_classes(text) or model_classes(text) or quantity and (quantity.get('prefix') or quantity.get('unit_explicit'))):continue
        b=t['bbox'];x,y=max(0,int(b['x1'])),max(0,int(b['y1']));z,v=min(image.shape[1],int(math.ceil(b['x2']))),min(image.shape[0],int(math.ceil(b['y2'])))
        clean[y:v,x:z]=255
    return clean

def proposals(b,image):
    # Short local alternatives let a compound learned box release its attached
    # wire/label context. Evidence is measured again inside each alternative.
    yield {k:float(b[k]) for k in ('x1','y1','x2','y2')}
    cx,cy=(b['x1']+b['x2'])/2,(b['y1']+b['y2'])/2
    for scale in (.70,1.15):
        w,h=(b['x2']-b['x1'])*scale,(b['y2']-b['y1'])*scale
        yield dict(x1=max(0,cx-w/2),y1=max(0,cy-h/2),x2=min(image.shape[1]-1,cx+w/2),y2=min(image.shape[0]-1,cy+h/2))

def evidence(image,box,kind):
    from backend.recognition import _validar_classe_geometrica
    if kind=='opamp':
        from backend.terminal_identity import read_opamp_signs
        marks=read_opamp_signs(image,{'class':'opamp','bbox':box})
        return dict(method='triangle_with_two_observed_input_signs',marks=marks) if {'+','-'}.issubset({t['text'] for t in marks}) else None
    if kind in MOS:
        from backend.terminal_identity import _crop,_marks,_mosfet,_sample
        mask,origin=_crop(image,box);_,labels=_marks(mask,origin,box);result=_mosfet(mask,origin,box,labels)
        if not result:return None
        ports=result[0];g=np.asarray(ports['G']);a,b=np.asarray(ports['D']),np.asarray(ports['S']);span=np.linalg.norm(a-b)
        if span<15 or np.linalg.norm(g-(a+b)/2)<span*.18:return None
        if not all(_sample(mask,p,origin,3) for p in ports.values()):return None
        return dict(method='isolated_gate_and_two_original_channel_leads',ports={k:p.tolist() for k,p in ports.items()})
    if kind=='battery':
        from backend.terminal_identity import _crop,_plate_axis,_sample
        mask,origin=_crop(image,box);pair=_plate_axis(mask,origin,box)
        if pair and pair[1] is not None and all(_sample(mask,p,origin,3) for p in pair[0]):
            return dict(method='unequal_parallel_plates_and_opposed_observed_leads',ports=[p.tolist() for p in pair[0]],positive=pair[1])
        return None
    valid=_validar_classe_geometrica(image,box,kind)
    if valid and valid[0]:return dict(method='independent_local_body_geometry',score=valid[1],details=valid[2])
    return None

def reconcile_annotation_groups(image,detections,rejected,tokens):
    from backend.drawing_mode import CURRENT
    if CURRENT.get()!='handwritten':return detections
    from backend.reference_labels import reference_classes,edge_distance,annotation_group_owner
    from backend.component_models import model_classes,model_ocr_reading
    result=deepcopy(detections);clean=annotation_free(image,tokens)
    references={str(i):t for i,t in enumerate(tokens) if confidence(t)>=.90 and reference_classes(t.get('text','')) and not t.get('inferred_prefix')}
    models=[]
    for t in tokens:
        if confidence(t)<.94:continue
        classes=model_classes(t.get('text',''))
        if not classes:continue
        refs={k:r for k,r in references.items() if reference_classes(r['text'])&classes}
        owner=annotation_group_owner(t,refs)
        if owner is None:continue
        ref=refs[owner];target=classes&reference_classes(ref['text'])
        if len(target)==1 and target&({'opamp'}|BJT|MOS):models.append((t,ref,next(iter(target))))
        elif target=={'diode','led'}:
            # 1N/BAV/UF model and D-number alone do not prove LED emission.
            models.append((t,ref,'diode'))
    # An explicit BAT-number with an aligned voltage is not a capacitor name.
    # Unequal physical plates are still required before correcting the class.
    from backend.measurement_text import parse_measurement
    for ref in references.values():
        targets=reference_classes(ref['text'])
        kind='battery' if targets=={'battery'} else 'inductor' if targets=={'inductor'} else None
        if kind is None:continue
        quantities=[t for t in tokens if confidence(t)>=.90 and (p:=parse_measurement(t.get('text',''),kind)) and p['unit_explicit']
                    and p['unit']==('henry' if kind=='inductor' else 'volt')
                    and annotation_group_owner(t,{kind:ref})==kind]
        if quantities:models.append((quantities[0],ref,kind))
    claimed=set()
    for token,reference,kind in models:
        family=BJT|{'opamp'} if kind=='opamp' else BJT if kind in BJT else MOS|CAP|{'diode'} if kind in MOS else CAP|{'battery'} if kind=='battery' else {'inductor','diode','led'} if kind=='inductor' else {'diode','led'}
        candidates=[(False,d) for d in result if d.get('class') in family]
        candidates += [(True,d) for d in rejected if d.get('class') in family]
        ranked=[]
        for recovered,d in candidates:
            if d.get('manual_reviewed') or (not recovered and d['detection_id'] in claimed):continue
            if recovered and any(other.get('manual_reviewed') and _iou(d['bbox'],other['bbox'])>=.30 for other in result):continue
            b=d['bbox'];size=max(b['x2']-b['x1'],b['y2']-b['y1']);height=max(6,token['bbox']['y2']-token['bbox']['y1'])
            distance=edge_distance(token['bbox'],b)
            if distance>min(130,max(40,size*.65,height*2)):continue
            if edge_distance(reference['bbox'],b)>max(65,size):continue
            for box in proposals(b,clean):
                # A model's rectangular OCR crop can overlap the triangle rim.
                # Signed-triangle evidence always uses unmodified photo ink.
                proof=evidence(image if kind=='opamp' else clean,box,kind)
                if not proof:continue
                if kind=='inductor':
                    details=proof.get('details',{})
                    if not details.get('straight_tip_pair') or details.get('curve_extrema',0)<6:continue
                if kind=='diode':
                    from backend.printed_symbols import printed_emission
                    if printed_emission(clean,box):continue
                center=math.hypot((b['x1']+b['x2']-token['bbox']['x1']-token['bbox']['x2'])/2,(b['y1']+b['y2']-token['bbox']['y1']-token['bbox']['y2'])/2)
                ranked.append((distance+.05*center+int(recovered)*2,recovered,d,box,proof));break
        if not ranked:continue
        _,recovered,d,box,proof=min(ranked,key=lambda r:r[0])
        if recovered:
            # A rejected proposal may share an old serial with a retained one.
            if any(edge_distance(d['bbox'],other['bbox'])==0 and other.get('class')==kind and _iou(d['bbox'],other['bbox'])>.40 for other in result):continue
            d=deepcopy(d);serial=1;used={o['detection_id'] for o in result}
            while 'TXT_REC_'+str(serial) in used:serial+=1
            d['detection_id']='TXT_REC_'+str(serial);result.append(d)
        if d['class']==kind and not recovered:claimed.add(d['detection_id']);continue
        old=d['class'];d.update(**{'class':kind,'class_id':IDS[kind]},resolved_from_class=old,
            bbox=frame(box),class_resolution='photographed_reference_model_pair_and_observed_symbol',
            annotation_type_evidence=dict(reference=reference['text'],model=token['text'],reference_box=reference['bbox'],model_box=token['bbox'],physical=proof))
        if kind=='diode':
            # A false LED proposal can include the model letters as emission
            # arrows. Once disproven, retain its actual triangle/bar and leads.
            from backend.diode_contacts import body_axes
            bodies=body_axes(clean,d['bbox'])
            if bodies:
                body=max(bodies,key=lambda p:p['area']);u=body['axis'];v=body['normal'];h=body['height'];w=body['width'];base=body['base']
                points=np.array([base+u*a+v*n for a in (-max(6,.30*h),h+max(6,.30*h)) for n in (-w*.65,w*.65)])
                d['bbox']=frame(dict(x1=float(points[:,0].min()),y1=float(points[:,1].min()),x2=float(points[:,0].max()),y2=float(points[:,1].max())))
                d['annotation_type_evidence']['body_triangle_bar_verified']=True
        claimed.add(d['detection_id'])
    return result

def _iou(a,b):
    inter=max(0,min(a['x2'],b['x2'])-max(a['x1'],b['x1']))*max(0,min(a['y2'],b['y2'])-max(a['y1'],b['y1']))
    return inter/max(1,(a['x2']-a['x1'])*(a['y2']-a['y1'])+(b['x2']-b['x1'])*(b['y2']-b['y1'])-inter)

def separate_annotation_frames(image,detections,tokens):
    """Use observed IEC outlines so a learned halo cannot swallow nearby text.

    Terminal searches retain the original frame as diagnostic context; short
    connected lead pixels remain in the physical component's visible frame.
    """
    from backend.drawing_mode import CURRENT
    if CURRENT.get()!='handwritten':return detections
    from backend.reference_labels import reference_classes,local_distance
    from backend.resistor_contacts import body_for_box
    result=deepcopy(detections)
    for d in result:
        if d.get('annotation_context_bbox') and str(d.get('symbol_box_refinement','')).startswith('observed_'):
            # Repeating OCR must not repeatedly contract a verified frame.
            continue
        if d.get('class') in MOS and not d.get('manual_reviewed'):
            named=[t for t in tokens if confidence(t)>=.90 and d['class'] in reference_classes(t.get('text','')) and local_distance(t,d) is not None]
            b=mos_body_frame(image,d) if named else None
            old=d['bbox']
            def swallowed(t):
                q=t['bbox'];area=max(1,(q['x2']-q['x1'])*(q['y2']-q['y1']))
                return max(0,min(old['x2'],q['x2'])-max(old['x1'],q['x1']))*max(0,min(old['y2'],q['y2'])-max(old['y1'],q['y1']))/area>=.20
            if b and any(swallowed(t) for t in named):
                d.update(annotation_context_bbox=dict(d['bbox']),bbox=frame(b),symbol_box_refinement='observed_mos_gate_channel_body_excludes_remote_wire_context')
        if d.get('class')=='capacitor' and not d.get('manual_reviewed'):
            named=[t for t in tokens if confidence(t)>=.80 and 'capacitor' in reference_classes(t.get('text','')) and local_distance(t,d) is not None]
            if named:
                from backend.capacitor_geometry import capacitor_body_evidence
                ok,e=capacitor_body_evidence(image,d['bbox'])
                if ok:
                    u=np.asarray(e['electrical_axis']);v=np.array([-u[1],u[0]]);size=max(e['outer_plate_lengths'])
                    points=np.array([np.asarray(c)+u*a+v*s for c,l in zip(e['outer_plate_centers'],e['outer_plate_lengths']) for a in (-max(6,.20*size),max(6,.20*size)) for s in (-l/2-4,l/2+4)])
                    b=dict(x1=float(points[:,0].min()),y1=float(points[:,1].min()),x2=float(points[:,0].max()),y2=float(points[:,1].max()))
                    before=d['bbox']
                    if (before['x2']-before['x1'])*(before['y2']-before['y1'])>1.3*(b['x2']-b['x1'])*(b['y2']-b['y1']):
                        d.update(annotation_context_bbox=dict(before),bbox=frame(retained_context(before,b)),symbol_box_refinement='observed_plate_pair_and_short_leads_exclude_annotation_context',annotation_frame_evidence=e)
        if d.get('class')!='resistor' or d.get('manual_reviewed'):continue
        if str(d.get('symbol_box_refinement','')).startswith('verified_'):continue
        nearby=[t for t in tokens if confidence(t)>=.90 and 'resistor' in reference_classes(t.get('text','')) and local_distance(t,d) is not None]
        if not nearby:continue
        body=body_for_box(image,d['bbox'])
        if not body or not body['evidence'].get('opposed_leads'):continue
        e=body['evidence'];length,width=e['body_size'];axis=np.asarray(e['body_axis']);normal=np.array([-axis[1],axis[0]]);center=np.asarray(body['center'])
        before=d['bbox']
        lead=min(max(6,length*.20),min(e.get('lead_support_px',[6,6]))*.5)
        points=np.array([center+axis*a+normal*n for a in (-length/2-lead,length/2+lead) for n in (-width/2-3,width/2+3)])
        b=frame(dict(x1=max(0,float(points[:,0].min())),y1=max(0,float(points[:,1].min())),x2=min(image.shape[1]-1,float(points[:,0].max())),y2=min(image.shape[0]-1,float(points[:,1].max()))))
        old_area=(before['x2']-before['x1'])*(before['y2']-before['y1']);new_area=b['width']*b['height']
        if old_area<new_area*1.25:continue
        d.update(bbox=frame(retained_context(before,b)),annotation_context_bbox=dict(before),annotation_frame_evidence=e,
                 symbol_box_refinement='observed_resistor_outline_short_leads_exclude_annotation_halo')
    return result

def mos_body_frame(image,detection):
    """Bound a proven MOSFET by its close parallel gate/channel electrodes."""
    from backend.terminal_identity import _crop,_lines,_angle_delta
    box=detection['bbox'];mask,origin=_crop(image,box)
    minimum=min(box['x2']-box['x1'],box['y2']-box['y1'])
    lines=[l for l in _lines(mask,origin) if l['length']>=max(18,minimum*.40)
           and box['x1']<l['center'][0]<box['x2'] and box['y1']<l['center'][1]<box['y2']]
    pairs=[]
    for i,a in enumerate(lines):
        u=(a['b']-a['a'])/a['length'];v=np.array([-u[1],u[0]])
        for b in lines[i+1:]:
            if _angle_delta(a['angle'],b['angle'])>math.radians(7):continue
            gap=abs((b['center']-a['center'])@v);shift=abs((b['center']-a['center'])@u)
            if not 5<=gap<=minimum*.20 or shift>.2*min(a['length'],b['length']):continue
            if min(a['length'],b['length'])<.75*max(a['length'],b['length']):continue
            pairs.append((min(a['length'],b['length']),a,b,u,v))
    if not pairs:return None
    length,a,b,u,v=max(pairs,key=lambda p:p[0]);middle=(a['center']+b['center'])/2
    parts=[l for l in lines if _angle_delta(l['angle'],a['angle'])<math.radians(7)
           and abs((l['center']-middle)@u)<length*.35 and abs((l['center']-middle)@v)<length*.65
           and l['length']<length*1.35]
    points=np.array([p for l in parts for p in (l['a'],l['b'])]);lo,hi=points.min(axis=0)-5,points.max(axis=0)+5
    before=(box['x2']-box['x1'])*(box['y2']-box['y1']);after=float(np.prod(hi-lo))
    if before<after*1.25 or np.any(hi<=lo):return None
    return dict(x1=max(0.,float(lo[0])),y1=max(0.,float(lo[1])),x2=min(image.shape[1]-1.,float(hi[0])),y2=min(image.shape[0]-1.,float(hi[1])))
