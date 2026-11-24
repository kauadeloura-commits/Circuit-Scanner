"""Recover an unproposed coil only from a continuous repeated curve with two leads."""
import cv2,numpy as np
from backend.resistor_geometry import symbol_mask
from backend.vcc_topology import external_leads

def recover_coils(image,detections,validator,class_id):
    original=symbol_mask(image)
    from backend.ink_color import estimate_component_ink,component_ink_mask
    profile=estimate_component_ink(image,{'detections':detections})
    found=[]
    if profile.get('active') and profile.get('chromatic'):
        found=_recover_with_mask(image,detections,validator,class_id,component_ink_mask(image,profile))
    found+=_recover_with_mask(image,detections+found,validator,class_id,original)
    from backend.paper_ink import fragmentation_policy,stroke_mask
    from backend.notebook_rulings import detect_notebook_rule_profile
    if fragmentation_policy(image,{'detections':detections},profile,detect_notebook_rule_profile(image,profile))['active']:
        found+=_recover_with_mask(image,detections+found,validator,class_id,stroke_mask(image))
    return found

def _recover_with_mask(image,detections,validator,class_id,mask):
    h,w=mask.shape
    length=max(18,min(45,int(max(h,w)*.019)))
    straight=cv2.bitwise_or(cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((1,length),np.uint8)),cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((length,1),np.uint8)))
    curves=cv2.morphologyEx(cv2.bitwise_and(mask,cv2.bitwise_not(straight)),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(curves,8);found=[]
    for i in range(1,count):
        x,y,bw,bh,area=stats[i]
        if area<35 or min(bw,bh)<5 or max(bw,bh)<24 or max(bw,bh)>max(160,.18*min(h,w)) or max(bw,bh)<1.8*min(bw,bh):continue
        horizontal=bw>=bh
        for pad in (2,4):
            b=dict(x1=float(max(0,x-pad if horizontal else x-4)),y1=float(max(0,y-4 if horizontal else y-pad)),x2=float(min(w-1,x+bw-pad if horizontal else x+bw+4)),y2=float(min(h-1,y+bh+4 if horizontal else y+bh-pad)))
            bodyarea=(b['x2']-b['x1'])*(b['y2']-b['y1'])
            if any(max(0,min(b['x2'],d['bbox']['x2'])-max(b['x1'],d['bbox']['x1']))*max(0,min(b['y2'],d['bbox']['y2'])-max(b['y1'],d['bbox']['y1']))>.15*bodyarea for d in detections+found):break
            valid,score,e=validator(image,b,mask)
            # Three photographed turns have five alternating extrema. The
            # two independent straight tips and continuous path still have
            # to be proven; requiring six discarded this valid coil shape.
            if not valid or e.get('curve_extrema',0)<5 or not e.get('straight_tip_pair'):continue
            from backend.symbol_bounds import clip_coil_box
            b,refined=clip_coil_box(image,b,detections+found,validator,mask)
            if refined is not None:score,e=refined
            axis='x' if horizontal else 'y';normal='y' if horizontal else 'x'
            span=b[axis+'2']-b[axis+'1'];short=b[normal+'2']-b[normal+'1']
            if span>2.5*short:
                context=dict(b);along=max(4,min(12,span*.07));across=max(5,min(14,span*.10))
                for coordinate,pad in ((axis,along),(normal,across)):
                    context[coordinate+'1']=max(0,context[coordinate+'1']-pad)
                    context[coordinate+'2']=min(image.shape[1 if coordinate=='x' else 0]-1,context[coordinate+'2']+pad)
                verified=external_leads(image,context,mask)==2
                if not verified:
                    # A nearby junction can create several edge fragments of
                    # the same coil lead. Recheck the complete repeated curve
                    # and opposed original crossings at the proposed frame.
                    from backend.terminal_wire_contacts import _contacts
                    good,_,proof=validator(image,context,mask)
                    ports=_contacts(image,context,None);coordinate=0 if horizontal else 1
                    center=(context[axis+'1']+context[axis+'2'])/2
                    opposed=[p for p in ports if p.get('face_axis')==coordinate]
                    verified=bool(good and proof.get('curve_extrema',0)>=5 and proof.get('straight_tip_pair') and
                        any(p['point'][coordinate]<center for p in opposed) and any(p['point'][coordinate]>center for p in opposed))
                if verified:b=context
            b.update(width=b['x2']-b['x1'],height=b['y2']-b['y1'],center_x=(b['x1']+b['x2'])/2,center_y=(b['y1']+b['y2'])/2)
            found.append(dict(class_id=class_id,**{'class':'inductor'},bbox=b,confidence=.90,best_view_confidence=.90,inference_sources=['continuous_coil_geometry'],geometry_validation=dict(valid=True,score=score,evidence=e)))
            break
    return found
