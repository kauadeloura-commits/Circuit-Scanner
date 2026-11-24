"""Reconcile a locally read model/reference with independently observed ink."""
from copy import deepcopy
import numpy as np

def reconcile(image,detections,rejected,tokens):
    from backend.drawing_mode import CURRENT
    if CURRENT.get()!='handwritten':return detections
    from backend.component_models import model_classes
    from backend.terminal_identity import _crop,_marks,_mosfet
    from backend.recognition import _validar_classe_geometrica
    from backend.reference_labels import edge_distance
    result=deepcopy(detections)
    from backend.reference_labels import reference_classes
    # Read body topology in a view without confident annotation glyphs.
    # Positions and final conductor extraction still use the original photo.
    geometry_image=image.copy()
    for t in tokens:
        confidence=float(t.get('confidence',0));confidence=confidence/100 if confidence>1 else confidence
        if confidence<.97 or not(reference_classes(t.get('text','')) or model_classes(t.get('text',''))):continue
        a=t['bbox'];x,y=max(0,int(a['x1'])),max(0,int(a['y1']));z,v=min(image.shape[1],int(a['x2'])),min(image.shape[0],int(a['y2']))
        geometry_image[y:v,x:z]=255
    candidates=result+[dict(d,_rejected_proposal=True) for d in rejected if d.get('class') in {'capacitor','capacitor_polarized','diode','mosfet_n','mosfet_p'}]
    known={'BS250':{'mosfet_p'}}
    for token in tokens:
        conf=float(token.get('confidence',0));conf=conf/100 if conf>1 else conf
        if conf<.97:continue
        text=''.join(str(token.get('text','')).split()).upper()
        classes=known.get(text,model_classes(text))
        if len(classes)!=1 or not classes.intersection({'mosfet_n','mosfet_p'}):continue
        kind=next(iter(classes));tb=token['bbox'];choices=[]
        for index,d in enumerate(candidates):
            if d.get('manual_reviewed') or d.get('class') not in {'capacitor','capacitor_polarized','diode','mosfet_n','mosfet_p'}:continue
            if d['class']==kind:continue
            b=d['bbox'];size=max(b['x2']-b['x1'],b['y2']-b['y1'])
            distance=edge_distance(tb,b)
            if distance>max(45,size*.85):continue
            # Expand only the proposed symbol context, never the text box.
            # Require an isolated gate and two channel contacts at the box.
            boxes=[{k:b[k] for k in ('x1','y1','x2','y2')}]
            mask,origin=_crop(geometry_image,b);_,labels=_marks(mask,origin,b);gate=_mosfet(mask,origin,b,labels)
            if gate:
                ports=gate[0];g=np.array(ports['G']);channels=np.array([ports['D'],ports['S']]);delta=channels.mean(axis=0)-g
                coordinate=int(np.argmax(np.abs(delta)));axis='xy'[coordinate];edge=axis+('2' if delta[coordinate]>0 else '1')
                if abs(channels[0][coordinate]-channels[1][coordinate])<size*.15:
                    extensions=[]
                    for ratio in (.35,.50,.65):
                        extended=dict(boxes[0]);growth=min(90,size*ratio)
                        extended[edge]=float(np.clip(extended[edge]+(growth if edge.endswith('2') else -growth),0,image.shape[1-coordinate]-1));extensions.append(extended)
                    boxes=extensions+boxes
            for scale in (1.35,1.7):
                x=(b['x1']+b['x2'])/2;y=(b['y1']+b['y2'])/2;w=(b['x2']-b['x1'])*scale;h=(b['y2']-b['y1'])*scale
                boxes.append(dict(x1=max(0,x-w/2),y1=max(0,y-h/2),x2=min(image.shape[1]-1,x+w/2),y2=min(image.shape[0]-1,y+h/2)))
            for box in boxes:
                mask,origin=_crop(geometry_image,box);_,labels=_marks(mask,origin,box);gate=_mosfet(mask,origin,box,labels)
                if gate is None:continue
                from backend.continuous_mosfet import read
                arrow=read(geometry_image,box)
                # The model must agree with a visible source arrow, or a
                # previously validated gate/channel topology and M reference.
                near_refs=[t for t in tokens if __import__('re').fullmatch(r'M\d+',str(t.get('text','')),__import__('re').I) and float(t.get('confidence',0))>=.95 and edge_distance(t['bbox'],tb)<max(45,size*.75)]
                physical=_validar_classe_geometrica(geometry_image,box,kind)
                if arrow and arrow['kind']!=kind:continue
                if not arrow and not (near_refs and physical and physical[0]):continue
                choices.append((distance,index,box,arrow,physical));break
        if not choices:continue
        _,index,box,arrow,physical=min(choices,key=lambda v:v[0]);d=candidates[index]
        if d.get('_rejected_proposal'):
            d=deepcopy(d);d.pop('_rejected_proposal',None)
            used={c.get('detection_id') for c in result};serial=1
            while f'REC_{serial:03d}' in used:serial+=1
            d['detection_id']=f'REC_{serial:03d}';result.append(d)
            candidates[index]=d
        box.update(width=box['x2']-box['x1'],height=box['y2']-box['y1'],center_x=(box['x1']+box['x2'])/2,center_y=(box['y1']+box['y2'])/2)
        d.update(resolved_from_class=d['class'],**{'class':kind,'class_id':12 if kind=='mosfet_n' else 13},bbox=box,
                 class_resolution='strong_local_ocr_model_and_independent_gate_channel_evidence',
                 typed_model_evidence=dict(text=token['text'],confidence=conf,arrow=arrow,geometry=physical))
    for d in result:
        if d.get('manual_reviewed') or d.get('class') not in {'capacitor','capacitor_polarized'}:continue
        b=d['bbox'];size=max(b['x2']-b['x1'],b['y2']-b['y1'])
        names=[t for t in tokens if __import__('re').fullmatch(r'(?:BAT|BT)\d+',str(t.get('text','')),__import__('re').I) and float(t.get('confidence',0))>=.97 and edge_distance(t['bbox'],b)<size*.8]
        proof=_validar_classe_geometrica(image,b,'battery') if names else None
        if proof and proof[0]:d.update(resolved_from_class=d['class'],**{'class':'battery','class_id':7},class_resolution='explicit_battery_reference_and_verified_unequal_plates',battery_ocr_evidence=dict(text=names[0]['text'],geometry=proof))
    # The filled source arrow can have its own diode proposal. A nearby real
    # diode still has a separate triangle/bar and opposed leads; retain it.
    from backend.diode_contacts import body_axes
    output=[]
    for d in result:
        owner=None
        if d.get('class') in {'diode','led'} and not d.get('manual_reviewed'):
            b=d['bbox'];area=(b['x2']-b['x1'])*(b['y2']-b['y1'])
            for m in result:
                if m.get('class') not in {'mosfet_n','mosfet_p'} or not m.get('typed_model_evidence'):continue
                a=m['bbox'];overlap=max(0,min(a['x2'],b['x2'])-max(a['x1'],b['x1']))*max(0,min(a['y2'],b['y2'])-max(a['y1'],b['y1']))
                if overlap/max(1,area)>=.80 and not body_axes(image,b):owner=m;break
        if owner is not None:
            owner.setdefault('subsymbol_proposal_rejections',[]).append(dict(d,reason='source_arrow_without_independent_diode_body'))
        else:output.append(d)
    return output
