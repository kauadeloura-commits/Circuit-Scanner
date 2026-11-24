"""Bound accepted power symbols by observed body and short lead context."""
from copy import deepcopy
import numpy as np

def refine(image,detections):
    from backend.drawing_mode import CURRENT
    if CURRENT.get()!='handwritten':return detections
    from backend.vcc_topology import external_leads
    from backend.resistor_geometry import symbol_mask
    result=deepcopy(detections);mask=symbol_mask(image)
    for d in result:
        if d.get('manual_reviewed') or d.get('symbol_box_refinement')=='observed_power_body_and_short_external_lead_context':continue
        b=d['bbox'];kind=d.get('class');proof=(d.get('geometry_validation') or {}).get('evidence') or {};head=None
        if kind=='vcc':head=proof.get('power_head_bbox')
        elif kind=='source_dc':head=(d.get('source_topology') or {}).get('source_body_bbox') or proof.get('source_body_bbox')
        if kind=='gnd':
            g=d.get('terminal_bar_ground') or {}
            if g.get('smaller_detached_marks')==0 and g.get('source_quarter_turn') in (1,3):
                body=g['ground_body_bbox'];candidate={k:body[k] for k in ('x1','y1','x2','y2')}
            else:continue
        elif head:
            directions=external_leads(image,head,mask,directions=True)
            if kind=='source_dc' and len(directions)!=2:
                diameter=min(head['x2']-head['x1'],head['y2']-head['y1']);padded={k:head[k]+(-1 if k.endswith('1') else 1)*diameter*.12 for k in ('x1','y1','x2','y2')}
                directions=external_leads(image,padded,mask,directions=True)
            if len(directions)!=(1 if kind=='vcc' else 2):continue
            if len(directions)==2 and np.dot(*directions)>-.65:continue
            axis=int(np.argmax(np.abs(directions[0])));name='xy'[axis];other='yx'[axis]
            diameter=min(head['x2']-head['x1'],head['y2']-head['y1'])
            if diameter<8:continue
            candidate={k:float(head[k]) for k in ('x1','y1','x2','y2')}
            across=diameter*(.20 if kind=='vcc' else .10)
            candidate[other+'1']-=across;candidate[other+'2']+=across
            if kind=='vcc':
                sign=directions[0][axis]
                candidate[name+'1']-=diameter*(.55 if sign<0 else .35)
                candidate[name+'2']+=diameter*(.55 if sign>0 else .35)
            else:
                candidate[name+'1']-=diameter*.25;candidate[name+'2']+=diameter*.25
        else:continue
        # A short power stem may immediately enter another proposed symbol.
        # Do not extend its frame deeper into that neighboring box: overlapping
        # masked regions can swallow the entire tiny inter-component wire.
        neighboring_frame=kind=='vcc' and any(o is not d and
            max(candidate['x1'],o['bbox']['x1'])<min(candidate['x2'],o['bbox']['x2']) and
            max(candidate['y1'],o['bbox']['y1'])<min(candidate['y2'],o['bbox']['y2']) for o in result)
        if kind=='source_dc' or neighboring_frame:
            candidate={k:(candidate[k]+b[k])*.5 for k in ('x1','y1','x2','y2')}
        for k in ('x1','y1'):candidate[k]=float(max(0,candidate[k]))
        candidate['x2']=float(min(image.shape[1]-1,candidate['x2']));candidate['y2']=float(min(image.shape[0]-1,candidate['y2']))
        # Keep a separate symbol's actual core out of the new frame.
        intrudes=False
        for o in result:
            if o is d:continue
            q=o['bbox'];pad=min(q['x2']-q['x1'],q['y2']-q['y1'])*.2
            if max(candidate['x1'],q['x1']+pad)<min(candidate['x2'],q['x2']-pad) and max(candidate['y1'],q['y1']+pad)<min(candidate['y2'],q['y2']-pad):
                intrudes=True;break
        if intrudes:continue
        lead_count=external_leads(image,candidate,mask)
        if kind=='vcc' and lead_count!=1:
            # Separate nearby voltage writing from the conductor belonging to
            # the already verified ring. Unconnected letters are not leads.
            from backend.terminal_wire_contacts import _contacts,_power_lead_contacts
            ports=_contacts(image,candidate,None)
            selected,evidence=_power_lead_contacts(image,dict(type='vcc',bbox=candidate),ports)
            if not evidence or evidence.get('method')!='actual_power_ring_connected_to_one_observed_lead' or len(selected)!=1:continue
        elif lead_count!=(2 if kind=='source_dc' else 1):continue
        candidate.update(width=candidate['x2']-candidate['x1'],height=candidate['y2']-candidate['y1'],center_x=(candidate['x1']+candidate['x2'])/2,center_y=(candidate['y1']+candidate['y2'])/2)
        d.update(bbox=candidate,symbol_box_refinement='observed_power_body_and_short_external_lead_context')
    return result
