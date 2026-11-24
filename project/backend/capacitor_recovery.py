"""Recover separate plates only when both actual outward leads are present."""
import math
import cv2
import numpy as np
from backend.resistor_geometry import symbol_mask
from backend.capacitor_geometry import capacitor_body_evidence
from backend.symbol_semantics import resolve_marked_families
from backend.vcc_topology import symbol_ink_labels,context_symbol_mask


def recover_capacitors(image,detections,class_ids):
    mask=symbol_mask(image);h,w=mask.shape
    trace_mask=context_symbol_mask(image,detections)
    raw=cv2.HoughLinesP(mask,1,np.pi/180,12,
                       minLineLength=max(8,int(min(h,w)*.012)),maxLineGap=4)
    if raw is None:return []
    _,labels=cv2.connectedComponents(mask,8)
    _,trace_labels=cv2.connectedComponents(trace_mask,8)
    known=symbol_ink_labels(trace_labels,detections)
    plates=[];by_label={}
    for x1,y1,x2,y2 in sorted(raw.reshape(-1,4),key=lambda l:float(l[2]-l[0])**2+float(l[3]-l[1])**2,reverse=True):
        vector=np.array([x2-x1,y2-y1],float);length=float(np.linalg.norm(vector))
        if not max(9,min(h,w)*.009)<=length<=max(120,min(h,w)*.30):continue
        center=np.array([(x1+x2)/2,(y1+y2)/2]);axis=vector/length
        px,py=np.rint(center).astype(int)
        values=labels[max(0,py-2):min(h,py+3),max(0,px-2):min(w,px+3)]
        values=values[values>0]
        if not len(values):continue
        label=int(np.bincount(values).argmax())
        trace_values=trace_labels[max(0,py-2):min(h,py+3),max(0,px-2):min(w,px+3)]
        trace_values=trace_values[trace_values>0]
        trace_label=int(np.bincount(trace_values).argmax()) if len(trace_values) else 0
        if any(abs(float(p['axis']@axis))>.97 and np.linalg.norm(p['center']-center)<max(5,length*.15) for p in by_label.get(label,())):continue
        plate=dict(center=center,axis=axis,length=length,label=label,trace_label=trace_label,ends=np.array([[x1,y1],[x2,y2]],float))
        plates.append(plate);by_label.setdefault(label,[]).append(plate)
    found=[]
    def lead(plate,normal,length,connected):
        # Sample only ink belonging to this plate, including the actual stem.
        distances=np.arange(max(5,length*.18),max(10,length*(.80 if connected else 1.20)),2)
        offsets=np.arange(-max(3,plate['length']*.18),max(3,plate['length']*.18)+1,2)
        points=plate['center']+distances[:,None,None]*normal+offsets[None,:,None]*plate['axis']
        xs=np.rint(points[:,:,0]).astype(int);ys=np.rint(points[:,:,1]).astype(int)
        valid=(xs>=0)&(xs<w)&(ys>=0)&(ys<h)
        hits=np.zeros(valid.shape,bool);hits[valid]=trace_mask[ys[valid],xs[valid]]>0
        return float(np.mean(np.any(hits,axis=1)))>=.75
    centers=np.array([p['center'] for p in plates]);axes=np.array([p['axis'] for p in plates]);plate_lengths=np.array([p['length'] for p in plates])
    for i,a in enumerate(plates):
        # The same geometric limits eliminate distant/nonparallel pairs in
        # one array pass, before any image crop or lead analysis is needed.
        normal=np.array([-a['axis'][1],a['axis'][0]])
        delta=centers-a['center'];gaps=delta@normal
        maximum=np.maximum(a['length'],plate_lengths)
        possible=(np.abs(axes@a['axis'])>=math.cos(math.radians(20))) & (np.minimum(a['length'],plate_lengths)>=maximum*.55)
        possible &= (np.abs(gaps)>=np.maximum(4,maximum*.07)) & (np.abs(gaps)<=maximum*.70) & (np.abs(delta@a['axis'])<=maximum*.32)
        for j in np.flatnonzero(possible):
            if j<=i:continue
            b=plates[j];length=maximum[j];gap=gaps[j]
            connected=bool({a['trace_label'],b['trace_label']}&known)
            side=1 if gap>0 else -1
            if not lead(a,-normal*side,length,connected) or not lead(b,normal*side,length,connected):continue
            ends=np.concatenate((a['ends'],b['ends']));pad=max(8,length*.25)
            box=dict(x1=float(max(0,ends[:,0].min()-pad)),y1=float(max(0,ends[:,1].min()-pad)),
                     x2=float(min(w-1,ends[:,0].max()+pad)),y2=float(min(h-1,ends[:,1].max()+pad)))
            center=(a['center']+b['center'])/2
            if any(d['bbox']['x1']<=center[0]<=d['bbox']['x2'] and d['bbox']['y1']<=center[1]<=d['bbox']['y2'] for d in detections+found):continue
            ok,evidence=capacitor_body_evidence(image,box)
            if not ok:continue
            lengths=evidence['outer_plate_lengths'];verified_length=max(lengths)
            if min(lengths)<verified_length*.60 or evidence['plate_gap_px']>verified_length*.70:continue
            if abs(float(np.asarray(evidence['electrical_axis'])@normal))<math.cos(math.radians(15)):continue
            evidence['connected_component_context']=connected
            # A short Hough fragment must not crop the verified plates or
            # prevent a nearby polarity sign from entering the search area.
            angle=math.radians(evidence['plate_axis_degrees'])
            axis=np.array([math.cos(angle),math.sin(angle)])
            verified_ends=np.array([np.asarray(p)+sign*axis*length/2
                                   for p,length in zip(evidence['outer_plate_centers'],lengths) for sign in (-1,1)])
            pad=max(8,verified_length*.25)
            box.update(x1=float(max(0,min(box['x1'],verified_ends[:,0].min()-pad))),
                       y1=float(max(0,min(box['y1'],verified_ends[:,1].min()-pad))),
                       x2=float(min(w-1,max(box['x2'],verified_ends[:,0].max()+pad))),
                       y2=float(min(h-1,max(box['y2'],verified_ends[:,1].max()+pad))))
            box.update(width=box['x2']-box['x1'],height=box['y2']-box['y1'],center_x=float(center[0]),center_y=float(center[1]))
            item=dict(detection_id='CAP_RECOVER_'+str(len(found)+1),**{'class':'capacitor'},class_id=class_ids['capacitor'],bbox=box,
                      confidence=.91,best_view_confidence=.91,inference_sources=['separate_plates_opposed_lead_geometry'],
                      geometry_validation=dict(valid=True,score=.91,evidence=evidence))
            resolved,_=resolve_marked_families(image,[item]+detections+found,class_ids)
            found.append(resolved[0])
    return found
