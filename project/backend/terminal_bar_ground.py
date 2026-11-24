"""Ground evidence from an actual terminating bar, including tiny pen dots.

A single bar requires a learned ground proposal and a physical connection
to another component. Two separate smaller strokes support a weak proposal.
No missing ink or electrical connection is synthesized.
"""
import cv2
import numpy as np
from backend.resistor_geometry import symbol_mask
from backend.vcc_topology import external_leads


def connected_body_evidence(image,bbox,detections,mask):
    from backend.resistor_geometry import session_result
    _,labels,_,_=session_result(('ground_ink_labels',id(mask)),lambda:cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8))
    h,w=mask.shape
    def ids(b):
        values=labels[max(0,int(b['y1'])):min(h,int(b['y2'])+1),max(0,int(b['x1'])):min(w,int(b['x2'])+1)]
        count=np.bincount(values.ravel());return {int(i) for i in np.flatnonzero(count>=5) if i}
    ground_ids=ids(bbox)
    for d in detections:
        if d.get('class') in {'gnd','vcc'}:continue
        b=d['bbox']
        if not ground_ids.intersection(ids(b)):continue
        # Text printed beside the diagram cannot supply two physical wires
        # leaving the same ink path of an electronic component.
        if external_leads(image,b,mask)<2:continue
        return dict(connected_to_component_ink=True,anchor_class=d.get('class'),anchor_bbox=b)
    return None


def _upright(mask,bbox):
    h,w=mask.shape
    bw,bh=bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1']
    pad=max(6,min(20,.3*max(bw,bh)))
    x1,y1=max(0,int(bbox['x1']-pad)),max(0,int(bbox['y1']-pad))
    x2,y2=min(w,int(bbox['x2']+pad+1)),min(h,int(bbox['y2']+pad+1))
    crop=mask[y1:y2,x1:x2]
    if not crop.size:return None
    n,labels,stats,centers=cv2.connectedComponentsWithStats((crop>0).astype(np.uint8),8)
    # Row runs measure short handwritten bars that Hough's vote threshold
    # misses. Only a connected perpendicular stem can establish the top bar.
    candidates=[]
    for yy in range(1,crop.shape[0]-1):
        xs=np.flatnonzero(np.any(crop[yy-1:yy+2]>0,axis=0))
        for run in np.split(xs,np.flatnonzero(np.diff(xs)>1)+1):
            if len(run)<max(7,.18*bw):continue
            left,right=int(run[0]),int(run[-1]);length=right-left+1;cx=(left+right)/2
            if length>max(bw*1.3,18):continue
            gx,gy=cx+x1,yy+y1
            if not (bbox['x1']-.1*bw<=gx<=bbox['x2']+.1*bw and bbox['y1']-.1*bh<=gy<=bbox['y2']+.35*bh):continue
            values=labels[yy-1:yy+2,max(0,round(cx)-1):round(cx)+2];values=values[values>0]
            label=int(np.bincount(values).argmax()) if len(values) else 0
            if not label:continue
            reach=max(8,int(length*.65));half=max(2,round(length*.20))
            top=max(0,yy-reach);stem=labels[top:max(top,yy-3),max(0,round(cx)-half):min(crop.shape[1],round(cx)+half+1)]
            if len(stem)<5 or np.mean(np.any(stem==label,axis=1))<.80:continue
            # The incoming wire ends here. A cross, E/F/H, a capacitor
            # connection or a continuing wire cannot pass as a terminal bar.
            below=labels[min(crop.shape[0],yy+4):min(crop.shape[0],yy+reach),max(0,left-2):min(crop.shape[1],right+3)]
            if np.count_nonzero(below==label)>max(4,.08*length*reach):continue
            lower=[]
            for j in range(1,n):
                if j==label:continue
                x,y,sw,sh,area=stats[j];dx,dy=centers[j]-[cx,yy]
                if area<1 or not 3<=dy<=length*1.3+5 or centers[j,1]+y1>bbox['y2']+max(3,.1*bh):continue
                if abs(dx)>max(3,length*.25) or sw>length*.85 or sh>max(5,length*.35):continue
                lower.append((float(centers[j,1]),int(x),int(y),int(sw),int(sh)))
            lower.sort();dots=[]
            for mark in lower:
                if not dots or mark[0]-dots[-1][0]>=2:dots.append(mark)
            # A run through the middle of a thick bar may have two edges.
            # Prefer the same bar with the longest observed span.
            bottom=max([yy+2]+[y+sh for _,_,y,_,sh in dots[:2]])
            body=dict(x1=float(max(0,x1+left-4)),x2=float(min(w-1,x1+right+5)),
                      y1=float(max(0,y1+yy-max(9,length*.65))),y2=float(min(h-1,y1+bottom+4)))
            directions=external_leads(None,body,mask,directions=True)
            if len(directions)!=1 or directions[0][1]>-.65:continue
            candidates.append((len(dots)>=2,length,body,len(dots)))
    if not candidates:return None
    _,length,body,dots=max(candidates,key=lambda v:(v[0],v[1]))
    return dict(terminating_bar_one_incoming_lead=True,smaller_detached_marks=dots,
                ground_body_bbox=body,bar_length_px=length,external_lead_count=1)


def terminal_bar_evidence(image,bbox,page_mask=None):
    from backend.ground_recovery import _rotate_box
    mask=symbol_mask(image) if page_mask is None else page_mask
    for turn in range(4):
        rotated=np.ascontiguousarray(np.rot90(mask,turn))
        b=_rotate_box(bbox,mask.shape,turn)
        evidence=_upright(rotated,b)
        if evidence:
            evidence['ground_body_bbox']=_rotate_box(evidence['ground_body_bbox'],rotated.shape,-turn)
            evidence['source_quarter_turn']=turn
            return evidence
    return None


def refine_learned_ground(image,det,page_mask=None,anchors=()):
    if det.get('manual_reviewed') or det.get('class')!='gnd':return det
    evidence=terminal_bar_evidence(image,det['bbox'],page_mask)
    if not evidence:
        repaired=_gapped_single_bar(image,det,anchors)
        return repaired if repaired is not None else det
    mask=symbol_mask(image) if page_mask is None else page_mask
    if external_leads(image,det['bbox'],mask)!=1 and evidence['smaller_detached_marks']<2:return det
    connected=connected_body_evidence(image,evidence['ground_body_bbox'],anchors,mask)
    if not connected:return det
    evidence.update(connected)
    # A plain terminal bar needs a learned ground proposal; two lower marks
    # provide the conventional three-stroke ground evidence instead.
    learned=[s for s in det.get('inference_sources',[]) if not any(t in s for t in ('geometry','bars','manual'))]
    confidence=float(det.get('best_learned_confidence',det.get('best_view_confidence',det.get('confidence',0))))
    physical=evidence['ground_body_bbox'];h,w=image.shape[:2]
    clipped=min(physical['x1'],physical['y1'],w-1-physical['x2'],h-1-physical['y2'])<=1
    if clipped and confidence<.8 and len(set(learned))<2:return det
    # The proposal supplies the class; the terminating bar, external wire
    # and connection to a two-lead body supply independent physical checks.
    # Requiring several rotated model views here also discarded faint but
    # complete grounds that already passed all three physical checks.
    if evidence['smaller_detached_marks']<2 and (not learned or confidence<.20):return det
    evidence.update(learned_ground_confidence=confidence,learned_ground_views=len(set(learned)))
    physical=evidence['ground_body_bbox'];old=det['bbox']
    # The original entry edge owns the wire contact. Moving it along the
    # stem either erases a short wire or moves the terminal into another
    # component. A tilted bar may supply only a partial horizontal run;
    # never shrink a wide learned body to that small fragment.
    body={k:old[k] for k in ('x1','y1','x2','y2')}
    bar_axis='y' if evidence['source_quarter_turn']%2 else 'x'
    # Proof uses a tight crop so a neighboring op-amp edge is not counted
    # as another lead. Display/localization padding is added afterwards.
    extent=dict(physical)
    extent[bar_axis+'1']=max(0,extent[bar_axis+'1']-2)
    extent[bar_axis+'2']=min(image.shape[0 if bar_axis=='y' else 1]-1,extent[bar_axis+'2']+2)
    ratio=(extent[bar_axis+'2']-extent[bar_axis+'1'])/max(1,old[bar_axis+'2']-old[bar_axis+'1'])
    if .65<=ratio<=1.4:
        body[bar_axis+'1']=extent[bar_axis+'1'];body[bar_axis+'2']=extent[bar_axis+'2']
    if evidence['smaller_detached_marks']>=2:
        distal=('y2','x1','y1','x2')[evidence['source_quarter_turn']]
        body[distal]=physical[distal]
    from backend.recognition import _validar_geometria_gnd
    if _validar_geometria_gnd(image,old)[0]:
        body={k:old[k] for k in ('x1','y1','x2','y2')}
        evidence['complete_ground_extent_preserved']=True
    if max(old['x2']-old['x1'],old['y2']-old['y1'])>64:
        # This short-row matcher confirms large photographed grounds but
        # cannot bound an entire long slanted bar reliably. Keep the learned
        # wire context; only small symbols receive the local extent repair.
        body={k:old[k] for k in ('x1','y1','x2','y2')}
        evidence['large_learned_ground_context_preserved']=True
    evidence['wire_entry_extent_preserved']=True
    evidence['bar_extent_adjusted']=.65<=ratio<=1.4
    body.update(width=body['x2']-body['x1'],height=body['y2']-body['y1'],center_x=(body['x1']+body['x2'])/2,center_y=(body['y1']+body['y2'])/2)
    return dict(det,bbox=body,terminal_bar_ground=evidence)


def _gapped_single_bar(image,det,anchors):
    """Bound one faint terminal bar; closing is confined to symbol evidence."""
    from backend.drawing_mode import CURRENT
    if CURRENT.get()!='handwritten':return None
    from backend.recognition import _validar_geometria_gnd
    old=det['bbox']
    if _validar_geometria_gnd(image,old)[0]:return None
    confidence=float(det.get('best_learned_confidence',det.get('confidence',0)))
    learned=[s for s in det.get('inference_sources',[]) if not any(t in s for t in ('geometry','bars','manual'))]
    if confidence<.5 or len(set(learned))<2:return None
    from backend.notebook_rulings import _paper_colors
    _,_,normalized=_paper_colors(image)
    mask=(normalized.mean(axis=2)<225).astype(np.uint8)*255
    evidence=terminal_bar_evidence(image,old,cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8)))
    if not evidence or evidence['smaller_detached_marks']!=0:return None
    physical=evidence['ground_body_bbox'];length=evidence['bar_length_px'];turn=evidence['source_quarter_turn']
    transverse=('y' if turn%2 else 'x')
    if length<.5*(old[transverse+'2']-old[transverse+'1']):return None
    from backend.ground_recovery import _rotate_box
    rotated=_rotate_box(physical,image.shape,turn)
    bar_y=rotated['y1']+max(9,length*.65)
    tight=dict(x1=rotated['x1']+4-length*.25,x2=rotated['x2']-5+length*.25,y1=bar_y-length*.25,y2=bar_y+length*.5)
    shape=np.rot90(mask,turn).shape
    for k in ('x1','y1'):tight[k]=max(0,tight[k])
    tight['x2']=min(shape[1]-1,tight['x2']);tight['y2']=min(shape[0]-1,tight['y2'])
    body=_rotate_box(tight,shape,-turn)
    closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    connected=connected_body_evidence(image,physical,anchors,closed)
    if not connected:return None
    evidence.update(connected,source='learned_single_bar_with_short_internal_pencil_gap',internal_closing_px=5,external_wires_unchanged=True)
    return dict(det,bbox=body,terminal_bar_ground=evidence)


