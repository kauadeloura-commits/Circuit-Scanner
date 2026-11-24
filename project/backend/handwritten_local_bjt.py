"""Recheck an ambiguous BJT locally, keeping independently observed leads."""
from copy import deepcopy
import numpy as np
from backend.resistor_geometry import cached_vision


def reconcile_weak_elbow(image,box,branches,chosen,arrows,strong):
    """A wire elbow must not overrule consistent, ink-matched B/C/E reads."""
    from backend.drawing_mode import CURRENT
    scale=min(box['x2']-box['x1'],box['y2']-box['y1'])
    head=arrows[strong]
    if CURRENT.get()!='handwritten' or scale<80 or head.get('score',0)>=9 or not head.get('rejected_base_or_plain_branch') or not head.get('head_center'):return strong,None
    branch=branches[strong];u=branch['far']-branch['p'];u/=max(1,np.linalg.norm(u));normal=np.array([-u[1],u[0]])
    peak=max(4,head.get('score',0)/1.25)
    if abs(float((np.asarray(head['head_center'])-branch['p'])@normal))<=peak*2.25:return strong,None
    from itertools import permutations
    observed=np.array([c['point'] for c in chosen]);votes=[];evidence=[]
    for pad in (0.,.15):
        pose=pose_read(image,box,pad)
        if not pose or pose['confidence']<.85 or min(pose['keypoint_confidence'])<.80:return strong,None
        anchors=np.array([pose['points'][pin] for pin in ('B','C','E')])
        ranked=sorted((float(np.linalg.norm(observed[list(order)]-anchors,axis=1).sum()),order) for order in permutations(range(3)))
        cost,order=ranked[0]
        if order[0]!=0 or max(np.linalg.norm(observed[list(order)]-anchors,axis=1))>scale*.45 or ranked[1][0]-cost<scale*.15:return strong,None
        votes.append(order[2]-1);evidence.append(pose)
    if len(set(votes))!=1 or votes[0]==strong:return strong,None
    return votes[0],dict(source='two_local_pose_reads_matched_to_three_original_wire_contacts',
                        rejected_elbow_offset_px=abs(float((np.asarray(head['head_center'])-branch['p'])@normal)),reads=evidence)


def _iou(a,b):
    area=max(0,min(a['x2'],b['x2'])-max(a['x1'],b['x1']))*max(0,min(a['y2'],b['y2'])-max(a['y1'],b['y1']))
    return area/max(1,(a['x2']-a['x1'])*(a['y2']-a['y1'])+(b['x2']-b['x1'])*(b['y2']-b['y1'])-area)


def _crop(image,b,pad):
    size=max(b['x2']-b['x1'],b['y2']-b['y1'])
    x=max(0,int(b['x1']-size*pad));y=max(0,int(b['y1']-size*pad))
    return image[y:min(image.shape[0],int(b['y2']+size*pad)),x:min(image.shape[1],int(b['x2']+size*pad))],x,y


@cached_vision
def pose_read(image,box,pad):
    from backend.terminal_learning import _load_keypoint_model
    model=_load_keypoint_model()
    if model is None:return None
    view,x,y=_crop(image,box,pad)
    if not view.size:return None
    result=model.predict(view,imgsz=416,conf=.20,iou=.5,verbose=False,device='cpu')[0]
    candidates=[]
    for index,b in enumerate(result.boxes):
        q=b.xyxy[0].cpu().numpy()+np.array([x,y,x,y]);kind=model.names[int(b.cls[0])]
        predicted=dict(zip(('x1','y1','x2','y2'),map(float,q)))
        if kind not in {'transistor_npn','transistor_pnp'} or _iou(predicted,box)<.40:continue
        if result.keypoints is None:continue
        conf=result.keypoints.conf[index].cpu().numpy();points=result.keypoints.xy[index].cpu().numpy()+np.array([x,y])
        if len(conf)!=3 or min(conf)<.40:continue
        candidates.append(dict(kind=kind,confidence=float(b.conf[0]),points=dict(zip(('B','C','E'),points.tolist())),keypoint_confidence=conf.tolist()))
    return max(candidates,key=lambda c:c['confidence']) if candidates else None


@cached_vision
def local_votes(image,box,family=('transistor_npn','transistor_pnp')):
    from backend.recognition import _carregar_modelo,_executar_predicao
    model=_carregar_modelo();votes=[]
    for pad in (0.,.15,.30):
        view,x,y=_crop(image,box,pad)
        if not view.size:continue
        candidates=[]
        for d in _executar_predicao(model,view,.05,416,'local_bjt_class_check'):
            if d['class'] not in family:continue
            b={k:d['bbox'][k]+(x if k.startswith('x') else y) for k in ('x1','y1','x2','y2')}
            if _iou(b,box)<.40:continue
            center=np.array([(b['x1']+b['x2'])/2,(b['y1']+b['y2'])/2]);original=np.array([(box['x1']+box['x2'])/2,(box['y1']+box['y2'])/2])
            if np.linalg.norm(center-original)>.25*max(box['x2']-box['x1'],box['y2']-box['y1']):continue
            candidates.append(dict(kind=d['class'],confidence=float(d['confidence']),bbox=b,pad=pad))
        if candidates:votes.append(max(candidates,key=lambda c:c['confidence']))
    return votes


def refine(image,detections,class_ids):
    from backend.drawing_mode import CURRENT
    if CURRENT.get()!='handwritten':return detections
    from backend.npn_wire_contacts import detect_bjt_wire_contacts
    output=deepcopy(detections)
    for d in output:
        kind=d.get('class')
        if not d.get('manual_reviewed') and kind in {'mosfet_n','mosfet_p'} and (d.get('geometry_validation') or {}).get('valid'):
            votes=local_votes(image,d['bbox'],('mosfet_n','mosfet_p'))
            if len(votes)==3 and len({v['kind'] for v in votes})==1 and min(v['confidence'] for v in votes)>=.60 and votes[0]['kind']!=kind:
                from backend.mosfet_semantics import visible_channel_type
                physical,proof=visible_channel_type(image,d['bbox']);target=votes[0]['kind']
                if not physical or physical==target:
                    d.update(**{'class':target,'class_id':class_ids[target]},resolved_from_class=kind,class_resolution='three_strong_local_mosfet_views_with_valid_gate_channel_body',local_mosfet_review=dict(votes=votes,physical=proof))
        if d.get('manual_reviewed') or kind not in {'transistor_npn','transistor_pnp'}:continue
        votes=local_votes(image,d['bbox'])
        if len(votes)!=3 or len({v['kind'] for v in votes})!=1 or votes[0]['kind']==kind:continue
        target=votes[0]['kind'];strong=sum(v['confidence']>=.55 for v in votes)>=2
        pose=pose_read(image,d['bbox'],.15) if not strong else None
        if not strong and not (pose and pose['kind']==target and pose['confidence']>=.75):continue
        chosen=max(votes,key=lambda v:v['confidence']);new_box=dict(d['bbox'])
        intersects_neighbor=False
        for other in output:
            if other is d or other.get('class') not in {'transistor_npn','transistor_pnp'}:continue
            b=other['bbox'];pad=.15*min(b['x2']-b['x1'],b['y2']-b['y1'])
            core=dict(x1=b['x1']+pad,y1=b['y1']+pad,x2=b['x2']-pad,y2=b['y2']-pad)
            if _iou(d['bbox'],core)>0:intersects_neighbor=True
        if intersects_neighbor:new_box=chosen['bbox']
        found,proof=detect_bjt_wire_contacts(image,new_box,None,target,color_reference=image)
        if not found or not proof.get('arrow_confirmed'):continue
        new_box.update(width=new_box['x2']-new_box['x1'],height=new_box['y2']-new_box['y1'],center_x=(new_box['x1']+new_box['x2'])/2,center_y=(new_box['y1']+new_box['y2'])/2)
        d.update(**{'class':target,'class_id':class_ids[target]},bbox=new_box,resolved_from_class=kind,
                 class_resolution='consistent_local_crops_and_observed_three_lead_bjt',
                 local_bjt_review=dict(votes=votes,pose=pose,geometry=proof,neighbor_excluded=intersects_neighbor))
    return output
