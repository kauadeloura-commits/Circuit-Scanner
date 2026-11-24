"""Stage 10: explicit Stage 9 connections and observed pin leads form nets."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import cv2
import numpy as np
from backend.ink_color import image_with_component_ink
from backend.notebook_rulings import apply_notebook_rule_profile, _paper_colors
from backend.junction_detection import _nearest_on_polyline


from backend.circuit_types import PIN_SCHEMAS
from backend.pen_gap_recovery import VERSION as PEN_GAP_VERSION, recover_pen_gaps
CONTACT_RECOVERY_VERSION='observed-external-contacts-2026-11-09'


def source_signature(wires: dict, terminals: dict) -> str:
    """Fingerprint only upstream inputs, never embedded derived stages."""
    payload = {'segments':wires.get('segments',[]), 'nodes':wires.get('nodes',[]),
               'stage9':wires.get('stage9'), 'ink_color_profile':wires.get('ink_color_profile'),
               'notebook_rule_profile':wires.get('notebook_rule_profile'),
               'components':terminals.get('components',[]),
               'pen_gap_policy_version':PEN_GAP_VERSION,
               'contact_recovery_version':CONTACT_RECOVERY_VERSION,
               'pen_gap_settings':wires.get('pen_gap_settings')}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,
                                   allow_nan=False,separators=(',',':')).encode('utf8')).hexdigest()


class _Groups:
    def __init__(self): self.parent={}
    def find(self, key):
        self.parent.setdefault(key,key)
        root=key
        while self.parent[root]!=root:root=self.parent[root]
        while self.parent[key]!=key:
            parent=self.parent[key];self.parent[key]=root;key=parent
        return root
    def union(self, keys):
        if not keys:return
        root=self.find(keys[0])
        for key in keys[1:]:self.parent[self.find(key)]=root


def _outside_normal(point, bbox):
    distances=[(point[0]-bbox['x1'],(-1.,0.)),(bbox['x2']-point[0],(1.,0.)),
               (point[1]-bbox['y1'],(0.,-1.)),(bbox['y2']-point[1],(0.,1.))]
    return np.array(min(distances,key=lambda item:item[0])[1])


def _lead_support(mask, point, target, normal, *, inward_slack=2):
    """Follow source ink outward; never walk through a component's body."""
    h,w=mask.shape
    p=np.rint(point).astype(int);q=np.rint(target).astype(int)
    x1=max(0,min(p[0],q[0])-5);x2=min(w,max(p[0],q[0])+6)
    y1=max(0,min(p[1],q[1])-5);y2=min(h,max(p[1],q[1])+6)
    if x2<=x1 or y2<=y1:return False
    crop=mask[y1:y2,x1:x2].copy()
    yy,xx=np.mgrid[y1:y2,x1:x2]
    crop[((xx-point[0])*normal[0]+(yy-point[1])*normal[1]) < -inward_slack]=0
    _,labels=cv2.connectedComponents(crop,connectivity=8)
    def nearby_labels(pos):
        x=int(pos[0]-x1);y=int(pos[1]-y1)
        return set(labels[max(0,y-1):y+2,max(0,x-1):x+2].ravel())-{0}
    return bool(nearby_labels(p)&nearby_labels(q))


def _coincident_contact(ink,p,q,a,b,protected_nodes):
    """Two facing external leads can share a pixel; ink must confirm contact."""
    if np.linalg.norm(q-p)>.75:return False
    if any(math.hypot(p[0]-n['position']['x'],p[1]-n['position']['y'])<=8
           for n in protected_nodes if n.get('position')):return False
    for c,position in ((a,p),(b,q)):
        box=c['bbox']
        edge=min(position[0]-box['x1'],box['x2']-position[0],position[1]-box['y1'],box['y2']-position[1])
        if not 0<edge<=4.5:return False
    na=_outside_normal(p,a['bbox']);nb=_outside_normal(q,b['bbox'])
    if float(na@nb)>-.75:return False
    return (_lead_support(ink,p,q,na) and _lead_support(ink,q,p,nb)
            and _lead_support(ink,p,p+na*3,na) and _lead_support(ink,q,q+nb*3,nb))


def construir_conectividade(image: np.ndarray, wires: dict, terminals: dict) -> dict:
    if image is None or image.size==0:raise ValueError('Imagem original indisponivel para associar pinos e fios.')
    from backend.drawing_mode import digital
    if digital():
        from backend.printed_connectivity import build
        return build(image,wires,terminals)
    from backend.paper_ink import wire_analysis_image
    image=wire_analysis_image(image,wires)
    stage9=wires.get('stage9')
    if not stage9:raise ValueError('Execute a etapa 9 de juncoes antes de construir as redes.')
    components=deepcopy(terminals.get('components') or [])
    if not components:raise ValueError('Execute a deteccao de terminais antes de construir as redes.')
    segments=wires.get('segments') or []
    by_wire={str(s['wire_id']):s for s in segments}
    wire_bounds={}
    for wid,segment in by_wire.items():
        points=segment.get('points') or []
        if points:
            wire_bounds[wid]=(min(p['x'] for p in points),min(p['y'] for p in points),
                              max(p['x'] for p in points),max(p['y'] for p in points))
    if len(by_wire)!=len(segments):raise ValueError('Identificadores de fios repetidos.')
    ids=[str(c.get('component_id') or c.get('detection_id') or '') for c in components]
    if any(not i for i in ids) or len(set(ids))!=len(ids):raise ValueError('Identificadores de componentes ausentes ou repetidos.')
    groups=_Groups()
    for wire in by_wire:groups.find('W:'+wire)
    issues=[];explicit={}
    def issue(code,message,**context):issues.append(dict(code=code,message=message,**context))
    known_status={'endpoint','wire_path','crossing_unconnected','junction_connected'}
    for node in stage9.get('nodes',[]):
        status=node.get('electrical_status')
        if status not in known_status or node.get('review_required'):
            issue('junction_pending','Juncao pendente de confirmacao.',node_id=node.get('node_id'))
            continue
        for connection in node.get('connection_groups') or []:
            if any(str(w) not in by_wire for w in connection):
                raise ValueError('Juncao referencia fio que nao existe na etapa 8.')
            groups.union(['W:'+str(w) for w in connection])
        if status=='junction_connected':
            connected=[str(w) for w in node.get('incident_wire_ids',[]) if str(w) in by_wire]
            for ref in node.get('attached_terminal_refs',[]):explicit.setdefault(str(ref),[]).extend(connected)
    profile=wires.get('ink_color_profile') or {}
    filtered,ruling_mask=apply_notebook_rule_profile(image,wires.get('notebook_rule_profile'),profile)
    # Neutral ink can change brightness across a shadowed photo. The ruling
    # exclusion and local paper contrast are more reliable there than one
    # global brightness window. Colored ink still requires its learned hue.
    if profile.get('chromatic'):filtered=image_with_component_ink(filtered,profile)
    _,_,normalized=_paper_colors(filtered)
    ink=(normalized.mean(axis=2)<242).astype(np.uint8)
    if filtered.ndim==3 and profile.get('active') and not profile.get('chromatic'):
        hsv=cv2.cvtColor(filtered,cv2.COLOR_BGR2HSV)
        ink[hsv[:,:,1]>45]=0
    _,_,raw_normalized=_paper_colors(image_with_component_ink(image,profile))
    raw_ink=(raw_normalized.mean(axis=2)<242).astype(np.uint8)
    from backend.external_conductor_ink import local_color_mask
    local_pen=local_color_mask(image,profile,ruling_mask)
    recovery_ink=np.maximum(raw_ink,local_pen)
    raw_external_ink=raw_ink.copy()
    neutral_external_ink=None
    raw_neutral_contact_ink=None
    if profile.get('active') and not profile.get('chromatic'):
        # A pencil wire may become lighter along the same photographed bend.
        # Global RGB tolerance must not erase its middle. Local paper contrast
        # supplies a separate neutral-ink view; colored rulings stay excluded.
        _,_,original_normalized=_paper_colors(image)
        saturation=cv2.cvtColor(np.rint(original_normalized).astype(np.uint8),cv2.COLOR_BGR2HSV)[:,:,1]
        neutral_external_ink=((original_normalized.mean(axis=2)<242)&(saturation<45)&(ruling_mask==0)).astype(np.uint8)
        raw_neutral_contact_ink=neutral_external_ink.copy()
    # Component cores cannot provide a hidden path between their pins.
    for c in components:
        if c.get('type')=='gnd':continue  # Grounds share one reference; retain their real stem. VCC outputs remain isolated.
        b=c.get('bbox') or {};keys=('x1','y1','x2','y2')
        if not all(k in b for k in keys):continue
        margin=max(3,min(12,min(b['x2']-b['x1'],b['y2']-b['y1'])*.15))
        ax,ay=max(0,int(b['x1']+margin)),max(0,int(b['y1']+margin))
        bx,by=min(ink.shape[1],int(b['x2']-margin)),min(ink.shape[0],int(b['y2']-margin))
        if ax<bx and ay<by:
            ink[ay:by,ax:bx]=0;raw_external_ink[ay:by,ax:bx]=0
            if neutral_external_ink is not None:neutral_external_ink[ay:by,ax:bx]=0
    # A component/ruling mask can leave two collinear ends on one visible
    # conductor. Recover that path without overriding any resolved crossing.
    incident={}
    for segment in segments:
        for key in ('start_node','end_node'):
            if segment.get(key):incident[segment[key]]=incident.get(segment[key],0)+1
    protected_nodes=[node for node in stage9.get('nodes',[]) if
                     node.get('electrical_status') in {'crossing_unconnected','review_required'} or node.get('review_required')]
    endpoints=[]
    for segment in segments:
        points=segment.get('points') or []
        if len(points)<2 or segment.get('orientation')=='closed_loop_or_curve':continue
        for side,key in ((0,'start_node'),(-1,'end_node')):
            if segment.get(key) and incident.get(segment[key],0)!=1:continue
            p=np.array([points[side]['x'],points[side]['y']],float)
            neighbor=points[1 if side==0 else -2]
            outward=p-np.array([neighbor['x'],neighbor['y']],float)
            length=float(np.linalg.norm(outward))
            if length>0:endpoints.append((segment['wire_id'],p,outward/length))
    snap=max(12,min(26,min(image.shape[:2])*.035));bridges=[]
    for i,(wa,p,na) in enumerate(endpoints):
        for wb,q,nb in endpoints[i+1:]:
            if wa==wb or groups.find('W:'+wa)==groups.find('W:'+wb):continue
            delta=q-p;distance=float(np.linalg.norm(delta))
            if not (0<=distance<=snap):continue
            da=float(delta@na)/max(distance,1e-9);db=float(-delta@nb)/max(distance,1e-9)
            collinear=da>=.75 and db>=.75
            tangent_slack=max(.25,min(.5,5/max(distance,1e-9)))
            elbow=abs(float(na@nb))<=.4 and min(da,db)>=-tangent_slack and max(da,db)>=.75
            # Thick photographed conductors can create two neighboring
            # skeleton paths or separate endpoint IDs at the same corner.
            # Actual continuous ink, outside the erased component cores,
            # establishes their equivalence even when their tangents match.
            adjacent_contact=distance<=8 and all(by_wire[w].get('start_node') and by_wire[w].get('end_node') for w in (wa,wb))
            if not(collinear or elbow or adjacent_contact):continue
            if any(math.hypot(p[0]-n['position']['x'],p[1]-n['position']['y'])<=snap or
                   math.hypot(q[0]-n['position']['x'],q[1]-n['position']['y'])<=snap for n in protected_nodes):continue
            # A photographed elbow need not be exactly perpendicular: allow
            # a few pixels of pen width behind its endpoint tangent. The
            # entire link still has to exist in the original external ink.
            masks=(ink,raw_external_ink)+( (neutral_external_ink,) if neutral_external_ink is not None else () )
            support=any(_lead_support(mask,p,q,na,inward_slack=5) and _lead_support(mask,q,p,nb,inward_slack=5) for mask in masks)
            if not support:continue
            groups.union(['W:'+wa,'W:'+wb]);bridges.append(dict(wire_ids=[wa,wb],distance_px=round(distance,3),evidence='continuous_source_ink_adjacent_endpoints' if adjacent_contact and not(collinear or elbow) else 'continuous_source_ink_at_elbow' if elbow and not collinear else 'continuous_source_ink_collinear'))
    connections=[];ground_keys=[];net_labels={};source_contrast_ink=None
    for c,cid in zip(components,ids):
        observed=[str(t.get('pin') or '') for t in c.get('terminals',[])]
        expected=PIN_SCHEMAS.get(c.get('type'))
        if expected is None:issue('unsupported_component','Classe do componente requer confirmacao.',component_id=cid)
        elif set(observed)!=set(expected):issue('pin_count_mismatch','Quantidade ou nomes de pinos incompletos.',component_id=cid,expected_pins=expected,observed_pins=observed)
        if len(set(observed))!=len(observed) or any(not p for p in observed):raise ValueError('Pinos ausentes ou repetidos no componente '+cid)
        if c.get('review_required') or (c.get('terminal_detection') or {}).get('review_required'):
            issue('component_pending','Componente ou identidade dos pinos requer confirmacao.',component_id=cid)
        for terminal in c.get('terminals',[]):
            pin=str(terminal['pin']);ref=cid+':'+pin;key='T:'+ref;groups.find(key)
            record=dict(terminal_ref=ref,component_id=cid,pin=pin,electrical_role=terminal.get('electrical_role'),
                        position=deepcopy(terminal.get('position')),status='unconnected',wire_id=None,distance_px=None)
            b=c.get('bbox') or {};position=terminal.get('position') or {}
            try:
                p=np.array([float(position['x']),float(position['y'])])
                valid=np.all(np.isfinite(p)) and 0<=p[0]<image.shape[1] and 0<=p[1]<image.shape[0]
                valid=valid and b['x1']<p[0]<b['x2'] and b['y1']<p[1]<b['y2']
            except (KeyError,TypeError,ValueError):valid=False
            if not valid:
                record['status']='invalid_position';issue('invalid_pin_position','Pino sem posicao valida dentro da caixa.',terminal_ref=ref)
            elif explicit.get(ref):
                groups.union([key]+['W:'+w for w in explicit[ref]])
                record.update(status='attached_by_junction',wire_id=explicit[ref][0],distance_px=0.)
            else:
                normal=_outside_normal(p,b)
                radius=max(18.,min(32.,float(terminal.get('search_radius_px') or 0)))
                radius=max(radius,min(24.,min(b['x2']-b['x1'],b['y2']-b['y1'])*.20))
                first_radius=radius
                radius=max(radius,min(80.,max(b['x2']-b['x1'],b['y2']-b['y1'])*.9))
                candidates={}
                for wid,segment in by_wire.items():
                    bounds=wire_bounds.get(wid)
                    if bounds and (p[0]<bounds[0]-radius or p[0]>bounds[2]+radius or
                                   p[1]<bounds[1]-radius or p[1]>bounds[3]+radius):continue
                    near=_nearest_on_polyline(p,segment)
                    if not near or near['distance']>radius:continue
                    q=np.array(near['point']);outward=float((q-p)@normal)
                    if outward < -2:continue
                    if near['distance']>first_radius:
                        delta=q-p;length=max(1e-9,float(delta@delta))
                        if any(np.linalg.norm(np.array([n['position']['x'],n['position']['y']])-(p+np.clip(float((np.array([n['position']['x'],n['position']['y']])-p)@delta)/length,0,1)*delta))<8 for n in protected_nodes if n.get('position')):continue
                    supported=_lead_support(ink,p,q,normal)
                    # This narrow overlap case was already confirmed on the
                    # original lead by the terminal stage and retained wire.
                    px,py=np.rint(p).astype(int)
                    restored=bool(terminal.get('wire_contact_confirmed') and near['distance']<=5
                                  and ruling_mask[py,px] and _lead_support(raw_ink,p,q,normal))
                    if (not supported and not restored and terminal.get('local_wire_shade_confirmed')
                            and terminal.get('wire_contact_confirmed') and near['distance']<=5
                            and not ruling_mask[py,px]):
                        # A tint-filtered lead was confirmed in the original
                        # image by the terminal stage. Recheck the short path
                        # to an existing wire on original paper contrast.
                        if source_contrast_ink is None:
                            _,_,original_normalized=_paper_colors(image)
                            source_contrast_ink=((original_normalized.mean(axis=2)<242)&(ruling_mask==0)).astype(np.uint8)
                        restored=_lead_support(source_contrast_ink,p,q,normal)
                    if not supported and not restored:continue
                    root=groups.find('W:'+wid)
                    tangent=np.array([math.cos(math.radians(near['angle'])),math.sin(math.radians(near['angle']))])
                    alignment=abs(float(tangent@normal))
                    score=near['distance']+3*(1-alignment)
                    candidate=(score,near['distance'],wid)
                    if root not in candidates or candidate<candidates[root]:candidates[root]=candidate
                ranked=sorted(candidates.values())
                if ranked and (len(ranked)==1 or ranked[1][0]-ranked[0][0]>2.0):
                    _,distance,wid=ranked[0];groups.union([key,'W:'+wid])
                    record.update(status='attached_surroundings' if distance>first_radius else 'attached',wire_id=wid,distance_px=round(float(distance),3))
                elif ranked:
                    record.update(status='ambiguous',candidate_wire_ids=[item[2] for item in ranked])
                    issue('ambiguous_pin','Mais de uma rede chega perto do pino; confirmar ligacao.',terminal_ref=ref)
                else:issue('unconnected_pin','Pino sem fio externo confirmado.',terminal_ref=ref)
            if c.get('type')=='gnd':ground_keys.append(key)
            # Only explicit electrical labels join distant power markers.
            # A VCC classification alone must not short all output circles.
            label=c.get('net_label') or terminal.get('net_label')
            if label and str(label).strip():net_labels.setdefault(str(label).strip(),[]).append(key)
            connections.append(record)
    # Two components can have adjacent/overlapping detection boxes that erase
    # their entire short external lead in Stage 8. Recover only a continuous
    # outward lead between facing pins; never join pins through either body.
    by_component={cid:c for c,cid in zip(components,ids)}
    direct_refs=set();coincident_contacts=[]
    # Pale external ink can fall outside the global pen saturation window.
    # For an exact contact only, use paper contrast in the original photo;
    # notebook rules and the facing-body boundary checks remain mandatory.
    coincident_ink=None
    for i,a in enumerate(connections):
        for b in connections[i+1:]:
            if a['component_id']==b['component_id']:continue
            already_attached=not ({a['status'],b['status']} & {'unconnected'})
            if already_attached and groups.find('T:'+a['terminal_ref'])==groups.find('T:'+b['terminal_ref']):continue
            if a['status'] in {'ambiguous','invalid_position'} or b['status'] in {'ambiguous','invalid_position'}:continue
            p=np.array([a['position']['x'],a['position']['y']],float)
            q=np.array([b['position']['x'],b['position']['y']],float);delta=q-p;distance=float(np.linalg.norm(delta))
            if distance>32:continue
            na=_outside_normal(p,by_component[a['component_id']]['bbox']);nb=_outside_normal(q,by_component[b['component_id']]['bbox'])
            coincident=distance<=.75
            if coincident:
                if coincident_ink is None:
                    _,_,original_normalized=_paper_colors(image)
                    coincident_ink=((original_normalized.mean(axis=2)<242)&(ruling_mask==0)).astype(np.uint8)
                if not _coincident_contact(coincident_ink,p,q,by_component[a['component_id']],by_component[b['component_id']],protected_nodes):continue
                coincident_contacts.append(dict(terminal_refs=[a['terminal_ref'],b['terminal_ref']],position=dict(x=float(p[0]),y=float(p[1])),evidence='continuous_ink_facing_external_leads'))
            else:
                ab=by_component[a['component_id']]['bbox'];bb=by_component[b['component_id']]['bbox']
                overlap_x=max(0,min(ab['x2'],bb['x2'])-max(ab['x1'],bb['x1']))
                overlap_y=max(0,min(ab['y2'],bb['y2'])-max(ab['y1'],bb['y1']))
                size=min(ab['x2']-ab['x1'],ab['y2']-ab['y1'],bb['x2']-bb['x1'],bb['y2']-bb['y1'])
                observed_contacts=all(any(t.get('pin')==r['pin'] and t.get('wire_contact_confirmed')
                    for t in by_component[r['component_id']].get('terminals',[])) for r in (a,b))
                overlap_limit=size*(.35 if observed_contacts else .18)
                overlap=bool(overlap_x>0 and overlap_y>0 and min(overlap_x,overlap_y)<=overlap_limit and float(na@nb)<-.75)
                if already_attached and not overlap:continue
                if overlap:
                    # Reversed face positions occur in the thin overlap of
                    # adjacent boxes. Verify the photographed lead itself;
                    # the global body eraser deliberately removed this strip.
                    direction=delta/distance
                    contact_masks=(raw_ink,recovery_ink*(ruling_mask==0))+( (raw_neutral_contact_ink,) if raw_neutral_contact_ink is not None else () )
                    if not any(_lead_support(m,p,q,direction) and _lead_support(m,q,p,-direction) for m in contact_masks):continue
                    if any(math.hypot(p[0]-n['position']['x'],p[1]-n['position']['y'])<=distance+4 for n in protected_nodes):continue
                else:
                    if float(delta@na)/distance<.5 or float(-delta@nb)/distance<.5:continue
                    if not _lead_support(ink,p,q,na) or not _lead_support(ink,q,p,nb):continue
            groups.union(['T:'+a['terminal_ref'],'T:'+b['terminal_ref']])
            for record in (a,b):
                if record['status']=='unconnected':
                    record.update(status='attached_coincident_contact' if coincident else 'attached_direct_lead',distance_px=round(distance,3))
                    direct_refs.add(record['terminal_ref'])
    from backend.observed_leads import recover_observed_leads
    observed_leads=[]
    by_ref={r['terminal_ref']:r for r in connections}
    # Preserve the established narrow-color paths first. A desaturated view
    # can reveal extra internal branches; those must not veto an independent
    # external path already established by the more selective original view.
    masks=[raw_ink]
    if np.any(recovery_ink!=raw_ink):masks.append(recovery_ink)
    for observed_ink in masks:
        found=recover_observed_leads(observed_ink*(ruling_mask==0),components,segments,protected_nodes,
            {key:groups.find(key) for key in list(groups.parent)},
            {'T:'+r['terminal_ref'] for r in connections if r['status'] in {'ambiguous','invalid_position'}})
        for lead in found:
            groups.union(lead['anchors']);observed_leads.append(lead)
            for key in lead['anchors']:
                if not key.startswith('T:'):continue
                ref=key[2:];record=by_ref[ref]
                if record['status']=='unconnected':
                    record.update(status='attached_observed_lead',distance_px=lead['length_px'])
                    direct_refs.add(ref)
    pen_gaps,gap_policy=recover_pen_gaps(recovery_ink,components,wires,connections,protected_nodes,groups.parent)
    for gap in pen_gaps:
        groups.union(gap['anchors'])
        for key in gap['anchors']:
            if not key.startswith('T:'):continue
            ref=key[2:];record=by_ref[ref]
            if record['status']=='unconnected':
                record.update(status='attached_pen_gap',distance_px=gap['length_px'])
                wire=next((a[2:] for a in gap['anchors'] if a.startswith('W:')),None)
                if wire is not None:record['wire_id']=wire
                direct_refs.add(ref)
    issues=[issue for issue in issues if not (issue['code']=='unconnected_pin' and issue.get('terminal_ref') in direct_refs)]
    groups.union(ground_keys)
    for label,keys in net_labels.items():
        if label.upper() in {'GND','0'}:
            groups.union(keys+ground_keys);ground_keys.extend(key for key in keys if key not in ground_keys)
        else:groups.union(keys)
    roots={}
    for wid in by_wire:roots.setdefault(groups.find('W:'+wid),dict(wire_ids=[],terminal_refs=[],labels=[]))['wire_ids'].append(wid)
    for record in connections:roots.setdefault(groups.find('T:'+record['terminal_ref']),dict(wire_ids=[],terminal_refs=[],labels=[]))['terminal_refs'].append(record['terminal_ref'])
    ground_root=groups.find(ground_keys[0]) if ground_keys else None
    for label,keys in net_labels.items():
        if keys:roots[groups.find(keys[0])]['labels'].append(label)
    def order(item):
        root,net=item
        return (root!=ground_root,min(net['wire_ids'] or ['~']),min(net['terminal_refs'] or ['~']))
    nets=[];by_root={};number=0
    for root,net in sorted(roots.items(),key=order):
        ground=root==ground_root
        if not ground:number+=1
        name='GND' if ground else f'NET_{number}'
        net.update(name=name,is_ground=ground,member_count=len(net['terminal_refs']),
                   wire_ids=sorted(net['wire_ids']),terminal_refs=sorted(net['terminal_refs']))
        nets.append(net);by_root[root]=name
    for record in connections:record['net']=by_root[groups.find('T:'+record['terminal_ref'])]
    for lead in observed_leads:lead['net']=by_root[groups.find(lead['anchors'][0])]
    for gap in pen_gaps:gap['net']=by_root[groups.find(gap['anchors'][0])]
    if not ground_keys and any(c.get('type') in {'source_dc','battery'} for c in components):
        issue('source_without_ground','Circuito com fonte sem referencia GND reconhecida.')
    for c,cid in zip(components,ids):
        pin_nets={r['net'] for r in connections if r['component_id']==cid}
        if len(c.get('terminals',[]))>=2 and len(pin_nets)==1:
            issue('component_pins_same_net','Todos os pinos do componente estao na mesma rede; verificar curto.',component_id=cid)
    unattached=sum(r['status'] in {'unconnected','invalid_position','ambiguous'} for r in connections)
    return dict(schema_version='stage10-connectivity-1.0',created_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                project_id=wires.get('project_id'),source_signature=source_signature(wires,terminals),
                components=components,nets=nets,terminal_connections=connections,warnings=issues,
                net_count=len(nets),terminal_count=len(connections),attached_terminal_count=len(connections)-unattached,
                unconnected_terminal_count=sum(r['status']=='unconnected' for r in connections),
                ambiguous_terminal_count=sum(r['status']=='ambiguous' for r in connections),
                wire_only_net_count=sum(not n['terminal_refs'] for n in nets),review_count=len(issues),
                recovered_wire_bridges=bridges,
                recovered_observed_leads=observed_leads,
                recovered_coincident_contacts=coincident_contacts,
                recovered_pen_gaps=pen_gaps,pen_gap_policy=gap_policy,
                criterion_met=bool(connections and not unattached and not any(i['code'] in {'junction_pending','invalid_pin_position','pin_count_mismatch'} for i in issues)))


def salvar_imagem_redes(image, connectivity, wires, output_path: Path):
    canvas=image.copy();segments={s['wire_id']:s for s in wires.get('segments',[])}
    palette=[(40,150,230),(180,80,40),(70,160,60),(180,60,170),(30,180,180),(140,120,30)]
    for i,net in enumerate(connectivity['nets']):
        color=(45,45,45) if net['is_ground'] else palette[i%len(palette)]
        for wid in net['wire_ids']:
            points=segments[wid].get('points') or []
            if len(points)>1:cv2.polylines(canvas,[np.array([[p['x'],p['y']] for p in points],np.int32)],False,color,3,cv2.LINE_AA)
        for lead in connectivity.get('recovered_observed_leads',[])+connectivity.get('recovered_pen_gaps',[]):
            if lead.get('net')!=net['name']:continue
            points=lead.get('points') or []
            if len(points)>1:cv2.polylines(canvas,[np.array([[p['x'],p['y']] for p in points],np.int32)],False,color,3,cv2.LINE_AA)
    for record in connectivity['terminal_connections']:
        p=record.get('position') or {}
        if not all(k in p and isinstance(p[k],(int,float)) and math.isfinite(p[k]) for k in ('x','y')):continue
        x,y=round(p['x']),round(p['y']);color=(20,20,230) if record['status'] in {'unconnected','ambiguous','invalid_position'} else (45,45,45)
        cv2.circle(canvas,(x,y),5,color,2,cv2.LINE_AA)
        cv2.putText(canvas,record['net'],(x+7,y-7),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1,cv2.LINE_AA)
    if not cv2.imwrite(str(output_path),canvas):raise ValueError('Nao foi possivel salvar a imagem das redes.')
