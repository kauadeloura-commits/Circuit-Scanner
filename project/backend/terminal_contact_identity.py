"""Reconcile BJT electrical names with the three final photographed wire contacts."""
from itertools import permutations
import numpy as np


def finalize_bjt_on_contacts(image,component,skeleton,reference):
    if component.get('type') not in {'transistor_npn','transistor_pnp'}:return None
    meta=dict(source='bjt_visible_arrow_base_bar_final_contacts',confirmed=False)
    terms=component.get('terminals') or []
    if len(terms)!=3:return meta
    from backend.drawing_mode import digital
    if digital():
        from backend.printed_bjt import read_roles
        anchors,proof=read_roles(image,component['bbox'])
        if anchors:
            points=np.array([[t['position']['x'],t['position']['y']] for t in terms],float)
            names=('B','C','E');wanted=np.array([anchors[p] for p in names])
            ranked=sorted((float(np.linalg.norm(points[list(order)]-wanted,axis=1).sum()),order) for order in permutations(range(3)))
            cost,order=ranked[0];scale=min(component['bbox']['x2']-component['bbox']['x1'],component['bbox']['y2']-component['bbox']['y1'])
            if np.max(np.linalg.norm(points[list(order)]-wanted,axis=1))<=max(6,scale*.30) and ranked[1][0]-cost>=4:
                for pin,j in zip(names,order):
                    terms[j].update(pin=pin,pin_id=component['component_id']+'.'+pin,index=names.index(pin)+1,electrical_role={'B':'base','C':'collector','E':'emitter'}[pin],identity_source=proof['source'],identity_uncertain=False)
                    if pin=='B':
                        p=anchors['B'];terms[j]['position']=dict(x=float(p[0]),y=float(p[1]))
                component['terminals']=sorted(terms,key=lambda t:t['index']);meta.update(confirmed=True,geometry=proof,source=proof['source'])
                component['terminal_contact_identity']=meta
                return meta
    geometry_before=component.get('transistor_geometry') or {}
    prior=geometry_before.get('original_ink_geometry') or {}
    selected=geometry_before.get('selected_wire_points') or {}
    if prior.get('arrow_confirmed') and max(prior.get('stage8_distances_px') or [float('inf')])<=5:
        agrees=all(t.get('pin') in selected and np.linalg.norm(
            np.array([t['position']['x'],t['position']['y']])-np.array([selected[t['pin']]['x'],selected[t['pin']]['y']]))<=5 for t in terms)
        if agrees:
            meta.update(confirmed=True,reason='visible_roles_already_match_final_contacts')
            return meta
    from backend.npn_wire_contacts import detect_bjt_wire_contacts
    found,geometry=detect_bjt_wire_contacts(image,component['bbox'],skeleton,component['type'],color_reference=reference)
    if found is None or not geometry.get('arrow_confirmed'):
        from backend.local_symbol_roles import small_bjt
        local,geometry=small_bjt(image,component,skeleton)
        if local is None:
            meta['reason']='three_leads_and_exclusive_arrow_not_confirmed';return meta
        anchors=np.array([local['B'],local['E']],float)
    else:
        chosen,evidence=found
        proof=evidence['hypotheses'][0]['points']
        anchors=np.array([chosen[0]['point'],[proof['E']['x'],proof['E']['y']]],float)
    points=np.array([[t['position']['x'],t['position']['y']] for t in terms],float)
    scale=max(1,min(component['bbox']['x2']-component['bbox']['x1'],component['bbox']['y2']-component['bbox']['y1']))
    ranked=sorted((float(np.linalg.norm(points[list(order)]-anchors,axis=1).sum()),order) for order in permutations(range(3),2))
    cost,order=ranked[0]
    distance=np.linalg.norm(points[list(order)]-anchors,axis=1)
    if distance.max()>max(5,scale*.15) or ranked[1][0]-cost<max(4,scale*.10):
        meta['reason']='final_contacts_do_not_match_visible_base_and_emitter';return meta
    names={order[0]:'B',order[1]:'E'}
    names[next(i for i in range(3) if i not in order)]='C'
    changed=[]
    roles={'B':'base','C':'collector','E':'emitter'};indexes={'B':1,'C':2,'E':3}
    for i,t in enumerate(terms):
        name=names[i]
        if t.get('pin')!=name:changed.append(dict(before=t.get('pin'),after=name))
        t.update(pin=name,pin_id=f"{component.get('component_id','COMP')}.{name}",index=indexes[name],
                 electrical_role=roles[name],identity_uncertain=bool(geometry.get('direction_conflict')),identity_source=meta['source'])
    component['terminals']=sorted(terms,key=lambda t:t['index'])
    meta.update(confirmed=True,reason='exclusive_arrow_emitter_base_bar_then_remaining_collector',changed_names=changed,
                direction_conflict=geometry.get('direction_conflict',False),geometry=geometry)
    component['terminal_contact_identity']=meta
    if geometry.get('direction_conflict'):
        reason='classe_bjt_conflita_com_sentido_estimado_da_seta'
        for target in (component,component.setdefault('terminal_detection',{})):
            target['review_required']=True
            target['review_reasons']=list(dict.fromkeys(target.get('review_reasons',[])+[reason]))
    return meta
