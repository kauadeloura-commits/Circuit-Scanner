"""Read small photographed symbol roles at a normalized local pixel scale."""
import cv2,numpy as np
from itertools import permutations

def local_view(image,box,scale=2,skeleton=None):
    pad=20;x=max(0,int(box['x1'])-pad);y=max(0,int(box['y1'])-pad)
    z=min(image.shape[1],int(np.ceil(box['x2']))+pad);t=min(image.shape[0],int(np.ceil(box['y2']))+pad)
    view=cv2.resize(image[y:t,x:z],None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
    b={k:(box[k]-(x if k.startswith('x') else y))*scale for k in ('x1','y1','x2','y2')}
    sk=None if skeleton is None else cv2.resize(skeleton[y:t,x:z],None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
    return view,b,sk,np.array([x,y],float)

def small_bjt(image,component,skeleton):
    from backend.npn_wire_contacts import detect_bjt_wire_contacts
    box=component['bbox'];minimum=min(box['x2']-box['x1'],box['y2']-box['y1'])
    if minimum>110:return None,{}
    view,b,sk,origin=local_view(image,box,2,skeleton)
    found,meta=detect_bjt_wire_contacts(view,b,sk,component['type'],color_reference=view)
    if found is None or not meta.get('arrow_confirmed'):return None,meta
    points=found[1]['hypotheses'][0]['points']
    anchors={name:np.array([p['x'],p['y']])/2+origin for name,p in points.items()}
    remaining=[np.array(p['point'])/2+origin for p in found[0] if all(np.linalg.norm(np.array(p['point'])/2+origin-a)>3 for a in anchors.values())]
    if len(remaining)!=1:return None,dict(meta,reason='third_independent_lead_not_separated')
    anchors['C']=remaining[0]
    return anchors,dict(meta,local_analysis_scale=2,source='small_bjt_local_scale_arrow_and_bar')

def small_mosfet(image,component):
    from backend.terminal_identity import _crop,_marks,_mosfet
    from backend.continuous_mosfet import read as read_continuous
    observed=read_continuous(image,component['bbox'])
    if observed and observed['kind']==component.get('type'):
        return {k:np.array(v) for k,v in observed['points'].items()},dict(confirmed=True,source=observed['method'],visible_source_arrow=observed)
    mask,offset=_crop(image,component['bbox']);_,labels=_marks(mask,offset,component['bbox'])
    direct=_mosfet(mask,offset,component['bbox'],labels)
    if direct is not None and direct[1]:
        return direct[0],dict(confirmed=True,source='isolated_gate_and_distinct_observed_current_branches',source_branch_radii=direct[2])
    box=component['bbox'];minimum=min(box['x2']-box['x1'],box['y2']-box['y1'])
    if minimum>115:return None,{}
    proposals=[]
    for scale in (2,3):
        view,b,_,origin=local_view(image,box,scale)
        mask,offset=_crop(view,b);_,labels=_marks(mask,offset,b)
        found=_mosfet(mask,offset,b,labels)
        if found is None:continue
        points,confirmed,radii=found
        proposals.append((scale,{name:point/scale+origin for name,point in points.items()},confirmed,radii))
    if not proposals:return None,{}
    if len(proposals)==2:
        if any(np.linalg.norm(proposals[0][1][name]-proposals[1][1][name])>5 for name in ('G','D','S')):
            return None,dict(confirmed=False,reason='local_scale_roles_disagree')
    scale,points,confirmed,radii=proposals[0]
    return points,dict(confirmed=confirmed or len(proposals)==2,local_analysis_scale=scale,agreement_scales=[p[0] for p in proposals],source='small_mosfet_isolated_gate_and_body_branch')

def apply_roles(component,anchors,meta):
    terms=component.get('terminals') or []
    if len(terms)!=3 or anchors is None:return False
    names=list(anchors);points=np.array([[t['position']['x'],t['position']['y']] for t in terms]);wanted=np.array(list(anchors.values()))
    ranked=sorted((float(np.linalg.norm(points[list(order)]-wanted,axis=1).sum()),order) for order in permutations(range(3),len(names)))
    cost,order=ranked[0]
    box=component['bbox'];tolerance=max(6,min(16,min(box['x2']-box['x1'],box['y2']-box['y1'])*.08))
    if np.max(np.linalg.norm(points[list(order)]-wanted,axis=1))>tolerance or ranked[1][0]-cost<5:return False
    roles={j:pin for j,pin in zip(order,names)}
    if component['type'].startswith('transistor') and len(order)<3:roles[next(i for i in range(3) if i not in order)]='C'
    indexes={'B':1,'C':2,'E':3,'G':1,'D':2,'S':3}
    for j,t in enumerate(terms):
        pin=roles[j];t.update(pin=pin,index=indexes[pin],pin_id=component['component_id']+'.'+pin,electrical_role={'B':'base','C':'collector','E':'emitter','G':'gate','D':'drain','S':'source'}[pin],identity_source=meta['source'],identity_uncertain=not meta.get('confirmed',meta.get('arrow_confirmed',False)))
    component['terminals']=sorted(terms,key=lambda t:t['index']);component['local_symbol_roles']=meta
    return True
