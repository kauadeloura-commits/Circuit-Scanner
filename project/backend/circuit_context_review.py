"""Resolve symbol identity against the final observed electrical contacts."""
import math
import cv2
import numpy as np


def finalize_diode_on_contacts(image,component):
    """A photographed triangular diode needs a bar across its actual apex.

    Neither page orientation nor which end happened to be called A/K by a
    keypoint model decides the cathode. Existing contact positions are kept.
    Without a closed triangular body and transverse bar, keep the old result.
    """
    terms=component.get('terminals') or []
    if component.get('type') not in {'diode','led'} or len(terms)!=2:return None
    if component.get('manual_reviewed') or any(t.get('manual_reviewed') for t in terms):return None
    from backend.drawing_mode import digital
    from backend.drawing_mode import CURRENT
    if digital() or CURRENT.get()=='handwritten':
        from backend.diode_contacts import select_diode_contacts
        contacts=[dict(point=[t['position']['x'],t['position']['y']]) for t in terms]
        _,selection=select_diode_contacts(image,component,contacts)
        if selection:
            for term in terms:
                p=np.array([term['position']['x'],term['position']['y']])
                name=min(('A','K'),key=lambda pin:np.linalg.norm(p-selection['selected_contacts'][pin]))
                term.update(pin=name,pin_id=component['component_id']+'.'+name,index=1 if name=='A' else 2,
                            electrical_role='anode' if name=='A' else 'cathode',identity_uncertain=False,polarity_source=selection['source'])
            component['terminals']=sorted(terms,key=lambda t:t['index'])
            component['terminal_identity_geometry']=selection
            return selection
    from backend.terminal_identity import _crop
    mask,origin=_crop(image,component['bbox'])
    pair=np.array([[t['position']['x'],t['position']['y']] for t in terms],float)
    vector=pair[1]-pair[0];length=float(np.linalg.norm(vector))
    if length<10:return None
    axis=vector/length;normal=np.array([-axis[1],axis[0]])
    b=component['bbox'];area=(b['x2']-b['x1'])*(b['y2']-b['y1'])
    contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    proposals=[]
    for contour in contours:
        a=abs(cv2.contourArea(contour))
        if not max(14,area*.035)<=a<=area*.65:continue
        polygon=cv2.approxPolyDP(contour,.045*cv2.arcLength(contour,True),True).reshape(-1,2)
        if len(polygon)!=3:continue
        points=polygon.astype(float)+origin
        for k,apex in enumerate(points):
            base=np.delete(points,k,axis=0);side=base[1]-base[0];width=float(np.linalg.norm(side))
            if width<8 or abs(float(side@axis))/width>.35:continue
            height=float((apex-base.mean(axis=0))@axis)
            if abs(height)<max(7,width*.3):continue
            direction=axis*(1 if height>0 else -1)
            # Inner triangle contours stop a few pen widths before the bar.
            # Both arms of a separate transverse bar must be observed beyond
            # the apex; the sloping sides alone cannot satisfy this test.
            best=0.
            for shift in np.linspace(-2,max(3,abs(height)*.22),9):
                center=apex+direction*shift-origin
                scores=[]
                for sign in (-1,1):
                    seen=[]
                    for offset in np.linspace(width*.22,width*.52,14):
                        x,y=np.rint(center+sign*normal*offset).astype(int)
                        if not(0<=x<mask.shape[1] and 0<=y<mask.shape[0]):seen.append(False);continue
                        zone=mask[max(0,y-2):y+3,max(0,x-2):x+3]
                        seen.append(bool(np.any(zone)))
                    scores.append(float(np.mean(seen)))
                best=max(best,min(scores))
            if best<.78:continue
            cathode=int(np.argmax((pair-base.mean(axis=0))@direction))
            proposals.append((best,a,cathode,apex))
    if not proposals:return None
    proposals.sort(key=lambda p:(p[0],p[1]),reverse=True)
    best=proposals[0]
    if any(p[2]!=best[2] and p[0]>=best[0]-.08 for p in proposals[1:]):return None
    for index,t in enumerate(terms):
        pin='K' if index==best[2] else 'A'
        t.update(pin=pin,pin_id=f"{component.get('component_id','COMP')}.{pin}",
                 index=2 if pin=='K' else 1,electrical_role='cathode' if pin=='K' else 'anode',
                 identity_uncertain=False,polarity_source='triangle_apex_and_observed_cathode_bar_after_contacts')
    component['terminals']=sorted(terms,key=lambda t:t['index'])
    meta=dict(source='triangle_apex_and_observed_cathode_bar_after_contacts',confirmed=True,
              transverse_bar_support=round(best[0],3),apex=[round(float(v),2) for v in best[3]])
    component['terminal_identity_geometry']=meta
    return meta
