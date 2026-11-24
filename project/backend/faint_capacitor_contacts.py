"""Recover faint photographed leads only outside two independently read plates."""
import cv2,numpy as np


def recover(image,component,contacts,skeleton):
    from backend.drawing_mode import CURRENT
    if CURRENT.get()!='handwritten' or component.get('type') not in {'capacitor','capacitor_polarized'} or len(contacts)>=2:return contacts,None
    if component.get('manual_reviewed'):return contacts,None
    from backend.capacitor_geometry import capacitor_body_evidence
    ok,body=capacitor_body_evidence(image,component['bbox'])
    if not ok or not body.get('outer_plate_centers') or not body.get('electrical_axis'):return contacts,None
    plates=np.asarray(body['outer_plate_centers']);axis=np.asarray(body['electrical_axis']);axis/=np.linalg.norm(axis)
    normal=np.array([-axis[1],axis[0]]);center=plates.mean(axis=0);offsets=sorted((plates-center)@axis)
    from backend.notebook_rulings import _paper_colors
    from backend.terminal_wire_contacts import _contacts
    _,_,normalized=_paper_colors(image);enhanced=np.full_like(image,255);enhanced[normalized.mean(axis=2)<225]=0
    possible=_contacts(enhanced,component['bbox'],skeleton);chosen=[]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    for sign,limit in ((-1,offsets[0]),(1,offsets[-1])):
        candidates=[c for c in possible if c.get('face_axis') is not None and abs(axis[c['face_axis']])>.8 and sign*((c['point']-center)@axis-limit)>2 and abs((c['point']-center)@normal)<max(body['outer_plate_lengths'])*.30]
        if not candidates:return contacts,None
        port=min(candidates,key=lambda c:abs((c['point']-center)@normal))
        p=port['point'];samples=np.array([p+v*axis+d*normal for v in (-3,0,3) for d in (-8,8)])
        xy=np.rint(samples).astype(int);valid=(xy[:,0]>=0)&(xy[:,0]<gray.shape[1])&(xy[:,1]>=0)&(xy[:,1]<gray.shape[0])
        x,y=np.rint(p).astype(int)
        if not np.any(valid) or gray[y,x]>210 or np.median(gray[xy[valid,1],xy[valid,0]])-float(gray[y,x])<8:return contacts,None
        chosen.append(port)
    if any(min(np.linalg.norm(old['point']-new['point']) for new in chosen)>5 for old in contacts):return contacts,None
    return chosen,dict(source='two_read_plates_and_contrast_confirmed_opposed_photographic_leads',previous_contacts=len(contacts),contacts=[c['point'].tolist() for c in chosen])
