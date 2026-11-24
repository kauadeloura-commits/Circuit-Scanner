"""Candidate: account for sign size and compact crosses touching page strokes."""
import cv2,numpy as np

def verified_plate_signs(image,bbox,marks):
    from backend.drawing_mode import CURRENT
    if CURRENT.get()!='handwritten' or not marks:return marks
    from backend.capacitor_geometry import capacitor_body_evidence
    ok,body=capacitor_body_evidence(image,bbox)
    if not ok or not body.get('outer_plate_centers'):return marks
    plates=np.asarray(body['outer_plate_centers'],float);axis=np.asarray(body['electrical_axis'],float);normal=np.array([-axis[1],axis[0]])
    center=plates.mean(axis=0);length=max(body['outer_plate_lengths'])
    from backend.symbol_semantics import _mark_crop,_search_box
    mask,origin=_mark_crop(image,_search_box(image,bbox),True)
    _,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    def paths(point):
        x,y=np.rint(np.asarray(point)-origin).astype(int)
        if not 0<=x<labels.shape[1] or not 0<=y<labels.shape[0]:return set()
        return set(labels[max(0,y-2):y+3,max(0,x-2):x+3].ravel())-{0}
    plate_paths=set().union(*(paths(p) for p in plates));output=[]
    for mark in marks:
        delta=np.asarray(mark['point'])-center
        bounds=mark.get('bounds') or [*mark['point'],*mark['point']]
        diameter=max(1,bounds[2]-bounds[0],bounds[3]-bounds[1])
        # A lead meeting the middle of a plate is a body junction. Resizing
        # that T can create four skeleton arms; it is not a separate '+' label.
        if mark.get('attached') and any(abs(float((np.asarray(mark['point'])-p)@normal))<=.30*plate_length and
               abs(float((np.asarray(mark['point'])-p)@axis))<=max(2,.10*plate_length)
               for p,plate_length in zip(plates,body['outer_plate_lengths'])):continue
        transverse=abs(float(delta@normal));core=max(10,.90*length)
        if transverse>core:
            # Extra distance accounts for the sign's own size, but a weak
            # letter-like junction must not use that additional allowance.
            if mark.get('score',0)<.58 or diameter>length or transverse>core+diameter*.5:continue
        if abs(float(delta@axis))<2:continue
        if min(np.linalg.norm(np.asarray(mark['point'])-p) for p in plates)>max(14,1.7*length):continue
        attached_to_long_stroke=False
        if mark.get('attached') and not paths(mark['point']).intersection(plate_paths):
            for label in paths(mark['point']):
                _,_,w,h,_=stats[label]
                if max(w,h)>=4*diameter and max(w,h)>=2.5*max(1,min(w,h)) and mark.get('score',0)>=.80:
                    attached_to_long_stroke=True
            if not attached_to_long_stroke:continue
        output.append(dict(mark,plate_owner_verified=True,compact_cross_on_long_ink_path=attached_to_long_stroke))
    return output
