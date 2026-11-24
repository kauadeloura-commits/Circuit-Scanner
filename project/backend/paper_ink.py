"""Local paper-relative strokes for photos whose background changes pen color."""
import cv2
import numpy as np
from backend.resistor_geometry import cached_vision


@cached_vision
def paper_relative_view(image):
    if image.ndim!=3:return image.copy()
    size=max(11,min(31,round(min(image.shape[:2])/35)))|1
    paper=cv2.medianBlur(image,size)
    return np.clip(image.astype(np.float32)*255/np.maximum(paper,1),0,255).astype(np.uint8)


@cached_vision
def stroke_mask(image):
    view=paper_relative_view(image)
    gray=view.mean(axis=2) if view.ndim==3 else view.astype(float)
    weak=(gray<220).astype(np.uint8)
    count,labels=cv2.connectedComponents(weak,connectivity=8)
    strong=np.unique(labels[gray<180]);keep=np.zeros(count,np.uint8);keep[strong]=255;keep[0]=0
    return keep[labels]


def fragmentation_policy(image,components,profile,rule_profile):
    meta=dict(active=False,method='local_paper_contrast_when_global_ink_fragments',sampled=0,fragmented=0)
    from backend.drawing_mode import digital
    if digital():return dict(meta,reason='explicit_digital_input')
    if image.ndim!=3 or rule_profile.get('active'):return meta
    from backend.ink_color import component_ink_mask,estimate_component_ink
    relative_profile=None
    if not profile.get('active'):
        if profile.get('reason')!='component_colors_without_consensus':return meta
        relative_profile=estimate_component_ink(paper_relative_view(image),components)
        if not(relative_profile.get('active') and relative_profile.get('chromatic') and relative_profile.get('sampled_component_count',0)>=4):return meta
    elif not profile.get('chromatic'):return meta
    original=component_ink_mask(image,profile);relative=stroke_mask(image)
    for c in components.get('components',components.get('detections',[])) if isinstance(components,dict) else []:
        if float(c.get('visual_confidence',c.get('class_confidence',c.get('confidence',0))) or 0)<.55:continue
        b=c.get('bbox') or {}
        if not all(k in b for k in ('x1','y1','x2','y2')):continue
        x1,y1=max(0,int(b['x1'])),max(0,int(b['y1']));x2,y2=min(image.shape[1],int(b['x2'])+1),min(image.shape[0],int(b['y2'])+1)
        local=relative[y1:y2,x1:x2]>0
        n=int(np.count_nonzero(local))
        if n<20:continue
        meta['sampled']+=1
        coverage=np.count_nonzero((original[y1:y2,x1:x2]>0)&local)/n if original is not None else 0.
        if coverage<.65:meta['fragmented']+=1
    meta['active']=meta['sampled']>=4 and meta['fragmented']>=max(2,int(np.ceil(meta['sampled']*.25)))
    if meta['active']:
        if relative_profile is None:relative_profile=estimate_component_ink(paper_relative_view(image),components)
        if relative_profile.get('active') and relative_profile.get('chromatic'):
            meta['relative_ink_hue']=relative_profile['reference_hsv'][0]
            hsv=cv2.cvtColor(paper_relative_view(image),cv2.COLOR_BGR2HSV)
            delta=np.abs(hsv[:,:,0].astype(float)-float(meta['relative_ink_hue']));delta=np.minimum(delta,180-delta)
            samples=[]
            for c in components.get('components',components.get('detections',[])) if isinstance(components,dict) else []:
                box=c.get('bbox') or {}
                if not all(k in box for k in ('x1','y1','x2','y2')):continue
                pad=max(6,min(16,min(box['x2']-box['x1'],box['y2']-box['y1'])*.15))
                x1,y1=max(0,int(box['x1']-pad)),max(0,int(box['y1']-pad));x2,y2=min(image.shape[1],int(box['x2']+pad)+1),min(image.shape[0],int(box['y2']+pad)+1)
                local=hsv[y1:y2,x1:x2];d=delta[y1:y2,x1:x2]
                chosen=(relative[y1:y2,x1:x2]>0)&(local[:,:,1]>=25)&(local[:,:,2]<=200)&(d<=40)
                if np.count_nonzero(chosen)>=12:samples.append(float(np.percentile(d[chosen],99)))
            # Measure the pen's blue-to-violet variation rather than clipping
            # a shadowed lead with a fixed global hue interval.
            meta['relative_hue_tolerance']=max(25,min(35,int(np.ceil(np.percentile(samples,99)))+3)) if len(samples)>=4 else 25
            meta['relative_hue_tolerance_source']='equal_weight_component_and_external_lead_hue_envelopes_plus_camera_margin'
        meta['original_color_consensus']=bool(profile.get('active'))
    return meta


@cached_vision
def conductor_mask(image,policy):
    mask=stroke_mask(image)
    if policy.get('relative_ink_hue') is None:return mask
    view=paper_relative_view(image);hsv=cv2.cvtColor(view,cv2.COLOR_BGR2HSV)
    delta=np.abs(hsv[:,:,0].astype(float)-float(policy['relative_ink_hue']))
    colored=(np.minimum(delta,180-delta)<=policy.get('relative_hue_tolerance',25))&(hsv[:,:,1]>=20)
    dark_neutral=(hsv[:,:,1]<20)&(hsv[:,:,2]<75)
    # A pale pen stroke may be lighter than the first segmentation threshold.
    # Retain it only in the measured pen hue and connected to strong real ink;
    # white gaps and neutral printed-paper borders remain absent.
    weak=(colored&(view.mean(axis=2)<245))|dark_neutral
    count,labels=cv2.connectedComponents(weak.astype(np.uint8),8)
    strong=np.unique(labels[(mask>0)&weak]);keep=np.zeros(count,np.uint8);keep[strong]=255;keep[0]=0
    return keep[labels]


def wire_analysis_image(image,wires):
    if not (wires or {}).get('conductor_ink_policy',{}).get('active'):return image
    return cv2.cvtColor(255-conductor_mask(image,wires['conductor_ink_policy']),cv2.COLOR_GRAY2BGR)
