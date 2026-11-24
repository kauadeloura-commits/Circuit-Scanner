"""Remove repeated notebook rules by paper-normalized color, preserving ink."""
from __future__ import annotations
import math
import cv2
import numpy as np
from backend.resistor_geometry import cached_vision

SCHEMA = 'notebook-color-rules-2.0'
REFERENCE_COLORS = {'traditional_gray': ['#B3B3B3','#CCCCCC'],
                    'light_gray': ['#D9D9D9','#E6E6E6'],
                    'school_cyan': ['#B3E6E6','#99FFFF']}


@cached_vision
def _paper_colors(image):
    bgr=image if image.ndim==3 else cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    # Correct exposure and paper color locally; printed colors in a shadowed
    # photograph cannot be compared directly with digital RGB swatches.
    size=max(17,min(41,int(min(image.shape[:2])/18)))|1
    paper=cv2.morphologyEx(bgr,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size)))
    paper=cv2.GaussianBlur(paper.astype(np.float32),(0,0),sigmaX=3)
    normalized=np.clip(bgr.astype(np.float32)*255/np.maximum(paper,1),0,255)
    blue,green,red=cv2.split(normalized)
    gray=(np.ptp(normalized,axis=2)<=18)&(normalized.mean(axis=2)>=167)&(normalized.mean(axis=2)<=242)
    cyan=(red>=140)&(red<=200)&(green>=210)&(blue>=210)&(np.abs(blue-green)<=28)&(blue-red>=25)
    return bgr,(gray|cyan).astype(np.uint8)*255,normalized


def protected_component_ink_mask(image,profile):
    if not profile.get('active'):return np.zeros(image.shape[:2],bool)
    from backend.ink_color import component_ink_mask
    match=component_ink_mask(image,profile)
    if match is None:return np.zeros(image.shape[:2],bool)
    # A loose wire color tolerance may admit a slightly lighter page rule.
    # Protect the actual reference color, rather than that entire tolerance.
    reference=np.array(profile['reference_rgb'][::-1],float)
    tolerance=max(8,min(16,float(profile['channel_tolerance'])*.4))
    return (match>0)&(np.max(np.abs(image.astype(float)-reference),axis=2)<=tolerance)


@cached_vision
def apply_notebook_rule_profile(image,profile,ink_profile=None):
    """Reuse exactly the detected page lines for terminals and junctions."""
    if not profile or not profile.get('active') or profile.get('image_size')!=list(image.shape[:2]):
        return image,np.zeros(image.shape[:2],np.uint8)
    bgr,palette,normalized=_paper_colors(image)
    tracked=profile.get('schema_version')==SCHEMA
    foreground=(normalized.mean(axis=2)<242).astype(np.uint8)*255
    thickness=cv2.distanceTransform(foreground,cv2.DIST_L2,3)
    expanded_thickness=cv2.dilate(thickness,np.ones((3,3),np.uint8))
    exact_ink=np.zeros(image.shape[:2],bool)
    if tracked and (ink_profile or {}).get('active'):
        ink_reference=np.array(ink_profile['reference_rgb'][::-1],float)
        # Repeated actual buses can resemble notebook geometry. A rule that
        # has the same measured ink code cannot safely be erased by color.
        exact_ink=np.max(np.abs(bgr.astype(float)-ink_reference),axis=2)<=6
    band=np.zeros(image.shape[:2],np.uint8)
    erase=np.zeros(image.shape[:2],bool)
    for family in profile.get('families',[]):
        colors=np.array([l['normalized_rgb'][::-1] for l in family['lines']])
        middle=np.median(colors,axis=0) if len(colors) else np.zeros(3)
        deviations=np.max(np.abs(colors-middle),axis=1) if len(colors) else np.zeros(0)
        med=float(np.median(deviations)) if len(colors) else 0
        mad=float(np.median(np.abs(deviations-med))) if len(colors) else 0
        for line in family['lines']:
            band.fill(0)
            if line.get('polyline'):
                cv2.polylines(band,[np.array(line['polyline'],np.int32)],False,255,7)
            else:cv2.line(band,tuple(line['start']),tuple(line['end']),255,7)
            yy,xx=np.nonzero((band>0)&((foreground>0) if tracked else (palette>0)))
            reference=np.array(line['normalized_rgb'][::-1])
            # A conductor lying along one rule can contaminate that rule's
            # dark color sample. It cannot redefine the color of the paper's
            # repeated family. Use its robust family color in that case.
            if len(colors)>=5 and np.max(np.abs(reference-middle))>max(25,med+4*mad):
                reference=middle
            close=np.max(np.abs(normalized[yy,xx]-reference),axis=1)<=line['color_tolerance']
            if tracked:
                half=float(line.get('half_width_px',1.8))
                # Broad, blurred printed rules vary more in width along a
                # photographed page. Thin crisp rules keep the stricter limit
                # needed to preserve a light pencil of similar gray.
                limit=max(2.6,half*(1.45 if half>=2.5 else 1.32)+.15)
                close&=expanded_thickness[yy,xx]<=limit
                close&=~exact_ink[yy,xx]
            erase[yy[close],xx[close]]=True
    if not tracked:erase&=~protected_component_ink_mask(bgr,ink_profile or {})
    mask=erase.astype(np.uint8)*255
    clean=image.copy();clean[erase]=255
    return clean,mask


@cached_vision
def detect_notebook_rule_profile(image,ink_profile=None):
    """Require long, parallel, regularly spaced strokes and measured local color.

    Color alone is insufficient: pencil and pale pens may use the same RGB.
    Isolated wires, component strokes, and dots are never classified by a
    swatch lookup alone. Rotations and both families of grid paper are allowed.
    """
    h,w=image.shape[:2]
    report=dict(schema_version=SCHEMA,active=False,image_size=[h,w],reference_colors=REFERENCE_COLORS,
                reason='no_repeated_color_rule_family',families=[],removed_pixel_count=0)
    from backend.drawing_mode import digital
    if digital():return dict(report,reason='explicit_digital_input')
    if min(h,w)<40:return report
    _,palette,normalized=_paper_colors(image)
    # Exposure-corrected foreground includes dark printed rules under shadows.
    candidates_mask=(normalized.mean(axis=2)<240).astype(np.uint8)*255
    distances_to_paper=cv2.distanceTransform((normalized.mean(axis=2)<242).astype(np.uint8)*255,cv2.DIST_L2,3)
    lines=cv2.HoughLinesP(candidates_mask,1,np.pi/360,threshold=max(25,int(min(h,w)*.06)),
                          minLineLength=max(35,int(min(h,w)*.48)),maxLineGap=max(10,int(min(h,w)*.04)))
    if lines is None:return report
    candidates=[]
    for x1,y1,x2,y2 in np.asarray(lines).reshape(-1,4):
        delta=np.array([x2-x1,y2-y1],float);length=np.linalg.norm(delta)
        if abs(delta[0])<w*.55 and abs(delta[1])<h*.55:continue
        angle=math.atan2(delta[1],delta[0])%math.pi
        candidates.append(dict(angle=angle,length=float(length),point=np.array([x1,y1],float)))
    groups=[]
    for line in sorted(candidates,key=lambda item:item['length'],reverse=True):
        group=next((g for g in groups if abs(math.sin(line['angle']-g[0]['angle']))<math.sin(math.radians(4))),None)
        if group is None:groups.append([line])
        else:group.append(line)
    for group in groups:
        angle=.5*math.atan2(sum(math.sin(2*l['angle'])*l['length'] for l in group),
                             sum(math.cos(2*l['angle'])*l['length'] for l in group))
        direction=np.array([math.cos(angle),math.sin(angle)]);normal=np.array([-direction[1],direction[0]])
        ordered=sorted([(float(l['point']@normal),l) for l in group],key=lambda item:item[0])
        clusters=[]
        for offset,line in ordered:
            if not clusters or offset-np.mean([v[0] for v in clusters[-1]])>max(5,min(h,w)*.009):clusters.append([])
            clusters[-1].append((offset,line))
        if len(clusters)<3:continue
        offsets=np.array([np.mean([v[0] for v in cluster]) for cluster in clusters])
        gaps=np.diff(offsets);spacing=float(np.median(gaps))
        if spacing<max(8,min(h,w)*.015):continue
        multiples=np.maximum(1,np.rint(gaps/spacing))
        regular=np.abs(gaps-spacing*multiples)<=spacing*.23
        if np.count_nonzero(regular)<max(2,len(gaps)*.75):continue
        family=dict(angle_degrees=round(math.degrees(angle),2),spacing_px=round(spacing,2),lines=[])
        extent=math.hypot(w,h)*2
        for offset,cluster in zip(offsets,clusters):
            representative=max(cluster,key=lambda item:item[1]['length'])[1]
            own=representative['angle']
            own_direction=np.array([math.cos(own),math.sin(own)])
            own_normal=np.array([-own_direction[1],own_direction[0]])
            center=own_normal*float(representative['point']@own_normal)
            start=np.rint(center-own_direction*extent).astype(int);end=np.rint(center+own_direction*extent).astype(int)
            ok,p,q=cv2.clipLine((0,0,w,h),tuple(start),tuple(end))
            if ok:
                band=np.zeros((h,w),np.uint8);cv2.line(band,p,q,255,3)
                samples=normalized[(band>0)&(candidates_mask>0)]
                if len(samples)<max(20,min(h,w)*.3):continue
                median=np.percentile(samples,10,axis=0)
                spread=float(np.median(np.max(np.abs(samples-median),axis=1)))
                # Follow mild perspective/lens curvature locally instead of
                # extrapolating every rule with one average page angle.
                path=[];widths=[];search=max(5,min(14,int(spacing*.12)))
                for t in np.linspace(0,1,max(12,int(np.linalg.norm(np.array(q)-p)/24))):
                    predicted=np.array(p)*(1-t)+np.array(q)*t
                    locations=np.rint(predicted+np.arange(-search,search+1)[:,None]*own_normal).astype(int)
                    valid=(locations[:,0]>=0)&(locations[:,0]<w)&(locations[:,1]>=0)&(locations[:,1]<h)
                    locations=locations[valid]
                    colors=normalized[locations[:,1],locations[:,0]]
                    distances=np.max(np.abs(colors-median),axis=1)
                    choices=distances+np.linalg.norm(locations-predicted,axis=1)*.9
                    index=int(np.argmin(choices))
                    path.append(locations[index].tolist() if distances[index]<=30 else np.rint(predicted).astype(int).tolist())
                    if distances[index]<=30:
                        nearby=(distances<=35)&(np.linalg.norm(locations-locations[index],axis=1)<=4)
                        widths.append(float(np.max(distances_to_paper[locations[nearby,1],locations[nearby,0]])))
                family['lines'].append(dict(start=list(p),end=list(q),
                    polyline=path,
                    half_width_px=round(float(np.percentile(widths,65)) if widths else 1.8,2),
                    normalized_rgb=[round(float(v),2) for v in median[::-1]],
                    color_tolerance=round(max(8,min(20,spread*2+6)),2)))
        if len(family['lines'])>=3:report['families'].append(family)
    if report['families']:
        report.update(active=True,reason='repeated_rules_matching_paper_normalized_colors')
        _,mask=apply_notebook_rule_profile(image,report,ink_profile)
        report['removed_pixel_count']=int(cv2.countNonZero(mask))
        if not report['removed_pixel_count']:report.update(active=False,reason='matching_color_protected_as_component_ink')
    return report
