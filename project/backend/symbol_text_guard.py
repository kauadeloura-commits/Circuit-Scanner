"""Separate circuit bodies from glyphs before confidence/consensus rescue.

Text strokes must not substitute for connected leads and a repeated coil.
The original page mask is read only; wire extraction remains independent.
"""
import cv2
import numpy as np


def isolated_text_run(image, bbox, page_mask=None):
    """Conservative OCR-free check for aligned, disconnected glyph groups.

    Connected leads protect capacitor/battery plates and adjacent labels.
    Single glyphs are handled by class topology and, when available, OCR.
    """
    from backend.resistor_geometry import symbol_mask
    from backend.vcc_topology import external_leads
    mask = symbol_mask(image) if page_mask is None else page_mask
    h,w = mask.shape
    x1,y1 = max(0,int(bbox['x1'])),max(0,int(bbox['y1']))
    x2,y2 = min(w,int(np.ceil(bbox['x2']))),min(h,int(np.ceil(bbox['y2'])))
    crop = mask[y1:y2,x1:x2]
    if not crop.size:
        return False, {}
    count,_,stats,_ = cv2.connectedComponentsWithStats((crop>0).astype(np.uint8),8)
    ink = np.count_nonzero(crop)
    for vertical in (False,True):
        major = crop.shape[0] if vertical else crop.shape[1]
        glyphs=[]
        for x,y,bw,bh,area in stats[1:]:
            length,height = (bh,bw) if vertical else (bw,bh)
            if area >= max(8,.025*ink) and .06*major <= length <= .48*major and .2 <= length/max(1,height) <= 3:
                glyphs.append((y+bh/2 if vertical else x+bw/2,
                               x+bw/2 if vertical else y+bh/2,height,area))
        if len(glyphs)<3 or sum(g[3] for g in glyphs)<.65*ink:
            continue
        glyphs.sort()
        heights=np.array([g[2] for g in glyphs]); baseline=np.array([g[1] for g in glyphs])
        aligned=np.ptp(baseline)<=.35*np.median(heights) and np.max(heights)<=2.5*np.min(heights)
        if aligned and external_leads(image,bbox,mask)==0:
            return True, {'isolated_aligned_glyphs':len(glyphs),'metric':'aligned_glyph_group_without_connected_leads',
                          'reason':'isolated_letter_number_group','hard_text_rejection':True}
    return False, {}


def coil_path_evidence(image, bbox, page_mask=None):
    from backend.resistor_geometry import symbol_mask
    from backend.vcc_topology import external_leads
    mask = symbol_mask(image) if page_mask is None else page_mask
    bw, bh = bbox['x2']-bbox['x1'], bbox['y2']-bbox['y1']
    pad_x, pad_y = (.35*bw, .35*bh)
    h, w = mask.shape
    x1, y1 = max(0, int(bbox['x1']-pad_x)), max(0, int(bbox['y1']-pad_y))
    x2, y2 = min(w, int(np.ceil(bbox['x2']+pad_x))), min(h, int(np.ceil(bbox['y2']+pad_y)))
    crop = (mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    if crop.size == 0:
        return False, {'reason': 'empty_crop', 'hard_text_rejection': True}
    count, labels, stats, _ = cv2.connectedComponentsWithStats(crop, 8)
    best = {'connected_curve_with_two_leads': False, 'curve_extrema': 0}
    for label in range(1, count):
        sx, sy, sw, sh, area = stats[label]
        if max(sw,sh) < .55*max(bw, bh) or min(sw,sh) < 5 or area < 25:
            continue
        yy, xx = np.where(labels == label)
        points = np.column_stack((xx,yy)).astype(float)
        inside=(points[:,0]+x1>=bbox['x1'])&(points[:,0]+x1<=bbox['x2'])&(points[:,1]+y1>=bbox['y1'])&(points[:,1]+y1<=bbox['y2'])
        if np.count_nonzero(inside)<max(20,.4*area):
            continue
        local_box={k:bbox[k]-(x1 if k.startswith('x') else y1) for k in ('x1','y1','x2','y2')}
        leads=external_leads(image[y1:y2,x1:x2],local_box,(labels==label).astype(np.uint8)*255)
        if leads>2:
            best.update(external_lead_count=leads,reason='junction_with_more_than_two_external_leads')
            continue
        centered = points-points.mean(axis=0)
        _,vectors=np.linalg.eigh(centered.T@centered)
        axis=vectors[:,-1]; normal=np.array([-axis[1],axis[0]])
        xx=np.rint(centered@axis).astype(int); xx-=xx.min()
        yy=centered@normal
        sx,sw,sh=0,int(xx.max())+1,float(np.ptp(yy))
        if sw < .55*max(bw,bh) or sh < 5:
            continue
        columns = np.arange(sx, sx+sw)
        sizes = np.bincount(xx, minlength=sw)[columns]
        sums = np.bincount(xx, weights=yy, minlength=sw)[columns]
        top=np.full(sw,np.inf); bottom=np.full(sw,-np.inf)
        np.minimum.at(top,xx,yy); np.maximum.at(bottom,xx,yy)
        widths=bottom-top
        if np.count_nonzero(sizes) < .95*sw:
            continue
        profile = sums / np.maximum(1, sizes)
        profile = np.convolve(np.pad(profile, (2, 2), mode='edge'), np.ones(5)/5, mode='valid')
        tip = max(8, int(.08*sw))
        # Both tips must be straight, on the same axis, on this very path.
        tolerance = max(1.6, .08*sh)
        def tip_runs(start, stop):
            return [i for i in range(start,stop)
                    if np.ptp(profile[i:i+tip]) <= tolerance
                    and np.median(widths[i:i+tip]) >= max(1.,.35*np.median(widths[np.isfinite(widths)]))
                    and np.percentile(widths[i:i+tip],85) <= max(5,.3*sh)]
        first_runs = tip_runs(0,max(1,int(.25*sw)-tip))
        last_runs = tip_runs(max(0,int(.75*sw)),sw-tip+1)
        pairs=[(a,b) for a in first_runs for b in last_runs
               if abs(np.mean(profile[a:a+tip])-np.mean(profile[b:b+tip])) <= max(4,.30*sh)]
        straight = bool(pairs)
        start,stop = max(pairs,key=lambda pair:pair[1]-pair[0]) if pairs else (0,sw-tip)
        # Only turns inside the detection can establish the body. Nearby
        # resistors or bends beyond its box must not certify label curves.
        body_columns=xx[inside]
        core_start=max(start+tip,int(body_columns.min()))
        core_stop=min(stop,int(body_columns.max())+1)
        core=profile[core_start:core_stop]
        if len(core)<8:
            continue
        prominence = max(2.2, .16*(np.percentile(core,85)-np.percentile(core,15)))
        extremes = []; previous = core[0]; direction = 0
        for value in core:
            difference = value-previous
            if direction == 0:
                if abs(difference) >= prominence:
                    direction = 1 if difference > 0 else -1; previous = value
            elif direction*difference >= 0:
                previous = value
            elif abs(difference) >= prominence:
                extremes.append(float(previous)); direction *= -1; previous = value
        if direction:
            extremes.append(float(previous))
        amplitude = float(np.ptp(core))
        ok = bool(straight and len(extremes) >= 4 and amplitude >= max(4, .15*sh))
        evidence = {'connected_curve_with_two_leads': ok, 'straight_tip_pair': straight,
                    'curve_extrema': len(extremes), 'curve_amplitude_px': round(amplitude, 2),
                    'path_span_px': int(sw), 'metric': 'same_ink_path_two_straight_tips_repeated_curves'}
        if ok:
            return True, evidence
        if len(extremes) > best['curve_extrema']:
            best = evidence
    return False, {**best, 'reason': 'glyph_curves_without_connected_coil_leads', 'hard_text_rejection': True}


def source_ring_evidence(image, bbox, page_mask=None):
    from backend.source_polarity_ring import polarity_ring_evidence
    signed=polarity_ring_evidence(image,bbox,page_mask)
    if signed:return True,signed
    from backend.resistor_geometry import symbol_mask
    from backend.vcc_topology import external_leads
    mask = symbol_mask(image) if page_mask is None else page_mask
    h, w = mask.shape
    size = max(bbox['x2']-bbox['x1'], bbox['y2']-bbox['y1'])
    pad = .4*size
    x1, y1 = max(0, int(bbox['x1']-pad)), max(0, int(bbox['y1']-pad))
    x2, y2 = min(w, int(bbox['x2']+pad)), min(h, int(bbox['y2']+pad))
    # A thin ruling may split a ring's hole, and its removal may leave tiny
    # gaps in the pen. Repair only the local shape used for validation.
    local = cv2.morphologyEx(mask[y1:y2,x1:x2], cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    local = cv2.morphologyEx(local, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
    contours, hierarchy = cv2.findContours(local, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    best = 0
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            if hierarchy[0,i,3] < 0:
                continue
            x,y,bw,bh = cv2.boundingRect(contour)
            area, perimeter = cv2.contourArea(contour), cv2.arcLength(contour, True)
            if min(bw,bh) < max(8, .2*min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])):
                continue
            if max(bw,bh) > 1.9*min(bw,bh) or not perimeter or 4*np.pi*area/perimeter**2 < .48:
                continue
            cx,cy = x+x1+bw/2, y+y1+bh/2
            if not (bbox['x1'] <= cx <= bbox['x2']+ .15*size and bbox['y1'] <= cy <= bbox['y2']+.15*size):
                continue
            ring = dict(x1=float(x+x1-3),y1=float(y+y1-3),x2=float(x+x1+bw+3),y2=float(y+y1+bh+3))
            leads = external_leads(image,ring,mask)
            best = max(best, leads)
            if leads >= 2:
                return True, {'ring_with_two_connected_leads': True, 'ring_external_leads': leads}
    return False, {'ring_with_two_connected_leads': False, 'ring_external_leads': best,
                   'reason': 'round_glyph_without_two_source_leads', 'hard_text_rejection': True}


def dc_source_topology(image, bbox, page_mask=None):
    """Distinguish a hollow two-lead source from a circular BJT enclosure.

    Arc segments cannot establish a transistor fork. Require a large, round,
    unpartitioned cavity and two opposed leads on the ring's ink component.
    A nearby isolated + or label inside the rectangular crop is not a lead.
    """
    from backend.source_polarity_ring import polarity_ring_evidence
    signed=polarity_ring_evidence(image,bbox,page_mask)
    if signed:return True,signed
    from backend.resistor_geometry import symbol_mask
    mask = symbol_mask(image) if page_mask is None else page_mask
    h,w=mask.shape;bw,bh=bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1']
    size=max(bw,bh);pad=.35*size
    x1,y1=max(0,int(bbox['x1']-pad)),max(0,int(bbox['y1']-pad))
    x2,y2=min(w,int(np.ceil(bbox['x2']+pad))),min(h,int(np.ceil(bbox['y2']+pad)))
    crop=mask[y1:y2,x1:x2]
    if not crop.size:return False,{}
    # Small pen gaps may close locally; the source image is never changed.
    kernel=max(3,min(5,int(min(bw,bh)*.06)))
    local=cv2.morphologyEx(crop,cv2.MORPH_CLOSE,np.ones((kernel,kernel),np.uint8))
    contours,hierarchy=cv2.findContours(local,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:return False,{}
    _,labels,_,_=cv2.connectedComponentsWithStats((local>0).astype(np.uint8),8)
    for i,contour in enumerate(contours):
        if hierarchy[0,i,3]<0:continue
        x,y,cw,ch=cv2.boundingRect(contour);area=cv2.contourArea(contour);perimeter=cv2.arcLength(contour,True)
        cx,cy=x+x1+cw/2,y+y1+ch/2
        if not perimeter or min(cw,ch)<max(10,.48*min(bw,bh)):continue
        circularity=4*np.pi*area/perimeter**2
        if circularity<.70 or max(cw,ch)>1.65*min(cw,ch) or area<.28*bw*bh:continue
        if abs(cx-(bbox['x1']+bbox['x2'])/2)>.2*bw or abs(cy-(bbox['y1']+bbox['y2'])/2)>.2*bh:continue
        # Count only branches belonging to the ring, excluding detached text.
        band=np.zeros_like(local);cv2.drawContours(band,[contour],-1,255,3)
        ids,counts=np.unique(labels[(band>0)&(local>0)],return_counts=True)
        if not len(ids):continue
        ring_id=int(ids[np.argmax(counts)])
        stroke=max(3,int(round(.08*min(cw,ch))))
        ring=(x-stroke,y-stroke,x+cw+stroke,y+ch+stroke)
        yy,xx=np.indices(local.shape)
        body=(xx>=ring[0]-2)&(xx<=ring[2]+2)&(yy>=ring[1]-2)&(yy<=ring[3]+2)
        outside=((labels==ring_id)&~body).astype(np.uint8)
        n,branches,stats,_=cv2.connectedComponentsWithStats(outside,8)
        distance=np.maximum.reduce([ring[0]-2-xx,xx-ring[2]-2,ring[1]-2-yy,yy-ring[3]-2])
        directions=[]
        for j in range(1,n):
            pixels=branches==j;near=pixels&(distance<=3)
            if stats[j,cv2.CC_STAT_AREA]<6 or not np.any(near) or not np.any(pixels&(distance>=8)):continue
            points=np.column_stack(np.where(near)[::-1]);v=points.mean(axis=0)-[x+cw/2,y+ch/2]
            norm=np.linalg.norm(v)
            if norm:directions.append(v/norm)
        if len(directions)!=2 or np.dot(*directions)>-.65:continue
        return True,dict(unpartitioned_source_ring=True,opposed_connected_leads=2,
                         cavity_circularity=round(float(circularity),3),cavity_area_ratio=round(area/(bw*bh),3),
                         metric='large_hollow_ring_two_opposed_leads_on_same_ink_path')
    return False,{}
