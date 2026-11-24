"""Capacitor plates need separate ink paths and opposed, connected leads."""
import math
import cv2
import numpy as np
from backend.resistor_geometry import symbol_mask,_thin_symbol,cached_vision


@cached_vision
def capacitor_body_evidence(image,bbox,*,maximum_gap_ratio=1.25,_digital_scaled=False):
    from backend.drawing_mode import digital
    if digital() and not _digital_scaled and min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])<75:
        pad=max(14,int(max(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])*.5))
        x,y=max(0,int(bbox['x1'])-pad),max(0,int(bbox['y1'])-pad)
        endx,endy=min(image.shape[1],int(math.ceil(bbox['x2']))+pad),min(image.shape[0],int(math.ceil(bbox['y2']))+pad)
        roi=image[y:endy,x:endx]
        if roi.size:
            scale=3
            box={k:(bbox[k]-(x if k.startswith('x') else y))*scale for k in ('x1','y1','x2','y2')}
            ok,e=capacitor_body_evidence(cv2.resize(roi,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC),box,maximum_gap_ratio=maximum_gap_ratio,_digital_scaled=True)
            for key in ('outer_plate_lengths',):
                if key in e:e[key]=[round(v/scale,3) for v in e[key]]
            if 'outer_plate_centers' in e:e['outer_plate_centers']=[[round(p[0]/scale+x,3),round(p[1]/scale+y,3)] for p in e['outer_plate_centers']]
            if 'plate_gap_px' in e:e['plate_gap_px']=round(e['plate_gap_px']/scale,3)
            e['local_print_scale']=scale
            if ok:return ok,e
    h,w=image.shape[:2];bw,bh=bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'];pad=max(12,int(max(bw,bh)*.4))
    x1,y1=max(0,int(bbox['x1'])-pad),max(0,int(bbox['y1'])-pad)
    x2,y2=min(w,int(math.ceil(bbox['x2']))+pad),min(h,int(math.ceil(bbox['y2']))+pad)
    roi=image[y1:y2,x1:x2];mask=symbol_mask(roi)
    if not mask.size:return False,{'reason':'empty_crop','hard_text_rejection':True}
    if digital():
        from backend.printed_capacitor import plate_ink
        mask=plate_ink(roi)
    # A dark notebook rule must not connect two blue pen plates.
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    colored=(hsv[:,:,1]>35)&(mask>0)
    # Tinted paper and blue notebook rulings do not define the component pen.
    # A color-only mask is safe only when it retains most of the symbol ink.
    if not digital() and np.count_nonzero(colored)>=20 and np.count_nonzero(colored)>=.75*np.count_nonzero(mask):
        mask=cv2.morphologyEx(np.where(colored,255,0).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    _,full_labels,_,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8)
    gy,gx=np.indices(mask.shape);in_box=(gx+x1>=bbox['x1']-2)&(gx+x1<=bbox['x2']+2)&(gy+y1>=bbox['y1']-2)&(gy+y1<=bbox['y2']+2)
    count,labels,stats,_=cv2.connectedComponentsWithStats(((mask>0)&in_box).astype(np.uint8),8);parts=[]
    for i in range(1,count):
        if stats[i,cv2.CC_STAT_AREA]<8:continue
        interior=labels==i
        contours,hierarchy=cv2.findContours(interior.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # Rounded digits have closed cells; capacitor plates are open strokes.
        if hierarchy is not None and any(hierarchy[0,j,3]>=0 and cv2.contourArea(c)>=max(4,.015*stats[i,cv2.CC_STAT_AREA]) for j,c in enumerate(contours)):
            # A printed positive plate may be an outlined narrow rectangle.
            # It still needs a separate opposing plate and connected leads below.
            holes=[c for j,c in enumerate(contours) if hierarchy[0,j,3]>=0 and cv2.contourArea(c)>=max(4,.015*stats[i,cv2.CC_STAT_AREA])]
            def rectangular_plate_hole(contour):
                _,_,a,b=cv2.boundingRect(contour)
                return min(a,b)<=.4*max(a,b) and cv2.contourArea(contour)>=.65*max(1,(a-1)*(b-1))
            if not (digital() and holes and all(rectangular_plate_hole(c) for c in holes)):continue
        paths=np.unique(full_labels[interior]);paths=paths[paths>0]
        extended=interior|(~in_box&np.isin(full_labels,paths))
        skeleton=(_thin_symbol(extended.astype(np.uint8)*255)>0).astype(np.uint8)
        degree=cv2.filter2D(skeleton,cv2.CV_16S,np.ones((3,3),np.int16))-skeleton
        ends=np.column_stack(np.where((skeleton>0)&(degree==1)));clusters=[]
        for end in ends:
            if all(np.linalg.norm(end-s)>max(3,.07*min(bw,bh)) for s in clusters):clusters.append(end)
        yy,xx=np.where(extended);points=np.column_stack((xx+x1,yy+y1)).astype(float)
        parts.append((points,len(clusters)>=3))
    evidence={'separate_plate_ink_paths':len(parts),'capacitor_body_with_opposed_leads':False}
    if len(parts)<2:
        from backend.handwritten_plate_context import connected_plate_evidence
        contextual=connected_plate_evidence(image,bbox)
        if contextual:return True,{**evidence,**contextual}
        return False,{**evidence,'reason':'connected_glyph_strokes_instead_of_separate_capacitor_plates','hard_text_rejection':True}
    angles=[0.,90.,45.,135.]
    lines=cv2.HoughLinesP(mask,1,np.pi/180,threshold=10,minLineLength=max(8,int(min(bw,bh)*.2)),maxLineGap=3)
    if lines is not None:
        candidates=sorted(np.asarray(lines).reshape(-1,4),key=lambda l:float(l[2]-l[0])**2+float(l[3]-l[1])**2,reverse=True)
        for a,b,c,d in candidates:
            angle=round(math.degrees(math.atan2(d-b,c-a))/3)*3%180
            if all(abs((angle-s+90)%180-90)>2 for s in angles):angles.append(angle)
            if len(angles)>=12:break
    for angle in angles:
        rad=math.radians(angle);axis=np.array([math.cos(rad),math.sin(rad)]);normal=np.array([-axis[1],axis[0]])
        plates=[]
        for points,branched in parts:
            a=points@axis;n=points@normal
            inside=(points[:,0]>=bbox['x1']-2)&(points[:,0]<=bbox['x2']+2)&(points[:,1]>=bbox['y1']-2)&(points[:,1]<=bbox['y2']+2)
            bins=np.rint(n[inside]).astype(int);offset=bins.min();counts=np.bincount(bins-offset)
            peak=int(np.argmax(counts))+offset;on_plate=(np.abs(n-peak)<=1.7)&inside
            low,high=np.percentile(a[on_plate],[2,98]);length=high-low
            if length<max(7,.25*min(bw,bh)) or length<.4*np.ptp(a[inside]):continue
            middle=(low+high)/2;lead=np.abs(a-middle)<=max(3,.12*length)
            if not np.any(lead):continue
            center=axis*middle+normal*peak
            if not bbox['x1']-2<=center[0]<=bbox['x2']+2 or not bbox['y1']-2<=center[1]<=bbox['y2']+2:continue
            plates.append((peak,middle,length,float(n[lead].min()),float(n[lead].max()),center,branched))
        for i,first in enumerate(plates):
            for second in plates[i+1:]:
                left,right=sorted((first,second),key=lambda p:p[0]);length=max(left[2],right[2]);gap=right[0]-left[0]
                if not max(3,.05*length)<=gap<=maximum_gap_ratio*length or abs(left[1]-right[1])>.42*length:continue
                # At least one plate must show the three-ended plate/lead
                # junction. Curved digits 2/3 only have two stroke ends.
                if not (left[6] or right[6]):continue
                def edge(center,direction,limit):
                    distances=[]
                    for coord,delta,maximum in zip(center,direction,(w-1,h-1)):
                        if abs(delta)>.05:distances.append(((maximum if delta>0 else 0)-coord)/delta)
                    return bool(distances and min(distances)<=limit)
                left_lead=left[0]-left[3]>=max(6,.3*left[2]) or edge(left[5],-normal,max(3,.04*left[2]))
                right_lead=right[4]-right[0]>=max(6,.3*right[2]) or edge(right[5],normal,max(3,.04*right[2]))
                if left_lead and right_lead:
                    return True,{**evidence,'capacitor_body_with_opposed_leads':True,'plate_gap_px':round(float(gap),2),'plate_axis_degrees':float(angle),
                        'outer_plate_centers':[[round(float(v),3) for v in p[5]] for p in (left,right)],
                        'outer_plate_lengths':[round(float(p[2]),3) for p in (left,right)],
                        'electrical_axis':[float(v) for v in normal],
                        'metric':'two_separate_plate_paths_with_outward_midpoint_leads'}
    from backend.handwritten_plate_context import connected_plate_evidence
    contextual=connected_plate_evidence(image,bbox)
    if contextual:return True,{**evidence,**contextual}
    return False,{**evidence,'reason':'plate_strokes_without_opposed_connected_capacitor_leads','hard_text_rejection':True}
