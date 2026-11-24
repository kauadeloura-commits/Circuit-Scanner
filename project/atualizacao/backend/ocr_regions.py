"""Find photographed text lines once, retaining original image coordinates."""
import cv2
import numpy as np
from backend.ink_color import estimate_component_ink,component_ink_mask


def text_ink_image(image,detections=()):
    bgr=image if image.ndim==3 else cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    profile=estimate_component_ink(bgr,{'detections':list(detections)}) if detections else {}
    mask=component_ink_mask(bgr,profile) if profile.get('active') and profile.get('chromatic') else None
    if mask is None:
        hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        # Saturated handwriting is separated from neutral paper/rules even
        # before component classification. Keep every substantial ink hue.
        colored=(hsv[:,:,1]>35)&(gray<200)
        paper=cv2.GaussianBlur(gray,(0,0),18)
        foreground=(paper.astype(float)-gray)>max(15,float(np.percentile(paper,50))*.12)
        # Neutral notebook rules may outnumber the colored writing. Exclude
        # long axis lines only while estimating the pen color coverage.
        policy_ink=foreground.astype(np.uint8)*255
        long_lines=cv2.morphologyEx(policy_ink,cv2.MORPH_OPEN,np.ones((1,max(40,gray.shape[1]//2)),np.uint8))
        long_lines|=cv2.morphologyEx(policy_ink,cv2.MORPH_OPEN,np.ones((max(40,gray.shape[0]//2),1),np.uint8))
        policy_ink=(policy_ink>0)&(cv2.dilate(long_lines,np.ones((3,3),np.uint8))==0)
        color_coverage=np.count_nonzero(colored&policy_ink)/max(1,np.count_nonzero(policy_ink))
        if np.count_nonzero(colored)>.002*gray.size and np.count_nonzero(colored)<.20*gray.size and color_coverage>=.55:
            mask=colored.astype(np.uint8)*255
        else:
            mask=foreground.astype(np.uint8)*255
    # Annotation ink may differ from the component pen. Retain strong neutral
    # strokes; do not let blue components erase black text.
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV);gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    neutral=((hsv[:,:,1]<30)&(gray<min(120,float(np.percentile(gray,75))*.65))).astype(np.uint8)*255
    count,labels,stats,_=cv2.connectedComponentsWithStats(neutral,8)
    kept=np.zeros(count,np.uint8)
    colored_neighborhood=cv2.dilate(mask,np.ones((5,5),np.uint8))
    for index,(x,y,w,h,area) in enumerate(stats[1:],1):
        # Long neutral notebook rules cannot be mistaken for writing regions.
        if area<7 or area>gray.size*.20 or max(w,h)>12*max(1,min(w,h)):continue
        if max(w,h)>max(60,min(gray.shape)*.06):
            yy,xx=np.nonzero(labels[y:y+h,x:x+w]==index)
            _,(major,minor),_=cv2.minAreaRect(np.column_stack((xx,yy)).astype(np.float32))
            if max(major,minor)>6*max(1,min(major,minor)):continue
        # Neutral edge pixels of colored strokes and crossing rules must not
        # join independent blue words to the notebook grid.
        if np.any(colored_neighborhood[y:y+h,x:x+w][labels[y:y+h,x:x+w]==index]):continue
        kept[index]=255
    mask=cv2.bitwise_or(mask,kept[labels])
    for detection in detections:
        evidence=detection.get('led_emission_validation') or {}
        if not evidence.get('valid'):continue
        for arrow in evidence.get('arrows',[]):
            if not arrow.get('bounds'):continue
            x1,y1,x2,y2=map(int,arrow['bounds'])
            mask[max(0,y1):min(mask.shape[0],y2),max(0,x1):min(mask.shape[1],x2)]=0
    return 255-mask


def text_line_regions(page,detections=()):
    mask=(page<128).astype(np.uint8)*255
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    minimum=max(5,min(page.shape)*.005)
    sizes=[h for x,y,w,h,area in stats[1:] if area>=10 and minimum<=h<=min(page.shape)*.10 and w<6*h]
    typical=float(np.median(sizes)) if sizes else 20.
    glyphs=[]
    for label in range(1,n):
        x,y,w,h,area=map(int,stats[label])
        if area<max(7,typical*.18) or max(w,h)<minimum:continue
        # Connected wires/physical bodies are much taller than handwriting.
        # Do not erase a complete detection box: beta/model text can be inside.
        if h>max(30,typical*2.3) or w>max(100,typical*9):continue
        point=centers[label]
        if any(d.get('class') in {'gnd','vcc'} and d['bbox']['x1']<=point[0]<=d['bbox']['x2'] and d['bbox']['y1']<=point[1]<=d['bbox']['y2'] for d in detections):continue
        glyphs.append(dict(x1=x,y1=y,x2=x+w,y2=y+h,area=area,label=label))
    groups=[]
    for glyph in sorted(glyphs,key=lambda g:(-(g['y2']-g['y1']),g['x1'])):
        chosen=[];height=glyph['y2']-glyph['y1']
        for i,group in enumerate(groups):
            for other in group:
                oh=other['y2']-other['y1']
                overlap=min(glyph['y2'],other['y2'])-max(glyph['y1'],other['y1'])
                gap=max(glyph['x1'],other['x1'])-min(glyph['x2'],other['x2'])
                if overlap>=.35*min(height,oh) and -.55*max(height,oh)<=gap<=.65*max(height,oh):
                    chosen.append(i);break
        if not chosen:groups.append([glyph])
        else:
            groups[chosen[0]].append(glyph)
            for i in reversed(chosen[1:]):groups[chosen[0]].extend(groups.pop(i))
    regions=[]
    for group in groups:
        x1=min(g['x1'] for g in group);x2=max(g['x2'] for g in group)
        y1=min(g['y1'] for g in group);y2=max(g['y2'] for g in group)
        if x2-x1<.6*(y2-y1) or y2-y1<minimum:continue
        regions.append(dict(x1=max(0,x1-5),y1=max(0,y1-5),x2=min(page.shape[1],x2+5),y2=min(page.shape[0],y2+5)))
    return sorted(regions,key=lambda b:(b['y1'],b['x1']))


def line_crop(page,box):
    x1,y1,x2,y2=(int(box[k]) for k in ('x1','y1','x2','y2'))
    crop=page[y1:y2,x1:x2]
    yy,xx=np.nonzero(crop<128)
    angle=0.
    if len(xx)>=15 and crop.shape[1]>1.8*crop.shape[0]:
        rect=cv2.minAreaRect(np.column_stack((xx,yy)).astype(np.float32))
        (_, _),(rw,rh),theta=rect
        theta=theta if rw>=rh else theta+90
        angle=(theta+90)%180-90
        if abs(angle)>15:angle=0.
    # A rotation never clips source glyphs; pad before turning the crop.
    pad=max(8,int(crop.shape[0]*.3));crop=cv2.copyMakeBorder(crop,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=255)
    if angle:
        crop=cv2.warpAffine(crop,cv2.getRotationMatrix2D((crop.shape[1]/2,crop.shape[0]/2),angle,1),
                            (crop.shape[1],crop.shape[0]),borderValue=255)
    scale=max(.5,min(3.,80/max(1,y2-y1)))
    crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    return crop,float(angle)
