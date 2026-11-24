"""Find text beside actual resistor bodies, excluding their outline and stems."""
import cv2,numpy as np

def views(image,detections):
    from backend.resistor_contacts import body_for_box
    cleaned=image.copy();bodies={}
    for d in detections:
        if d.get('class')!='resistor':continue
        b=d['bbox'];body=body_for_box(image,b)
        if body:
            polygon=np.asarray(body['evidence']['body_corners']);lo=polygon.min(axis=0);hi=polygon.max(axis=0)
            mask=np.zeros(image.shape[:2],np.uint8);cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],255)
            mask=cv2.dilate(mask,np.ones((3,3),np.uint8));cleaned[mask>0]=255
            bound=dict(x1=lo[0]-2,y1=lo[1]-2,x2=hi[0]+2,y2=hi[1]+2)
        else:
            if float(d.get('confidence',0))<.60:continue
            w,h=b['x2']-b['x1'],b['y2']-b['y1'];bound={k:float(b[k]) for k in ('x1','y1','x2','y2')}
            if max(w,h)<1.35*min(w,h):continue
            axis='x' if h>w else 'y';mid=(b[axis+'1']+b[axis+'2'])/2;short=min(w,h)
            bound[axis+'1']=mid-.40*short;bound[axis+'2']=mid+.40*short
            cv2.rectangle(cleaned,(int(bound['x1']),int(bound['y1'])),(int(np.ceil(bound['x2'])),int(np.ceil(bound['y2']))),(255,255,255),-1)
        bodies[d['detection_id']]=bound
    return cleaned,bodies

def regions(image,detections):
    from backend.ocr_regions import text_ink_image,text_line_regions
    from backend.reference_labels import edge_distance
    cleaned,bodies=views(image,detections);page=text_ink_image(image,detections)
    page[np.all(cleaned==255,axis=2)&np.any(image!=255,axis=2)]=255
    from backend.ink_color import estimate_component_ink
    profile=estimate_component_ink(image,{'detections':detections})
    if profile.get('active') and profile.get('chromatic'):
        ref=np.asarray(profile['reference_rgb'][::-1],np.uint8).reshape(1,1,3)
        limit=min(210,float(cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)[0,0])+60)
        page[cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)>limit]=255
    output=[]
    for d in detections:
        b=bodies.get(d.get('detection_id'))
        if not b:continue
        size=max(b['x2']-b['x1'],b['y2']-b['y1']);margin=max(48,min(100,size*1.8))
        lanes=[(b['x1']-margin,b['y1']-40,b['x1']-1,b['y2']+40),(b['x2']+1,b['y1']-40,b['x2']+margin,b['y2']+40),
               (b['x1']-margin,b['y1']-48,b['x2']+margin,b['y1']-1),(b['x1']-margin,b['y2']+1,b['x2']+margin,b['y2']+48)]
        for lane in lanes:
            x,y,z,t=[int(v) for v in lane];x,y=max(0,x),max(0,y);z,t=min(page.shape[1],z),min(page.shape[0],t)
            crop=page[y:t,x:z]
            if min(crop.shape[:2],default=0)<6:continue
            mask=(crop<128).astype(np.uint8)*255
            horizontal=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((1,max(35,round(mask.shape[1]*.75))),np.uint8))
            gray=cv2.cvtColor(cleaned[y:t,x:z],cv2.COLOR_BGR2GRAY)
            ink_values=gray[mask>0]
            limit=min(180,float(np.percentile(ink_values,20))+35) if len(ink_values) else 0
            mask[(horizontal>0)&(gray>limit)]=0
            for r in text_line_regions(255-mask):
                box={k:r[k]+(x if k.startswith('x') else y) for k in ('x1','y1','x2','y2')};h=box['y2']-box['y1'];w=box['x2']-box['x1']
                if h<12 or w<.45*h or w>5.5*h or edge_distance(box,b)>max(38,size):continue
                # This owner must be at least as close as another resistor.
                rivals=[edge_distance(box,q) for cid,q in bodies.items() if cid!=d['detection_id']]
                if rivals and min(rivals)+8<edge_distance(box,b):continue
                if any(max(0,min(box['x2'],a['x2'])-max(box['x1'],a['x1']))*max(0,min(box['y2'],a['y2'])-max(box['y1'],a['y1']))>.75*min(w*h,(a['x2']-a['x1'])*(a['y2']-a['y1'])) for a in output):continue
                box.update(region_source='physical_component_annotation_lane',component_context=dict(detection_id=d['detection_id'],kind=d['class'],ambiguous=False))
                output.append(box)
    return cleaned,output
