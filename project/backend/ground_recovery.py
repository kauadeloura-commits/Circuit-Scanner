"""Indexed three-bar ground recovery, avoiding page-wide cubic line searches."""
import cv2,numpy as np
from backend.resistor_geometry import symbol_mask
from backend.vcc_topology import external_leads

def _recover_upright(image,detections,class_id,mask):
    h,w=mask.shape;raw=cv2.HoughLinesP(mask,1,np.pi/180,12,minLineLength=12,maxLineGap=3)
    if raw is None:return []
    _,ink_labels,ink_stats,ink_centers=cv2.connectedComponentsWithStats(mask,8)
    shape_cache={}
    def simple_bar(label,length):
        if label not in shape_cache:
            x,y,bw,bh,area=ink_stats[label]
            blob=(ink_labels[y:y+bh,x:x+bw]==label).astype(np.uint8)
            contours,hierarchy=cv2.findContours(blob,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            hole=hierarchy is not None and any(hierarchy[0,i,3]>=0 and cv2.contourArea(c)>=max(14,.20*area) for i,c in enumerate(contours))
            shape_cache[label]=(bw,bh,hole)
        bw,bh,hole=shape_cache[label]
        return not hole and max(bw,bh)<=length*1.2 and (min(bw,bh)<=max(bw,bh)*.55 or max(bw,bh)<=length*.55)
    def stem(top):
        ax,ay,bx,by=top['ends'];axis=np.array([bx-ax,by-ay],float)
        axis/=max(1,np.linalg.norm(axis));normal=np.array([-axis[1],axis[0]])
        if normal[1]<0:normal=-normal
        distances=np.arange(max(6,top['length']*.12),max(12,top['length']*.95),3)
        offsets=np.arange(-max(3,top['length']*.20),max(3,top['length']*.20)+1,2)
        points=np.array([top['x'],top['y']])-distances[:,None,None]*normal+offsets[None,:,None]*axis
        xs=np.rint(points[:,:,0]).astype(int);ys=np.rint(points[:,:,1]).astype(int)
        valid=(xs>=0)&(xs<w)&(ys>=0)&(ys<h);hits=np.zeros(valid.shape,bool)
        hits[valid]=ink_labels[ys[valid],xs[valid]]==top['label']
        return np.mean(np.any(hits,axis=1))>=.75
    bars=[];bar_cells={}
    for x1,y1,x2,y2 in raw.reshape(-1,4):
        length=abs(x2-x1)
        if not 4<=length<=max(96,min(180,min(h,w)*.25)) or abs(y2-y1)>max(3,length*.30):continue
        bar=dict(x=(x1+x2)/2,y=(y1+y2)/2,length=length,ends=(int(min(x1,x2)),int(min(y1,y2)),int(max(x1,x2)),int(max(y1,y2))))
        px,py=round(bar['x']),round(bar['y'])
        values=ink_labels[max(0,py-2):min(h,py+3),max(0,px-3):min(w,px+4)]
        values=values[values>0]
        bar['label']=int(np.bincount(values).argmax()) if len(values) else 0
        cell=(int(bar['x']//4),int(bar['y']//4))
        neighbors=(b for dx in (-1,0,1) for dy in (-1,0,1)
                   for b in bar_cells.get((cell[0]+dx,cell[1]+dy),()))
        if not any(abs(b['x']-bar['x'])<4 and abs(b['y']-bar['y'])<4 and abs(b['length']-length)<8 for b in neighbors):
            bars.append(bar);bar_cells.setdefault(cell,[]).append(bar)
    # Read the more permissive line candidates once per orientation. Repeating
    # Hough in thousands of overlapping crops was the dominant recovery cost.
    refinement=cv2.HoughLinesP(mask,1,np.pi/180,8,minLineLength=9,maxLineGap=3)
    refined=[]
    for x1,y1,x2,y2 in (() if refinement is None else refinement.reshape(-1,4)):
        length=abs(x2-x1)
        if not length or abs(y2-y1)>max(4,length*.45):continue
        cx,cy=(x1+x2)/2,(y1+y2)/2
        values=ink_labels[max(0,round(cy)-2):min(h,round(cy)+3),max(0,round(cx)-3):min(w,round(cx)+4)]
        values=values[values>0]
        if len(values):refined.append(dict(x=cx,y=cy,length=length,ends=(int(min(x1,x2)),int(min(y1,y2)),int(max(x1,x2)),int(max(y1,y2))),label=int(np.bincount(values).argmax())))
    refinement_by_label={}
    for line in refined:refinement_by_label.setdefault(line['label'],[]).append(line)
    refinement_arrays={label:np.array([[b['x'],b['y'],b['length'],*b['ends']] for b in lines])
                       for label,lines in refinement_by_label.items()}
    for bar in list(bars):
        if bar['length']<12 or not bar['label'] or not stem(bar):continue
        radius=int(max(32,min(120,bar['length']*2)))
        x,y=int(bar['x']),int(bar['y']);ax,ay=max(0,x-radius),max(0,y-radius)
        bx,by=min(w,x+radius+1),min(h,y+radius+1)
        values=refinement_arrays.get(bar['label'])
        if values is None:continue
        cx,cy,length,x1,y1,x2,y2=values.T
        valid=(x1>=ax)&(x2<bx)&(y1>=ay)&(y2<by)&(length>bar['length'])
        valid &= (np.abs(cy-bar['y'])<=np.maximum(8,length*.25)) & (np.abs(cx-bar['x'])<=length*.4)
        indices=np.flatnonzero(valid)
        if len(indices):
            best=indices[int(np.argmax(length[indices]))]
            bars.append(refinement_by_label[bar['label']][best])
    # Short lower bars and handwritten dots can fall below Hough's vote
    # threshold. Their complete isolated ink path provides the third stroke.
    for label,(x,y,bw,bh,area) in enumerate(ink_stats[1:],1):
        if area<4 or not 2<=bw<=max(96,min(180,min(h,w)*.25)) or not (bh<=max(4,bw*.75) or max(bw,bh)<=18):continue
        cx,cy=ink_centers[label]
        bars.append(dict(x=cx,y=cy,length=bw,ends=(int(x),int(y),int(x+bw-1),int(y+bh-1)),label=label))
    # Hough often splits the longest bar at its attached stem. Reunite only
    # collinear, adjoining pieces of the same photographed ink path.
    merged=[];by_label={}
    for bar in sorted(bars,key=lambda b:b['length'],reverse=True):
        a=np.array(bar['ends'][:2],float);b=np.array(bar['ends'][2:],float)
        axis=(b-a)/max(1,np.linalg.norm(b-a));normal=np.array([-axis[1],axis[0]])
        group=next((g for g in by_label.get(bar['label'],()) if
                    abs(float((np.array([g['x'],g['y']])-(a+b)/2)@normal))<=max(4,bar['length']*.10) and
                    abs(g['x']-bar['x'])<=.6*(g['length']+bar['length'])+5),None)
        if group is None:
            group=dict(bar);merged.append(group);by_label.setdefault(bar['label'],[]).append(group);continue
        points=np.array([a,b,group['ends'][:2],group['ends'][2:]],float)
        low,high=points[:,0].min(),points[:,0].max()
        group.update(x=(low+high)/2,y=float(points[:,1].mean()),length=high-low,
                     ends=(int(low),int(points[:,1].min()),int(high),int(points[:,1].max())))
    bars=merged
    found=[]
    for top in bars:
        if top['length']<12 or not top['label'] or not stem(top):continue
        # A diode cathode, capacitor plate or resistor edge belongs to its
        # component. It cannot supply the top bar of a second ground below.
        from backend.drawing_mode import CURRENT
        if CURRENT.get()=='handwritten' and any(d.get('class') not in {'gnd','vcc','mosfet_n','mosfet_p'} and
               d['bbox']['x1']+.15*(d['bbox']['x2']-d['bbox']['x1'])<=top['x']<=d['bbox']['x2']-.15*(d['bbox']['x2']-d['bbox']['x1']) and
               d['bbox']['y1']+.15*(d['bbox']['y2']-d['bbox']['y1'])<=top['y']<=d['bbox']['y2']-.15*(d['bbox']['y2']-d['bbox']['y1']) for d in detections):continue
        nearby=[b for b in bars if 4<b['y']-top['y']<max(24,top['length']*1.85) and abs(b['x']-top['x'])<max(6,top['length']*.30)]
        for middle in nearby:
            if middle['length']>top['length']*1.10 or not middle['label'] or middle['label']==top['label'] or not simple_bar(middle['label'],top['length']):continue
            for bottom in nearby:
                gap1=middle['y']-top['y'];gap2=bottom['y']-middle['y']
                if not gap2>=4 or not .4<gap1/gap2<2.5 or bottom['length']>middle['length']*1.15 or middle['length']>top['length']*1.10 or top['length']<bottom['length']*1.4:continue
                owned=[bar['label'] for bar in (top,middle,bottom)]
                # F/E/H and numbers have joined strokes. A ground has three
                # separate bars; only the first is joined to the circuit lead.
                if 0 in owned or len(set(owned))!=3:continue
                if not simple_bar(bottom['label'],top['length']):continue
                x1=min(b['ends'][0] for b in (top,middle,bottom))-5;x2=max(b['ends'][2] for b in (top,middle,bottom))+5
                y1=top['y']-max(10,min(24,top['length']*.35));y2=bottom['y']+6
                box=dict(x1=float(max(0,x1)),y1=float(max(0,y1)),x2=float(min(w-1,x2)),y2=float(min(h-1,y2)))
                cx,cy=(box['x1']+box['x2'])/2,(box['y1']+box['y2'])/2
                enclosed=[d for d in detections+found if d['bbox']['x1']<=cx<=d['bbox']['x2'] and d['bbox']['y1']<=cy<=d['bbox']['y2']]
                if any(not (d.get('class') in {'mosfet_n','mosfet_p'} and not d.get('manual_reviewed') and box['y1']>d['bbox']['y1']+.55*(d['bbox']['y2']-d['bbox']['y1'])) for d in enclosed):continue
                if external_leads(image,box,mask)!=1:continue
                box.update(width=box['x2']-box['x1'],height=box['y2']-box['y1'],center_x=cx,center_y=cy)
                found.append(dict(**{'class':'gnd'},class_id=class_id,bbox=box,confidence=.88,best_view_confidence=.88,
                    inference_sources=['three_descending_bars_one_lead'],geometry_validation=dict(valid=True,score=.88,evidence=dict(three_descending_bars=True,external_lead_count=1))))
    return found


def _rotate_box(box,shape,turn):
    h,w=shape[:2]
    points=np.array([[x,y] for x in (box['x1'],box['x2']) for y in (box['y1'],box['y2'])])
    for _ in range(turn%4):
        points=np.column_stack((points[:,1],w-1-points[:,0]));h,w=w,h
    b=dict(x1=float(points[:,0].min()),y1=float(points[:,1].min()),x2=float(points[:,0].max()),y2=float(points[:,1].max()))
    b.update(width=b['x2']-b['x1'],height=b['y2']-b['y1'],center_x=(b['x1']+b['x2'])/2,center_y=(b['y1']+b['y2'])/2)
    return b


def recover_grounds(image,detections,class_id,_page_mask=None):
    from backend.ink_color import estimate_component_ink
    masks=[symbol_mask(image) if _page_mask is None else _page_mask]
    if _page_mask is None:
        profile=estimate_component_ink(image,{'detections':detections})
        if profile.get('active') and profile.get('chromatic'):
            from backend.notebook_rulings import _paper_colors
            _,_,normalized=_paper_colors(image)
            hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            corrected=cv2.cvtColor(normalized.astype(np.uint8),cv2.COLOR_BGR2HSV)
            delta=np.abs(hsv[:,:,0].astype(float)-profile['reference_hsv'][0])
            hue=np.minimum(delta,180-delta)<=max(15,profile['hue_tolerance']*1.5)
            # Match hue after local paper correction. Fixed RGB distances
            # erased light bars on a different-colored area of the same page.
            chromatic=(hue&(corrected[:,:,1]>=25)&(normalized.mean(axis=2)<235)).astype(np.uint8)*255
            masks.append(cv2.morphologyEx(chromatic,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)))
    found=[]
    for mask in masks:
        for turn in range(4):
            rotated=np.ascontiguousarray(np.rot90(image,turn))
            rotated_mask=np.ascontiguousarray(np.rot90(mask,turn))
            existing=[dict(d,bbox=_rotate_box(d['bbox'],image.shape,turn)) for d in detections+found]
            candidates=_recover_upright(rotated,existing,class_id,rotated_mask)
            for candidate in candidates:
                candidate['bbox']=_rotate_box(candidate['bbox'],rotated.shape,-turn)
                candidate['geometry_validation']['evidence']['source_quarter_turn']=turn
                found.append(candidate)
    return found
