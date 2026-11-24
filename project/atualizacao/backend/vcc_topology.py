"""Count physical leads leaving a symbol in the original image context."""
import cv2
import numpy as np
from backend.resistor_geometry import symbol_mask

def ring_bounds(x,y,width,height,shape,mask=None):
    # The hole excludes the pen stroke. A fixed four-pixel margin counted the
    # thick, uneven rim of a photographed ring as a second external wire.
    pad=max(4,min(24,int(round(min(width,height)*.42))))
    if mask is not None:
        # Measure the photographed rim instead of consuming a short stem
        # with a margin proportional to the circle diameter.
        margin=max(4,min(16,round(min(width,height)*.35)))
        patch=mask[max(0,y-margin):min(mask.shape[0],y+height+margin),max(0,x-margin):min(mask.shape[1],x+width+margin)]
        if patch.size:
            distance=cv2.distanceTransform(patch,cv2.DIST_L2,3)
            values=distance[(patch>0)&(distance>0)]
            if len(values):pad=min(pad,max(4,int(np.ceil(2*np.percentile(values,65)))))
    h,w=shape[:2]
    return dict(x1=float(max(0,x-pad)),y1=float(max(0,y-pad)),
                x2=float(min(w-1,x+width+pad)),y2=float(min(h-1,y+height+pad)))


def symbol_ink_labels(labels,detections):
    """Dominant ink belonging to actual component bodies, excluding annotations."""
    owned=set();h,w=labels.shape
    for item in detections:
        if item.get('class') in {'gnd','vcc'}:continue
        b=item['bbox'];x1,y1=max(0,int(b['x1'])),max(0,int(b['y1']))
        x2,y2=min(w,int(b['x2'])+1),min(h,int(b['y2'])+1)
        values=labels[y1:y2,x1:x2];values=values[values>0]
        if not len(values):continue
        counts=np.bincount(values);counts[0]=0
        owned.update(int(v) for v in np.flatnonzero(counts>=max(10,counts.max()*.30)))
    return owned


def context_symbol_mask(image,detections):
    mask=symbol_mask(image)
    from backend.ink_color import estimate_component_ink
    from backend.notebook_rulings import _paper_colors
    profile=estimate_component_ink(image,{'detections':detections})
    if not profile.get('active') or not profile.get('chromatic'):return mask
    _,_,normalized=_paper_colors(image)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    corrected=cv2.cvtColor(normalized.astype(np.uint8),cv2.COLOR_BGR2HSV)
    delta=np.abs(hsv[:,:,0].astype(float)-profile['reference_hsv'][0])
    hue=np.minimum(delta,180-delta)<=max(15,profile['hue_tolerance']*1.5)
    colored=(hue&(corrected[:,:,1]>=25)&(normalized.mean(axis=2)<250)).astype(np.uint8)*255
    return cv2.morphologyEx(mask|colored,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))

def external_leads(image,bbox,page_mask=None):
    mask=symbol_mask(image) if page_mask is None else page_mask
    h,w=mask.shape;size=max(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])
    margin=max(25,min(100,int(size*.8)))
    x1=max(0,int(np.floor(bbox['x1']))-margin);y1=max(0,int(np.floor(bbox['y1']))-margin)
    x2=min(w,int(np.ceil(bbox['x2']))+margin+1);y2=min(h,int(np.ceil(bbox['y2']))+margin+1)
    crop=(mask[y1:y2,x1:x2]>0).astype(np.uint8)
    if not crop.size:return 0
    yy,xx=np.indices(crop.shape);gx=xx+x1;gy=yy+y1
    body=(gx>=bbox['x1']-2)&(gx<=bbox['x2']+2)&(gy>=bbox['y1']-2)&(gy<=bbox['y2']+2)
    distance=np.maximum.reduce([bbox['x1']-2-gx,gx-bbox['x2']-2,bbox['y1']-2-gy,gy-bbox['y2']-2])
    original_count,original_labels,_,_=cv2.connectedComponentsWithStats(crop,8)
    inside_labels=set(np.unique(original_labels[body&(crop>0)]));inside_labels.discard(0)
    work=crop.copy();work[body]=0
    count,labels,stats,_=cv2.connectedComponentsWithStats(work,8);branches=0
    for i in range(1,count):
        pixels=labels==i
        if stats[i,cv2.CC_STAT_AREA]<6 or not np.any(pixels&(distance<=3)) or not np.any(pixels&(distance>=8)):continue
        if not any(label in inside_labels for label in np.unique(original_labels[pixels])):continue
        branches+=1
    return branches

def hollow_power_body(image,bbox,page_mask=None):
    mask=symbol_mask(image) if page_mask is None else page_mask
    h,w=mask.shape;x1=max(0,int(bbox['x1'])-3);y1=max(0,int(bbox['y1'])-3)
    x2=min(w,int(np.ceil(bbox['x2']))+4);y2=min(h,int(np.ceil(bbox['y2']))+4)
    contours,hierarchy=cv2.findContours(mask[y1:y2,x1:x2],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:return None
    # A transistor ring divided by its base/fork has several large holes.
    # A power terminal has one unpartitioned hollow center.
    holes=[cv2.contourArea(c) for i,c in enumerate(contours) if hierarchy[0,i,3]>=0]
    if holes and sum(a>=max(8,.2*max(holes)) for a in holes)>1:return None
    candidates=[]
    for i,c in enumerate(contours):
        if hierarchy[0,i,3]<0:continue
        x,y,bw,bh=cv2.boundingRect(c);area=cv2.contourArea(c);per=cv2.arcLength(c,True)
        if min(bw,bh)<3 or max(bw,bh)>1.8*min(bw,bh) or per==0 or 4*np.pi*area/(per*per)<.55:continue
        if area<max(8,.025*(bbox['x2']-bbox['x1'])*(bbox['y2']-bbox['y1'])):continue
        # Inner hole expanded to include the ring stroke.
        candidates.append(ring_bounds(x+x1,y+y1,bw,bh,mask.shape,mask))
    return max(candidates,key=lambda b:(b['x2']-b['x1'])*(b['y2']-b['y1'])) if candidates else None
