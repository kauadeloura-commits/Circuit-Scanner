"""Recover local writing and give every component a turn in crowded pages."""
from collections import defaultdict

def recover_regions(page,regions,detections):
    if len(detections)<10:return regions
    from backend.ocr_regions import text_line_regions
    h,w=page.shape;output=list(regions)
    def overlap(a,b):
        area=max(0,min(a['x2'],b['x2'])-max(a['x1'],b['x1']))*max(0,min(a['y2'],b['y2'])-max(a['y1'],b['y1']))
        return area/max(1,min((a['x2']-a['x1'])*(a['y2']-a['y1']),(b['x2']-b['x1'])*(b['y2']-b['y1'])))
    for d in detections:
        if d.get('class')=='gnd':continue
        b=d['bbox'];margin=max(30,min(160,max(b['x2']-b['x1'],b['y2']-b['y1'])*.9))
        x1,y1=max(0,int(b['x1']-margin)),max(0,int(b['y1']-margin));x2,y2=min(w,int(b['x2']+margin)),min(h,int(b['y2']+margin))
        local=text_line_regions(page[y1:y2,x1:x2])
        for r in local:
            # A crop edge can cut a word or a wire into a false letter.
            if (x1>0 and r['x1']<6 or y1>0 and r['y1']<6 or
                x2<w and r['x2']>x2-x1-6 or y2<h and r['y2']>y2-y1-6):continue
            import cv2,numpy as np
            crop=(page[y1+r['y1']:y1+r['y2'],x1+r['x1']:x1+r['x2']]<128).astype(np.uint8)
            _,_,stats,_=cv2.connectedComponentsWithStats(crop,8)
            height=r['y2']-r['y1']
            glyphs=[s for s in stats[1:] if s[4]>=max(12,height*.3) and s[3]>=max(8,height*.45)]
            if len(glyphs)<2:continue
            r=dict(r,x1=r['x1']+x1,x2=r['x2']+x1,y1=r['y1']+y1,y2=r['y2']+y1,region_source='component_neighborhood')
            if any((r['x1']+r['x2'])/2>=other['bbox']['x1'] and (r['x1']+r['x2'])/2<=other['bbox']['x2'] and
                (r['y1']+r['y2'])/2>=other['bbox']['y1'] and (r['y1']+r['y2'])/2<=other['bbox']['y2'] for other in detections):continue
            if not any(overlap(r,old)>.45 for old in output):output.append(r)
    return output

def schedule_regions(regions):
    groups=defaultdict(list);unclaimed=[]
    for r in regions:
        context=r.get('component_context')
        if context:groups[context.get('detection_id')].append(r)
        else:unclaimed.append(r)
    output=[]
    # References/models and values of distant parts must not be pushed out by
    # a cluster of many fragments around the first component.
    while groups:
        for key in list(groups):
            output.append(groups[key].pop(0))
            if not groups[key]:del groups[key]
    return output+unclaimed
