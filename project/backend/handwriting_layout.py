"""Keep complete electrical inscriptions, including a unit written underneath."""
import cv2
import numpy as np

def join_stacked_units(image,regions,refs):
    from backend.handwriting_reference import ink,candidates,_clean_line
    masks={i:ink(image[b['y1']:b['y2'],b['x1']:b['x2']]) for i,b in enumerate(regions)}
    units=[]
    for i,b in enumerate(regions):
        if b['x2']-b['x1']>1.4*(b['y2']-b['y1']):continue
        cleaned,_=_clean_line(masks[i])
        _,labels,stats,_=cv2.connectedComponentsWithStats(cleaned,8)
        lower=[j for j,s in enumerate(stats[1:],1) if s[4]>=8 and s[1]+s[3]/2>cleaned.shape[0]*.45]
        if lower and any(s[1]==0 and s[3]<cleaned.shape[0]*.45 for s in stats[1:]):cleaned=np.isin(labels,lower).astype(np.uint8)*255
        ranking=candidates(cleaned,refs)
        _,_,stats,_=cv2.connectedComponentsWithStats(cleaned,8)
        substantial=[s for s in stats[1:] if s[4]>=max(8,cleaned.shape[0]*.2)]
        if len(substantial)>2:continue
        choices=[r for r in ranking if r['text'] in {'F','H','V','Ω'} and r['score']>=.55 and r['score']>=ranking[0]['score']-.25]
        if choices:units.append((i,choices[0]));masks[i]=cleaned
    output=[];used=set()
    for i,b in enumerate(regions):
        if i in used:continue
        if b['x2']-b['x1']<1.7*(b['y2']-b['y1']):output.append(b);continue
        first=masks[i];clean,height=_clean_line(first)
        _,_,stats,_=cv2.connectedComponentsWithStats(clean,8)
        glyphs=sorted([s for s in stats[1:] if s[4]>=8 and s[3]>height*.5],key=lambda s:s[0])
        if len(glyphs)<2:output.append(b);continue
        lx,ly,lw,lh,_=glyphs[-1];suffix=candidates(clean[ly:ly+lh,lx:lx+lw],refs)
        if not suffix or not any(r['text'] in {'n','m','µ','p','K','M'} and r['score']>=.60 and r['score']>=suffix[0]['score']-.15 for r in suffix):output.append(b);continue
        x,y,w,h,_=glyphs[0];ranking=candidates(clean[y:y+h,x:x+w],refs)
        if not ranking or ranking[0]['text'] not in '0123456789' or ranking[0]['score']<.52:output.append(b);continue
        matches=[]
        for j,unit in units:
            u=regions[j]
            if j==i or j in used:continue
            gap=u['y1']-b['y2'];uh=u['y2']-u['y1']
            if -uh*.4<=gap<=uh*.8 and u['x1']>b['x1']+(b['x2']-b['x1'])*.55 and u['x1']<b['x2']+uh*.2:matches.append((abs(gap),j,unit))
        if not matches:output.append(b);continue
        _,j,unit=min(matches);u=regions[j];um=masks[j]
        points=cv2.findNonZero(um)
        if points is None:output.append(b);continue
        x,y,w,h=cv2.boundingRect(points);unit_mask=cv2.resize(um[y:y+h,x:x+w],(max(1,round(w*height/h)),max(1,round(height))))
        canvas=np.zeros((max(first.shape[0],unit_mask.shape[0]+10),first.shape[1]+unit_mask.shape[1]+8),np.uint8)
        canvas[:first.shape[0],:first.shape[1]]=first
        uy=(canvas.shape[0]-unit_mask.shape[0])//2
        canvas[uy:uy+unit_mask.shape[0],first.shape[1]+5:first.shape[1]+5+unit_mask.shape[1]]=unit_mask
        merged=dict(b,x1=min(b['x1'],u['x1']),y1=min(b['y1'],u['y1']),x2=max(b['x2'],u['x2']),y2=max(b['y2'],u['y2']),prepared_mask=canvas)
        output.append(merged);used.add(j)
    return [b for b in output if not any(b is regions[i] for i in used)]

def split_component_inscriptions(regions,detections):
    output=[]
    for box in regions:
        height=box['y2']-box['y1'];width=box['x2']-box['x1']
        if width<height*3 or 'prepared_mask' in box:output.append(box);continue
        centers=[]
        for d in detections:
            if d['class'] not in {'resistor','capacitor','capacitor_polarized','inductor'}:continue
            b=d['bbox'];cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
            if (box['x1']+height*1.2<cx<box['x2']-height*1.2 and box['y1']<=cy<=box['y2']):centers.append(cx)
        cuts=[box['x1']]+sorted(set(round(x) for x in centers))+[box['x2']]
        output.extend(dict(box,x1=a+(min(8,round(height*.2)) if a!=box['x1'] else 0),x2=b) for a,b in zip(cuts,cuts[1:]) if b>a)
    return output

def join_horizontal_units(image,regions,refs,detections):
    from backend.handwriting_reference import ink,candidates
    output=[];used=set()
    for i,a in enumerate(regions):
        if i in used:continue
        h=a['y2']-a['y1'];w=a['x2']-a['x1']
        if w>h*6 or 'prepared_mask' in a:output.append(a);continue
        center=((a['x1']+a['x2'])/2,(a['y1']+a['y2'])/2)
        if any(d['bbox']['x1']<=center[0]<=d['bbox']['x2'] and d['bbox']['y1']<=center[1]<=d['bbox']['y2'] for d in detections):output.append(a);continue
        numeric=ink(image[a['y1']:a['y2'],a['x1']:a['x2']])
        if w>h*1.2:
            from backend.handwriting_reference import _clean_line
            numeric,gh=_clean_line(numeric)
            _,labels,stats,_=cv2.connectedComponentsWithStats(numeric,8)
            glyphs=sorted([(i,s) for i,s in enumerate(stats[1:],1) if s[4]>=8 and s[3]>=gh*.45],key=lambda p:p[1][0])
            if not glyphs:output.append(a);continue
            i0,(x,y,gw,gh,_)=glyphs[0];numeric=(labels[y:y+gh,x:x+gw]==i0).astype(np.uint8)*255
        rank=candidates(numeric,refs)
        if not rank or rank[0]['text'] not in '0123456789' or rank[0]['score']<.62:output.append(a);continue
        options=[]
        for j,b in enumerate(regions):
            if i==j or j in used or 'prepared_mask' in b:continue
            bh=b['y2']-b['y1'];gap=b['x1']-a['x2']
            if gap<0 or gap>max(h,bh)*.9 or min(a['y2'],b['y2'])-max(a['y1'],b['y1'])<min(h,bh)*.35:continue
            unit=candidates(ink(image[b['y1']:b['y2'],b['x1']:b['x2']]),refs)
            if unit and unit[0]['text'] in {'V','F','H','Ω'} and unit[0]['score']>=.63:options.append((gap,j))
        if not options:output.append(a);continue
        _,j=min(options);b=regions[j];output.append(dict(a,x2=b['x2'],y1=min(a['y1'],b['y1']),y2=max(a['y2'],b['y2'])));used.add(j)
    return [b for b in output if not any(b is regions[i] for i in used)]
