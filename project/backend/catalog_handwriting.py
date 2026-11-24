"""Constrain already visible model digits to known names, using user references."""
import re

def read_contextual_gain(mask,refs,kind):
    if kind not in {'transistor_npn','transistor_pnp'}:return None
    import cv2,numpy as np
    from backend.handwriting_reference import _clean_line,_Segments,candidates,_result
    clean,height=_clean_line(mask)
    _,original_labels=cv2.connectedComponents(clean,8)
    bars=cv2.morphologyEx(clean,cv2.MORPH_OPEN,np.ones((1,max(5,round(height*.22))),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(bars,8);options=[]
    strokes=[s for s in stats[1:] if s[2]>=6 and s[2]>s[3]*1.5]
    for a in strokes:
        for b in strokes:
            ax,ay,aw,ah,_=a;bx,by,bw,bh,_=b
            overlap=min(ax+aw,bx+bw)-max(ax,bx)
            if not (.10*height<by-ay<.65*height and overlap>.55*min(aw,bw) and min(aw,bw)>.35*max(aw,bw)):continue
            labels=[]
            for x,y,w,h in ((ax,ay,aw,ah),(bx,by,bw,bh)):
                values=original_labels[y:y+h,x:x+w];values=values[values>0]
                labels.append(int(np.bincount(values).argmax()) if len(values) else 0)
            if 0 in labels or labels[0]==labels[1]:continue
            start=max(ax,bx);end=max(ax+aw,bx+bw)
            if not height*.20<start<clean.shape[1]*.60:continue
            prefix=candidates(clean[:,:start],refs)
            gain=[r for r in prefix if r['text'] in {'B','b','β','P','p','hFE','HFE'}]
            if not gain or gain[0]['score']<.50 or gain[0]['score']<prefix[0]['score']-.25:continue
            tail=clean[:,end+1:];lattice=_Segments(tail,refs,height);lattice.numeric_margin=.24
            for cost,digits,ids,scores in lattice.digit_paths(lattice.end):
                if not re.fullmatch(r'\d{1,5}',digits) or not 0<int(digits)<=100000 or np.mean(scores)<.64:continue
                options.append((cost/len(digits),digits,ids,scores,gain[0]))
    if not options:return None
    options.sort(key=lambda o:o[0]);_,digits,ids,scores,prefix=options[0]
    result=_result('β='+digits,[prefix['reference_id'],'geometric_parallel_bars']+ids,[prefix['score'],.96]+scores,'bjt_local_equals_then_reference_digits')
    result.update(unresolved=False,needs_review=True,alternatives=['β='+o[1] for o in options[:3]])
    return result

def read_catalog_model(mask,refs,kind,partial):
    from backend.component_models import CATALOG
    from backend.handwriting_reference import _clean_line,_Segments,_result
    digits=re.findall(r'\d{3,}',partial)
    names=[name for name,spec in CATALOG.items() if spec.get('type',spec.get('kind'))==kind and any(d in name for d in digits)]
    # Reading a visible model name does not require parameter catalog coverage.
    # Compare every prefix character; never invent a model from digits alone.
    if kind=='opamp':
        names=list(dict.fromkeys(names+[prefix+d[i:] for d in digits for i in range(max(1,len(d)-1))
                                      if 2<=len(d[i:])<=5 for prefix in ('LM','TL','NE','OPA','AD')]))
    known={e['text'].upper() for e,_ in refs}
    names=[n for n in names if all(ch in known for ch in n)]
    if not names:return None
    clean,height=_clean_line(mask);lattice=_Segments(clean,refs,height);choices=[]
    for name in names:
        states={lattice.start:(0.,[],[])}
        for char in name:
            next_states={}
            for a,(cost,ids,scores) in states.items():
                for b in lattice.cuts:
                    if b<=a or b-a>height*1.6 or not lattice.single_glyph(a,b):continue
                    ranking=lattice.rank(a,b);matches=[r for r in ranking if r['text'].upper()==char]
                    if not matches or matches[0]['score']<.50 or matches[0]['score']<ranking[0]['score']-.22:continue
                    r=matches[0];entry=(cost+1-r['score'],ids+[r['reference_id']],scores+[r['score']])
                    if b not in next_states or entry[0]<next_states[b][0]:next_states[b]=entry
            states=next_states
        if lattice.end in states:
            cost,ids,scores=states[lattice.end];choices.append((cost/len(name),name,ids,scores))
    if not choices:return None
    choices.sort();cost,name,ids,scores=choices[0]
    if cost>.34:return None
    # Do not invent missing catalog letters: every character above is compared
    # against a supplied sample. Close variants stay explicit alternatives.
    result=_result(name,ids,scores,'catalog_model_with_reference_character_alignment')
    result.update(unresolved=False,needs_review=True,alternatives=[c[1] for c in choices[:3]])
    return result
