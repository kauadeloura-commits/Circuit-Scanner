"""Supplied handwriting outlines plus a small supervised glyph classifier."""
from pathlib import Path
from functools import lru_cache
import json,math,re
import cv2
import numpy as np

BANK=Path(__file__).resolve().parent/'handwriting_references'
SIZE=48
READER_VERSION='complete-margins-and-typed-model-recheck-14'

@lru_cache(maxsize=16)
def _reference_pack(keys):
    arrays=[[],[],[],[],[],[],[],[]]
    for label,blob,aspect in keys:
        template=np.frombuffer(blob,np.uint8).reshape(SIZE,SIZE)
        td=cv2.distanceTransform(255-template,cv2.DIST_L2,3)
        tr,tdr=_rectangular(blob);ts,tsd=_stroke_form(blob)
        for group,value in zip(arrays[:6],(template,td,tr,tdr,ts,tsd)):group.append(value.ravel())
        if label in {'V','H','Ω','M','W','0','8'}:
            mirrored,md,mr,mrd,ms,msd=_reflection(blob)
        else:mirrored,md,mr,mrd,ms,msd=template,td,tr,tdr,ts,tsd
        arrays[6].append([v.ravel() for v in (mirrored,mr,ms)])
        arrays[7].append([v.ravel() for v in (md,mrd,msd)])
    packed=[np.asarray(v) for v in arrays]
    for i in (0,2,4,6):packed[i]=packed[i]>0
    return packed


@lru_cache(maxsize=16)
def _reference_ink_positions(keys):
    packed=_reference_pack(keys)
    tables=[packed[i] for i in (0,2,4)]+[packed[6][:,i] for i in range(3)]
    result=[]
    for table in tables:
        positions=[np.flatnonzero(row) for row in table]
        counts=np.asarray([len(p) for p in positions])
        starts=np.concatenate(([0],np.cumsum(counts)[:-1]))
        result.append((np.concatenate(positions),starts,counts))
    return result

def _batched_scores(query,dist,rect,rd,stroke,sd,refs,indices=None):
    """The same chamfer/Dice comparisons, evaluated as arrays for all samples."""
    keys=tuple((e['text'],s[0].tobytes(),s[2]) for e,s in refs)
    tm,td,tr,tdr,ts,tsd,mirrors,mirror_dist=_reference_pack(keys)
    selected=np.asarray(indices if indices is not None else range(len(refs)),dtype=np.intp)
    sparse=_reference_ink_positions(keys)
    def similarity(q,d,t,dt,index,rows,stroke_only=False):
        q=q.ravel()>0;d=d.ravel();pixels=np.flatnonzero(q)
        positions,starts,counts=sparse[index]
        sums=np.add.reduceat(d[positions],starts)[rows];counts=counts[rows]
        # Gather only ink columns. Copying all eight dense reference arrays
        # for each possible character cut was most of the OCR comparison work.
        chamfer=(dt[np.ix_(rows,pixels)].mean(axis=1)+sums/np.maximum(1,counts))/2
        if stroke_only:return np.exp(-chamfer/3.5)
        dice=2*t[np.ix_(rows,pixels)].sum(axis=1)/np.maximum(1,q.sum()+counts)
        return .65*np.exp(-chamfer/2.8)+.35*dice
    actual=.45*similarity(query,dist,tm,td,0,selected)+.30*similarity(rect,rd,tr,tdr,1,selected)+.25*similarity(stroke,sd,ts,tsd,2,selected,True)
    reflected=actual.copy()
    symmetric=np.asarray([j for j,i in enumerate(selected) if refs[i][0]['text'] in {'V','H','Ω','M','W','0','8'}],dtype=np.intp)
    if len(symmetric):
        rows=selected[symmetric]
        reflected[symmetric]=(.45*similarity(query,dist,mirrors[:,0],mirror_dist[:,0],3,rows)+.30*similarity(rect,rd,mirrors[:,1],mirror_dist[:,1],4,rows)+.25*similarity(stroke,sd,mirrors[:,2],mirror_dist[:,2],5,rows,True))
    return actual,reflected

@lru_cache(maxsize=256)
def _rectangular(blob):
    shape=np.frombuffer(blob,np.uint8).reshape(SIZE,SIZE)
    x,y,w,h=cv2.boundingRect(cv2.findNonZero(shape))
    patch=cv2.resize(shape[y:y+h,x:x+w],(32,40),interpolation=cv2.INTER_AREA)
    result=np.zeros((SIZE,SIZE),np.uint8);result[4:44,8:40]=(patch>100)*255
    return result,cv2.distanceTransform(255-result,cv2.DIST_L2,3)

def _similarity(query,dist,template,td):
    chamfer=(float(td[query>0].mean())+float(dist[template>0].mean()))/2
    dice=2*np.count_nonzero((query>0)&(template>0))/max(1,np.count_nonzero(query)+np.count_nonzero(template))
    return .65*math.exp(-chamfer/2.8)+.35*dice

@lru_cache(maxsize=512)
def _stroke_form(blob):
    thin=_thin_shape(blob)
    # A tiny stroke can disappear during thinning. Retain the original glyph
    # rather than aborting the entire text page on an empty skeleton.
    signature=normalize(thin)
    normalized=signature[0] if signature is not None else np.frombuffer(blob,np.uint8).reshape(SIZE,SIZE)
    return _rectangular(normalized.tobytes())

@lru_cache(maxsize=2048)
def _thin_shape(blob):
    from backend.thinning import thin_binary
    return thin_binary(np.frombuffer(blob,np.uint8).reshape(SIZE,SIZE))*255

def _holes(mask):
    padded=cv2.copyMakeBorder(mask,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
    n,_,stats,_=cv2.connectedComponentsWithStats(255-padded,8)
    return sum(s[4]>=max(3,np.count_nonzero(mask)*.015) for s in stats[1:])-1

@lru_cache(maxsize=1024)
def _shape_holes(blob):return _holes(np.frombuffer(blob,np.uint8).reshape(SIZE,SIZE))

@lru_cache(maxsize=128)
def _reflection(blob):
    template=np.frombuffer(blob,np.uint8).reshape(SIZE,SIZE)
    mirrored=template[:,::-1].copy();md=cv2.distanceTransform(255-mirrored,cv2.DIST_L2,3)
    mr,mrd=_rectangular(mirrored.tobytes());ms,msd=_stroke_form(mirrored.tobytes())
    return mirrored,md,mr,mrd,ms,msd

def ink(image):
    if image.ndim==2:return (image<128).astype(np.uint8)*255
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    color=(hsv[:,:,1]>40)&(hsv[:,:,2]<235)
    dark=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    # A little saturated ink must not reduce a faded word to a few pixels.
    coverage=np.count_nonzero(color&(dark>0))/max(1,np.count_nonzero(dark))
    if .0003<np.mean(color)<.45 and coverage>=.55:return color.astype(np.uint8)*255
    return dark

def normalize(mask):
    pts=cv2.findNonZero(mask)
    if pts is None:return None
    x,y,w,h=cv2.boundingRect(pts);crop=mask[y:y+h,x:x+w]
    scale=(SIZE-8)/max(w,h);nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    result=np.zeros((SIZE,SIZE),np.uint8);patch=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_AREA)
    result[(SIZE-nh)//2:(SIZE-nh)//2+nh,(SIZE-nw)//2:(SIZE-nw)//2+nw]=(patch>100)*255
    return result,w/h

def signature(mask):
    norm=normalize(mask)
    if norm is None:return None
    shape,aspect=norm;distance=cv2.distanceTransform(255-shape,cv2.DIST_L2,3)
    return shape,distance,aspect

@lru_cache(maxsize=4)
def _load(stamp,folder,trained=False):
    folder=Path(folder);manifest=json.loads((folder/'index.json').read_text(encoding='utf8'));refs=[]
    samples=manifest.get('trained_samples',manifest['samples']) if trained else manifest['samples']
    for entry in samples:
        if not trained and not entry.get('general_ocr_reference',True):continue
        if Path(entry['file']).name!=entry['file']:continue
        mask=cv2.imread(str(folder/entry['file']),cv2.IMREAD_GRAYSCALE)
        sig=signature(mask) if mask is not None else None
        if sig:refs.append((dict(entry,_bank_directory=str(folder),_trained_branch=trained),sig))
    return manifest,refs

def library(folder=BANK,*,trained=False):
    path=Path(folder)/'index.json'
    if not path.is_file():return {},[]
    from backend.glyph_classifier import model_stamp
    return _load((path.stat().st_mtime_ns,model_stamp(folder)),str(folder),trained)

_CANDIDATE_CACHE={}

def candidates(mask,refs):
    if not mask.size or not np.any(mask):return []
    key=(mask.shape,mask.tobytes(),tuple(id(s[0]) for _,s in refs))
    if key in _CANDIDATE_CACHE:return _CANDIDATE_CACHE[key]
    photographs=[r for r in refs if r[0].get('id','').startswith('PHOTO')]
    baseline=[r for r in refs if not r[0].get('id','').startswith('PHOTO')]
    # A calibration photograph adds a writing style, not a new alphabet that
    # can overturn the supplied references on every unrelated circuit.
    result=_compare_candidates(mask,baseline or refs)
    if photographs and baseline:
        extra=_compare_candidates(mask,photographs)
        if extra and extra[0]['score']>=.85 and (not result or extra[0]['score']>=result[0]['score']+.06):
            merged={r['text']:r for r in result}
            for candidate in extra:
                if candidate['text'] not in merged or candidate['score']>merged[candidate['text']]['score']:merged[candidate['text']]=candidate
            result=sorted(merged.values(),key=lambda r:r['score'],reverse=True)
    if len(_CANDIDATE_CACHE)>=2048:_CANDIDATE_CACHE.pop(next(iter(_CANDIDATE_CACHE)))
    _CANDIDATE_CACHE[key]=result
    return result

def _compare_candidates(mask,refs):
    if not mask.size or not np.any(mask):return []
    # An equality delimiter is a geometric pair of parallel bars, not an
    # inferred letter E. It does not add an external reference image.
    _,_,stats,_=cv2.connectedComponentsWithStats(mask,8)
    strokes=[s for s in stats[1:] if s[4]>=3]
    if len(strokes)==2:
        a,b=sorted(strokes,key=lambda s:s[1]);span=b[1]+b[3]-a[1]
        if (min(a[2],b[2])>1.6*max(a[3],b[3]) and .45<a[2]/b[2]<2.2
            and abs((a[0]+a[2]/2)-(b[0]+b[2]/2))<.25*max(a[2],b[2])
            and span>1.8*max(a[3],b[3]) and span<1.7*max(a[2],b[2])):
            return [dict(text='=',score=.96,reference_id='geometric_parallel_bars')]
    sig=signature(mask)
    if sig is None:return []
    query,dist,aspect=sig;labels={};rect,rd=_rectangular(query.tobytes());stroke,sd=_stroke_form(query.tobytes())
    from backend.glyph_classifier import support
    learned=support(query,aspect,refs[0][0].get('_bank_directory',BANK)) if refs and refs[0][0].get('_trained_branch') else None
    k_strokes=_angular_k(query) and not(learned and learned['text'] in '0235689' and learned['margin']>=4)
    indices=[i for i,(e,s) in enumerate(refs) if abs(math.log(aspect/s[2]))<=1.05]
    eligible=[refs[i] for i in indices]
    if not eligible:return []
    actual,reflected=_batched_scores(query,dist,rect,rd,stroke,sd,refs,indices)
    for index,(entry,(template,td,ta)) in enumerate(eligible):
        deviation=abs(math.log(aspect/ta))
        if deviation>1.05:continue
        # Symmetric chamfer tolerates differences in stroke thickness and scale;
        # aspect ratio prevents a narrow 1 being stretched into a letter.
        score=float(actual[index])-.045*deviation
        label=entry['text']
        # These glyphs retain their meaning when the two arms lean differently.
        # Do not mirror asymmetric letters or digits (R, 2, 5, 6, 9, etc.).
        if label in {'V','H','Ω','M','W','0','8'}:
            score=max(score,float(reflected[index])-.045*deviation-.01)
        if label=='0':score-=.18*max(0,_shape_holes(query.tobytes())-_shape_holes(template.tobytes()))
        if label=='8' and _shape_holes(query.tobytes())==0 and not (entry.get('id','').startswith('PHOTO') and _shape_holes(template.tobytes())==0):score-=.10
        # A detached top bar is common in handwritten 5. Preserve that stroke
        # rather than ranking the lower curve alone as a 3 or 4.
        if label=='5' and len(strokes)>1:
            top=min(strokes,key=lambda s:s[1])
            points=cv2.findNonZero(mask);_,gy,_,gh=cv2.boundingRect(points)
            if top[2]>top[3]*1.5 and top[1]+top[3]<gy+gh*.45:score+=.045
        if label=='5' and _five_corner(query):score+=.055
        if label in {'0','6','9'} and _shape_holes(query.tobytes())==0:score-=.035
        if label=='K' and k_strokes:score=max(score,.72)
        if label not in labels or score>labels[label]['score']:labels[label]=dict(text=label,score=score,reference_id=entry['id'])
    if learned and learned.get('font_specialist'):
        label=learned['text'];match=labels.get(label);best=max(r['score'] for r in labels.values())
        if match and match['score']>=.40 and match['score']>=best-.25:
            match.update(score=max(match['score'],.84),trained_glyph_support=True)
        elif not match:
            labels[label]=dict(text=label,score=.84,reference_id='supervised_font_glyph_'+label,trained_glyph_support=True)
    if learned and learned['text'] in labels and (learned['text'] not in '0123456789' or learned['margin']>=2):
        result=labels[learned['text']];best=max(r['score'] for r in labels.values())
        if result['score']>=.55 and result['score']>=best-.10:
            result['score']=min(.995,result['score']+learned['boost'])
            result['trained_glyph_support']=True
    return sorted(labels.values(),key=lambda r:r['score'],reverse=True)

@lru_cache(maxsize=512)
def _angular_k_cached(blob):
    from backend.thinning import thin_binary
    mask=np.frombuffer(blob,np.uint8).reshape(SIZE,SIZE)
    pts=cv2.findNonZero(mask);x,y,w,h=cv2.boundingRect(pts)
    sk=_thin_shape(blob)
    lines=cv2.HoughLinesP(sk,1,np.pi/180,4,minLineLength=max(4,h*.12),maxLineGap=3)
    if lines is None:return False
    stem=False;upper=False;lower=False
    sy,sx=np.nonzero(sk)
    stem_pixels=(sx>=x+w*.27)&(sx<=x+w*.53)
    if np.count_nonzero(stem_pixels)>=h*.5:
        stem=bool(np.ptp(sy[stem_pixels])>h*.65)
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        if x1>x2:x1,y1,x2,y2=x2,y2,x1,y1
        dx=x2-x1;dy=y2-y1
        if abs(dy)>h*.35 and dx<max(3,abs(dy)*.2) and (x1+x2)/2<x+w*.52:stem=True
        if dx>w*.20 and x1<x+w*.65 and x2>x+w*.6:
            if -.9*h<dy<-h*.17:upper=True
            if h*.10<dy<h*.9:lower=True
    if stem and lower and np.count_nonzero((sx>x+w*.72)&(sy<y+h*.38))>=4:upper=True
    return stem and upper and lower

def _angular_k(mask):return _angular_k_cached(mask.tobytes())

@lru_cache(maxsize=512)
def _five_corner_cached(blob):
    from backend.thinning import thin_binary
    mask=np.frombuffer(blob,np.uint8).reshape(SIZE,SIZE)
    pts=cv2.findNonZero(mask);x,y,w,h=cv2.boundingRect(pts)
    if _holes(mask)>0:return False
    sk=_thin_shape(blob)
    if np.count_nonzero(sk[y+round(h*.22):y+round(h*.55),x:x+round(w*.42)])<h*.20:return False
    yy,xx=np.nonzero(sk[y:y+round(h*.27),x:x+w])
    if len(xx)<4:return False
    _,(rw,rh),angle=cv2.minAreaRect(np.column_stack((xx,yy)).astype(np.float32))
    angle=(angle+(90 if rw<rh else 0)+90)%180-90
    return max(rw,rh)>w*.45 and max(rw,rh)>2.5*max(1,min(rw,rh)) and abs(angle)<40

def _five_corner(mask):return _five_corner_cached(mask.tobytes())

def accepted(ranking,minimum=.73,margin=.075):
    return bool(ranking and ranking[0]['score']>=minimum and (len(ranking)<2 or ranking[0]['score']-ranking[1]['score']>=margin))

def _legacy_read_line(mask,refs,kind=None):
    whole=candidates(mask,refs)
    if accepted(whole,.83,.08) and len(whole[0]['text'])>1:
        return dict(text=whole[0]['text'],confidence=whole[0]['score']*100,reference_ids=[whole[0]['reference_id']],unresolved=False)
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8)
    boxes=[(int(x),int(y),int(w),int(h)) for x,y,w,h,a in stats[1:] if a>=max(3,mask.size*.0001)]
    boxes.sort();groups=[]
    for x,y,w,h in boxes:
        if groups and x<=groups[-1][2]+1:
            g=groups[-1];groups[-1]=(min(x,g[0]),min(y,g[1]),max(x+w,g[2]),max(y+h,g[3]))
        else:groups.append((x,y,x+w,y+h))
    # Multiple cut candidates handle touching glyphs without inventing a label.
    expanded=[]
    for x1,y1,x2,y2 in groups:
        if x2-x1>1.3*(y2-y1):
            projection=np.count_nonzero(mask[y1:y2,x1:x2],axis=0);cuts=[]
            for x in range(5,len(projection)-5):
                if projection[x]<=max(1,(y2-y1)*.12) and projection[x]==min(projection[max(0,x-3):x+4]):
                    if not cuts or x-cuts[-1]>.3*(y2-y1):cuts.append(x)
            starts=[x1]+[x1+x for x in cuts]+[x2]
            expanded.extend((a,y1,b,y2) for a,b in zip(starts,starts[1:]) if b>a)
        else:expanded.append((x1,y1,x2,y2))
    groups=expanded;count=len(groups);dp={0:(0.,[],[],[])}
    # Locate a unit before reading its numeric prefix. Case of m/M is preserved.
    numeric_end=None;unit_label=None
    if groups:
        x1,y1,x2,y2=groups[-1];unit=candidates(mask[y1:y2,x1:x2],refs)
        if accepted(unit,.78,.06) and unit[0]['text'] in {'F','H','V','Ω','Hz'}:numeric_end=count-1;unit_label=unit[0]['text']
        expected={'capacitor':'F','capacitor_polarized':'F','inductor':'H','source_dc':'V','battery':'V','resistor':'Ω'}.get(kind)
        if numeric_end is None and expected:
            unit_only=candidates(mask[y1:y2,x1:x2],[(e,s) for e,s in refs if e['text']==expected])
            if accepted(unit_only,.70,0):numeric_end=count-1;unit_label=expected
    numeric_labels=set('0123456789pnmKM')|{'µ'}
    for i in range(count):
        if i not in dp:continue
        cost,texts,ids,scores=dp[i]
        for end in range(i+1,min(count,i+4)+1):
            parts=groups[i:end];x1=min(g[0] for g in parts);y1=min(g[1] for g in parts);x2=max(g[2] for g in parts);y2=max(g[3] for g in parts)
            if end>i+1 and any(b[0]-a[2]>.42*max(y2-y1,1) for a,b in zip(parts,parts[1:])):break
            pool=[(e,s) for e,s in refs if e['text'] in numeric_labels] if numeric_end is not None and end<=numeric_end else refs
            is_unit=numeric_end is not None and i==numeric_end and end==count
            if is_unit:pool=[(e,s) for e,s in refs if e['text']==unit_label]
            ranks=candidates(mask[y1:y2,x1:x2],pool);valid=accepted(ranks,.70,0) if is_unit else accepted(ranks)
            if not valid and end>i+1:continue
            label=ranks[0]['text'] if valid else '�';score=ranks[0]['score'] if valid else 0.;reference=ranks[0]['reference_id'] if valid else None
            next_cost=cost+((1-score)*max(1,len(label)) if valid else 1.2)
            if end not in dp or next_cost<dp[end][0]:dp[end]=(next_cost,texts+[label],ids+[reference],scores+[score])
    _,texts,ids,scores=dp.get(count,(0,[],[],[]))
    text=''.join(texts)
    ambiguous_unit=kind in {'inductor','capacitor','capacitor_polarized'} and numeric_end is None and bool(__import__('re').fullmatch(r'\d+[pnmµKM]\d+',text))
    return dict(text=text,confidence=100*sum(scores)/max(1,len(scores)),reference_ids=[x for x in ids if x],unresolved='�' in texts or not texts or ambiguous_unit)

def _clean_line(mask):
    mask=(mask>0).astype(np.uint8)*255
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    tall=[s[3] for s in stats[1:] if s[4]>=12 and s[3]>=mask.shape[0]*.3]
    height=float(np.median(tall)) if tall else float(mask.shape[0])
    main_areas=[s[4] for s in stats[1:] if s[3]>=height*.65 and s[4]>=12]
    typical_area=float(np.median(main_areas)) if main_areas else 0.
    keep=np.zeros(n,np.uint8)
    for i,(x,y,w,h,area) in enumerate(stats[1:],1):
        if area<max(3,height*.13):continue
        if h>height*1.5 and w<height*.3 and (x<height*.15 or x+w>mask.shape[1]-height*.15):continue
        if (area<typical_area*.24 and (x==0 or y==0 or x+w==mask.shape[1] or y+h==mask.shape[0])
            and min(w,h)<height*.25):continue
        # A neighboring wire above/below the writing is not part of a glyph.
        if w>1.65*height and h<=height*.30:continue
        if w>1.35*height and h<height*.40 and (y==0 or y+h==mask.shape[0]):continue
        if any(j!=i and main[4]>area*3 and main[3]>h*4 and
               main[0]<=x+w/2<=main[0]+main[2] and
               -height*.25<y-(main[1]+main[3])<height*.25
               for j,main in enumerate(stats[1:],1)):continue
        keep[i]=255
    return keep[labels],height

def _symbol_only(mask,height):
    pts=cv2.findNonZero(mask)
    if pts is None:return 'empty_fragment'
    x,y,w,h=cv2.boundingRect(pts)
    if h<height*.4 or (h<.35*w and w<2*height):return 'isolated_stroke'
    # Cross-shaped polarity marks belong to terminal analysis, not values OCR.
    components=cv2.connectedComponentsWithStats(mask[y:y+h,x:x+w],8)[0]-1
    if .65<w/h<1.5 and components==1 and _holes(mask[y:y+h,x:x+w])==0:
        from backend.thinning import thin_binary
        sk=thin_binary(cv2.copyMakeBorder(mask[y:y+h,x:x+w],3,3,3,3,cv2.BORDER_CONSTANT,value=0))
        neighbors=cv2.filter2D(sk,cv2.CV_16S,np.ones((3,3),np.int16))-sk
        ends=np.column_stack(np.nonzero((sk>0)&(neighbors==1)))
        if len(ends)==4:
            cy,cx=np.mean(ends,axis=0)
            directions=[(ex-cx,ey-cy) for ey,ex in ends]
            if (sum(abs(dx)>1.6*abs(dy) for dx,dy in directions)==2 and
                sum(abs(dy)>1.6*abs(dx) for dx,dy in directions)==2):return 'polarity_mark'
    return None

def _stem_displacement(mask):
    points=cv2.findNonZero(mask)
    if points is None:return None
    x,y,w,h=cv2.boundingRect(points)
    if h<8 or w>h*.9:return None
    medians=[]
    for low,high in ((.30,.50),(.70,.90)):
        _,xx=np.nonzero(mask[y+round(h*low):y+round(h*high),x:x+w])
        if not len(xx):return None
        medians.append(float(np.median(xx)))
    return (medians[1]-medians[0])/h

@lru_cache(maxsize=16)
def _stem_references(keys):
    values={label:[] for label in ('1','7')}
    for label,blob in keys:
        value=_stem_displacement(np.frombuffer(blob,np.uint8).reshape(SIZE,SIZE))
        if value is not None:values[label].append(value)
    return {label:float(np.median(v)) for label,v in values.items() if v}

def _numeric_stem_reference(mask,refs):
    """Use the author's lower stroke when a neighbor changes the glyph's head."""
    if _holes(mask):return None
    value=_stem_displacement(mask)
    if value is None:return None
    keys=tuple((e['text'],s[0].tobytes()) for e,s in refs if e['text'] in {'1','7'})
    model=_stem_references(keys)
    if set(model)!= {'1','7'}:return None
    one,seven=abs(value-model['1']),abs(value-model['7'])
    return '1' if one<=.035 and seven-one>=.075 else None

class _Segments:
    """A bounded lattice of gap/valley cuts; preserve overlapping glyph parts."""
    def __init__(self,mask,refs,height):
        self.mask=mask;self.refs=[(e,s) for e,s in refs if len(e['text'])==1];self.height=height
        self.cache={};self.numeric_cache={};self.start=self.end=0;self.cuts=[]
        self.numeric_margin=.16
        self.reference_numbers=False
        self.projection=np.count_nonzero(mask,axis=0)
        # Removing neighboring glyphs can leave a candidate suffix without ink.
        # An empty lattice has no character paths; it is not an OCR failure.
        if not mask.size or not np.any(mask):
            self.labels=np.zeros(mask.shape,np.int32);self.stats=np.empty((0,5),np.int32);self.centers=np.empty((0,2))
            return
        _,self.labels,self.stats,self.centers=cv2.connectedComponentsWithStats(mask,8)
        xs=np.flatnonzero(np.any(mask,axis=0));self.start=int(xs[0]);self.end=int(xs[-1])+1
        projection=np.count_nonzero(mask,axis=0);cuts={self.start,self.end}
        # Blank gaps are boundaries, never individual characters.
        for a,b in zip(xs,xs[1:]):
            if b>a+1:cuts.add(int((a+b+1)//2))
        major=sorted([s for s in self.stats[1:] if s[3]>.48*height],key=lambda s:s[0])
        for a,b in zip(major,major[1:]):
            if a[0]+a[2]/2<b[0] and a[0]+a[2]>b[0]:cuts.add(int((a[0]+a[2]+b[0])/2))
        radius=max(2,round(height*.07));last=-99
        for x in range(self.start+radius,self.end-radius):
            if projection[x]<=height*.85 and projection[x]==min(projection[x-radius:x+radius+1]) and x-last>height*.12:
                cuts.add(x);last=x
        protected=[s for s in self.stats[1:] if s[3]>.48*height and s[2]<1.25*s[3]]
        cuts={x for x in cuts if x in {self.start,self.end} or not any(s[0]+s[2]*.15<x<s[0]+s[2]*.85 for s in protected)}
        self.cuts=sorted(cuts);self.cache={};self.numeric_cache={};self.projection=projection
    def crop(self,a,b):
        if self.start==self.end:return np.zeros_like(self.mask[:,a:b])
        # Bounding boxes may overlap: remove neighboring connected components
        # whose centers and most pixels belong to the other character.
        keep=np.zeros(len(self.stats),np.uint8)
        for i,(x,y,w,h,area) in enumerate(self.stats[1:],1):
            overlap=max(0,min(x+w,b)-max(x,a))
            if a<=self.centers[i][0]<b or overlap>=w*.7 or w>1.3*h:keep[i]=255
        return keep[self.labels[:,a:b]]
    def rank(self,a,b):
        key=(a,b)
        if key not in self.cache:
            crop=self.crop(a,b);self.cache[key]=candidates(crop,self.refs)
        return self.cache[key]
    def single_glyph(self,a,b):
        crop=self.crop(a,b)
        _,_,stats,_=cv2.connectedComponentsWithStats(crop,8)
        major=sorted([s for s in stats[1:] if s[3]>.48*self.height],key=lambda s:s[0])
        return not any(a[0]+a[2]<=b[0] for a,b in zip(major,major[1:]))
    def digit_paths(self,end):
        if end<=self.start:return []
        if end in self.numeric_cache:return self.numeric_cache[end]
        # Numeric grammar stops m, rc, Z etc replacing an entire digit sequence.
        states={self.start:[(0.,'',[],[])]}
        for a in self.cuts:
            if a>=end or a not in states:continue
            for b in self.cuts:
                if b<=a or b>end:continue
                if b-a>self.height*2.1:break
                if not self.single_glyph(a,b):continue
                ranking=self.rank(a,b)
                actual=self.crop(a,b);points=cv2.findNonZero(actual)
                if points is None:continue
                _,_,glyph_w,glyph_h=cv2.boundingRect(points)
                _,glyph_y,_,_=cv2.boundingRect(points)
                decimal=glyph_h<=self.height*.23 and glyph_w<=self.height*.23 and glyph_y>=self.mask.shape[0]*.55
                minus=glyph_h<=self.height*.23 and glyph_w>glyph_h*2.5 and abs(glyph_y-self.mask.shape[0]*.48)<self.height*.3
                if decimal:digits=[dict(text='.',score=.93,reference_id='geometric_decimal_separator')]
                elif minus:digits=[dict(text='-',score=.93,reference_id='geometric_numeric_sign')]
                else:
                    if glyph_w>glyph_h*1.45 or glyph_h<self.height*.45:continue
                    digits=[r for r in ranking if r['text'] in '0123456789'][:2]
                    straight_one=False
                    if getattr(self,'value_numbers',False) and digits and digits[0]['text'] in {'1','7'} and digits[0]['score']>=.60 and digits[0]['score']>=ranking[0]['score']-.10:
                        cue=_numeric_stem_reference(actual,self.refs)
                        one=next((r for r in ranking if r['text']=='1'),None)
                        if cue=='1' and one:
                            one=dict(one,score=max(.85,one['score']),reading_evidence='lower_stem_matches_supplied_ones')
                            digits=[one]+[r for r in digits if r['text']!='1'];straight_one=True
                    from backend.glyph_classifier import support
                    norm=normalize(actual)
                    if getattr(self,'value_numbers',False) and norm:
                        entire=support(norm[0],norm[1],self.refs[0][0].get('_bank_directory',BANK))
                        if ranking and ranking[0]['text'] in {'K','k','µ','n','p'} and ranking[0]['score']>=.7:
                            digits=[r for r in digits if r['score']>=ranking[0]['score']-.05]
                        if entire and entire['text'] not in '0123456789' and entire['margin']>=2:
                            visual=next((r for r in ranking if r['text']==entire['text']),None)
                            if visual and visual['score']>=.64 and visual['score']>=ranking[0]['score']-.04:continue
                    learned=support(norm[0],norm[1],self.refs[0][0].get('_bank_directory',BANK),allowed=set('0123456789')) if self.reference_numbers and norm and self.refs and (self.refs[0][0].get('_trained_branch') or getattr(self,'value_numbers',False)) else None
                    if straight_one:learned=None
                    threshold=(4 if learned and learned['text']=='4' else 2) if getattr(self,'value_numbers',False) else .6 if self.reference_numbers else 2
                    if learned and learned['margin']>=threshold:
                        visual=next((r for r in ranking if r['text']==learned['text']),None)
                        if getattr(self,'value_numbers',False) and visual and digits and visual['score']<digits[0]['score']-(.25 if learned.get('font_specialist') else .08):visual=None
                        if visual and visual['score']>=.38 and visual['score']>=ranking[0]['score']-.30:
                            floor=.80 if learned.get('font_specialist') else .69 if getattr(self,'value_numbers',False) else .76 if self.reference_numbers else .69
                            visual=dict(visual,score=max(visual['score'],min(.88,floor+.018*learned['margin'])),trained_glyph_support=True)
                            digits=sorted([visual]+[r for r in digits if r['text']!=visual['text']],key=lambda r:r['score'],reverse=True)[:2]
                for r in digits:
                    if r['score']<.52:continue
                    if r['text'] not in {'.','-'} and ranking and r['score']<ranking[0]['score']-self.numeric_margin:continue
                    for cost,text,ids,scores in states[a]:
                        if r['text']=='.' and (not text or text=='-' or '.' in text):continue
                        if r['text']=='-' and text:continue
                        # Prefer a supported unsplit glyph over multiple cuts.
                        penalty=.08+float(self.projection[min(b,self.mask.shape[1]-1)])/self.height*.15 if b<end else .08
                        entry=(cost+1-float(r['score'])+penalty,text+r['text'],ids+[r['reference_id']],scores+[float(r['score'])])
                        states.setdefault(b,[]).append(entry)
                if b in states:
                    best={}
                    for p in states[b]:
                        if p[1] not in best or p[0]<best[p[1]][0]:best[p[1]]=p
                    states[b]=sorted(best.values())[:6]
        self.numeric_cache[end]=[p for p in states.get(end,[]) if re.fullmatch(r'-?\d+(?:\.\d+)?',p[1])]
        return self.numeric_cache[end]

def _result(text,ids,scores,grammar):
    return dict(text=text,confidence=100*sum(scores)/max(1,len(scores)),reference_ids=list(dict.fromkeys(ids)),unresolved=False,reading_method=grammar)



def _separated_units(mask,refs,expected=None,diagnostics=None,left_labels=None):
    """Follow low-ink seams between touching glyphs, preserving slanted strokes."""
    points=cv2.findNonZero(mask)
    if points is None:return []
    x,y,w,h=cv2.boundingRect(points)
    if w<.65*h or w>2.8*h:return []
    source=mask[y:y+h,x:x+w];output={}
    prefix_refs=[(e,s) for e,s in refs if e['text'] in (left_labels if left_labels is not None else {'p','n','µ','m','M','K'})]
    unit_refs=[(e,s) for e,s in refs if e['text'] in ({expected} if expected else {'F','H','V','Ω','W'})]
    for fraction in (.36,.44,.52,.60,.68):
        center=w*fraction;columns=np.arange(w)
        cost=(source>0).astype(float)*4.+abs(columns-center)[None,:]/max(1,w*.15)
        cost[:,(columns<w*.18)|(columns>w*.82)]=1e6
        previous=cost[0].copy();parents=[]
        for row in cost[1:]:
            choices=np.stack((np.r_[1e6,previous[:-1]]+.12,previous,np.r_[previous[1:],1e6]+.12))
            selected=choices.argmin(axis=0);parents.append(selected.astype(np.int8)-1)
            previous=row+choices.min(axis=0)
        seam=np.zeros(h,np.int32);seam[-1]=int(previous.argmin())
        for iy in range(h-2,-1,-1):seam[iy]=seam[iy+1]+int(parents[iy][seam[iy+1]])
        left=source.copy();right=source.copy()
        for iy,cut in enumerate(seam):left[iy,cut:]=0;right[iy,:cut+1]=0
        lp=candidates(left,refs);rp=candidates(right,refs)
        if not lp or not rp:continue
        prefix=candidates(left,prefix_refs);unit=candidates(right,unit_refs)
        if not prefix or not unit:continue
        p,u=prefix[0],unit[0]
        if diagnostics is not None:diagnostics.append(dict(fraction=fraction,left=left,right=right,prefix=p,unit=u,left_best=lp[0],right_best=rp[0]))
        if p['score']<.55 or u['score']<(.55 if expected else .63) or p['score']<lp[0]['score']-.08 or u['score']<rp[0]['score']-.12:continue
        label=p['text']+u['text'];score=(float(p['score'])+float(u['score']))/2
        crossed=sum(source[iy,cut]>0 for iy,cut in enumerate(seam))/h
        score-=crossed*.08
        item=dict(text=label,score=score,reference_id=p['reference_id']+'+'+u['reference_id'])
        if label not in output or score>output[label]['score']:output[label]=item
    return sorted(output.values(),key=lambda r:r['score'],reverse=True)

def _structured_line(mask,refs,kind,height,diagnostics=None,reference_kinds=()):
    lattice=_Segments(mask,refs,height);options=[];start,end=lattice.start,lattice.end
    prefixes={'n','p','m','M','K','µ'};units={'F','H','V','Ω','W'}
    expected={'capacitor':'F','capacitor_polarized':'F','inductor':'H','source_dc':'V','vcc':'V','battery':'V','resistor':'Ω'}.get(kind)
    if expected:lattice.numeric_margin=.24
    expected_reference={'capacitor':{'C','c'},'capacitor_polarized':{'C','c'},'resistor':{'R','r'},'inductor':{'L'},'source_dc':{'V'},'battery':{'B'},'transistor_npn':{'Q','q','T','t'},'transistor_pnp':{'Q','q','T','t'},'diode':{'D','d'},'led':{'D','d'},'mosfet_n':{'M','Q','q'},'mosfet_p':{'M','Q','q'},'opamp':{'U'}}.get(kind,set())
    primary_reference=set(expected_reference)
    for other in reference_kinds:
        expected_reference=expected_reference|{'capacitor':{'C','c'},'capacitor_polarized':{'C','c'},'resistor':{'R','r'},'inductor':{'L'},'source_dc':{'V'},'battery':{'B'}}.get(other,set())
    for a in lattice.cuts:
        if a<=start or end-a>height*1.6:continue
        ranking=lattice.rank(a,end)
        unit_points=cv2.findNonZero(lattice.crop(a,end))
        if unit_points is None:continue
        _,_,unit_w,unit_h=cv2.boundingRect(unit_points)
        if unit_w>unit_h*1.35:continue
        for unit in [r for r in ranking if r['text'] in units]:
            if expected and unit['text']!=expected and unit['score']<.72:continue
            minimum=.50 if expected==unit['text'] in {'H','Ω'} else .59
            tolerance=.25 if expected==unit['text'] else .14
            if unit['score']<minimum or unit['score']<ranking[0]['score']-tolerance:continue
            prefixes_at=[(a,None)]
            for b in lattice.cuts:
                if b<=start or b>=a or a-b>height*1.8:continue
                ranks=lattice.rank(b,a)
                norm=normalize(lattice.crop(b,a))
                supported_prefix=bool(ranks and any(r['text'] in prefixes and r['score']>=.68 and r['score']>=ranks[0]['score']-.03 for r in ranks))
                if not lattice.single_glyph(b,a) and not supported_prefix and not (norm is not None and _angular_k(norm[0])):continue
                for prefix in [r for r in ranks if r['text'] in prefixes][:2]:
                    threshold=.59 if expected else .65
                    if prefix['score']>=threshold and prefix['score']>=ranks[0]['score']-(.09 if expected else .05):prefixes_at.append((b,prefix))
            for numeric_end,prefix in prefixes_at:
                for cost,number,ids,scores in lattice.digit_paths(numeric_end):
                    if len(number)>10:continue
                    suffix=[prefix,unit] if prefix else [unit]
                    score=(cost+sum(1-float(r['score'])+.08 for r in suffix))/(len(number)+len(suffix))
                    score+=.06*max(0,float(ranking[0]['score'])-float(unit['score']))/.14
                    if expected==unit['text']:score-=.04
                    if prefix and prefix['text'] in {'K','M','m','µ','n','p'}:score-=.02
                    options.append((score,number+''.join(r['text'] for r in suffix),ids+[r['reference_id'] for r in suffix],scores+[float(r['score']) for r in suffix],'unit_then_numeric_grammar'))
    # When a prefix touches a unit (e.g. KΩ), compare their combined outlines
    # rather than requiring a clean vertical cut between the two characters.
    for a in lattice.cuts:
        if a<=start or end-a<height*.7 or end-a>height*2.8:continue
        crop=lattice.crop(a,end)
        if not lattice.single_glyph(a,end):continue
        single=lattice.rank(a,end)
        points=cv2.findNonZero(crop)
        _,_,uw,uh=cv2.boundingRect(points)
        if uw<uh*1.25 and single and any(r['text'] in units and r['score']>=.60 and r['score']>=single[0]['score']-.05 for r in single):continue
        rank=_separated_units(crop,refs,expected)
        if not rank or rank[0]['score']<(.55 if expected else .60):continue
        for suffix in rank[:2]:
            if suffix['score']<rank[0]['score']-.04:continue
            for cost,number,ids,scores in lattice.digit_paths(a):
                score=(cost+2*(1-float(suffix['score'])+.08))/(len(number)+2)
                score-=.045
                options.append((score,number+suffix['text'],ids+suffix['reference_id'].split('+'),scores+[float(suffix['score'])]*2,'touching_unit_reference_composition'))
    # Reference identifiers use the reference alphabet only at the first glyph.
    for a in lattice.cuts:
        if a<=start or a>=end or a-start>height*1.5:continue
        ranking=lattice.rank(start,a)
        for prefix in [r for r in ranking if r['text'] in {'R','r','C','c','L','V','B','Q','q','D','d','M','U'}][:2]:
            compatible=prefix['text'] in expected_reference
            if prefix['score']<(.49 if compatible else .64) or prefix['score']<ranking[0]['score']-(.28 if compatible else .07):continue
            # Read suffix from a, rather than treating the reference as a digit.
            tail=lattice.crop(a,end);sub=_Segments(tail,refs,height);sub.reference_numbers=True
            for cost,digits,ids,scores in sub.digit_paths(sub.end):
                if len(digits)>5 or not digits.isdigit():continue
                score=(cost+1-float(prefix['score'])+.08)/(len(digits)+1)
                if compatible:score-=.14 if prefix['text'] in primary_reference else .04
                label=prefix['text'].upper()+digits
                options.append((score,label,[prefix['reference_id']]+ids,[float(prefix['score'])]+scores,'reference_then_numeric_grammar'))
    # A pair of separated horizontal bars establishes a gain assignment.
    for a in lattice.cuts:
        if a<=start:continue
        for b in lattice.cuts:
            if b<=a or b>=end or b-a>height:continue
            eq=candidates(lattice.crop(a,b),refs)
            if not eq or eq[0]['text']!='=':continue
            left=candidates(lattice.crop(start,a),refs)
            if not left:continue
            gain=[r for r in left if r['text'] in {'β','B','b','hFE','HFE'}]
            contextual=kind in {'transistor_npn','transistor_pnp'}
            if not gain or gain[0]['score']<(.53 if contextual else .65) or gain[0]['score']<left[0]['score']-(.20 if contextual else .08):continue
            tail=lattice.crop(b,end);sub=_Segments(tail,refs,height)
            if contextual:sub.numeric_margin=.24
            for cost,digits,ids,scores in sub.digit_paths(sub.end):
                score=(cost+1-float(gain[0]['score'])+.04)/(len(digits)+2)
                options.append((score,'β='+digits,[gain[0]['reference_id'],eq[0]['reference_id']]+ids,[float(gain[0]['score']),.96]+scores,'gain_assignment_numeric_grammar'))
    if not options:return None
    best={}
    for option in options:
        if option[1] not in best or option[0]<best[option[1]][0]:best[option[1]]=option
    options=sorted(best.values())
    measurements=[o for o in options if 'unit' in o[4] and o[0]<.49]
    references=[o for o in options if o[4]=='reference_then_numeric_grammar']
    strong_reference=bool(references and sum(references[0][3])/len(references[0][3])>=.72 and references[0][0]<measurements[0][0]-.08) if measurements else False
    if measurements and not strong_reference:options=measurements
    elif references and measurements:options=references;measurements=[]
    winner=options[0]
    if diagnostics is not None:diagnostics.extend(options)
    if winner[0]>(.49 if measurements else .445):return None
    # Reject a competing full interpretation, not equivalent C/c templates.
    if len(options)>1 and (options[1][0]-winner[0])*max(2,len(winner[1]))<.03:
        result=_result(winner[1],winner[2],winner[3],winner[4])
        result.update(needs_review=True,alternatives=[o[1] for o in options[:3]])
        if measurements:result['unresolved']=True
        return result
    return _result(winner[1],winner[2],winner[3],winner[4])

def _trained_reference(mask,refs,kind,height,reference_kinds):
    """Read actual R/C/L ink before its numeric suffix; never supply a missing digit."""
    from backend.glyph_classifier import model_stamp,support
    if not refs:return None
    folder=refs[0][0].get('_bank_directory',BANK) if refs else BANK
    if not model_stamp(folder):return None
    prefixes={'resistor':{'R'},'capacitor':{'C'},'capacitor_polarized':{'C'},'inductor':{'L'},'transistor_npn':{'Q','T'},'transistor_pnp':{'Q','T'},'diode':{'D'},'led':{'D'},'mosfet_n':{'M','Q'},'mosfet_p':{'M','Q'},'opamp':{'U'}}
    expected=set().union(*(prefixes[k] for k in (kind,*reference_kinds) if k in prefixes))
    if not expected:return None
    points=cv2.findNonZero(mask)
    if points is None:return None
    _,_,w,h=cv2.boundingRect(points)
    if w>h*4.5:return None
    _,_,stats,_=cv2.connectedComponentsWithStats(mask,8)
    major=[s for s in stats[1:] if s[3]>=height*.45 and s[4]>=8]
    if len(major)<2 or len(major)>6:return None
    _,trained=library(folder,trained=True);lattice=_Segments(mask,trained,height);options=[]
    for cut in lattice.cuts:
        if cut<=lattice.start or cut>=lattice.end or cut-lattice.start>height*1.3:continue
        if lattice.single_glyph(cut,lattice.end) and lattice.end-cut>height*1.5:continue
        crop=lattice.crop(lattice.start,cut);norm=normalize(crop)
        if norm is None:continue
        learned=support(norm[0],norm[1],folder)
        # Avoid comparing a numeric value/model against every reference prefix.
        if not learned or learned['text'].upper() not in expected or learned['margin']<.6:continue
        ranking=lattice.rank(lattice.start,cut)
        prefix=next((r for r in ranking if r['text'].upper()==learned['text'].upper()),None)
        if not prefix or prefix['score']<.62 or prefix['score']<ranking[0]['score']-.08:continue
        tail=lattice.crop(cut,lattice.end);sub=_Segments(tail,trained,height);sub.reference_numbers=True
        for cost,digits,ids,scores in sub.digit_paths(sub.end):
            if not digits.isdigit() or len(digits)>5:continue
            score=(cost+1-float(prefix['score'])+.08)/(len(digits)+1)
            options.append((score,prefix['text'].upper()+digits,[prefix['reference_id']]+ids,[float(prefix['score'])]+scores))
    if not options:return None
    options.sort();winner=options[0]
    if winner[0]>.34:return None
    result=_result(winner[1],winner[2],winner[3],'reference_then_numeric_grammar')
    result['trained_glyph_classifier']=True
    if len(options)>1 and options[1][1]!=winner[1] and options[1][0]-winner[0]<.025:result.update(needs_review=True,alternatives=[o[1] for o in options[:3]])
    return result

def read_line(mask,refs,kind=None,reference_kinds=(),composite=True):
    if not mask.size:return dict(text='',confidence=0.,reference_ids=[],unresolved=True)
    # Read the photographic ink before cleanup can delete a touching zero,
    # a model suffix, or the unit. The same calibrated gate remains required.
    from backend.sequence_ocr import read_lines
    original_sequence=read_lines([mask],[None])[0]
    clean,height=_clean_line(mask)
    if not np.any(clean):return original_sequence or dict(text='',confidence=0.,reference_ids=[],unresolved=True)
    ignored=_symbol_only(clean,height)
    if ignored:return original_sequence or dict(text='',confidence=0.,reference_ids=[],unresolved=False,ignored=True,reason=ignored)
    whole=candidates(clean,refs)
    # A verified complete token preserves touching/slanted characters better
    # than an early per-glyph interpretation (for example a handwritten 1/7).
    # Both absolute agreement and a separate competing-token margin are needed.
    if whole and len(whole[0]['text'])>1 and accepted(whole,.94,.08):
        return _result(whole[0]['text'],[whole[0]['reference_id']],
                       [float(whole[0]['score'])],'whole_verified_reference_token')
    # A verified whole inscription and the explicit beta/equals specialist
    # retain priority over a generic sequence decoder trained on many words.
    from backend.catalog_handwriting import read_contextual_gain
    gain=read_contextual_gain(clean,refs,kind)
    if gain:return gain
    if original_sequence:return dict(original_sequence,input_variant='original_ink_before_cleanup')
    sequence=read_lines([clean],[kind])[0]
    if sequence:return sequence
    complete_reference=_trained_reference(clean,refs,kind,height,reference_kinds)
    if complete_reference:return complete_reference
    pts=cv2.findNonZero(clean);_,_,gw,gh=cv2.boundingRect(pts)
    initial=[r for r in whole if r['text'] in {'R','r','C','c','L'} and r['score']>=(.45 if kind=='resistor' and r['text'] in {'R','r'} else .52) and r['score']>=whole[0]['score']-(.30 if kind=='resistor' else .22)]
    if initial and gw<gh*1.10:
        return dict(text=initial[0]['text'].upper(),confidence=float(initial[0]['score'])*100,reference_ids=[initial[0]['reference_id']],unresolved=True,reading_method='reference_initial_needs_complete_suffix')
    legacy_model=_legacy_read_line(clean,refs,kind)
    if kind in {'transistor_npn','transistor_pnp','mosfet_n','mosfet_p','opamp','diode'} and re.search(r'\d{3,}',legacy_model.get('text','')):
        from backend.catalog_handwriting import read_catalog_model
        model=read_catalog_model(clean,refs,kind,legacy_model['text'])
        if model:return model
        _,trained=library(refs[0][0].get('_bank_directory',BANK),trained=True)
        model=read_catalog_model(clean,trained,kind,legacy_model['text'])
        if model:return dict(model,trained_glyph_classifier=True)
    if not legacy_model.get('unresolved'):
        from backend.component_models import CATALOG,model_name
        if model_name(legacy_model['text']) in CATALOG or re.fullmatch(r'1[nN]\d{3,5}[A-Za-z]?',legacy_model['text']):return dict(legacy_model,reading_method='complete_model_reference_comparison')
    if accepted(whole,.91,.075):
        return _result(whole[0]['text'],[whole[0]['reference_id']],[float(whole[0]['score'])],'direct_reference_match')
    if accepted(whole,.86,.06) and len(whole[0]['text'])>1:
        return _result(whole[0]['text'],[whole[0]['reference_id']],[float(whole[0]['score'])],'whole_reference_token')
    if not np.any(clean):return dict(text='',confidence=0.,reference_ids=[],unresolved=True)
    structured=_structured_line(clean,refs,kind,height,reference_kinds=reference_kinds)
    if structured and kind=='opamp' and re.search(r'\d{3,}',structured.get('text','')):
        from backend.catalog_handwriting import read_catalog_model
        _,trained=library(refs[0][0].get('_bank_directory',BANK),trained=True)
        model=read_catalog_model(clean,trained,kind,structured['text'])
        if model:return dict(model,trained_glyph_classifier=True)
    if structured:return structured
    fallback=legacy_model
    if not fallback.get('unresolved'):return fallback
    if kind in {'diode','transistor_npn','transistor_pnp','mosfet_n','mosfet_p','opamp'} and refs:
        from backend.glyph_classifier import model_stamp
        folder=refs[0][0].get('_bank_directory',BANK)
        if model_stamp(folder):
            _,trained=library(folder,trained=True)
            model_refs=[(dict(e,text=e['text'].upper()),sig) for e,sig in trained if re.fullmatch(r'[A-Za-z0-9]+',e['text'])]
            completed=_legacy_read_line(clean,model_refs,kind)
            from backend.component_models import CATALOG,model_name
            text=completed.get('text','').upper()
            explicit_diode=kind=='diode' and re.fullmatch(r'1N\d{3,5}[A-Z]?',text)
            if not completed.get('unresolved') and (model_name(text) in CATALOG or explicit_diode):
                return dict(completed,text=text,reading_method='complete_model_trained_glyph_comparison',trained_glyph_classifier=True)
    return fallback

def _composite_line(mask,refs,kind,height,reference_kinds):
    from backend.measurement_text import parse_measurement
    lattice=_Segments(mask,refs,height)
    if lattice.end-lattice.start<height*2.5:return None
    if sum(s[3]>=height*.45 and s[4]>=8 for s in lattice.stats[1:])<4:return None
    splits=[x for x in lattice.cuts if lattice.start+height*.8<x<lattice.end-height*.8]
    splits=sorted(splits,key=lambda x:(lattice.projection[x],abs(x-(lattice.start+lattice.end)*.6)))[:10]
    for cut in splits:
        left=read_line(lattice.crop(lattice.start,cut),refs,kind,reference_kinds,False)
        right=read_line(lattice.crop(cut,lattice.end),refs,kind,reference_kinds,False)
        if left.get('unresolved') or right.get('unresolved') or left.get('ignored') or right.get('ignored'):continue
        for reference,value in ((left,right),(right,left)):
            if not re.fullmatch(r'[RCL]\d{1,4}',reference['text']):continue
            measurement=parse_measurement(value['text'])
            expected_unit={'R':'ohm','C':'farad','L':'henry'}[reference['text'][0]]
            if not measurement or not measurement.get('unit_explicit') or measurement['unit']!=expected_unit:continue
            result=_result(reference['text']+' '+value['text'],reference['reference_ids']+value['reference_ids'],[reference['confidence']/100,value['confidence']/100],'complete_reference_and_value_inscription')
            result['needs_review']=left.get('needs_review',False) or right.get('needs_review',False)
            return result
    return None

def class_reference(mask,refs,kind):
    """Read the suffix beside a standard prefix missing from the photo bank."""
    if kind not in {'led','diode'}:return None
    clean,height=_clean_line(mask);lattice=_Segments(clean,refs,height)
    if not lattice.cuts:return None
    glyphs=[s for s in lattice.stats[1:] if s[3]>=height*.45 and s[4]>=8]
    if len(glyphs)<(3 if kind=='led' else 2):return None
    first_ok=False
    for cut in lattice.cuts:
        if cut<=lattice.start or cut-lattice.start>height*1.2:continue
        ranking=lattice.rank(lattice.start,cut)
        wanted='L' if kind=='led' else '0'
        choices=[r for r in ranking if r['text']==wanted and r['score']>=.64 and r['score']>=ranking[0]['score']-.10]
        if choices:first_ok=True;break
    if not first_ok:return None
    for cut in lattice.cuts:
        if cut<=lattice.start+height*(1.5 if kind=='led' else .45) or lattice.end-cut>height:continue
        ranking=lattice.rank(cut,lattice.end)
        if not ranking or ranking[0]['text'] not in '0123456789' or ranking[0]['score']<.70:continue
        digit=ranking[0];prefix='LED' if kind=='led' else 'D'
        result=_result(prefix+digit['text'],[digit['reference_id']],[float(digit['score'])],'component_class_prefix_and_read_numeric_suffix')
        result.update(needs_review=True,inferred_prefix=prefix,prefix_source='detected_component_class')
        return result
    return None

def recognize(image,detections=(),folder=BANK,detected_text_regions=()):
    from backend.local_word_ocr import locate_words
    from backend.handwriting_deskew import prepare,restore
    regions=list(detected_text_regions)
    if not any('text_angle_degrees' in r for r in regions):regions.extend(locate_words(image))
    prepared=prepare(image,detections,regions)
    if prepared is None:return _recognize_upright(image,detections,folder,regions)
    view,mapped,proposals,inverse,angle=prepared
    tokens,pending=_recognize_upright(view,mapped,folder,proposals)
    return restore(tokens,inverse,angle),restore(pending,inverse,angle)


def _recognize_upright(image,detections=(),folder=BANK,detected_text_regions=()):
    from backend.processing_progress import report
    report('Localizando as escritas próximas dos componentes…')
    from backend.ocr_regions import text_ink_image,text_line_regions
    from backend.ocr_context import prioritized_regions
    manifest,refs=library(folder)
    if not refs:raise ValueError('A biblioteca de escrita está vazia. Adicione as referências antes de ler o OCR.')
    from backend.handwriting_orientation import orient,transform_box
    original_height,original_width=image.shape[:2]
    page=text_ink_image(image,detections)
    image,page,detections,rotation=orient(image,page,detections,refs)
    regions=text_line_regions(page,detections)
    # Neural text proposals retain a whole word even when its zero touches
    # a wire and disappears from the connected-character grouping.
    for proposal in detected_text_regions:
        if float(proposal.get('confidence',0))<.60:continue
        box=proposal.get('bbox',proposal)
        if rotation:box=transform_box(box,rotation,original_height,original_width)
        x1=max(0,int(box['x1'])-3);y1=max(0,int(box['y1'])-3)
        x2=min(image.shape[1],int(np.ceil(box['x2']))+4);y2=min(image.shape[0],int(np.ceil(box['y2']))+4)
        if x2-x1<3 or y2-y1<4:continue
        candidate=dict(x1=x1,y1=y1,x2=x2,y2=y2,region_source='trained_whole_text_detector')
        area=(x2-x1)*(y2-y1)
        overlaps=lambda b:max(0,min(x2,b['x2'])-max(x1,b['x1']))*max(0,min(y2,b['y2'])-max(y1,b['y1']))/max(1,min(area,(b['x2']-b['x1'])*(b['y2']-b['y1'])))
        duplicates=[b for b in regions if overlaps(b)>.75]
        if duplicates:
            # Replace a partial connected blob only when the complete neural
            # word is of a comparable size, never a page-sized text proposal.
            if all(area<3*max(1,(b['x2']-b['x1'])*(b['y2']-b['y1'])) for b in duplicates):
                regions=[b for b in regions if b not in duplicates];regions.append(candidate)
        else:regions.append(candidate)
    from backend.crowded_text import recover_regions,schedule_regions
    regions=recover_regions(page,regions,detections)
    from backend.handwriting_layout import join_stacked_units
    regions=join_stacked_units(image,regions,refs)
    from backend.handwriting_layout import split_component_inscriptions
    regions=split_component_inscriptions(regions,detections)
    from backend.handwriting_layout import join_horizontal_units
    regions=join_horizontal_units(image,regions,refs,detections)
    if detections:regions,_=prioritized_regions(page,regions,detections)
    regions=schedule_regions(regions)
    from backend.local_word_ocr import read_words,accept_word
    whole_readings=read_words([image[int(b['y1']):int(b['y2']),int(b['x1']):int(b['x2'])] for b in regions])
    from backend.domain_word_ocr import read_words as read_circuit_words,choose_word
    domain_indices=[i for i,b in enumerate(regions) if b.get('region_source')=='trained_whole_text_detector']
    domain_values=read_circuit_words([image[int(regions[i]['y1']):int(regions[i]['y2']),int(regions[i]['x1']):int(regions[i]['x2'])] for i in domain_indices])
    domain_readings=dict(zip(domain_indices,domain_values))
    from backend.annotation_word_reader import read_regions
    precise_readings=read_regions(image,regions,detections,raw_general=whole_readings,raw_domain=domain_readings,text_page=page)
    tokens=[];pending=[]
    typical=float(np.median([b['y2']-b['y1'] for b in regions])) if regions else 20.
    for i,box in enumerate(regions):
        report('Lendo referências, valores e modelos…',i,len(regions))
        x1,y1,x2,y2=(int(box[k]) for k in ('x1','y1','x2','y2'))
        if y2-y1<(max(5,typical*.25) if box.get('region_source')=='component_neighborhood' and len(detections)>=20 else typical*.42):continue
        context=box.get('component_context') or {};body_fragment=False
        if context.get('distance')==0:
            det=next((d for d in detections if d['detection_id']==context.get('detection_id')),None)
            if det:
                b=det['bbox'];area=max(1,(x2-x1)*(y2-y1))
                overlap=max(0,min(x2,b['x2'])-max(x1,b['x1']))*max(0,min(y2,b['y2'])-max(y1,b['y1']))/area
                _,_,stats,_=cv2.connectedComponentsWithStats((page[y1:y2,x1:x2]<128).astype(np.uint8),8)
                substantial=[s for s in stats[1:] if s[4]>=max(8,(y2-y1)*.2)]
                body_fragment=overlap>.72 and len(substantial)<=2
        # Region detection can use a strict circuit-ink mask. Reading uses the
        # actual photograph so weak pen edges are retained inside each crop.
        mask=box.get('prepared_mask')
        if mask is None:mask=ink(image[y1:y2,x1:x2])
        kind=(box.get('component_context') or {}).get('kind')
        if kind=='opamp' and not body_fragment:
            det=next((d for d in detections if d.get('detection_id')==context.get('detection_id')),None)
            if det:
                b=det['bbox'];cx,cy=(x1+x2)/2,(y1+y2)/2
                if not (b['x1']<=cx<=b['x2'] and b['y1']<=cy<=b['y2']):
                    mask=mask.copy();ax,ay=max(0,int(b['x1'])-x1),max(0,int(b['y1'])-y1)
                    bx,by=min(mask.shape[1],int(b['x2'])-x1),min(mask.shape[0],int(b['y2'])-y1)
                    if ax<bx and ay<by:mask[ay:by,ax:bx]=0
        nearby=[]
        for d in detections:
            b=d['bbox'];dx=max(b['x1']-x2,x1-b['x2'],0);dy=max(b['y1']-y2,y1-b['y2'],0)
            if math.hypot(dx,dy)<max(70,min(180,max(b['x2']-b['x1'],b['y2']-b['y1'])*1.8)):nearby.append(d['class'])
        from backend.handwriting_values import read_photo_value,UNITS
        value_kind=kind if kind in UNITS else 'resistor' if 'resistor' in nearby and kind=='opamp' else None
        neural_word=(precise_readings.get(i) or choose_word(whole_readings[i],domain_readings.get(i))) if not body_fragment else None
        if body_fragment or context.get('distance')==0 and kind in {'diode','led'}:
            from backend.annotation_word_reader import coil_reference_in_conflicting_body
            det=next((d for d in detections if d.get('detection_id')==context.get('detection_id')),None)
            rescued=coil_reference_in_conflicting_body(image,det,whole_readings[i],domain_readings.get(i))
            if rescued:neural_word=rescued;body_fragment=False
        if neural_word and neural_word.get('ignored'):continue
        result=neural_word or read_line(mask,refs,kind if not body_fragment else None,nearby if not body_fragment else (),composite=not body_fragment);variant='photograph_color'
        from backend.measurement_text import parse_measurement
        measurement=parse_measurement(result.get('text',''))
        from backend.handwriting_values import complete_reading_has_priority
        protected_reading=complete_reading_has_priority(result,kind)
        check_prefix=bool(not protected_reading and measurement and measurement['unit_explicit'] and not measurement['prefix'])
        if result.get('unresolved') or check_prefix:
            filtered=(page[y1:y2,x1:x2]<128).astype(np.uint8)*255
            retry=read_line(filtered,refs,kind if not body_fragment else None,nearby if not body_fragment else (),composite=not body_fragment)
            alternative=parse_measurement(retry.get('text',''))
            better_prefix=bool(check_prefix and alternative and alternative['prefix'] and alternative['unit']==measurement['unit'] and alternative['numeric_reading']==measurement['numeric_reading'] and retry.get('confidence',0)>=result.get('confidence',0)-5)
            if not retry.get('unresolved') and (result.get('unresolved') or better_prefix):
                result=retry;variant='filtered_ink_fallback'
        # A unit touching a component's lead is excluded from page-wide blob
        # grouping. Reread the complete annotation up to the body edge, using
        # the photograph rather than the already truncated connected blob.
        if kind in {'resistor','capacitor','capacitor_polarized','inductor','vcc','source_dc','battery'}:
            det=next((d for d in detections if d.get('detection_id')==context.get('detection_id')),None)
            if det and not body_fragment and (result.get('unresolved') or not measurement):
                b=det['bbox'];height=y2-y1;edge=int(b['x1'])-2
                if 0<edge-x2<height*.9 and b['y1']-height<y2 and y1<b['y2']+height:
                    extended=read_line(ink(image[y1:y2,x1:edge]),refs,kind,nearby,False)
                    value=parse_measurement(extended.get('text',''))
                    if value and value['unit_explicit'] and (not extended.get('unresolved') or extended.get('needs_review') and extended.get('confidence',0)>=64):
                        if extended.get('unresolved'):
                            extended=dict(extended,unresolved=False,needs_review=True,numeric_alternatives=extended.get('alternatives',[]))
                        result=extended;box=dict(box,x2=edge);variant='unit_recovered_beside_component_lead'
        # A digit touching a nearby wire can disappear from global connected
        # components. Recover the complete identifier beside its initial.
        whole_letter=candidates(_clean_line(mask)[0],refs)
        letters=[r for r in whole_letter if r['text'] in {'R','r','C','c','L'} and r['score']>=(.45 if kind=='resistor' and r['text'] in {'R','r'} else .52) and r['score']>=whole_letter[0]['score']-(.30 if kind=='resistor' else .22)]
        if not body_fragment and letters and x2-x1<1.7*(y2-y1):
            ex1=max(0,x1-round((y2-y1)*.15));ey1=max(0,y1-round((y2-y1)*.50))
            ex2=min(image.shape[1],x2+round((y2-y1)*1.25));ey2=min(image.shape[0],y2+round((y2-y1)*.80))
            extended=read_line(ink(image[ey1:ey2,ex1:ex2]),refs,kind,nearby,False)
            prefix=letters[0]['text'].upper()
            if (re.fullmatch(prefix+r'\d{1,3}',extended.get('text','')) and not extended.get('unresolved')
                and (extended['confidence']>result.get('confidence',0)+3 or result.get('unresolved') and extended['confidence']>=65)):
                result=extended;box=dict(box,x1=ex1,y1=ey1,x2=ex2,y2=ey2);variant='complete_identifier_neighborhood'
        if re.fullmatch(r'1[nN]\d{3,5}',result.get('text','')):
            ex2=min(image.shape[1],x2+round((y2-y1)*.9))
            extended=read_line(ink(image[y1:y2,x1:ex2]),refs,kind,nearby,False)
            if (re.fullmatch(r'1[nN]\d{4,5}',extended.get('text','')) and not extended.get('unresolved')
                and extended['confidence']>=result.get('confidence',0)-5):
                result=extended;box=dict(box,x2=ex2);variant='complete_model_neighborhood'
        # Keep the established full-unit recovery first. Apply the numeric
        # classifier once the complete source annotation has been recovered.
        prior=parse_measurement(result.get('text',''))
        trusted=bool(prior and prior.get('unit_explicit')
                     and (not result.get('unresolved') and result.get('confidence',0)>=70
                          or prior.get('prefix') and (result.get('reading_method')!='touching_unit_reference_composition' or prior.get('unit')!='volt')))
        from backend.handwriting_values import prefix_needs_photographic_recheck
        unusual_prefix=prefix_needs_photographic_recheck(result,kind)
        if unusual_prefix:trusted=False
        protected_reading=complete_reading_has_priority(result,kind)
        numeric=read_photo_value(image,box,detections,refs,value_kind) if not protected_reading and not body_fragment and (not trusted or kind in {'vcc','source_dc','battery','resistor'}) else None
        if numeric:
            replacement=parse_measurement(numeric['text'],kind=value_kind)
            prefix_correction=bool(kind=='resistor' and prior and not prior.get('prefix') and replacement and replacement.get('prefix') in {'k','M'}
                                   and numeric['reading_method']=='trained_digits_component_unit_prefix' and numeric['confidence']>=80)
            voltage_digits=bool(kind in {'vcc','source_dc','battery'} and numeric['confidence']>=80 and numeric.get('numeric_anchor_margin',0)>=3)
            complete_zero=bool(kind=='resistor' and prior and replacement and prior.get('prefix')==replacement.get('prefix')
                               and numeric['reading_method']=='trained_digits_component_unit_prefix' and numeric['confidence']>=82
                               and numeric.get('numeric_anchor_margin',0)>=3 and prior['numeric_reading'].isdigit() and replacement['numeric_reading'].isdigit()
                               and replacement['numeric_reading'].startswith(prior['numeric_reading'])
                               and set(replacement['numeric_reading'][len(prior['numeric_reading']):])=={'0'})
            embedded_correction=bool(numeric.get('reading_method')=='trained_digits_embedded_prefix_and_unit'
                                     and numeric['confidence']>=result.get('confidence',0)+8)
            leading_correction=bool(numeric.get('leading_digit_verified') and prior and replacement
                                    and prior['unit']==replacement['unit'] and prior.get('prefix')==replacement.get('prefix')
                                    and prior['numeric_reading'][1:]==replacement['numeric_reading'][1:])
            if not trusted or prefix_correction or voltage_digits or complete_zero or embedded_correction or leading_correction:result=numeric;variant='complete_numeric_annotation'
        elif unusual_prefix:
            result=dict(result,needs_review=True,prefix_recheck_unresolved=True)
        explicit_value=parse_measurement(result.get('text',''))
        from backend.reference_labels import reference_classes
        compatible_names=reference_classes(result.get('text',''))
        complete_name=bool(compatible_names) and (kind is None or kind in compatible_names or result.get('conflicting_body_reference_evidence')) and not result.get('unresolved') and not result.get('inferred_prefix')
        inferred=None if complete_name or explicit_value and explicit_value.get('unit_explicit') else class_reference(mask,refs,kind)
        if inferred and not re.fullmatch(r'1[nN]\d{3,5}',result.get('text','')):
            from backend.annotation_word_reader import verify_class_prefix
            result=verify_class_prefix(image[y1:y2,x1:x2],inferred,domain_readings.get(i))
        if result.get('ignored'):continue
        if body_fragment and (result.get('unresolved') or not result.get('reading_method')):continue
        if context.get('distance')==0 and re.fullmatch(r'[RCLB]\d+',result.get('text','')):
            allowed={'resistor':'R','capacitor':'C','capacitor_polarized':'C','inductor':'L','battery':'B'}.get(kind)
            if (not allowed or result['text'][0]!=allowed or result['confidence']<75) and not result.get('conflicting_body_reference_evidence'):continue
        clean_box={k:v for k,v in box.items() if k!='prepared_mask'}
        original_box=transform_box(clean_box,(-rotation)%4,*image.shape[:2]) if rotation else clean_box
        item=dict(result,bbox=original_box,line_region_id=i,source='handwriting_reference',reference_bank_version=manifest.get('version'),reader_version=READER_VERSION,mask_variant=variant,text_rotation_degrees=rotation*90,component_context=box.get('component_context'))
        if result['unresolved']:pending.append(item)
        else:tokens.append(item)
    from backend.annotation_recovery import recover_values
    report('Conferindo valores próximos dos componentes…')
    upright=[dict(t,bbox=transform_box(t['bbox'],rotation,original_height,original_width) if rotation else t['bbox']) for t in tokens]
    for item in recover_values(image,detections,upright,refs):
        item.update(reference_bank_version=manifest.get('version'),reader_version=READER_VERSION,text_rotation_degrees=rotation*90)
        upright.append(item)
    from backend.physical_annotation_reader import recover as recover_physical_words
    # Body-free verification needs the same upright coordinates as its image
    # and detections. Previously it was skipped for every sideways photograph.
    upright_pending=[dict(t,bbox=transform_box(t['bbox'],rotation,original_height,original_width) if rotation else t['bbox']) for t in pending]
    upright,upright_pending=recover_physical_words(image,detections,upright,upright_pending,refs,word_regions=regions)
    def original_coordinates(items):
        output=[]
        for t in items:
            item=dict(t)
            if rotation:
                item['bbox']=transform_box(item['bbox'],(-rotation)%4,*image.shape[:2])
                if item.get('replaced_readings'):
                    item['replaced_readings']=[dict(r,bbox=transform_box(r['bbox'],(-rotation)%4,*image.shape[:2])) for r in item['replaced_readings']]
            item.setdefault('text_rotation_degrees',rotation*90)
            output.append(item)
        return output
    final=original_coordinates(upright);uncertain=original_coordinates(upright_pending)
    from backend.inscription_guard import separate
    # Use the source orientation for both pixels and boxes.
    final,rejected=separate(np.rot90(image,(-rotation)%4).copy() if rotation else image,
        [dict(d,bbox=transform_box(d['bbox'],(-rotation)%4,*image.shape[:2])) for d in detections] if rotation else detections,final)
    uncertain.extend(rejected)
    return final,uncertain
