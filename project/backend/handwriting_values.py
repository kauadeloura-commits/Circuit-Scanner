"""Complete numeric annotations with observed SI prefixes, never supplied digits."""
import re
from copy import deepcopy
import cv2
import numpy as np

UNITS={'resistor':'Ω','capacitor':'F','capacitor_polarized':'F','inductor':'H',
       'vcc':'V','source_dc':'V','battery':'V'}
_READINGS={}

def complete_reading_has_priority(reading,kind):
    """Keep a complete reference or calibrated word ahead of digit retries."""
    if reading.get('unresolved'):return False
    text=str(reading.get('text','')).strip()
    prefix={'resistor':'R','capacitor':'C','capacitor_polarized':'C','inductor':'L'}.get(kind)
    if prefix and re.fullmatch(prefix+r'\d{1,4}',text,re.I):return True
    method=reading.get('reading_method')
    if method=='nearby_whole_word_consensus':
        return float(reading.get('confidence',0))>=96 and (reading.get('independent_readers',0)>=2 or float(reading.get('confidence',0))>=99)
    if method=='complete_reference_and_value_inscription':return True
    from backend.measurement_text import parse_measurement
    if not parse_measurement(text,kind=kind):return False
    if method=='trained_whole_inscription_ctc':
        return float(reading.get('confidence',0))>=99 and float(reading.get('min_character_confidence',0))>=.85
    if method=='local_neural_whole_inscription':
        return float(reading.get('confidence',0))>=99 and float(reading.get('min_character_confidence',0))>=.94
    return method=='whole_verified_reference_token' and float(reading.get('confidence',0))>=94


def prefix_needs_photographic_recheck(reading,kind):
    """An unusual prefix triggers a fresh reading, never an invented value.

    kF and kH are valid SI expressions. A low-confidence reading of them in
    a small schematic must not bypass the prefix-aware photographic reader.
    """
    from backend.measurement_text import parse_measurement
    parsed=parse_measurement(reading.get('text',''),kind=kind)
    if not parsed or float(reading.get('confidence',0))>=95:return False
    common={'capacitor':{'','p','n','u','m'},'capacitor_polarized':{'','p','n','u','m'},'inductor':{'','n','u','m'}}
    return kind in common and parsed.get('prefix','') not in common[kind]

def read_value(mask,refs,kind):
    if kind not in UNITS or not refs:return None
    from backend.glyph_classifier import model_stamp
    folder=refs[0][0].get('_bank_directory')
    key=(mask.shape,mask.tobytes(),kind,tuple(id(s[0]) for _,s in refs),model_stamp(folder) if folder else None)
    if key not in _READINGS:
        if len(_READINGS)>=512:_READINGS.pop(next(iter(_READINGS)))
        _READINGS[key]=_read_value(mask,refs,kind)
    return deepcopy(_READINGS[key])

def _read_value(mask,refs,kind):
    if kind not in UNITS or not refs:return None
    from backend import handwriting_reference as hw
    from backend.glyph_classifier import model_stamp,support
    folder=refs[0][0].get('_bank_directory',hw.BANK)
    if not model_stamp(folder):return None
    clean,height=hw._clean_line(mask)
    if not np.any(clean):return None
    n,labels,stats,_=cv2.connectedComponentsWithStats(clean,8)
    main=[s for s in stats[1:] if s[3]>=height*.6 and s[4]>=20]
    keep=np.ones(n,np.uint8)*255;keep[0]=0
    for i,s in enumerate(stats[1:],1):
        x,y,w,h,area=s
        if h>height*.35:continue
        if any(area<m[4]*.18 and m[0]<=x+w/2<=m[0]+m[2] and 0<=m[1]-(y+h)<height*.5 for m in main):keep[i]=0
    clean=keep[labels]
    _,trained=hw.library(folder,trained=True)
    # For values, retain template rankings independently of the classifier.
    # Only a strongly corroborated digit may change that ranking below.
    trained=[(dict(e,_trained_branch=False),s) for e,s in trained]
    lat=hw._Segments(clean,trained,height);lat.reference_numbers=True;lat.value_numbers=True;lat.numeric_margin=.30
    # References do not need an SI-value lattice or a second threshold pass.
    for cut in lat.cuts:
        if cut<=lat.start or cut-lat.start>height*1.3:continue
        norm=hw.normalize(lat.crop(lat.start,cut))
        if norm:
            first=support(norm[0],norm[1],folder)
            if first and first['text'] in {'R','r','C','c','L'} and first['margin']>=1:return None
        break
    wanted=UNITS[kind];choices=[]
    prefixes={'K','k','M'} if kind=='resistor' else {'p','n','µ','m'}
    # A whole visible V/F/H/Ω is one glyph. Do not split it to manufacture mV.
    def glyph(a,b,allowed):
        if b<=a:return None
        actual=lat.crop(a,b)
        n,labels,stats,_=cv2.connectedComponentsWithStats(actual,8)
        substantial=[s for s in stats[1:] if s[3]>=height*.45]
        area=max((s[4] for s in substantial),default=0)
        if area:
            keep=np.array([0]+[255 if s[4]>=area*.14 or s[3]>=height*.45 else 0 for s in stats[1:]],np.uint8)
            actual=keep[labels]
        norm=hw.normalize(actual)
        if norm is None or not lat.single_glyph(a,b):return None
        gx,gy,gw,gh=cv2.boundingRect(cv2.findNonZero(actual))
        if gw>gh*1.4:return None
        learned=support(norm[0],norm[1],folder)
        restricted=support(norm[0],norm[1],folder,allowed=allowed)
        if restricted and restricted.get('font_specialist'):learned=restricted
        small_n=kind in {'capacitor','capacitor_polarized'} and 'n' in allowed and learned and learned['text']=='h' and gh<=height*1.25
        threshold=.6 if wanted in allowed else 3
        pool=[(e,s) for e,s in lat.refs if e['text'] in allowed]
        ranks=hw.candidates(actual,pool)
        found=[r for r in ranks if r['text'] in allowed]
        if not found:return None
        r=found[0];best=ranks[0]['score']
        if learned and learned['text'] not in allowed and learned['margin']>threshold and not small_n and r['score']<.85:return None
        corroborated=learned and learned['text'] in allowed and learned['margin']>=.5
        resistance_prefix=kind=='resistor' and r['text'] in {'K','k','M'}
        unit_context=wanted in allowed and r['text'].upper()==wanted
        minimum=.56 if unit_context and r['score']>=best-.04 else .58 if corroborated else .60 if resistance_prefix else .62 if small_n else .66
        if r['score']<minimum or r['score']<best-(.18 if corroborated else .08 if resistance_prefix else .055):return None
        if r['text'] in {'n','m'} and hw._holes(actual)>0:return None
        label=r['text'].upper() if r['text'] in {'k','v','f','h'} else r['text']
        return dict(r,text=label)
    def append(number,suffix,ids,scores,cost,grammar):
        if not re.fullmatch(r'\d+(?:\.\d+)?',number):return
        text=number+''.join(s['text'] for s in suffix)
        if len(number)>7:return
        confidence=float(np.mean(scores+[s['score'] for s in suffix]))
        if min(scores or [0])<.52 or confidence<.69:return
        loss=(cost+sum(1-s['score']+.08 for s in suffix))/(len(scores)+len(suffix))
        choices.append((loss,text,ids+[s['reference_id'] for s in suffix],scores+[s['score'] for s in suffix],grammar,
                        max((s.get('classifier_margin',0) for s in suffix),default=0)))
    cuts=lat.cuts
    for end in cuts:
        if end<=lat.start:continue
        suffix=glyph(end,lat.end,{wanted,wanted.lower()})
        if kind=='resistor' and end>lat.start:
            rank=lat.rank(end,lat.end)
            unit=next((r for r in rank if r['text']=='Ω'),None)
            if unit and unit['score']>=.55 and unit['score']>=rank[0]['score']-.2:
                # A handwritten 1 and the left arm of K can overlap. Follow
                # a low-ink seam instead of adding that arm to the digit.
                for pair in hw._separated_units(lat.crop(lat.start,end),lat.refs,'K',left_labels=set('0123456789')):
                    if not re.fullmatch(r'\dK',pair['text']):continue
                    text=pair['text']+'Ω';scores=[float(pair['score'])]*2+[float(unit['score'])]
                    loss=sum(1-s+.08 for s in scores)/3
                    choices.append((loss,text,pair['reference_id'].split('+')+[unit['reference_id']],scores,'overlapping_numeric_prefix_and_unit',0))
        joined=lat.crop(end,lat.end)
        if joined.size and np.any(joined) and lat.single_glyph(end,lat.end):
            _,_,jw,jh=cv2.boundingRect(cv2.findNonZero(joined))
            singles=lat.rank(end,lat.end)
            whole_unit=bool(jw<jh*1.35 and singles and any(r['text'] in {'V','v','F','f','H','h','Ω'} and r['score']>=.60 and r['score']>=singles[0]['score']-.05 for r in singles))
            if height*.7<jw<height*2.8 and not whole_unit:
                for combined in hw._separated_units(joined,lat.refs,wanted)[:1]:
                    if combined['text'][:-1] not in prefixes:continue
                    for cost,num,ids,scores in lat.digit_paths(end):
                        append(num,[combined],ids,scores,cost,'trained_digits_observed_prefix_and_unit')
        if suffix:
            for cost,num,ids,scores in lat.digit_paths(end):append(num,[suffix],ids,scores,cost,'trained_digits_observed_unit')
            for start in cuts:
                if not lat.start<start<end:continue
                prefix=glyph(start,end,prefixes)
                if prefix:
                    for cost,num,ids,scores in lat.digit_paths(start):append(num,[prefix,suffix],ids,scores,cost,'trained_digits_observed_prefix_and_unit')
            # IEC resistance notation embeds its decimal separator in the
            # prefix: 4K7Ω is 4.7 kΩ, not the unrelated integer 417Ω.
            # Every digit, prefix and final Ω must have observed ink.
            if kind=='resistor':
                for start in cuts:
                    if not lat.start<start<end:continue
                    for stop in cuts:
                        if not start<stop<end:continue
                        prefix=glyph(start,stop,{'K','k','M'})
                        if not prefix:continue
                        tail=hw._Segments(lat.crop(stop,end),trained,height);tail.reference_numbers=True;tail.value_numbers=True;tail.numeric_margin=.28
                        for cost,num,ids,scores in lat.digit_paths(start):
                            for tc,tn,ti,ts in tail.digit_paths(tail.end):
                                if not tn.isdigit() or len(tn)>2:continue
                                append(num,[prefix,dict(text=tn,score=float(np.mean(ts)),reference_id='+'.join(ti)),suffix],ids,scores,cost+tc,'trained_digits_embedded_prefix_and_unit')
        # Bare prefixes are conventional component values: 10K, 100n, 2K2.
        prefix=glyph(end,lat.end,prefixes)
        if prefix:
            for cost,num,ids,scores in lat.digit_paths(end):append(num,[prefix],ids,scores,cost,'trained_digits_component_unit_prefix')
        if kind=='resistor':
            joined=lat.crop(end,lat.end)
            if joined.size and np.any(joined) and lat.single_glyph(end,lat.end):
                jw=cv2.boundingRect(cv2.findNonZero(joined))[2]
                if height*1.3<jw<height*2.8:
                    right=hw.candidates(joined[:,max(0,round(jw*.45)):],lat.refs)
                    digits=[r['text'] for r in right if r['text'] in '0123456789'][:2]
                    for digit in digits:
                        for combined in hw._separated_units(joined,lat.refs,digit)[:1]:
                            if combined['text'] not in {'K'+digit,'M'+digit}:continue
                            norm=hw.normalize(joined[:,max(0,round(jw*.45)):]);vote=support(norm[0],norm[1],folder) if norm else None
                            combined=dict(combined,classifier_margin=vote['margin'] if vote and vote['text']==digit else 0)
                            for cost,num,ids,scores in lat.digit_paths(end):
                                append(num,[combined],ids,scores,cost,'trained_digits_embedded_resistance_prefix')
            for stop in cuts:
                if not end<stop<lat.end:continue
                prefix=glyph(end,stop,{'K','k','M'})
                if not prefix:continue
                tail=hw._Segments(lat.crop(stop,lat.end),trained,height);tail.reference_numbers=True;tail.value_numbers=True;tail.numeric_margin=.28
                for cost,num,ids,scores in lat.digit_paths(end):
                    for tc,tn,ti,ts in tail.digit_paths(tail.end):
                        if not tn.isdigit() or len(tn)>2:continue
                        norm=hw.normalize(tail.mask);vote=support(norm[0],norm[1],folder) if norm else None
                        append(num,[prefix,dict(text=tn,score=float(np.mean(ts)),reference_id='+'.join(ti),
                                                classifier_margin=vote['margin'] if vote and vote['text']==tn and (tn!='4' or vote['margin']>=4) else 0)],ids,scores,cost,'trained_digits_embedded_resistance_prefix')
    if not choices:return None
    choices.sort();best=choices[0]
    explicit=[c for c in choices if c[4]=='trained_digits_observed_prefix_and_unit' and c[0]<=best[0]+.04]
    if explicit:best=min(explicit)
    embedded=[c for c in choices if c[4] in {'trained_digits_embedded_resistance_prefix','trained_digits_embedded_prefix_and_unit'} and c[0]<=best[0]+.05]
    if embedded and not explicit:best=min(embedded)
    if best[0]>.34:return None
    result=hw._result(best[1],best[2],best[3],best[4]);result['trained_glyph_classifier']=True
    result['embedded_digit_margin']=best[5]
    major=sorted([s for s in lat.stats[1:] if s[3]>=height*.48 and s[4]>=12],key=lambda s:s[0])
    if major:
        first=major[0];stop=round((first[0]+first[2]+major[1][0])/2) if len(major)>1 else first[0]+first[2]
        norm=hw.normalize(lat.crop(lat.start,stop));anchor=support(norm[0],norm[1],folder) if norm else None
        result['numeric_anchor_margin']=anchor['margin'] if anchor and anchor['text']==best[1][0] else 0
    alternatives=list(dict.fromkeys(c[1] for c in choices if c[0]<best[0]+.018))
    if len(alternatives)>1:result.update(needs_review=True,alternatives=alternatives[:3])
    return result

def read_photo_value(image,box,detections,refs,kind):
    """Recover faded strokes around a text ROI, keeping component bodies out."""
    if kind not in UNITS:return None
    x1,y1,x2,y2=[int(box[k]) for k in ('x1','y1','x2','y2')]
    height=y2-y1;results=[]
    for pad in (0,max(3,round(height*.18))):
        ax,ay=max(0,x1-pad),max(0,y1-pad);bx,by=min(image.shape[1],x2+pad),min(image.shape[0],y2+pad)
        patch=image[ay:by,ax:bx];gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
        from backend.handwriting_reference import ink
        for mode,mask in (('color',ink(patch)),('contrast',cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1])):
            for d in detections:
                if d.get('class')=='opamp':continue
                b=d['bbox'];mx1,my1=max(0,int(b['x1'])-ax),max(0,int(b['y1'])-ay)
                mx2,my2=min(mask.shape[1],int(np.ceil(b['x2']))-ax),min(mask.shape[0],int(np.ceil(b['y2']))-ay)
                if mx1<mx2 and my1<my2:mask[my1:my2,mx1:mx2]=0
            reading=read_value(mask,refs,kind)
            if reading:results.append(dict(reading,photo_value_mask=mode,photo_value_padding=pad))
        if results and any(not r.get('needs_review') for r in results):break
    if not results:return None
    results.sort(key=lambda r:(r['reading_method'] not in {'trained_digits_embedded_prefix_and_unit','trained_digits_observed_prefix_and_unit','trained_digits_observed_unit'},
                               r.get('numeric_anchor_margin',0)<2,-r.get('embedded_digit_margin',0),bool(r.get('needs_review')),r['reading_method']!='trained_digits_observed_prefix_and_unit',
                               r['photo_value_mask']!='contrast',-r['confidence']))
    from backend.numeric_word_consensus import verify_leading_digit
    return verify_leading_digit(image,box,results[0],kind)
