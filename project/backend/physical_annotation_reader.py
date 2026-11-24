"""Recover nearby words from physical symbol-free crops, without supplied text."""
import cv2,numpy as np
from collections import defaultdict

def _confidence(t):
    value=float(t.get('confidence',0));return value/100 if value>1 else value

def _overlap(a,b):
    inter=max(0,min(a['x2'],b['x2'])-max(a['x1'],b['x1']))*max(0,min(a['y2'],b['y2'])-max(a['y1'],b['y1']))
    return inter/max(1,min((a['x2']-a['x1'])*(a['y2']-a['y1']),(b['x2']-b['x1'])*(b['y2']-b['y1'])))

def _key(text,kind):
    if not isinstance(text,str) or not text.strip():return None
    from backend.reference_labels import reference_classes
    from backend.measurement_text import parse_measurement,DIMENSIONS
    classes=reference_classes(text)
    if kind in classes:return ('reference',''.join(text.split()).upper())
    from backend.component_models import model_ocr_reading
    model=model_ocr_reading(text)
    if kind in model['classes']:return ('model',model['text'])
    value=parse_measurement(text,kind)
    if value and value.get('unit')==DIMENSIONS.get(kind) and (value.get('unit_explicit') or value.get('prefix') or kind=='resistor' and __import__('re').fullmatch(r'\d+(?:[.,]\d+)?',text.strip())):
        return ('value',value['unit'],value['normalized'])
    return None

def word_consensus(rows,kind):
    groups=defaultdict(list)
    for engine,variant,row in rows:
        if not row:continue
        key=_key(row.get('text',''),kind)
        if key and row.get('confidence',0)>=.70 and row.get('min_character_confidence',0)>=.40:groups[key].append((engine,variant,row))
    accepted=[]
    for key,items in groups.items():
        if key[0]=='model' and not any(not variant.startswith('edge') for _,variant,_ in items):continue
        strong=[r for r in items if r[2]['confidence']>=.99 and r[2]['min_character_confidence']>=.94]
        independent=len({r[0] for r in items if r[2]['confidence']>=.75})>=2
        precise=[r for r in items if r[0]=='circuit_word' and r[2]['confidence']>=.97 and r[2]['min_character_confidence']>=.97]
        repeated=len({r[1] for r in items if r[0]=='circuit_word' and r[2]['confidence']>=.95 and r[2]['min_character_confidence']>=.95})>=2
        from backend.component_models import CATALOG
        model_support=[r for r in items if r[2]['confidence']>=.90 and r[2]['min_character_confidence']>=.80]
        model_agreement=key[0]=='model' and key[1] in CATALOG and len({r[0] for r in model_support})>=2
        # PARSeq reports a product over all characters, while the general
        # reader reports a mean. A long code can have sound character scores
        # and still fail the short-word probability gate. Require repeated,
        # exact agreement from both readers, including strong PARSeq letters.
        model_general={v for e,v,r in items if e=='general_word' and r['confidence']>=.82 and r['min_character_confidence']>=.40}
        model_domain={v for e,v,r in items if e=='circuit_word' and r['confidence']>=.70 and r['min_character_confidence']>=.80}
        untrimmed_model=any(e=='general_word' and not v.startswith('edge') and r['confidence']>=.82 and r['min_character_confidence']>=.40 for e,v,r in items)
        repeated_model=key[0]=='model' and len(str(key[1]))>=5 and len(model_general)>=2 and len(model_domain)>=2 and untrimmed_model
        stable_engines={engine for engine in {r[0] for r in items} if len({v for e,v,row in items if e==engine and row['confidence']>=.80 and row['min_character_confidence']>=.80})>=2}
        stable_support=[r for r in items if r[2]['confidence']>=.95 and r[2]['min_character_confidence']>=.85]
        dimensioned=key[0]=='value' and any((p:=__import__('backend.measurement_text',fromlist=['parse_measurement']).parse_measurement(r[2]['text'],kind)) and (p.get('unit_explicit') or p.get('prefix')) for r in items)
        stable_value=dimensioned and len(stable_engines)>=2 and bool(stable_support)
        # A small complete prefixed value may have one faint character. Accept
        # it only when both independent readers agree and a second image view
        # repeats that same value; never use this gate for bare symbol digits.
        repeated_views={engine for engine in {r[0] for r in items} if len({v for e,v,row in items if e==engine and row['confidence']>=.85 and row['min_character_confidence']>=.60})>=2}
        corroborated_value=dimensioned and independent and bool(repeated_views) and any(
            r[2]['confidence']>=.88 and r[2]['min_character_confidence']>=.80 for r in items)
        original_rule=bool(strong or independent and precise or repeated and any(r[2]['confidence']>=.98 and r[2]['min_character_confidence']>=.98 for r in precise))
        if not original_rule and not model_agreement and not repeated_model and not stable_value and not corroborated_value:continue
        best=max(strong or precise or (model_support if model_agreement else stable_support) or items,key=lambda r:r[2]['confidence'])
        accepted.append((len({r[0] for r in items}),best[2]['confidence'],key,best,items,not original_rule))
    accepted.sort(key=lambda r:(r[0],r[1]),reverse=True)
    if not accepted or len(accepted)>1 and accepted[1][0]==accepted[0][0] and accepted[1][1]>=accepted[0][1]-.02:return None
    _,_,key,best,items,corroborated=accepted[0]
    return dict(text=best[2]['text'],confidence=best[2]['confidence']*100,min_character_confidence=best[2]['min_character_confidence'],unresolved=False,
                reading_method='physical_crop_whole_word_consensus',word_evidence=[dict(engine=e,variant=v,**r) for e,v,r in items],reference_ids=[],
                independent_readers=len({r[0] for r in items}),needs_review=corroborated,
                evidence_rule='independent_readers_and_repeated_observed_word' if corroborated else 'precise_whole_word')

def complete_word_proposals(local,global_regions):
    """Retain a complete row when the page detector has an ambiguous owner.

    Symbol-centered lanes establish candidate owners. The page detector may
    provide a better word boundary. Both hypotheses are read; the final typed
    reference assignment decides ownership, rather than discarding the word.
    """
    output=[]
    for box in global_regions:
        context=box.get('component_context') or {}
        for fragment in local:
            owner=fragment.get('component_context') or {}
            if owner.get('detection_id')==context.get('detection_id'):continue
            h=fragment['y2']-fragment['y1'];height=box['y2']-box['y1']
            vertical=max(0,min(box['y2'],fragment['y2'])-max(box['y1'],fragment['y1']))
            if not .75<=height/max(1,h)<=1.35 or vertical<.80*min(height,h):continue
            if _overlap(box,fragment)<.65:continue
            output.append(dict(box,component_context=dict(owner,word_boundary_alternative=True,
                original_region_owner=context.get('detection_id'))))
    return output

def same_region_candidate(a,b):
    return ((a.get('component_context') or {}).get('detection_id')==(b.get('component_context') or {}).get('detection_id')
        and _overlap(a,b)>.93 and abs((a['x1']+a['x2'])-(b['x1']+b['x2']))<8)


def requires_confirmed_reference(reading,kind):
    """Symbol-like bare digits and glyph-only guesses need a local name.

    Complete dimensioned words read by the word readers remain independent of
    reference recognition. Individual-glyph calibration is weaker evidence.
    """
    from backend.measurement_text import parse_measurement
    parsed=parse_measurement(reading.get('text',''),kind)
    if not parsed:return False
    bare=kind=='resistor' and not parsed.get('prefix') and not parsed.get('unit_explicit')
    return bool(bare or reading.get('calibrated_character_evidence'))

def prioritize_reference_recrops(proposals,tokens):
    """Read complete rows near a weak photographed name before body fragments.

    This schedules existing pixels; the readers still supply every character.
    """
    from backend.reference_labels import reference_classes
    weak=[t for t in tokens if .55<=_confidence(t)<.95 and reference_classes(t.get('text','')) and t.get('component_context')]
    def rank(box):
        cid=(box.get('component_context') or {}).get('detection_id')
        for t in weak:
            if t['component_context'].get('detection_id')!=cid:continue
            a=t['bbox'];h=max(1,a['y2']-a['y1']);height=box['y2']-box['y1']
            vertical=max(0,min(a['y2'],box['y2'])-max(a['y1'],box['y1']))
            if .65<=height/h<=1.4 and vertical>=.8*min(h,height) and _overlap(a,box)>=.65:return 0
        return 1
    return sorted(proposals,key=rank)


def recover(image,detections,tokens,pending,refs,word_regions=()):
    from backend.drawing_mode import CURRENT
    if CURRENT.get()!='handwritten':return tokens,pending
    from backend.reference_labels import reference_classes,local_distance
    from backend.measurement_text import parse_measurement,DIMENSIONS
    model_kinds={'diode','transistor_npn','transistor_pnp','mosfet_n','mosfet_p','opamp'}
    supported={'resistor','capacitor','capacitor_polarized','inductor','vcc','source_dc','battery'}|model_kinds;targets={}
    for d in detections:
        kind=d.get('class')
        if kind not in supported:continue
        local=[t for t in tokens if (t.get('component_context') or {}).get('detection_id')==d['detection_id']]
        name=kind in {'vcc','source_dc','battery'} or any(kind in reference_classes(t.get('text','')) and _confidence(t)>=.95 and not t.get('inferred_prefix') for t in local)
        if kind in model_kinds:
            from backend.component_models import model_classes
            model=any(kind in model_classes(t.get('text','')) and _confidence(t)>=.95 for t in local)
            # A first region owner can be wrong in a crowded drawing. The
            # typed D/Q/U reference assignment occurs later; its absence here
            # must not prevent looking for an actual model beside the symbol.
            nearby_typed=any(_confidence(t)>=.55 and local_distance(t,d) is not None and
                (kind in reference_classes(t.get('text','')) or kind in model_classes(t.get('text',''))) for t in tokens+pending)
            if not model and (name or nearby_typed):targets[d['detection_id']]=d
            continue
        value=any(not reference_classes(t.get('text','')) and (p:=parse_measurement(t.get('text',''),kind)) and p.get('unit')==DIMENSIONS.get(kind) and _confidence(t)>=.95 and not t.get('needs_review') for t in local)
        decimal_prefix=any((p:=parse_measurement(t.get('text',''),kind)) and p.get('unit') in {'henry','farad'} and not p.get('prefix') and '.' in str(p.get('numeric_reading','')) for t in local)
        if not name or not value or decimal_prefix:targets[d['detection_id']]=d
    if not targets:return tokens,pending
    from backend.annotation_lanes import regions
    from backend.ocr_regions import text_ink_image,text_line_regions
    from backend.ocr_context import prioritized_regions
    cleaned,proposals=regions(image,detections)
    page=text_ink_image(image,detections);page[np.all(cleaned==255,axis=2)&np.any(image!=255,axis=2)]=255
    extra,_=prioritized_regions(page,text_line_regions(page,detections),detections)
    proposals.extend(complete_word_proposals(proposals,extra))
    proposals.extend(b for b in extra if b.get('component_context'))
    proposals.extend(dict(t['bbox'],component_context=t.get('component_context')) for t in tokens+pending if t.get('component_context'))
    for d in targets.values():
        if d['class'] not in model_kinds:continue
        for b in word_regions:
            if b.get('prepared_mask') is not None:continue
            distance=local_distance({'bbox':b},d)
            if distance is None or distance>max(45,min(100,2*(b['y2']-b['y1']))):continue
            if b['x2']-b['x1']<1.25*(b['y2']-b['y1']):continue
            proposals.append(dict(b,component_context=dict(detection_id=d['detection_id'],kind=d['class'],ambiguous=False,typed_neighbour_recheck=True)))
    proposals=prioritize_reference_recrops(proposals,tokens)
    # Whole-word boundaries from the page detector can cross an initially
    # incorrect symbol owner. Read these model candidates before spending
    # the bounded retry budget on fragments from symbol-centred lanes.
    proposals.sort(key=lambda b: 0 if (b.get('component_context') or {}).get('typed_neighbour_recheck') and targets.get((b.get('component_context') or {}).get('detection_id'),{}).get('class') in model_kinds else 1)
    selected=[];counts=defaultdict(int)
    for box in proposals:
        cid=(box.get('component_context') or {}).get('detection_id')
        if cid not in targets or counts[cid]>=7:continue
        x1,y1,x2,y2=(int(box[k]) for k in ('x1','y1','x2','y2'))
        if min(x2-x1,y2-y1)<9 or y2-y1>85 or x2-x1>6*(y2-y1):continue
        if any(same_region_candidate(box,b) for b in selected):continue
        selected.append(dict(box));counts[cid]+=1
    from backend.local_word_ocr import read_words as general
    from backend.domain_word_ocr import read_words as domain
    originals=[];original_indexes=[]
    for i,b in enumerate(selected):
        x,y,z,t=(int(b[k]) for k in ('x1','y1','x2','y2'));roi=cleaned[y:t,x:z]
        if not roi.size:continue
        originals.append(roi);original_indexes.append(i)
    readings=defaultdict(list)
    for i,g,d in zip(original_indexes,general(originals),domain(originals)):
        readings[i].extend([('general_word','original',g),('circuit_word','original',d)])
    patches=[];indexes=[]
    for i,roi in zip(original_indexes,originals):
        kind=targets[selected[i]['component_context']['detection_id']]['class']
        first=readings[i]
        model_word=kind in model_kinds and any(r and len(r.get('text',''))>=4 and sum(c.isdigit() for c in r['text'])>=2 and any(c.isalpha() for c in r['text']) for _,_,r in first)
        # Two precise, independent readers already agree on the actual word.
        # Reserve extra image variants for words that need another observation.
        if (all(r and r.get('confidence',0)>=.99 and r.get('min_character_confidence',0)>=.94 for _,_,r in first)
                and first[0][2]['text']==first[1][2]['text']):
            continue
        variants=[('contrast',cv2.cvtColor(cv2.normalize(cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY),None,0,255,cv2.NORM_MINMAX),cv2.COLOR_GRAY2BGR)),('sharpened',cv2.addWeighted(roi,1.8,cv2.GaussianBlur(roi,(0,0),1),-.8,0))]
        box=selected[i];x,y,z,t=(int(box[k]) for k in ('x1','y1','x2','y2'))
        pad=max(3,min(6,round((t-y)*.10)))
        loose=cleaned[max(0,y-pad):min(cleaned.shape[0],t+pad),max(0,x-pad):min(cleaned.shape[1],z+pad)]
        if loose.size and model_word:
            variants.append(('complete_margin',cv2.copyMakeBorder(loose,3,3,3,3,cv2.BORDER_CONSTANT,value=(255,255,255))))
            gray=cv2.normalize(cv2.cvtColor(loose,cv2.COLOR_BGR2GRAY),None,0,255,cv2.NORM_MINMAX)
            variants.append(('complete_margin_contrast',cv2.copyMakeBorder(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),3,3,3,3,cv2.BORDER_CONSTANT,value=(255,255,255))))
        if model_word:
            variants.append(('white_margin',cv2.copyMakeBorder(roi,3,3,3,3,cv2.BORDER_CONSTANT,value=(255,255,255))))
        # Read a model's edge separately when a neighbouring vertical lead
        # contaminates the word. The untrimmed word must independently agree;
        # these hypotheses cannot supply a missing final digit on their own.
        if model_word:
            for fraction in (.08,.16):
                for label,source in [('edge',loose),('edge_original',roi)]:
                    end=source.shape[1]-round(source.shape[0]*fraction)
                    if end<source.shape[1]*.75:continue
                    crop=source[:,:end]
                    gray=cv2.normalize(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),None,0,255,cv2.NORM_MINMAX)
                    variants.append((label+str(fraction),cv2.copyMakeBorder(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),3,3,3,3,cv2.BORDER_CONSTANT,value=(255,255,255))))
        for variant,patch in variants:patches.append(patch);indexes.append((i,variant))
    a=general(patches);b=domain(patches)
    for (i,variant),g,d in zip(indexes,a,b):readings[i].extend([('general_word',variant,g),('circuit_word',variant,d)])
    recovered=[];numeric_attempts=defaultdict(int)
    for i,box in enumerate(selected):
        context=box['component_context'];kind=targets[context['detection_id']]['class'];reading=word_consensus(readings[i],kind)
        parsed=parse_measurement(reading.get('text',''),kind) if reading else None
        decimal_prefix=bool(parsed and parsed.get('unit') in {'henry','farad'} and not parsed.get('prefix') and '.' in str(parsed.get('numeric_reading','')))
        texts=[str(r.get('text','')) for _,_,r in readings[i] if r]
        import re
        value_shape=any(len(text.strip())>=2 and re.match(r'^(?:\d|[TIli](?:[kKM]|[ΩΩ]))',text) for text in texts)
        if kind in {'resistor','capacitor','capacitor_polarized','inductor'} and value_shape and (reading is None or decimal_prefix) and numeric_attempts[context['detection_id']]<2:
            x,y,z,t=(int(box[k]) for k in ('x1','y1','x2','y2'));roi=cleaned[y:t,x:z]
            calibrated=calibrated_numeric(roi,refs,kind)
            numeric_attempts[context['detection_id']]+=1
            if calibrated:
                if reading and _key(reading['text'],kind)!=_key(calibrated['text'],kind):calibrated.update(needs_review=True,neural_alternative=reading['text'])
                reading=calibrated
        if not reading:continue
        recovered.append(dict(reading,bbox={k:box[k] for k in ('x1','y1','x2','y2')},source='handwriting_reference',component_context=dict(context,kind=kind,ambiguous=False),mask_variant='verified_body_removed_only',physical_crop=True))
    output=list(tokens);review=list(pending)
    named={}
    for t in list(tokens)+recovered:
        cid=(t.get('component_context') or {}).get('detection_id')
        if cid in targets and _confidence(t)>=.95 and targets[cid]['class'] in reference_classes(t.get('text','')):
            named[cid]=True
    for item in sorted(recovered,key=lambda t:-t['confidence']):
        context=item['component_context']
        if requires_confirmed_reference(item,context['kind']) and not named.get(context['detection_id']):
            review.append(dict(item,unresolved=True,reason='symbol_like_numeric_crop_without_confirmed_local_reference'))
            continue
        if any(_key(t.get('text',''),item['component_context']['kind'])==_key(item['text'],item['component_context']['kind']) and _overlap(t['bbox'],item['bbox'])>.6 and _confidence(t)>=_confidence(item) for t in output):continue
        replaced=[]
        for t in output:
            calibrated=bool(item.get('calibrated_character_evidence'))
            if _overlap(t['bbox'],item['bbox'])>=.60 and (_confidence(t)<.95 and _confidence(item)>=.97 or calibrated and _confidence(t)<.90):
                replaced.append(t)
        if replaced:
            item['replaced_readings']=[dict(text=t['text'],confidence=t.get('confidence'),bbox=t['bbox']) for t in replaced]
            output=[t for t in output if all(t is not old for old in replaced)]
            review.extend(dict(t,unresolved=True,reason='replaced_by_complete_symbol_free_photographic_word') for t in replaced)
        output.append(item)
    return output,review


def calibrated_numeric(crop,refs,kind):
    """A supplied individual-glyph match can resolve a whole-word ambiguity."""
    if not crop.size:return None
    from backend import handwriting_reference as hw
    from backend.handwriting_values import read_value
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    rows=[]
    for variant,mask in [('photographic_color',hw.ink(crop)),('photographic_contrast',cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1])]:
        row=read_value(mask,refs,kind)
        if row and not row.get('unresolved') and _key(row.get('text',''),kind):rows.append((variant,row))
    grouped=defaultdict(list)
    for variant,row in rows:grouped[_key(row['text'],kind)].append((variant,row))
    eligible=[]
    for key,items in grouped.items():
        supplied=[r for _,r in items if any('PHOTO' in s for s in r.get('reference_ids',[])) and _confidence(r)>=.80]
        if not supplied:continue
        best=max(supplied,key=_confidence)
        alternatives=[r for other,group in grouped.items() if other!=key for _,r in group]
        if len(items)<2 and alternatives and _confidence(best)<max(_confidence(r) for r in alternatives)+.045:continue
        eligible.append((len(items),_confidence(best),best,items))
    if not eligible:return None
    eligible.sort(key=lambda r:(r[0],r[1]),reverse=True);_,_,best,items=eligible[0]
    result=dict(best,reading_method='physical_crop_calibrated_character_and_observed_unit',numeric_focus_verified=True,
                calibrated_character_evidence=[dict(variant=v,text=r['text'],confidence=float(r['confidence']),reference_ids=r.get('reference_ids',[])) for v,r in items])
    if len(grouped)>1:result.update(needs_review=True,alternatives=list(dict.fromkeys([best['text']]+[r['text'] for _,r in rows])))
    return result
