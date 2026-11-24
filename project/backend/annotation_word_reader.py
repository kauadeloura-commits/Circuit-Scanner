"""Read complete nearby labels from original pixels with two local readers.

Different contrast views improve small/light digits without replacing them
with a common resistor value or guessing the next reference number.
"""
import re
import cv2
import numpy as np


def verify_class_prefix(crop,reading,domain=None):
    """A full photographed word can corroborate a separately read suffix."""
    if not reading.get('inferred_prefix'):return reading
    if domain is None:
        from backend.domain_word_ocr import read_words
        domain=read_words([crop])[0]
    if not domain:return reading
    expected=''.join(reading['text'].split()).upper()
    actual=''.join(domain.get('text','').split()).upper()
    if actual!=expected or domain.get('confidence',0)<.80 or domain.get('min_character_confidence',0)<.80:return reading
    result=dict(reading);result['proposed_prefix']=result.pop('inferred_prefix')
    result.update(prefix_source='complete_photographed_word',prefix_word_evidence=domain,
                  reading_method='whole_reference_word_and_independent_numeric_suffix',
                  confidence=min(float(reading['confidence']),float(domain['confidence'])*100))
    return result


def _key(text):
    from backend.reference_labels import reference_classes
    from backend.measurement_text import parse_measurement
    from backend.component_models import model_classes, model_name
    if reference_classes(text): return ('reference', ''.join(text.split()).upper())
    if len(text)<=80 and model_classes(text): return ('model', model_name(text))
    phrase=re.fullmatch(r'\s*((?:LED|BAT|BT|CP|VS|IC|[RCLDQTMUV])\s*\d{1,5})\s*[:=]?\s+(.+)',text,re.I)
    if phrase and reference_classes(phrase[1]):
        suffix=_key(phrase[2])
        if suffix and suffix[0] in {'value','implicit_value','model'}:
            return ('annotation',''.join(phrase[1].split()).upper(),suffix)
    value = parse_measurement(text)
    if value and value.get('unit_explicit'):
        return ('value', value['unit'], value['normalized'])
    # 4k7, 100k and 10u have an explicit prefix but no printed unit.
    # Keep the literal prefix; the nearby component supplies the dimension
    # only during association, never in this visual word comparison.
    value=parse_measurement(text,'resistor')
    if value and value.get('prefix'):
        return ('implicit_value',value['numeric_reading'],value['prefix'])
    return None


def coil_reference_in_conflicting_body(image,detection,general,domain):
    """A proven coil may release its confidently read L-number from a diode box.

    The later L-name/H-value pairing must still confirm the class change.
    A component's learned class cannot veto all contrary text evidence.
    """
    if not detection or detection.get('class') not in {'diode','led'}:return None
    reading=consensus([('general_word','original',general),('circuit_word','original',domain)])
    if not reading or reading['confidence']<99 or reading.get('min_character_confidence',0)<.94:return None
    from backend.reference_labels import reference_classes
    if reference_classes(reading['text'])!={'inductor'}:return None
    from backend.recognition import _validar_geometria_indutor
    valid,score,proof=_validar_geometria_indutor(image,detection['bbox'])
    if not valid or not proof.get('straight_tip_pair') or proof.get('curve_extrema',0)<6:return None
    return dict(reading,conflicting_body_reference_evidence=dict(previous_class=detection['class'],coil=proof))


def consensus(rows,kind=None):
    """Two independent readers must agree when either one is below its gate."""
    groups = {}
    for engine, variant, row in rows:
        if not row: continue
        key = _key(row.get('text', ''))
        if not key: continue
        confidence = float(row.get('confidence', 0))
        minimum = float(row.get('min_character_confidence', 0))
        if confidence < .96 or minimum < .85: continue
        groups.setdefault(key, []).append((engine, variant, row))
    accepted = []
    for key, support in groups.items():
        engines = {r[0] for r in support}
        strong = [r for r in support if r[2]['confidence'] >= .99 and r[2]['min_character_confidence'] >= .94]
        if key[0]=='model' and kind is not None:
            from backend.component_models import model_classes
            # Sequence probabilities multiply across characters. A longer
            # model can have every character >98.5% yet fall below 99% as a
            # complete sequence. Retain only a compatible electrical model
            # with a stricter per-character gate from the domain reader.
            strong += [r for r in support if r[0]=='circuit_word' and kind in model_classes(r[2]['text'])
                       and r[2]['confidence']>=.96 and r[2]['min_character_confidence']>=.985]
        if len(engines) < 2 and not strong: continue
        best = max(support, key=lambda r:r[2]['confidence'])
        accepted.append((len(engines), float(best[2]['confidence']), best, support))
    if not accepted:
        ordinary=[r for e,v,r in rows if r and r.get('confidence',0)>=.99
                  and r.get('min_character_confidence',0)>=.94
                  and re.search(r'[A-Za-z]{3,}',r.get('text',''))
                  and not re.search(r'\d',r.get('text','')) and _key(r['text']) is None]
        if ordinary:
            return dict(text=ordinary[0]['text'],unresolved=False,ignored=True,
                        reason='confident_non_electrical_word',confidence=ordinary[0]['confidence']*100)
        return None
    accepted.sort(key=lambda x:(x[0], x[1]), reverse=True)
    winner = accepted[0]
    # Two independently supported but different numeric readings are ambiguous.
    if len(accepted) > 1 and accepted[1][0] >= winner[0]: return None
    _, confidence, (_, variant, best), support = winner
    return dict(text=best['text'].strip(), confidence=confidence*100,
        min_character_confidence=min(float(r[2]['min_character_confidence']) for r in support),
        unresolved=False, reference_ids=[], reading_method='complete_reference_and_value_inscription' if _key(best['text'])[0]=='annotation' else 'nearby_whole_word_consensus',
        input_variant=variant, independent_readers=len({r[0] for r in support}),
        word_evidence=[dict(engine=e,variant=v,**r) for e,v,r in support])


def read_regions(image, regions, detections, *, raw_general=None, raw_domain=None, text_page=None):
    from backend.local_word_ocr import read_words
    from backend.domain_word_ocr import read_words as domain_words
    from backend.reference_labels import local_distance
    patches = []; selected = []; ready={}
    for i, box in enumerate(regions):
        if 'prepared_mask' in box: continue
        if detections and not any(local_distance({'bbox':box}, d) is not None for d in detections): continue
        context=box.get('component_context') or {}
        if text_page is not None and context.get('distance')==0:
            det=next((d for d in detections if d.get('detection_id')==context.get('detection_id')),None)
            if det:
                b=det['bbox'];area=max(1,(box['x2']-box['x1'])*(box['y2']-box['y1']))
                overlap=max(0,min(box['x2'],b['x2'])-max(box['x1'],b['x1']))*max(0,min(box['y2'],b['y2'])-max(box['y1'],b['y1']))/area
                crop=text_page[int(box['y1']):int(box['y2']),int(box['x1']):int(box['x2'])]
                _,_,stats,_=cv2.connectedComponentsWithStats((crop<128).astype(np.uint8),8)
                substantial=[s for s in stats[1:] if s[4]>=max(8,crop.shape[0]*.2)]
                if overlap>.72 and len(substantial)<=2:continue
        a=raw_general[i] if raw_general is not None else None
        b=(raw_domain or {}).get(i)
        initial=consensus([('general_word','original',a),('circuit_word','original',b)],kind=context.get('kind') if not context.get('ambiguous') else None)
        if initial and (initial.get('ignored') or initial['confidence']>=99 and initial.get('min_character_confidence',0)>=.94):
            ready[i]=initial;continue
        # Region proposals already include a border. Padding them again can
        # bring a resistor edge into a word and turn an omega into a zero.
        h = box['y2']-box['y1']
        x1,y1 = max(0,int(box['x1'])),max(0,int(box['y1']))
        x2,y2 = min(image.shape[1],int(box['x2'])),min(image.shape[0],int(box['y2']))
        crop=image[y1:y2,x1:x2]
        if not crop.size: continue
        patches.append(crop);selected.append((i,'original'))
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
        gray=cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)
        patches.append(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR));selected.append((i,'local_contrast'))
        # Small printed glyphs can occupy less than half a detector crop.
        # Tighten to real ink before the reader's fixed-height resize; do not
        # enlarge the paper margin along with the characters.
        if h <= 40 and np.percentile(gray,75) >= 210:
            mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
            points=cv2.findNonZero(mask)
            if points is not None:
                x,y,w,gh=cv2.boundingRect(points)
                if gh>=5 and w>=3:
                    tight=gray[y:y+gh,x:x+w]
                    tight=cv2.copyMakeBorder(tight,3,3,3,3,cv2.BORDER_CONSTANT,value=255)
                    patches.append(cv2.cvtColor(tight,cv2.COLOR_GRAY2BGR));selected.append((i,'tight_actual_ink'))
    def with_cache(reader,cache):
        result=[None]*len(patches);pending=[]
        for j,(i,variant) in enumerate(selected):
            if variant=='original' and i in cache:result[j]=cache[i]
            else:pending.append(j)
        values=reader([patches[j] for j in pending])
        for j,value in zip(pending,values):result[j]=value
        return result
    general=with_cache(read_words,dict(enumerate(raw_general)) if raw_general is not None else {})
    special=with_cache(domain_words,raw_domain or {})
    rows={}
    for (i,variant),a,b in zip(selected,general,special):
        rows.setdefault(i,[]).extend([('general_word',variant,a),('circuit_word',variant,b)])
    for i,values in rows.items():
        context=regions[i].get('component_context') or {}
        reading=consensus(values,kind=context.get('kind') if not context.get('ambiguous') else None)
        if reading:ready[i]=reading
    return ready
