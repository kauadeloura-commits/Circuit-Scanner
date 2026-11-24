"""Resolve a cropped reference only with separate local value evidence."""
import copy
import re
from backend.measurement_text import parse_measurement
from backend.reference_labels import reference_classes


def recover_reference_readings(tokens, detections):
    from backend.drawing_mode import CURRENT
    if CURRENT.get()!='handwritten':return tokens
    kinds={d['detection_id']:d['class'] for d in detections}
    output=[]
    for token in tokens:
        context=token.get('component_context') or {}
        owner=context.get('detection_id');kind=kinds.get(owner)
        # Embedded-R notation is a valid resistance. Never turn it into a
        # reference from spelling alone or from an expected numbering order.
        if kind!='resistor' or context.get('ambiguous') or not re.fullmatch(r'\d+[Rr]\d+',token.get('text','')):
            output.append(token);continue
        alternatives=[a for a in token.get('spatial_alternatives',[])
            if kind in reference_classes(a.get('text','')) and a.get('confidence',0)>=.99
            and a.get('confidence',0)>=token.get('confidence',0) and a.get('bbox')]
        separate_value=False
        for other in tokens:
            oc=other.get('component_context') or {}
            if other is token or oc.get('detection_id')!=owner or oc.get('ambiguous'):continue
            value=parse_measurement(other.get('text',''),kind)
            if not value or not (value.get('prefix') or value.get('unit_explicit')) or other.get('confidence',0)<.65:continue
            a,b=token['bbox'],other['bbox']
            intersection=max(0,min(a['x2'],b['x2'])-max(a['x1'],b['x1']))*max(0,min(a['y2'],b['y2'])-max(a['y1'],b['y1']))
            area=min((a['x2']-a['x1'])*(a['y2']-a['y1']),(b['x2']-b['x1'])*(b['y2']-b['y1']))
            if area>0 and intersection/area<.40:separate_value=True;break
        if len(alternatives)!=1 or not separate_value:
            output.append(token);continue
        best=alternatives[0]
        # Both readings remain available for inspection; this changes field
        # ownership, not the pixels, the model or the raw OCR evidence.
        recovered=copy.deepcopy(token)
        recovered.update(text=best['text'],bbox=copy.deepcopy(best['bbox']),confidence=best['confidence'],
            raw_text=token.get('raw_text',token['text']),reading_method='read_reference_crop_with_distinct_local_value',
            reference_context_evidence=dict(previous_reading=token['text'],alternative=copy.deepcopy(best),separate_value=other['text']))
        output.append(recovered)
    return output
