"""Verify an ambiguous leading digit using complete-word views of actual ink.

This cannot invent a prefix, drop zeros, move a decimal point or replace the
remaining digits. Two image views must agree on the leading character, and
one must independently read the entire original suffix and numeric tail.
"""
from copy import deepcopy
import cv2
import numpy as np


def _raw_views(masks):
    from backend import sequence_ocr as reader
    if not reader.MODEL.is_file():return []
    import torch
    prepared=[reader.prepare(m) for m in masks]
    if any(m is None for m in prepared):return []
    model,alphabet,_=reader._load(reader.MODEL.stat().st_mtime_ns)
    with torch.inference_mode():
        logits=model(torch.from_numpy(np.stack(prepared).astype(np.float32)/255).unsqueeze(1))
        rows=reader.decode(logits,alphabet)
        probabilities,indices=torch.softmax(logits,dim=-1).max(-1)
    for i,row in enumerate(rows):
        ids=indices[:,i].tolist();scores=probabilities[:,i].tolist()
        row['leading_character_confidence']=next((float(p) for k,p in zip(ids,scores) if k),0.)
    return rows


def verify_leading_digit(image,box,reading,kind):
    if not reading or not str(reading.get('reading_method','')).startswith('trained_digits'):return reading
    margin=reading.get('numeric_anchor_margin')
    if not isinstance(margin,(int,float)) or not 0<=margin<3:return reading
    from backend.measurement_text import parse_measurement
    prior=parse_measurement(reading.get('text',''),kind=kind)
    if not prior or not prior.get('unit_explicit'):return reading
    digits=str(prior.get('numeric_reading',''))
    if not digits or not digits[0].isdigit():return reading
    h,w=image.shape[:2]
    x1,y1=max(0,int(box['x1'])),max(0,int(box['y1']))
    x2,y2=min(w,int(box['x2'])),min(h,int(box['y2']))
    crop=image[y1:y2,x1:x2]
    if not crop.size:return reading
    from backend.handwriting_reference import ink
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
    masks=[ink(crop),cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]]
    rows=_raw_views(masks)
    if len(rows)!=2:return reading
    parsed=[]
    for row in rows:
        value=parse_measurement(row.get('text',''),kind=kind)
        if (not value or value.get('prefix')!=prior.get('prefix') or value['unit']!=prior['unit']
            or row.get('leading_character_confidence',0)<.94 or row.get('confidence',0)<.75):return reading
        number=str(value.get('numeric_reading',''))
        if not number or not number[0].isdigit():return reading
        parsed.append((row,value,number))
    first={number[0] for _,_,number in parsed}
    if len(first)!=1 or next(iter(first))==digits[0]:return reading
    complete=[(row,value) for row,value,number in parsed if number[1:]==digits[1:]
              and row.get('confidence',0)>=.80 and row.get('min_character_confidence',0)>=.50]
    if not complete:return reading
    row,value=max(complete,key=lambda p:p[0]['confidence'])
    result=deepcopy(reading)
    result.update(text=value['text'],reading_method='whole_word_views_verify_ambiguous_leading_digit',
                  leading_digit_before=reading['text'],leading_digit_verified=True,
                  leading_digit_evidence=[dict(text=r['text'],confidence=r['confidence'],leading_character_confidence=r['leading_character_confidence']) for r in rows])
    return result
