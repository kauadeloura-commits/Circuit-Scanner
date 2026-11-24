"""Reject symbol strokes as handwriting while retaining their raw evidence."""
import cv2
import numpy as np

def separate(image,detections,tokens):
    kept=[];rejected=[]
    for token in tokens:
        text=token.get('text','')
        if text in {'+','-','='}:kept.append(token);continue
        b=token['bbox'];area=max(1,(b['x2']-b['x1'])*(b['y2']-b['y1']))
        covered=max((max(0,min(b['x2'],d['bbox']['x2'])-max(b['x1'],d['bbox']['x1']))*
            max(0,min(b['y2'],d['bbox']['y2'])-max(b['y1'],d['bbox']['y1']))/area for d in detections),default=0.)
        roi=image[max(0,int(b['y1'])):int(b['y2']),max(0,int(b['x1'])):int(b['x2'])]
        reason=None
        confidence=float(token.get('confidence',0));confidence=confidence/100 if confidence>1 else confidence
        independent=len({r.get('engine') for r in token.get('word_evidence',[]) if r.get('engine')})>=2 or token.get('independent_readers',0)>=2
        precise_word=token.get('reading_method') in {'local_neural_whole_inscription','physical_crop_whole_word_consensus','nearby_whole_word_consensus'} and confidence>=.99 and token.get('min_character_confidence',0)>=.94
        context=token.get('component_context') or b.get('component_context') or {}
        if covered>=.85 and context.get('distance')==0 and not (independent or precise_word):
            reason='word_region_inside_symbol_body'
        elif roi.size and len(text)>1:
            gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) if roi.ndim==3 else roi
            ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
            y,x=np.nonzero(ink)
            if len(x)>=8:
                owned=np.zeros_like(ink)
                for d in detections:
                    box=d['bbox'];x1=max(0,int(box['x1']-b['x1']));y1=max(0,int(box['y1']-b['y1']))
                    x2=min(ink.shape[1],int(box['x2']-b['x1'])+1);y2=min(ink.shape[0],int(box['y2']-b['y1'])+1)
                    if x1<x2 and y1<y2:owned[y1:y2,x1:x2]=1
                if context.get('distance')==0 and covered>.35 and np.count_nonzero(owned&(ink>0))/len(x)>=.75 and not (independent or precise_word):
                    rejected.append(dict(token,rejection_reason='ink_belongs_to_symbol_and_external_lead'));continue
                points=np.c_[x,y].astype(float);eigen=np.linalg.eigvalsh(np.cov(points.T))
                # A single uninterrupted almost straight stroke cannot encode
                # a multi-character inscription. Do not manufacture digits.
                n,labels,stats,_=cv2.connectedComponentsWithStats((ink>0).astype(np.uint8),8)
                if n>1 and max(stats[1:,4])/len(x)>.93 and eigen[-1]/max(.25,eigen[0])>35:
                    reason='single_conductor_stroke_not_complete_word'
        if reason:rejected.append(dict(token,rejection_reason=reason))
        else:kept.append(token)
    return kept,rejected
