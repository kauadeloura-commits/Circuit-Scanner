"""Read small and vertical printed annotations without rotating circuit geometry."""
import cv2,numpy as np
from backend.resistor_geometry import cached_vision


def original_box(box,quarter,scale,shape):
    height,width=shape[:2]
    points=np.array([[box[x]/scale,box[y]/scale] for x,y in [('x1','y1'),('x1','y2'),('x2','y1'),('x2','y2')]])
    x,y=points.T
    if quarter==1:points=np.stack((width-1-y,x),axis=1)
    elif quarter==2:points=np.stack((width-1-x,height-1-y),axis=1)
    elif quarter==3:points=np.stack((y,height-1-x),axis=1)
    lo,hi=points.min(axis=0),points.max(axis=0)
    return dict(x1=float(max(0,lo[0])),y1=float(max(0,lo[1])),x2=float(min(width,hi[0])),y2=float(min(height,hi[1])))


def annotation_region(text,box,detections):
    """The inside of a symbol is not a printed word beside that symbol."""
    x,y=(box['x1']+box['x2'])/2,(box['y1']+box['y2'])/2
    for d in detections:
        b=d['bbox'];w,h=b['x2']-b['x1'],b['y2']-b['y1']
        if b['x1']+.15*w<x<b['x2']-.15*w and b['y1']+.15*h<y<b['y2']-.15*h:
            return False
    if str(text).upper() in {'Q','T'}:
        return not any(d['bbox']['x1']<x<d['bbox']['x2'] and d['bbox']['y1']<y<d['bbox']['y2'] for d in detections)
    return True


@cached_vision
def read_annotations(image,detections):
    from backend.local_word_ocr import locate_words,read_words
    from backend.ocr import _limpar_texto,_classes_sugeridas_por_texto
    from backend.reference_labels import local_distance,reference_classes
    from backend.component_models import model_classes
    from backend.measurement_text import parse_measurement
    output=[];native_semantics=0
    for quarter in (0,3,1,2):
        if quarter==2 and native_semantics>=max(3,len(detections)//4):continue
        native=np.ascontiguousarray(np.rot90(image,quarter))
        scale=min(3.,1280/max(native.shape[:2]))
        enlarged=cv2.resize(native,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        regions=locate_words(enlarged);crops=[];selected=[]
        for region in regions:
            b=region['bbox'];mapped=original_box(b,quarter,scale,image.shape)
            if not annotation_region('',mapped,detections):continue
            if detections and not any(local_distance(dict(bbox=mapped),d) is not None for d in detections):continue
            x,y=max(0,int(b['x1'])-2),max(0,int(b['y1'])-2)
            z,t=min(enlarged.shape[1],int(b['x2'])+3),min(enlarged.shape[0],int(b['y2'])+3)
            if z<=x or t<=y:continue
            crops.append(enlarged[y:t,x:z]);selected.append(mapped)
        for row,box in zip(read_words(crops),selected):
            if not row or float(row['confidence'])<(.88 if quarter==0 else .94):continue
            text=_limpar_texto(row['text'])
            classes,kind=_classes_sugeridas_por_texto(text)
            models=model_classes(text)
            if models and not reference_classes(text):classes,kind=set(),'model'
            if not classes and not models:continue
            if str(text).upper() in {'Q','T'} and (quarter!=0 or row['confidence']<.92):continue
            if not annotation_region(text,box,detections):continue
            # A rotated symbol can resemble a digit. Bare values need at
            # least two real digits and high confidence; one stroke is not 1Ω.
            if kind=='value' and not parse_measurement(text) and not any(parse_measurement(text,d['class']) for d in detections):continue
            if kind=='value' and text.isdigit() and (len(text)<2 or row['confidence']<.98):continue
            compatible=classes or models
            if not any(d['class'] in compatible and local_distance(dict(bbox=box),d) is not None for d in detections):continue
            output.append(dict(text=text,raw_text=row['text'],confidence=float(row['confidence']),bbox=box,
                               engine='local_printed_word',variant='printed_multiscale_orientation',orientation='normal',
                               text_quarter_turns=quarter,upsample_scale=scale,min_character_confidence=row.get('min_character_confidence')))
            if quarter==0:native_semantics+=1
    return output
