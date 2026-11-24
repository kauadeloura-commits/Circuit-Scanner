"""Printed supply markers require a physical head and an owned power label."""
import re,math,cv2,numpy as np
from backend.resistor_geometry import cached_vision

@cached_vision
def filled_power_head(image,box):
 from backend.printed_capacitor import plate_ink
 from backend.vcc_topology import external_leads
 mask=plate_ink(image);h,w=mask.shape
 x1,y1=max(0,int(box['x1'])-3),max(0,int(box['y1'])-3)
 x2,y2=min(w,int(box['x2'])+4),min(h,int(box['y2'])+4);crop=mask[y1:y2,x1:x2]
 if not crop.size:return None
 core=cv2.morphologyEx(crop,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)));contours,_=cv2.findContours(core,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 for c in contours:
  area=cv2.contourArea(c)
  if area<7:continue
  hull=cv2.convexHull(c);poly=cv2.approxPolyDP(hull,cv2.arcLength(hull,True)*.07,True)
  triangle_area,_=cv2.minEnclosingTriangle(hull)
  if (len(poly)!=3 and (triangle_area<=0 or cv2.contourArea(hull)/triangle_area<.70)) or area/max(1,cv2.contourArea(hull))<.78:continue
  x,y,bw,bh=cv2.boundingRect(hull)
  if min(bw,bh)<3 or max(bw,bh)>min(bw,bh)*3:continue
  b=dict(x1=float(max(0,x+x1-3)),y1=float(max(0,y+y1-3)),x2=float(min(w-1,x+x1+bw+2)),y2=float(min(h-1,y+y1+bh+2)))
  if external_leads(image,b,mask)!=1:continue
  return dict(validation_mode='printed_filled_supply_arrow_one_external_lead',power_head_bbox=b,external_lead_count=1,solid_triangle=True)
 return None


def power_text(text):
 text=re.sub(r'\s+','',str(text)).replace('−','-')
 return bool(re.fullmatch(r'[+-]?V(?:CC|EE|DD|SS)\d*',text,re.I) or re.fullmatch(r'[+-]?(?:\d+(?:[.,]\d+)?|O)(?:[aA~][+-]?\d+(?:[.,]\d+)?)?[mMkK]?V',text,re.I))


def box_exit_count(mask,box):
 h,w=mask.shape;points=[]
 for axis,edge in [(0,box['x1']),(0,box['x2']),(1,box['y1']),(1,box['y2'])]:
  other=1-axis;low,high=(box['y1'],box['y2']) if axis==0 else (box['x1'],box['x2'])
  seen=[]
  for t in range(max(0,int(low)),min(h if other==1 else w,int(np.ceil(high))+1)):
   p=np.rint([edge,t] if axis==0 else [t,edge]).astype(int)
   if not(1<=p[0]<w-1 and 1<=p[1]<h-1):continue
   if mask[p[1]-1:p[1]+2,p[0]-1:p[0]+2].any():seen.append(p)
  groups=[]
  for p in seen:
   if not groups or np.linalg.norm(p-groups[-1][-1])>4:groups.append([])
   groups[-1].append(p)
  for group in groups:
   p=np.mean(group,axis=0)
   if all(np.linalg.norm(p-q)>5 for q in points):points.append(p)
 return len(points)


def filter_ports(image,detections,tokens):
 """An unlabeled input/output circle is not automatically a DC supply."""
 from backend.drawing_mode import digital
 if not digital():return detections
 candidates=[d for d in detections if d.get('class')=='vcc' and not d.get('manual_reviewed')]
 owners={}
 for token in [*tokens,*getattr(tokens,'raw_tokens',[])]:
  if not power_text(token.get('text')) or float(token.get('confidence',0))<.60:continue
  b=token.get('bbox') or {}
  if not all(k in b for k in ('x1','y1','x2','y2')):continue
  ranked=[]
  for d in candidates:
   a=d['bbox'];distance=math.hypot(max(a['x1']-b['x2'],b['x1']-a['x2'],0),max(a['y1']-b['y2'],b['y1']-a['y2'],0))
   if distance<=max(24,min(90,4*(b['y2']-b['y1']))):ranked.append((distance,id(d),d))
  if ranked:
   _,key,d=min(ranked,key=lambda item:item[:2]);owners.setdefault(key,[]).append(token['text'])
 result=[]
 for d in detections:
  if d.get('class')!='vcc' or d.get('manual_reviewed'):result.append(d);continue
  labels=owners.get(id(d),[])
  filled=filled_power_head(image,d['bbox'])
  if not labels and not filled:
   from backend.symbol_semantics import visible_plus
   if not visible_plus(image,d['bbox'],[other['bbox'] for other in detections if other is not d]):continue
   labels=['+']
  new=dict(d,printed_power_labels=labels)
  box=dict(d['bbox'])
  evidence=(d.get('geometry_validation') or {}).get('evidence') or {}
  if evidence.get('validation_mode')=='connection_dot_topology' and max(box['x2']-box['x1'],box['y2']-box['y1'])<28:
   from backend.printed_capacitor import plate_ink
   mask=plate_ink(image);h,w=mask.shape
   sides=[]
   for name,axis,edge,sign in [('x1',0,box['x1'],-1),('x2',0,box['x2'],1),('y1',1,box['y1'],-1),('y2',1,box['y2'],1)]:
    lo,hi=(box['y1'],box['y2']) if axis==0 else (box['x1'],box['x2']);score=0
    for q in np.arange(lo+2,hi-1):
     p=(int(round(edge+sign*2)),int(round(q))) if axis==0 else (int(round(q)),int(round(edge+sign*2)))
     if 0<=p[0]<w and 0<=p[1]<h:score+=bool(mask[p[1],p[0]])
    sides.append((score,name,sign))
   score,name,sign=max(sides)
   if score:
    extent=max(8,.60*(box['y2']-box['y1'] if name.startswith('y') else box['x2']-box['x1']))
    box[name]=float(np.clip(box[name]+sign*extent,0,(h if name.startswith('y') else w)-1))
    for edge in (('x1','x2') if name.startswith('y') else ('y1','y2')):box[edge]=float(np.clip(box[edge]+(-3 if edge.endswith('1') else 3),0,(w if edge.startswith('x') else h)-1))
    box.update(width=box['x2']-box['x1'],height=box['y2']-box['y1'],center_x=(box['x1']+box['x2'])/2,center_y=(box['y1']+box['y2'])/2)
    new['bbox']=box;new['printed_stem_included']=True
  # Include an actual stem past a tiny head; do not leave the terminal on
  # the circle or arrowhead. Stop at the first junction, never at another part.
  result.append(new)
 return result
