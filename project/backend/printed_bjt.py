"""Read printed BJT bar/arrow at local scale; circle is a symbol enclosure."""
import cv2,numpy as np
from backend.resistor_geometry import cached_vision


@cached_vision
def body_view(image,box,scale=4,*,with_head=False):
 from backend.local_symbol_roles import local_view
 view,b,_,origin=local_view(image,box,scale)
 gray=cv2.cvtColor(view,cv2.COLOR_BGR2GRAY)
 # Keep strokes at their observed locations while giving tiny arrowheads
 # enough pixels for the orientation-independent bar/arrow reader.
 from backend.printed_capacitor import plate_ink
 core=plate_ink(view)
 circles=cv2.HoughCircles(cv2.GaussianBlur(gray,(5,5),0),cv2.HOUGH_GRADIENT,1,
  max(15,min(view.shape[:2])*.3),param1=80,param2=22,
  minRadius=max(8,int(min(b['x2']-b['x1'],b['y2']-b['y1'])*.23)),
  maxRadius=max(10,int(max(b['x2']-b['x1'],b['y2']-b['y1'])*.58)))
 circle=None
 if circles is not None:
  center=np.array([(b['x1']+b['x2'])/2,(b['y1']+b['y2'])/2])
  def rim_coverage(c):
   angles=np.linspace(0,2*np.pi,120,endpoint=False)
   coords=np.stack((c[0]+c[2]*np.cos(angles),c[1]+c[2]*np.sin(angles)),axis=1)
   support=cv2.dilate(core,np.ones((max(3,scale*2+1),max(3,scale*2+1)),np.uint8))
   coords=np.rint(coords).astype(int);valid=(coords[:,0]>=0)&(coords[:,1]>=0)&(coords[:,0]<core.shape[1])&(coords[:,1]<core.shape[0])
   return float(np.mean(support[coords[valid,1],coords[valid,0]]>0)) if valid.all() else 0
  options=[c for c in circles[0] if np.linalg.norm(c[:2]-center)<c[2]*.5 and rim_coverage(c)>.83]
  if options:circle=min(options,key=lambda c:np.linalg.norm(c[:2]-center))
 head_core=None
 if circle is not None:
  cx,cy,r=map(float,circle);yy,xx=np.indices(core.shape)
  distance=np.hypot(xx-cx,yy-cy)
  if np.median(gray[distance<r*.65])<220:
   values=gray[(distance<r*.8)&(gray<225)]
   level=min(220,float(np.percentile(values,10))+40) if len(values) else 0
   core=np.where(gray<level,255,0).astype(np.uint8)
  head_core=core.copy();head_core[distance>.93*r]=0
  # The enclosing circle is not part of the internal base/branch geometry.
  erase=(distance>.86*r)&(distance<1.12*r)
  # Keep the enclosure and its lead intersections out of the internal bar
  # reader. External contacts remain in the original image, across the cut.
  core[erase]=0
 enhanced=cv2.cvtColor(255-core,cv2.COLOR_GRAY2BGR)
 result=(enhanced,b,origin,dict(local_scale=scale,circle_excluded=None if circle is None else [float(v/scale) for v in circle],
         circle_center=None if circle is None else (np.asarray(circle[:2])/scale+origin).tolist(),circle_radius=None if circle is None else float(circle[2]/scale)))
 return (*result,cv2.cvtColor(255-head_core,cv2.COLOR_GRAY2BGR) if head_core is not None else enhanced) if with_head else result


@cached_vision
def read_roles(image,box):
 from backend.drawing_mode import digital
 if not digital():return None,dict(confirmed=False)
 from backend.printed_circle_roles import read
 circular,proof=read(image,box)
 if circular:return circular,proof
 from backend.npn_wire_contacts import _detect_npn_wire_contacts
 from backend.wire_detection import _skeletonize
 attempts=[]
 for scale in (4,3):
  view,b,origin,details,head=body_view(image,box,scale,with_head=True)
  from backend.local_symbol_roles import local_view
  original,_,_,_=local_view(image,box,scale)
  from backend.printed_capacitor import plate_ink
  sk=_skeletonize(plate_ink(original))
  found,meta=_detect_npn_wire_contacts(head,b,sk,arrow_mask_image=head,contact_image=original)
  if found is None or not meta.get('arrow_confirmed'):
   found,meta=_detect_npn_wire_contacts(view,b,sk,arrow_mask_image=head,contact_image=original)
  attempts.append(dict(meta,**details))
  if found is None or not meta.get('arrow_confirmed'):continue
  contacts,e=found;proof=e['hypotheses'][0]['points']
  if not meta.get('detected_arrow_direction') and not details.get('circle_center'):
   from backend.printed_head import direction as triangle_direction
   subtype,triangle=triangle_direction(head,b,meta)
   if subtype:meta.update(detected_arrow_direction=subtype,printed_arrow_triangle=triangle)
  if not all(p in proof for p in ('B','E')):continue
  anchors={p:np.array([value['x'],value['y']])/scale+origin for p,value in proof.items()}
  remaining=[np.asarray(c['point'])/scale+origin for c in contacts if all(np.linalg.norm(np.asarray(c['point'])/scale+origin-p)>2 for p in anchors.values())]
  if len(remaining)!=1:continue
  anchors['C']=remaining[0]
  direction,shape=direction_from_head(head,b,meta)
  # Taper alone remains diagnostic: a skewed shaft can reverse this score.
  # Subtype changes require the independently verified arrow-wing direction.
  bar=np.asarray(meta.get('base_bar_points'),float)/scale+origin
  if bar.shape==(2,2):
   middle=np.asarray(meta.get('base_lead_near',np.mean(np.asarray(meta['base_bar_points']),axis=0)),float)/scale+origin
   bar_u=(bar[1]-bar[0])/max(1,np.linalg.norm(bar[1]-bar[0]));normal=np.array([-bar_u[1],bar_u[0]])
   if normal@((anchors['E']+anchors['C'])/2-middle)>0:normal=-normal
   distances=[]
   for axis,(lo,hi) in enumerate(((box['x1']+1.1,box['x2']-1.1),(box['y1']+1.1,box['y2']-1.1))):
    if abs(normal[axis])>.05:
     t=((hi if normal[axis]>0 else lo)-middle[axis])/normal[axis]
     if t>0:distances.append(t)
   if distances:
    p=middle+normal*min(distances)
    from backend.printed_capacitor import plate_ink
    ink=plate_ink(image);x,y=np.rint(p).astype(int)
    if 0<=x<ink.shape[1] and 0<=y<ink.shape[0] and ink[max(0,y-2):y+3,max(0,x-2):x+3].any():anchors['B']=p
  return anchors,dict(meta,**details,head_taper=shape,confirmed=True,source='printed_local_bar_arrow_enclosure_separated')
 return None,dict(confirmed=False,reason='printed_bar_arrow_not_proven',attempts=attempts)


def direction_from_head(view,box,meta):
 from backend.printed_capacitor import plate_ink
 branches=meta.get('branch_anchors',[]);arrows=meta.get('arrow_evidence',[])
 if len(branches)!=2 or len(arrows)!=2:return None,{}
 branch=branches[int(arrows[1].get('score',0)>arrows[0].get('score',0))]
 a,b=np.asarray(branch['near']),np.asarray(branch['far']);length=np.linalg.norm(b-a)
 if length<10:return None,{}
 u=(b-a)/length;v=np.array([-u[1],u[0]]);mask=plate_ink(view)
 along=np.arange(max(4,length*.18),length*.90,.5)
 across=np.arange(-max(18,length*.3),max(18,length*.3)+.5,.5)
 pts=a+along[:,None,None]*u+across[None,:,None]*v
 xx=np.rint(pts[:,:,0]).astype(int);yy=np.rint(pts[:,:,1]).astype(int)
 valid=(xx>=0)&(yy>=0)&(xx<mask.shape[1])&(yy<mask.shape[0]);on=np.zeros(xx.shape,bool);on[valid]=mask[yy[valid],xx[valid]]>0
 widths=[]
 for row in on:
  indexes=np.flatnonzero(row)
  widths.append(float(across[indexes[-1]]-across[indexes[0]]) if len(indexes) else 0.)
 widths=np.asarray(widths);positive=widths[widths>0]
 if len(positive)<8:return None,{}
 base=float(np.percentile(positive,25));peak=int(np.argmax(widths));maximum=float(widths[peak])
 if maximum<max(6,base*1.5):return None,dict(maximum=maximum,shaft=base)
 low=peak;high=peak
 while low>0 and widths[low-1]>max(base*1.3,maximum*.2):low-=1
 while high+1<len(widths) and widths[high+1]>max(base*1.3,maximum*.2):high+=1
 before=peak-low;after=high-peak
 result=dict(maximum=maximum,shaft=base,head_start=float(along[low]),peak=float(along[peak]),head_end=float(along[high]),before=int(before),after=int(after))
 direction='outward' if after>before else 'inward'
 if max(before,after)>=max(3,1.7*min(before,after)) and min(peak,len(widths)-1-peak)>2:return direction,result
 return None,result


@cached_vision
def circle_leads(image,box):
 from backend.printed_capacitor import plate_ink
 _,_,_,meta=body_view(image,box,3)
 if meta['circle_center'] is None:return [],meta
 center=np.asarray(meta['circle_center']);radius=meta['circle_radius']
 x=max(0,int(center[0]-radius*1.7));y=max(0,int(center[1]-radius*1.7))
 z=min(image.shape[1],int(center[0]+radius*1.7)+1);t=min(image.shape[0],int(center[1]+radius*1.7)+1)
 roi=image[y:t,x:z];mask=plate_ink(roi)
 yy,xx=np.indices(mask.shape);rr=np.hypot(xx+x-center[0],yy+y-center[1])
 mask[(rr<radius*1.08)|(rr>radius*1.60)]=0
 count,labels,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8)
 contacts=[]
 for i in range(1,count):
  if stats[i,cv2.CC_STAT_AREA]<3:continue
  seen=rr[labels==i]
  if min(seen)>radius*1.18 or max(seen)<radius*1.35:continue
  row,col=np.nonzero((labels==i)&(rr<=radius*1.18));points=np.stack((col+x,row+y),axis=1)
  if not len(points):continue
  p=points[np.argmin(np.linalg.norm(points-points.mean(axis=0),axis=1))]
  contacts.append(p.astype(float))
 return contacts,meta


def finalize_circle_contacts(image,component):
 from itertools import permutations
 if component.get('type') not in {'transistor_npn','transistor_pnp'}:return False
 terms=component.get('terminals',[])
 if len(terms)!=3:return False
 from backend.terminal_wire_contacts import _manual
 if _manual(component) or component.get('manual_reviewed') or any(t.get('manual_reviewed') or str(t.get('source','')).startswith('manual') for t in terms):return False
 anchors,proof=read_roles(image,component['bbox'])
 if anchors and proof.get('source')=='printed_circle_internal_bar_and_exclusive_emitter_head':
  box=dict(component['bbox'])
  for term in terms:
   p=anchors[term['pin']];term.update(position=dict(x=float(p[0]),y=float(p[1])),original_ink_confirmed=True,wire_contact_confirmed=True,identity_source=proof['source'])
   box.update(x1=min(box['x1'],p[0]-2),y1=min(box['y1'],p[1]-2),x2=max(box['x2'],p[0]+2),y2=max(box['y2'],p[1]+2))
  box.update(width=box['x2']-box['x1'],height=box['y2']-box['y1'],center_x=(box['x1']+box['x2'])/2,center_y=(box['y1']+box['y2'])/2)
  component['bbox']=box;component['printed_symbol_geometry']=proof
  return True
 contacts,meta=circle_leads(image,component['bbox'])
 if not 3<=len(contacts)<=6:return False
 anchors,proof=read_roles(image,component['bbox'])
 names=('B','C','E')
 if not anchors:anchors={t['pin']:np.array([t['position']['x'],t['position']['y']]) for t in terms}
 if not all(p in anchors for p in names):return False
 center=np.asarray(meta['circle_center']);candidates=np.asarray(contacts)-center
 candidates/=np.maximum(1,np.linalg.norm(candidates,axis=1))[:,None]
 wanted=np.asarray([anchors[p]-center for p in names]);wanted/=np.maximum(1,np.linalg.norm(wanted,axis=1))[:,None]
 ranked=sorted((float(np.linalg.norm(candidates[list(order)]-wanted,axis=1).sum()),order) for order in permutations(range(len(contacts)),3))
 cost,order=ranked[0]
 if cost>1.25 or ranked[1][0]-cost<.15:return False
 by_name={t['pin']:t for t in terms};box=dict(component['bbox'])
 for pin,j in zip(names,order):
  p=contacts[j];term=by_name[pin];term.update(position=dict(x=float(p[0]),y=float(p[1])),original_ink_confirmed=True,wire_contact_confirmed=True,identity_source='printed_circle_three_external_ink_leads')
  box.update(x1=min(box['x1'],p[0]-2),y1=min(box['y1'],p[1]-2),x2=max(box['x2'],p[0]+2),y2=max(box['y2'],p[1]+2))
 box.update(width=box['x2']-box['x1'],height=box['y2']-box['y1'],center_x=(box['x1']+box['x2'])/2,center_y=(box['y1']+box['y2'])/2)
 component['bbox']=box;component['printed_symbol_geometry']=dict(meta,method='circular_body_insulated_three_observed_external_leads')
 return True
