"""Circular BJT roles from the interior base bar and a thicker emitter head."""
import cv2,numpy as np
from backend.resistor_geometry import cached_vision


@cached_vision
def read(image,box):
 from backend.printed_bjt import body_view
 from backend.printed_capacitor import plate_ink
 from backend.local_symbol_roles import local_view
 _,b,origin,meta=body_view(image,box,4)
 if meta.get('circle_center') is None:return None,{}
 scale=4;center=(np.asarray(meta['circle_center'])-origin)*scale;radius=meta['circle_radius']*scale
 view,_,_,_=local_view(image,box,scale);ink=plate_ink(view)
 gray=cv2.cvtColor(view,cv2.COLOR_BGR2GRAY);yy,xx=np.indices(ink.shape);rr=np.hypot(xx-center[0],yy-center[1])
 if np.median(gray[rr<radius*.65])<220:
  values=gray[(rr<radius*.8)&(gray<225)]
  if len(values):ink=np.where(gray<min(220,float(np.percentile(values,10))+40),255,0).astype(np.uint8)
 core=ink.copy();core[rr>radius*.82]=0
 lines=cv2.HoughLinesP(core,1,np.pi/180,threshold=max(10,int(radius*.45)),minLineLength=radius*.80,maxLineGap=max(3,radius*.10))
 if lines is None:return None,{}
 distance=cv2.distanceTransform((ink>0).astype(np.uint8),cv2.DIST_L2,5)
 candidates=[]
 for raw in np.asarray(lines).reshape(-1,4):
  a=raw[:2].astype(float);z=raw[2:].astype(float);v=(z-a)/max(1,np.linalg.norm(z-a));n=np.array([-v[1],v[0]])
  middle=(a+z)/2;offset=float(n@(center-middle))
  if abs(offset)<radius*.18 or abs(offset)>radius*.65:continue
  if offset<0:n=-n
  # A base wire crosses the bar and continues out through the opposite side.
  best=None
  for t in np.arange(-radius*.25,np.linalg.norm(z-a)+radius*.25,2):
   q=a+v*t;lengths=np.arange(radius*.10,radius*1.30,2);points=q-lengths[:,None]*n
   xy=np.rint(points).astype(int);valid=(xy[:,0]>=1)&(xy[:,0]<ink.shape[1]-1)&(xy[:,1]>=1)&(xy[:,1]<ink.shape[0]-1)
   on=np.zeros(len(points),bool)
   for i in np.flatnonzero(valid):
    x,y=xy[i];on[i]=bool(ink[y-1:y+2,x-1:x+2].any())
   score=float(on.mean())
   if best is None or score>best[0]:best=(score,q)
  if best[0]<.85:continue
  q=best[1]
  delta=np.stack((xx-q[0],yy-q[1]),axis=2);across=delta@n;along=delta@v
  half=[]
  for sign in (-1,1):
   zone=(rr<radius*.90)&(across>radius*.20)&(along*sign>radius*.08)
   values=np.where(zone,distance,0);peak=float(values.max())
   y,x=np.unravel_index(int(values.argmax()),values.shape)
   half.append((peak,np.array([x,y],float),sign))
  low,high=sorted(half,key=lambda p:p[0]);ratio=high[0]/max(1,low[0])
  if high[0]<max(3,low[0]*1.25) or high[0]-low[0]<1.5:continue
  candidates.append((best[0]+min(ratio,3)*.20,q,n,v,high,dict(meta,bar=[(a/scale+origin).tolist(),(z/scale+origin).tolist()],head_width_ratio=ratio)))
 if not candidates:return None,{}
 _,base,n,v,head,proof=max(candidates,key=lambda p:p[0])
 # A narrow ring excludes the enclosure and cannot merge two leads at a
 # junction farther outside the symbol. Each candidate is original ink.
 ring=(rr>radius*1.18)&(rr<radius*1.28)&(ink>0)
 count,labels,stats,_=cv2.connectedComponentsWithStats(ring.astype(np.uint8),8)
 ports=[]
 for i in range(1,count):
  if stats[i,cv2.CC_STAT_AREA]<scale*2:continue
  y,x=np.nonzero(labels==i);p=np.array([x.mean(),y.mean()])
  # Snap to the center of the observed stroke, rather than the enclosure.
  pixels=np.stack((x,y),axis=1);p=pixels[np.argmin(np.linalg.norm(pixels-p,axis=1))].astype(float)
  ports.append(p)
 base_ports=[p for p in ports if (p-base)@n<-radius*.5 and abs((p-base)@v)<radius*.25]
 if not base_ports:return None,{}
 B=min(base_ports,key=lambda p:abs((p-base)@v))
 sides=[]
 for sign in (-1,1):
  options=[p for p in ports if (p-base)@n>radius*.25 and (p-base)@v*sign>radius*.35]
  if not options:return None,{}
  # Real top/bottom leads occupy the extremity along the bar. Text at the
  # side of a symbol cannot replace them with a neighboring glyph.
  sides.append(max(options,key=lambda p:(p-base)@v*sign))
 E=sides[0 if head[2]==-1 else 1];C=sides[1 if head[2]==-1 else 0]
 anchors={pin:p/scale+origin for pin,p in [('B',B),('C',C),('E',E)]}
 return anchors,dict(proof,confirmed=True,source='printed_circle_internal_bar_and_exclusive_emitter_head',emitter_head=(head[1]/scale+origin).tolist())
