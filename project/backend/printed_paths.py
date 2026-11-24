"""Thin printed zigzags: one continuous periodic path with opposed leads."""
import cv2,numpy as np
from scipy.signal import find_peaks
from backend.resistor_geometry import cached_vision

@cached_vision
def zigzag(image,box):
 from backend.printed_capacitor import plate_ink
 x,y=max(0,int(box['x1'])),max(0,int(box['y1']))
 z,t=min(image.shape[1],int(np.ceil(box['x2']))+1),min(image.shape[0],int(np.ceil(box['y2']))+1)
 roi=image[y:t,x:z]
 if not roi.size:return None
 mask=plate_ink(roi)
 if mask.shape[0]>mask.shape[1]:mask=mask.T
 height,width=mask.shape
 if width<16 or height<5:return None
 count,labels,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8)
 for i in range(1,count):
  a,b,w,h,area=stats[i]
  if w<width*.75 or h<max(3,height*.15) or a>width*.2 or a+w<width*.8:continue
  path=[]
  for column in range(a,a+w):
   seen=np.flatnonzero(labels[:,column]==i)
   path.append(float(np.median(seen)) if len(seen) else np.nan)
  path=np.asarray(path)
  if np.count_nonzero(np.isfinite(path))<w*.95:continue
  path=np.interp(np.arange(w),np.flatnonzero(np.isfinite(path)),path[np.isfinite(path)])
  middle=path[int(w*.12):max(int(w*.12)+1,int(w*.88))]
  amplitude=float(np.ptp(middle))
  if amplitude<max(2,height*.12):continue
  peaks,_=find_peaks(middle,prominence=max(1,amplitude*.30),distance=max(2,int(width*.045)))
  valleys,_=find_peaks(-middle,prominence=max(1,amplitude*.30),distance=max(2,int(width*.045)))
  turns=sorted([(int(p),1) for p in peaks]+[(int(p),-1) for p in valleys])
  cropped=box['x1']<=1 or box['y1']<=1 or box['x2']>=image.shape[1]-2 or box['y2']>=image.shape[0]-2
  if len(turns)<(3 if cropped else 5) or any(a[1]==b[1] for a,b in zip(turns,turns[1:])):continue
  spans=np.diff([q[0] for q in turns])
  if np.std(spans)/max(1,np.mean(spans))>.65:continue
  return dict(method='printed_continuous_periodic_zigzag',distinct_zigzag_segments=len(turns)+1,
              one_original_ink_path=True,opposed_extent=True,amplitude_px=amplitude,span_px=int(w),period_cv=float(np.std(spans)/max(1,np.mean(spans))))
 return None
