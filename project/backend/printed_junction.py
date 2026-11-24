"""A dot must add ink beyond the two extrapolated conductor strips."""
import cv2,numpy as np


def dot_shape(mask,point):
 x,y=point;yy,xx=np.indices(mask.shape);rr=np.hypot(xx-x,yy-y)
 ring=((rr>=9)&(rr<=12)&(mask>0)).astype(np.uint8)
 count,labels,stats,centers=cv2.connectedComponentsWithStats(ring,8)
 arms=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=3]
 if len(arms)!=4:return None,dict(arms=len(arms))
 directions=np.asarray([centers[i]-point for i in arms]);directions/=np.maximum(1,np.linalg.norm(directions,axis=1))[:,None]
 options=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]
 pairing=min(options,key=lambda pairs:sum(float(directions[a]@directions[b]) for a,b in pairs))
 if any(directions[a]@directions[b]>-.75 for a,b in pairing):return None,dict(arms=4,reason='opposed_arms_not_confirmed')
 predicted=np.zeros(mask.shape,bool);widths=[]
 grid=np.stack((xx,yy),axis=2)
 for a,b in pairing:
  p,q=centers[arms[a]],centers[arms[b]];u=(q-p)/np.linalg.norm(q-p);normal=np.array([-u[1],u[0]])
  r,c=np.nonzero((labels==arms[a])|(labels==arms[b]));points=np.stack((c,r),axis=1)
  middle=points.mean(axis=0);middle+=float(np.median((points-middle)@normal))*normal
  width=float(np.percentile(abs((points-middle)@normal),95));widths.append(width)
  predicted|=abs((grid-middle)@normal)<=width+.5
 extra=(mask>0)&(rr<=5)&~predicted
 amount=int(extra.sum())
 return amount>=6,dict(method='four_observed_arms_and_extra_dot_ink',excess_ink_pixels=amount,arm_half_widths=widths)
