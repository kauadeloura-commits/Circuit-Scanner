"""Conservative filled-head fallback when a bent external shaft hides direction."""
import cv2,numpy as np


def direction(view,box,meta):
 from backend.printed_capacitor import plate_ink
 branches=meta.get('branch_anchors',[]);arrows=meta.get('arrow_evidence',[])
 if len(branches)!=2 or len(arrows)!=2:return None,{}
 branch=branches[int(arrows[1].get('score',0)>arrows[0].get('score',0))]
 near,far=np.asarray(branch['near']),np.asarray(branch['far']);length=np.linalg.norm(far-near)
 if length<12:return None,{}
 u=(far-near)/length;v=np.array([-u[1],u[0]])
 ink=plate_ink(view);yy,xx=np.indices(ink.shape);delta=np.stack((xx-near[0],yy-near[1]),axis=2);along,across=delta@u,delta@v
 zone=(along>length*.15)&(along<length*.85)&(abs(across)<length*.26)
 mask=np.where(zone,ink,0).astype(np.uint8)
 widths=cv2.distanceTransform(mask,cv2.DIST_L2,5);values=widths[mask>0]
 shaft=float(np.percentile(values,60)) if len(values) else 1
 kernel=max(3,int(shaft*3)|1)
 opened=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel)))
 contours,_=cv2.findContours(opened,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);found=[]
 for c in contours:
  area=cv2.contourArea(c)
  if area<kernel*kernel:continue
  total,tri=cv2.minEnclosingTriangle(cv2.convexHull(c))
  if total<=0 or not .60<=area/total<=.97:continue
  tri=tri.reshape(3,2)
  local=tri.mean(axis=0)-near;local/=max(1,np.linalg.norm(local));normal=np.array([-local[1],local[0]])
  for i,p in enumerate(tri):
   a,b=np.delete(tri,i,axis=0)-p;la,lb=float(a@normal),float(b@normal);aa,bb=float(a@local),float(b@local)
   if la*lb<0 and min(abs(la),abs(lb))>kernel*.6 and aa*bb>0 and min(abs(aa),abs(bb))>kernel*.6 and abs(float((p-near)@normal))<kernel*1.5:
    found.append(('outward' if aa<0 else 'inward',area/total,tri.tolist()))
 if found and len({f[0] for f in found})==1:return found[0][0],dict(method='local_filled_triangle_before_external_bend',triangle_fit=found[0][1],vertices=found[0][2])
 return None,{}
