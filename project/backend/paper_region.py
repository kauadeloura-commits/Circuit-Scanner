"""Conservative paper isolation. No component predictions or annotations are used.
Keeps original pixels inside the paper; whitens outside, then crops by translation.
No perspective warp, rescaling, rotation, or circuit-dependent crop boundaries.
"""
import cv2
import numpy as np
import json,hashlib
from pathlib import Path
VERSION='white-paper-region-2'
def isolate_paper(image):
    h,w=image.shape[:2]
    scale=min(1.,800/max(h,w));small=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale)))) if scale<1 else image
    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    neutral=((hsv[:,:,1]<=65)&(hsv[:,:,2]>=58)).astype(np.uint8)*255
    neutral=cv2.morphologyEx(neutral,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(neutral,8)
    total=neutral.size;large=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=total*.025]
    meta=dict(version=VERSION,original_size=[w,h],active=False,crop=[0,0,w,h],reason='no_confident_paper',paper_fraction=0.)
    if not large:return image.copy(),meta,np.ones((h,w),np.uint8)*255
    points=[]
    for i in large:
        ys,xs=np.where(labels==i)
        points.append(np.column_stack([xs,ys]).astype(np.int32))
    hull=cv2.convexHull(np.concatenate(points))
    paper=np.zeros(neutral.shape,np.uint8);cv2.fillConvexPoly(paper,hull,255)
    fraction=float(np.mean(paper>0));meta['paper_fraction']=round(fraction,5)
    if fraction<.42:return image.copy(),meta,np.ones((h,w),np.uint8)*255
    paper=cv2.resize(paper,(w,h),interpolation=cv2.INTER_NEAREST)
    # Preserve pen touching a paper boundary; margin never depends on test labels.
    margin=max(3,round(min(h,w)*.006))
    expanded=cv2.dilate(paper,np.ones((2*margin+1,2*margin+1),np.uint8))
    original_hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    blue=(original_hsv[:,:,0]>=88)&(original_hsv[:,:,0]<=155)&(original_hsv[:,:,1]>=35)&(original_hsv[:,:,2]<245)
    # A white padding margin must not reintroduce the green cover or brown desk.
    # Only actual pen that touches the paper boundary is retained beyond the hull.
    blue_page=np.count_nonzero(blue&(paper>0))>=h*w*.0002
    edge_ink=blue if blue_page else ((original_hsv[:,:,1]<=65)&(original_hsv[:,:,2]<85))
    paper[(expanded>0)&edge_ink]=255
    if np.mean(paper==0)<.007:
        meta.update(reason='already_paper',paper_fraction=1.)
        return image.copy(),meta,np.ones((h,w),np.uint8)*255
    ys,xs=np.where(paper>0);x1,y1,x2,y2=int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1
    masked=image.copy();masked[paper==0]=255
    meta.update(active=True,crop=[x1,y1,x2,y2],reason='neutral_paper_hull',masked_fraction=round(float(np.mean(paper==0)),5),output_size=[x2-x1,y2-y1],margin_px=margin)
    return masked[y1:y2,x1:x2].copy(),meta,paper

def prepare_project_image(upload_path,output_path):
    """Keep the received file intact and create one coordinate-consistent PNG.
    All later stages and the displayed reference use this SAME analysis image.
    """
    upload_path=Path(upload_path);output_path=Path(output_path)
    image=cv2.imread(str(upload_path))
    if image is None:raise ValueError('Não foi possível abrir a imagem enviada.')
    cropped,meta,_=isolate_paper(image)
    if not cv2.imwrite(str(output_path),cropped,[cv2.IMWRITE_PNG_COMPRESSION,3]):
        raise OSError('Não foi possível salvar a imagem do circuito.')
    meta.update(upload_filename=upload_path.name,analysis_filename=output_path.name,
        upload_sha256=hashlib.sha256(upload_path.read_bytes()).hexdigest(),
        analysis_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
        coordinate_frame='analysis_image_pixels',original_upload_preserved=True)
    (output_path.parent/'paper_region.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    return meta
