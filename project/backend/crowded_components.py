"""Adaptive crop scale for dense circuits; no changes to trained weights."""
import math
import numpy as np

def tile_policy(image_shape,detections,imgsz=640):
    h,w=image_shape[:2]
    bodies=[d for d in detections if d.get('class') not in {'gnd','vcc'} and float(d.get('confidence',0))>=.45]
    if len(bodies)<8:return None
    sizes=[max(d['bbox']['x2']-d['bbox']['x1'],d['bbox']['y2']-d['bbox']['y1']) for d in bodies]
    median=float(np.median(sizes));projected=median*imgsz/max(h,w)
    if projected>=85:return None
    tile=max(384,min(960,64*round(median*imgsz/90/64)))
    # Retain full-page coverage; limit the crop count by increasing crop size,
    # never by discarding an area of the circuit.
    while math.ceil(max(1,h-tile)/(tile*.65)+1)*math.ceil(max(1,w-tile)/(tile*.65)+1)>24:tile+=64
    return dict(tile_size=min(max(h,w),tile),overlap=.35,imgsz=640,
                component_count=len(bodies),median_body_px=round(median,2),
                projected_body_px=round(projected,2),method='density_and_visible_symbol_scale')

def clipped_on_internal_edge(detection,x,y,crop_shape,image_shape):
    h,w=crop_shape[:2];ih,iw=image_shape[:2];b=detection['bbox'];margin=3
    return (x>0 and b['x1']<=margin or y>0 and b['y1']<=margin or
            x+w<iw and b['x2']>=w-margin or y+h<ih and b['y2']>=h-margin)
