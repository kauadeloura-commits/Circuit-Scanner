"""Remove excess label context only when the two physical plates prove it."""
import numpy as np


def bjt_bounds(detection,anchors,proof):
    if not anchors or proof.get('circle_center') or not proof.get('confirmed'):
        return detection
    box=detection['bbox'];points=np.asarray(list(anchors.values()))
    lo,hi=points.min(axis=0),points.max(axis=0)
    minimum=min(box['x2']-box['x1'],box['y2']-box['y1']);pad=max(4,.10*minimum)
    b=dict(box)
    for axis in range(2):
        start,end=('x1','x2') if axis==0 else ('y1','y2')
        if lo[axis]-box[start]>.25*minimum:b[start]=float(lo[axis]-pad)
        if box[end]-hi[axis]>.25*minimum:b[end]=float(hi[axis]+pad)
    b.update(width=b['x2']-b['x1'],height=b['y2']-b['y1'],center_x=(b['x1']+b['x2'])/2,center_y=(b['y1']+b['y2'])/2)
    return dict(detection,bbox=b) if b!=box else detection


def capacitor_bounds(image, detection):
    from backend.drawing_mode import digital
    if not digital() or detection.get('manual_reviewed') or detection.get('class') not in {'capacitor','capacitor_polarized'}:
        return detection
    from backend.capacitor_geometry import capacitor_body_evidence
    box = detection['bbox']
    ok, body = capacitor_body_evidence(image, box)
    if not ok:
        return detection
    u = np.asarray(body['electrical_axis'], float)
    v = np.array([-u[1], u[0]])
    lengths = body['outer_plate_lengths']
    span = abs(v[0]) * (box['x2']-box['x1']) + abs(v[1]) * (box['y2']-box['y1'])
    if span <= max(lengths)*2.2:
        return detection
    points=[]
    for center,length in zip(body['outer_plate_centers'], lengths):
        for along in (-.40,.40):
            for across in (-.65,.65):
                points.append(np.asarray(center)+u*(along*max(lengths))+v*(across*length))
    points=np.asarray(points)
    lo,hi=points.min(axis=0)-2,points.max(axis=0)+2
    b=dict(x1=float(max(box['x1'],lo[0])),y1=float(max(box['y1'],lo[1])),
           x2=float(min(box['x2'],hi[0])),y2=float(min(box['y2'],hi[1])))
    b.update(width=b['x2']-b['x1'],height=b['y2']-b['y1'],center_x=(b['x1']+b['x2'])/2,center_y=(b['y1']+b['y2'])/2)
    return dict(detection,bbox=b,context_bbox_before_plate_refinement=dict(box),printed_bounds_evidence=body)
