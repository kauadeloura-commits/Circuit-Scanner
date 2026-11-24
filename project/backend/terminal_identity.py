"""Visible signs and symbol topology determine electrical terminal identity."""
from __future__ import annotations

import math
from functools import lru_cache
from itertools import permutations
import cv2
import numpy as np
from backend.resistor_geometry import cached_vision

TARGET_CLASSES = {"capacitor_polarized", "source_dc", "battery", "opamp", "mosfet_n", "mosfet_p"}
ROLES = {"+": "positive", "-": "negative", "OUT": "output", "G": "gate", "D": "drain", "S": "source"}


@cached_vision
def _crop(image, bbox):
    h, w = image.shape[:2]
    pad = max(8, min(32, int(min(bbox["x2"] - bbox["x1"], bbox["y2"] - bbox["y1"]) * .18)))
    x1, y1 = max(0, int(bbox["x1"]) - pad), max(0, int(bbox["y1"]) - pad)
    x2, y2 = min(w, int(math.ceil(bbox["x2"])) + pad), min(h, int(math.ceil(bbox["y2"])) + pad)
    patch = image[y1:y2, x1:x2]
    from backend.drawing_mode import digital,printed_ink
    if digital():return printed_ink(patch),np.array([x1,y1],dtype=float)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = (gray <= min(210, threshold + 5)).astype(np.uint8) * 255
    # Remove pale page-wide strokes inside the crop, including vertical and
    # tilted notebook rules; protect the darker ink of crossing symbols.
    ink_values = gray[mask > 0]
    if len(ink_values):
        dark_limit = float(np.percentile(ink_values, 25)) + 15
        lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=30,
                                minLineLength=max(30, int(min(mask.shape)*.70)), maxLineGap=12)
        if lines is not None:
            for ax, ay, bx, by in np.asarray(lines).reshape(-1, 4):
                if abs(bx-ax) < mask.shape[1]*.80 and abs(by-ay) < mask.shape[0]*.80:
                    continue
                band = np.zeros_like(mask)
                cv2.line(band, (int(ax), int(ay)), (int(bx), int(by)), 255, 7)
                mask[(band > 0) & (gray > dark_limit)] = 0
    return mask, np.array([x1, y1], dtype=float)


@lru_cache(maxsize=1)
def _plus_templates():
    templates=[];samples=[]
    for angle in range(0,90,15):
        theta=math.radians(angle)
        directions=[np.array([math.cos(theta),math.sin(theta)]),np.array([-math.sin(theta),math.cos(theta)])]
        for thickness in (3,5,7,9):
            template=np.zeros((32,32),np.uint8)
            cv2.line(template,(5,16),(27,16),255,thickness)
            cv2.line(template,(16,5),(16,27),255,thickness)
            rotated=cv2.warpAffine(template,cv2.getRotationMatrix2D((16,16),angle,1),(32,32))
            for dx in (-3,0,3):
                for dy in (-3,0,3):
                    center=np.array([16+dx,16+dy])
                    samples.append(np.rint([[center+sign*direction*d for d in range(6,13)] for direction in directions for sign in (-1,1)]).astype(int))
                    templates.append(cv2.warpAffine(rotated,np.float32([[1,0,dx],[0,1,dy]]),(32,32))>80)
    return np.asarray(templates),np.asarray(samples)


def _plus_score(blob):
    h, w = blob.shape
    if min(h, w) < 4 or min(h, w) / max(h, w) < .45:
        return 0.0
    side = max(h, w) + 4
    square = np.zeros((side, side), np.uint8)
    square[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = blob
    actual = cv2.resize(square, (32, 32), interpolation=cv2.INTER_NEAREST) > 0
    templates,samples=_plus_templates()
    supported=cv2.dilate(actual.astype(np.uint8),np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT)
    arms=np.mean(supported[samples[:,:,:,1],samples[:,:,:,0]],axis=2)
    eligible=np.min(arms,axis=1)>=.42
    if not np.any(eligible):return 0.0
    proposals=templates[eligible]
    intersection=np.count_nonzero(proposals&actual,axis=(1,2))
    union=np.count_nonzero(proposals,axis=(1,2))+np.count_nonzero(actual)-intersection
    return float(np.max(intersection/np.maximum(1,union)))


def _four_arm_junction(skeleton, point, radius):
    """Count branches connected through the center, excluding short corner spurs."""
    x, y = np.rint(point).astype(int)
    radius = max(7, int(radius))
    x1, y1 = max(0, x-radius), max(0, y-radius)
    x2, y2 = min(skeleton.shape[1], x+radius+1), min(skeleton.shape[0], y+radius+1)
    patch = skeleton[y1:y2, x1:x2].copy()
    yy, xx = np.indices(patch.shape)
    distance = np.hypot(xx+x1-point[0], yy+y1-point[1])
    cut = max(2, radius*.18)
    patch[distance <= cut] = 0
    count, labels, _, _ = cv2.connectedComponentsWithStats(patch.astype(np.uint8), 8)
    branches = 0
    for label in range(1, count):
        radii = distance[labels == label]
        # Skeleton thinning shortens a small sign's arms. Require meaningful
        # branches beyond the junction, rather than a percentage of the crop.
        if radii.min() <= cut+2 and radii.max() >= max(cut+3, radius*.40):
            branches += 1
    return branches >= 4


@cached_vision
def _marks(mask, origin, bbox, *, include_attached=False, attached_margin=0, maximum_fraction=.34):
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    maximum = max(12.0, min(bbox["x2"] - bbox["x1"], bbox["y2"] - bbox["y1"]) * maximum_fraction)
    marks = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < 7 or max(w, h) > maximum or min(w, h) < 2:
            continue
        blob = (labels[y:y+h, x:x+w] == label).astype(np.uint8) * 255
        score = _plus_score(blob)
        position = centers[label] + origin
        if score >= .52:
            marks.append({"kind": "+", "point": position, "score": score, "label": label})
        elif max(w, h) / max(1, min(w, h)) >= 2.0:
            marks.append({"kind": "-", "point": position, "score": .75, "label": label})
    if include_attached:
        # A handwritten '+' can touch a plate. Locate its four arms locally
        # without requiring it to be a separate connected component.
        from backend.wire_detection import _skeletonize
        skeleton = (_skeletonize(mask) > 0).astype(np.uint8)
        neighbors = cv2.filter2D(skeleton, cv2.CV_16S, np.ones((3, 3), np.int16)) - skeleton
        junctions = ((skeleton > 0) & (neighbors >= 3)).astype(np.uint8)
        n, _, _, centers = cv2.connectedComponentsWithStats(junctions, 8)
        radius = max(6, min(14, int(maximum * .4)))
        for local in centers[1:n]:
            x, y = np.rint(local).astype(int)
            position = local + origin
            if not (bbox["x1"]-attached_margin <= position[0] <= bbox["x2"]+attached_margin
                    and bbox["y1"]-attached_margin <= position[1] <= bbox["y2"]+attached_margin):
                continue
            if x-radius < 0 or y-radius < 0 or x+radius >= mask.shape[1] or y+radius >= mask.shape[0]:
                continue
            if any(m['kind']=='+' and m['label']==int(labels[y,x]) for m in marks):
                continue
            if not _four_arm_junction(skeleton, local, radius):
                continue
            score = _plus_score(mask[y-radius:y+radius+1, x-radius:x+radius+1])
            if score >= .58:
                marks.append({"kind": "+", "point": position, "score": score,
                              "label": int(labels[y, x]), "attached": True})
    return marks, labels


def _sample(mask, point, origin, radius=2):
    x, y = np.rint(np.asarray(point) - origin).astype(int)
    if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
        return False
    return bool(np.any(mask[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1]))


def _support(mask, start, direction, distances, origin):
    return np.mean([_sample(mask, start + direction * d, origin) for d in distances])


def _exit(start, direction, bbox):
    times = []
    for axis, keys in [(0, ("x1", "x2")), (1, ("y1", "y2"))]:
        if abs(direction[axis]) < 1e-6:
            continue
        for key in keys:
            t = (float(bbox[key]) - start[axis]) / direction[axis]
            point = start + direction * t
            if t >= -1e-6 and bbox["x1"]-1 <= point[0] <= bbox["x2"]+1 and bbox["y1"]-1 <= point[1] <= bbox["y2"]+1:
                times.append((t, point))
    return min(times, key=lambda item: item[0])[1] if times else start


def _wire_exit(mask, anchor, direction, bbox, origin):
    point = _exit(anchor, direction, bbox)
    # Move only across the expected wire, never toward a sign or another pin.
    perpendicular = np.array([-direction[1], direction[0]])
    for offset in sorted(range(-5, 6), key=abs):
        candidate = point + perpendicular * offset
        if _sample(mask, candidate, origin, radius=1):
            return candidate
    return point


def _plus_side(marks, points):
    p1, p2 = np.array(points[0], float), np.array(points[1], float)
    vector = p2 - p1
    length = float(np.linalg.norm(vector))
    if length < 5:
        return None, None
    unit = vector / length
    candidates = []
    for mark in marks:
        if mark["kind"] != "+":
            continue
        projection = float(np.dot(mark["point"] - (p1+p2)/2, unit))
        if mark.get("attached"):
            perpendicular = np.array([-unit[1], unit[0]])
            if abs(np.dot(mark["point"]-(p1+p2)/2, perpendicular)) < 5:
                continue  # A crossing on the lead itself is not a polarity sign.
        if abs(projection) < max(3.0, length * .03):
            continue  # A centered '+' cannot distinguish two opposed leads.
        distance = float(min(np.linalg.norm(mark["point"]-p1), np.linalg.norm(mark["point"]-p2)))
        if distance > length * .65:
            continue
        candidates.append((distance, 1 if projection > 0 else 0, mark))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1 and candidates[1][1] != candidates[0][1] and candidates[1][0] < candidates[0][0] + length * .12:
        return None, None
    return candidates[0][1], candidates[0][2]


def _circle_axis(mask, origin, bbox):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    area_box = (bbox["x2"]-bbox["x1"])*(bbox["y2"]-bbox["y1"])
    for contour in contours:
        area = abs(cv2.contourArea(contour))
        perimeter = cv2.arcLength(contour, True)
        if len(contour) < 5 or not .05*area_box < area < .75*area_box or perimeter == 0:
            continue
        (cx, cy), (a, b), _ = cv2.fitEllipse(contour)
        if min(a, b)/max(a, b) < .65 or 4*math.pi*area/perimeter**2 < .62:
            continue
        center = np.array([cx, cy]) + origin
        if not bbox["x1"] < center[0] < bbox["x2"] or not bbox["y1"] < center[1] < bbox["y2"]:
            continue
        radius = (a+b)/4
        for angle in range(0, 180, 5):
            unit = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
            distances = np.arange(radius + 4, radius + max(12, min(a, b)*.32), 2)
            score = min(_support(mask, center, unit, distances, origin),
                        _support(mask, center, -unit, distances, origin))
            if score >= .70:
                candidates.append((score, area, center, unit))
    if not candidates:
        # Handwritten source circles sometimes have a gap. Accept their two
        # visible opposite leads only if body arcs extend to both sides of
        # the lead axis; never use a bare bbox direction as evidence.
        _, labels = cv2.connectedComponents(mask, 8)
        ports = _border_ports(mask, origin, bbox, labels)
        if len(ports) != 2 or {p["side"] for p in ports} not in ({"left", "right"}, {"top", "bottom"}):
            return None
        points = [p["point"] for p in ports]
        vector = points[1]-points[0]
        length = np.linalg.norm(vector)
        if length < 12:
            return None
        unit = vector/length
        normal = np.array([-unit[1], unit[0]])
        yy, xx = np.nonzero(mask)
        delta = np.stack((xx+origin[0], yy+origin[1]), axis=1) - np.mean(points, axis=0)
        middle = np.abs(delta @ unit) < length*.35
        offsets = (delta @ normal)[middle]
        minimum = min(bbox["x2"]-bbox["x1"], bbox["y2"]-bbox["y1"])
        if np.count_nonzero(offsets > minimum*.18) >= 5 and np.count_nonzero(offsets < -minimum*.18) >= 5:
            return points
        return None
    _, _, center, unit = max(candidates, key=lambda c: (c[0], c[1]))
    return [_wire_exit(mask, center, -unit, bbox, origin), _wire_exit(mask, center, unit, bbox, origin)]


def _lines(mask, origin):
    result = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(mask)[0]
    if result is None:
        return []
    lines = []
    for ax, ay, bx, by in np.asarray(result).reshape(-1, 4):
        a, b = np.array([ax, ay]) + origin, np.array([bx, by]) + origin
        length = np.linalg.norm(b-a)
        if length >= 7:
            lines.append({"a": a, "b": b, "center": (a+b)/2, "length": float(length),
                          "angle": math.atan2(b[1]-a[1], b[0]-a[0]) % math.pi})
    return lines


def _angle_delta(a, b):
    return abs((a-b+math.pi/2) % math.pi - math.pi/2)


def _plate_axis(mask, origin, bbox):
    minimum = min(bbox["x2"]-bbox["x1"], bbox["y2"]-bbox["y1"])
    lines = [l for l in _lines(mask, origin) if l["length"] >= max(10, minimum*.20)
             and bbox["x1"] <= l["center"][0] <= bbox["x2"] and bbox["y1"] <= l["center"][1] <= bbox["y2"]]
    candidates = []
    for anchor in lines:
        v = np.array([math.cos(anchor["angle"]), math.sin(anchor["angle"])])
        u = np.array([-v[1], v[0]])
        if u[np.argmax(np.abs(u))] < 0:
            u = -u
        aligned = [l for l in lines if _angle_delta(l["angle"], anchor["angle"]) < math.radians(15)
                   and abs(np.dot(l["center"]-anchor["center"], v)) < (anchor["length"]+l["length"])*.42]
        aligned.sort(key=lambda l: np.dot(l["center"], u))
        groups = []
        for line in aligned:
            if not groups or abs(np.dot(line["center"]-groups[-1][0]["center"], u)) > max(5, minimum*.05):
                groups.append([line])
            else:
                groups[-1].append(line)
        if len(groups) < 2:
            continue
        plates = []
        for group in groups:
            tangential = [np.dot(p, v) for line in group for p in (line["a"], line["b"])]
            normal = np.mean([np.dot(line["center"], u) for line in group])
            plates.append({"center": v*(min(tangential)+max(tangential))/2 + u*normal,
                           "length": max(tangential)-min(tangential)})
        span = abs(np.dot(plates[-1]["center"]-plates[0]["center"], u))
        if not max(6, minimum*.08) <= span <= minimum*.85:
            continue
        candidates.append((len(plates) * np.median([p["length"] for p in plates]), plates, u))
    if not candidates:
        return None
    _, plates, u = max(candidates, key=lambda item: item[0])
    center = np.mean([p["center"] for p in plates], axis=0)
    points = [_wire_exit(mask, center, -u, bbox, origin), _wire_exit(mask, center, u, bbox, origin)]
    positive = None
    first, last = plates[0]["length"], plates[-1]["length"]
    if max(first, last)/max(1, min(first, last)) >= 1.35:
        positive = 0 if first > last else 1
    return points, positive


@cached_vision
def _triangle(mask, origin, bbox, marks):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    area_box = (bbox["x2"]-bbox["x1"])*(bbox["y2"]-bbox["y1"])
    for contour in contours:
        area = abs(cv2.contourArea(contour))
        if not .06*area_box < area < .85*area_box:
            continue
        approx = cv2.approxPolyDP(contour, .045*cv2.arcLength(contour, True), True)
        if len(approx) != 3:
            continue
        vertices = approx.reshape(3, 2).astype(float) + origin
        for index in range(3):
            apex = vertices[index]
            base = np.array([vertices[(index+1)%3], vertices[(index+2)%3]])
            midpoint = base.mean(axis=0)
            direction = apex-midpoint
            length = np.linalg.norm(direction)
            unit = direction/max(1, length)
            ports = []
            for t in np.linspace(.12, .88, 65):
                point = base[0] + (base[1]-base[0])*t
                support = _support(mask, point, -unit, np.arange(5, max(12, length*.25), 2), origin)
                if support >= .70:
                    ports.append((t, point))
            clusters = []
            for t, point in ports:
                if not clusters or t-clusters[-1][-1][0] > .045:
                    clusters.append([])
                clusters[-1].append((t, point))
            if len(clusters) < 2:
                continue
            incoming = [np.mean([p for _, p in group], axis=0) for group in clusters]
            out_support = _support(mask, apex, unit, np.arange(4, max(10, length*.22), 2), origin)
            if out_support < .65:
                continue
            near_marks = [m for m in marks if cv2.pointPolygonTest(vertices.astype(np.float32),tuple(map(float,m['point'])),True)>=-2
                          and -.32*length < np.dot(m["point"]-midpoint, unit) < .6*length
                          and abs((base[1][0]-base[0][0])*(m["point"][1]-base[0][1])
                                  -(base[1][1]-base[0][1])*(m["point"][0]-base[0][0]))/max(1, np.linalg.norm(base[1]-base[0])) < .45*length]
            plus = [m for m in near_marks if m["kind"] == "+"]
            if not plus:
                continue
            plus.sort(key=lambda m: min(np.linalg.norm(m["point"]-p) for p in incoming))
            chosen = plus[0]
            distances = np.array([np.linalg.norm(chosen["point"]-p) for p in incoming])
            if np.sort(distances)[1]-np.min(distances) < 4:
                continue
            positive = int(np.argmin(distances))
            minus = [m for m in near_marks if m["kind"] == "-"
                     and int(np.argmin([np.linalg.norm(m["point"]-p) for p in incoming])) != positive]
            if not minus:
                continue  # Require both marks before certifying the input identity.
            minus.sort(key=lambda m: min(np.linalg.norm(m["point"]-p) for p in incoming))
            negative = int(np.argmin([np.linalg.norm(minus[0]["point"]-p) for p in incoming]))
            points = {"+": _opamp_lead_exit(mask,incoming[positive],-unit,bbox,origin,vertices),
                      "-": _opamp_lead_exit(mask,incoming[negative],-unit,bbox,origin,vertices),
                      "OUT": _opamp_lead_exit(mask,apex,unit,bbox,origin,vertices)}
            candidates.append((area, points, apex))
    if not candidates:
        return None
    _, points, apex = max(candidates, key=lambda c: c[0])
    return points, apex

def _opamp_lead_exit(mask,anchor,direction,bbox,origin,vertices):
    """Follow a signed input's actual ink even when its lead bends in the box."""
    body=np.zeros_like(mask)
    cv2.fillPoly(body,[np.rint(vertices-origin).astype(np.int32)],255)
    body=cv2.dilate(body,np.ones((7,7),np.uint8))
    outside=mask.copy();outside[body>0]=0
    yy,xx=np.indices(mask.shape)
    outside[(xx+origin[0]<bbox['x1']-1)|(xx+origin[0]>bbox['x2']+1)|
            (yy+origin[1]<bbox['y1']-1)|(yy+origin[1]>bbox['y2']+1)]=0
    count,labels=cv2.connectedComponents(outside,8)
    seed=anchor+direction*8-origin
    ys,xs=np.nonzero(outside)
    if len(xs):
        distances=np.hypot(xs-seed[0],ys-seed[1]);closest=int(np.argmin(distances))
        if distances[closest]<=7:
            label=labels[ys[closest],xs[closest]]
            ports=[p for p in _border_ports(outside,origin,bbox,labels) if p['label']==label]
            if len(ports)==1:return ports[0]['point']
    return _wire_exit(mask,anchor,direction,bbox,origin)


def _border_ports(mask, origin, bbox, labels):
    ports = []
    distance = cv2.distanceTransform((mask>0).astype(np.uint8), cv2.DIST_L2, 5)
    stroke_radius = float(np.percentile(distance[distance>0], 70)) if np.any(distance) else 1.0
    for side, start, end in [("top", [bbox["x1"], bbox["y1"]], [bbox["x2"], bbox["y1"]]),
                              ("bottom", [bbox["x1"], bbox["y2"]], [bbox["x2"], bbox["y2"]]),
                              ("left", [bbox["x1"], bbox["y1"]], [bbox["x1"], bbox["y2"]]),
                              ("right", [bbox["x2"], bbox["y1"]], [bbox["x2"], bbox["y2"]])]:
        a, b = np.array(start, float), np.array(end, float)
        samples = np.linspace(a, b, max(2, int(np.linalg.norm(b-a))))
        groups = []
        for point in samples:
            if _sample(mask, point, origin, 1):
                if not groups or np.linalg.norm(point-groups[-1][-1]) > 3:
                    groups.append([])
                groups[-1].append(point)
        for group in groups:
            point = np.mean(group, axis=0)
            inward = {"top": np.array([0,1]), "bottom": np.array([0,-1]),
                      "left": np.array([1,0]), "right": np.array([-1,0])}[side]
            if _support(mask, point, inward, np.arange(3, 15, 2), origin) < .80:
                continue
            x, y = np.rint(point-origin).astype(int)
            width_patch = distance[max(0,y-3):y+4, max(0,x-3):x+4]
            if width_patch.size == 0 or np.max(width_patch) < max(1.0, .80*stroke_radius):
                continue
            patch = labels[max(0, y-3):y+4, max(0, x-3):x+4]
            values = patch[patch>0]
            if len(values):
                label = int(np.bincount(values).argmax())
                if not any(np.linalg.norm(point-p["point"]) < 8 for p in ports):
                    ports.append({"point": point, "label": label, "side": side})
    return ports


def _mosfet(mask, origin, bbox, labels):
    ports = _border_ports(mask, origin, bbox, labels)
    minimum = min(bbox["x2"]-bbox["x1"], bbox["y2"]-bbox["y1"])
    lines = [l for l in _lines(mask, origin) if l["length"] >= minimum*.32]
    candidates = []
    for line in lines:
        center = line["center"]
        x, y = np.rint(center-origin).astype(int)
        region = labels[max(0, y-3):y+4, max(0, x-3):x+4]
        found = region[region>0]
        if not len(found):
            continue
        gate_label = int(np.bincount(found).argmax())
        gate_ports = [p for p in ports if p["label"] == gate_label]
        # An external source branch may leave a broad detector frame twice.
        # Those two exits are one observed ink island, not two device pins.
        by_label={}
        for port in ports:
            if port['label']==gate_label:continue
            previous=by_label.get(port['label'])
            if previous is None or np.linalg.norm(port['point']-center)<np.linalg.norm(previous['point']-center):
                by_label[port['label']]=port
        others=list(by_label.values())
        if len(gate_ports) != 1 or len(others) != 2:
            continue
        v = np.array([math.cos(line["angle"]), math.sin(line["angle"])])
        u = np.array([-v[1], v[0]])
        offsets = [np.dot(p["point"]-center, u) for p in others]
        if offsets[0]*offsets[1] <= 0:
            continue
        u *= 1 if np.mean(offsets) > 0 else -1
        if abs(np.dot(others[0]["point"]-others[1]["point"], v)) < minimum*.35:
            continue
        # The isolated gate bar faces the channel. The body/source arrow must
        # have its own connector to one of the two current-carrying terminals.
        yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
        delta = np.stack((xx+origin[0]-center[0], yy+origin[1]-center[1]), axis=2)
        transverse = delta @ v
        normal = delta @ u
        zone = (np.abs(transverse) < line["length"]*.32) & (normal > 3) & (normal < minimum*.65)
        radii = []
        for port in others:
            distance = cv2.distanceTransform((labels == port["label"]).astype(np.uint8), cv2.DIST_L2, 5)
            radii.append(float(np.max(distance[zone])) if np.any(zone) else 0.0)
        source = int(np.argmax(radii))
        branch_distance = cv2.distanceTransform((labels == others[source]["label"]).astype(np.uint8), cv2.DIST_L2, 5)
        branch_values = branch_distance[branch_distance > 0]
        typical_radius = float(np.percentile(branch_values, 70)) if len(branch_values) else 1.0
        identifiable = (others[0]["label"] != others[1]["label"]
                        and radii[source] >= max(2.5, radii[1-source]*1.45, typical_radius*1.45))
        candidates.append((line["length"], {"G": gate_ports[0]["point"],
                                            "S": others[source]["point"], "D": others[1-source]["point"]},
                           identifiable, radii))
    if not candidates:
        return None
    _, points, identified, radii = max(candidates, key=lambda c: c[0])
    return points, identified, radii


def resolve_terminal_identity(image, component, terminals):
    kind = component.get("type")
    meta = {"source": "visible_symbol_terminal_identity", "confirmed": False}
    if kind not in TARGET_CLASSES or image is None:
        meta["reason"] = "image_required_for_visible_identity"
        return terminals, meta
    bbox = component["bbox"]
    mask, origin = _crop(image, bbox)
    marks, labels = _marks(mask, origin, bbox, include_attached=kind in {"capacitor_polarized", "source_dc", "opamp"})
    if kind=='source_dc' and not any(m['kind']=='+' for m in marks):
        marks += _small_source_marks(mask,origin,bbox)
    if kind=='capacitor_polarized':
        from backend.symbol_semantics import visible_plus
        marks += [dict(kind='+',point=np.asarray(m['point']),score=m['score']) for m in visible_plus(image,bbox)]
    margin = max(6, min(bbox["x2"]-bbox["x1"], bbox["y2"]-bbox["y1"])*.10)
    marks = [m for m in marks if bbox["x1"]-margin <= m["point"][0] <= bbox["x2"]+margin
             and bbox["y1"]-margin <= m["point"][1] <= bbox["y2"]+margin]
    if kind=='opamp':
        marks=[m for m in marks if bbox['x1']<m['point'][0]<bbox['x2'] and bbox['y1']<m['point'][1]<bbox['y2']]
        meta['sign_reading_method']='local_symbol_ocr_inside_triangle_then_trace_observed_leads'
    meta["marks"] = [{"kind": m["kind"], "x": round(float(m["point"][0]), 2),
                      "y": round(float(m["point"][1]), 2)} for m in marks]
    points, confirmed, reason = None, False, "visible_polarity_mark_not_confirmed"
    if kind in {"capacitor_polarized", "source_dc", "battery"}:
        pair = [np.array([t["position"]["x"], t["position"]["y"]]) for t in terminals[:2]]
        # Capacitor plate geometry already supplied the two lead positions.
        # Preserve those positions and correct their electrical names only.
        plate = _plate_axis(mask, origin, bbox) if kind == "battery" else None
        circle = _circle_axis(mask, origin, bbox) if kind == "source_dc" else None
        if circle is not None:
            pair = circle
        elif plate is not None:
            pair = plate[0]
        if len(pair) == 2:
            positive, sign = _plus_side(marks, pair)
            if positive is not None and (kind != "source_dc" or circle is not None):
                confirmed, reason = True, "positive_terminal_from_visible_plus"
            elif kind == "source_dc":
                positive = None
            elif kind == "battery" and plate is not None and plate[1] is not None:
                positive, confirmed, reason = plate[1], True, "battery_long_plate_positive"
            elif kind == 'capacitor_polarized' and not any(m['kind']=='+' for m in marks):
                from backend.printed_capacitor import plate_polarity,positive_contact
                evidence=plate_polarity(image,bbox)
                positive=positive_contact(evidence,pair)
                if positive is not None:
                    confirmed,reason=True,evidence['method']
                    meta['printed_plate_evidence']=evidence
            if positive is not None:
                points = {"+": pair[positive], "-": pair[1-positive]}
    elif kind == "opamp":
        from backend.opamp_terminal_geometry import signed_triangle
        photographed = signed_triangle(mask, origin, bbox, marks)
        triangle = None
        if photographed is not None:
            points, apex, local_marks = photographed
            triangle = points, apex
            meta['marks'] = [{"kind": m['kind'], "x": round(float(m['point'][0]), 2),
                              "y": round(float(m['point'][1]), 2)} for m in local_marks]
        if triangle is None:triangle = _triangle(mask, origin, bbox, marks)
        if triangle is not None:
            points, apex = triangle
            confirmed, reason = True, "opamp_input_signs_and_triangle_output"
            meta["triangle_apex"] = [round(float(v), 2) for v in apex]
        else:
            reason = "opamp_triangle_leads_or_input_signs_not_confirmed"
    else:
        mosfet = _mosfet(mask, origin, bbox, labels)
        if mosfet is not None:
            points, confirmed, radii = mosfet
            meta["source_branch_radii"] = radii
            reason = "mosfet_isolated_gate_and_source_body_connector" if confirmed else "mosfet_gate_found_source_drain_ambiguous"
        else:
            reason = "mosfet_visible_gate_channel_not_confirmed"
    meta.update(confirmed=bool(confirmed), reason=reason)
    if points is None:
        for terminal in terminals:
            terminal["identity_uncertain"] = True
        return terminals, meta
    names = ("+", "-", "OUT") if kind == "opamp" else ("G", "D", "S") if kind.startswith("mosfet") else ("+", "-")
    new_terminals = []
    for index, pin in enumerate(names, 1):
        x, y = points[pin]
        x, y = float(np.clip(x, 0, image.shape[1]-1)), float(np.clip(y, 0, image.shape[0]-1))
        distances = {"left": abs(x-bbox["x1"]), "right": abs(x-bbox["x2"]),
                     "top": abs(y-bbox["y1"]), "bottom": abs(y-bbox["y2"])}
        new_terminals.append({"pin_id": f"{component.get('component_id', 'COMP')}.{pin}", "pin": pin,
                              "index": index, "electrical_role": ({'+':'non_inverting_input','-':'inverting_input','OUT':'output'}[pin]
                                                                      if kind=='opamp' else ROLES[pin]),
                              "position": {"x": round(x, 2), "y": round(y, 2)},
                              "source": "visible_symbol_terminal_identity", "terminal_side": min(distances, key=distances.get),
                              "identity_uncertain": not confirmed and pin != "G",
                              "search_radius_px": 10, "confidence": .93 if confirmed else .45})
    return new_terminals, meta


POLARIZED_TWO_PIN_CLASSES={"source_dc","capacitor_polarized","battery"}


def _small_source_marks(mask,origin,bbox):
    """Read a tiny photographed '+' without the native minimum-size cutoff."""
    if min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])>85:return []
    scaled=cv2.resize(mask,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
    marks,_=_marks(scaled,origin*2,{k:v*2 for k,v in bbox.items()},include_attached=False)
    return [dict(m,point=(m['point']-.5)/2,source='local_source_symbol_ocr_2x') for m in marks if m['kind']=='+'
            and bbox['x1']<(m['point'][0]-.5)/2<bbox['x2'] and bbox['y1']<(m['point'][1]-.5)/2<bbox['y2']]

def read_opamp_signs(image,detection):
    """Local symbol OCR: read both glyphs only when their triangle owns them."""
    if detection.get('class')!='opamp':return []
    bbox=detection['bbox'];mask,origin=_crop(image,bbox)
    marks,_=_marks(mask,origin,bbox,include_attached=True)
    marks=[m for m in marks if bbox['x1']<m['point'][0]<bbox['x2'] and bbox['y1']<m['point'][1]<bbox['y2']]
    if _triangle(mask,origin,bbox,marks) is None:
        from backend.opamp_geometry import signed_triangle
        observed=signed_triangle(mask,origin,bbox,marks)
        if observed is None:return []
        marks=observed[2]
    return [dict(text=m['kind'],confidence=float(m['score'])*100,reference_ids=['geometric_polarity_glyph'],unresolved=False,
                 reading_method='opamp_local_symbol_ocr',detection_id=detection.get('detection_id'),
                 bbox=dict(x1=float(m['point'][0]-3),y1=float(m['point'][1]-3),x2=float(m['point'][0]+3),y2=float(m['point'][1]+3)))
            for m in marks]


def finalize_opamp_on_contacts(image,component):
    """Assign signs and apex to the three final observed leads, jointly."""
    if component.get('type')!='opamp':return None
    terms=component.get('terminals') or []
    meta={'source':'opamp_original_signs_triangle_and_joint_contact_assignment','confirmed':False}
    if len(terms)!=3:return meta
    proposed,geometry=resolve_terminal_identity(image,component,[])
    if not geometry.get('confirmed'):
        meta['reason']=geometry.get('reason');return meta
    names=('+','-','OUT')
    anchors=np.array([[t['position']['x'],t['position']['y']] for t in proposed])
    contacts=np.array([[t['position']['x'],t['position']['y']] for t in terms])
    scale=max(1,min(component['bbox']['x2']-component['bbox']['x1'],component['bbox']['y2']-component['bbox']['y1']))
    ranked=sorted((float(np.linalg.norm(contacts[list(order)]-anchors,axis=1).sum()),order)
                  for order in permutations(range(3)))
    cost,order=ranked[0];margin=(ranked[1][0]-cost)/scale
    distances=np.linalg.norm(contacts[list(order)]-anchors,axis=1)
    meta.update(metric='two_signed_inputs_one_apex_output_joint_bijection',
                contact_cost_normalized=round(cost/scale,4),assignment_margin_normalized=round(margin,4),
                maximum_contact_distance_px=round(float(distances.max()),3),triangle_apex=geometry['triangle_apex'],
                marks=geometry.get('marks',[]))
    if distances.max()>.30*scale or margin<.08:
        meta['reason']='opamp_observed_contacts_do_not_separate_three_roles';return meta
    for index,(pin,j) in enumerate(zip(names,order),1):
        terms[j].update(pin=pin,pin_id=f"{component.get('component_id','COMP')}.{pin}",
                        index=index,electrical_role={'+':'non_inverting_input','-':'inverting_input','OUT':'output'}[pin],identity_uncertain=False,
                        identity_source=meta['source'],confidence=max(.93,float(terms[j].get('confidence') or 0)))
    component['terminals']=sorted(terms,key=lambda t:t['index'])
    meta.update(confirmed=True,reason='opamp_roles_confirmed_on_final_original_ink_contacts')
    component['terminal_identity_geometry']=meta
    td=component.setdefault('terminal_detection',{})
    cleared={'orientacao_geometrica_ambigua','confirmar_identidade_eletrica_dos_lados',
             'opamp_triangle_leads_or_input_signs_not_confirmed','atribuicao_de_pinos_aos_fios_ambigua'}
    if all(t.get('original_ink_confirmed') for t in terms):cleared.add('contato_interno_sem_tinta_confirmada')
    for target in (component,td):
        target['review_reasons']=[r for r in target.get('review_reasons',[]) if r not in cleared]
        target['review_required']=bool(target['review_reasons'])
    td['identity_method']=meta['metric']
    component['orientation']={'axis':'horizontal' if abs(anchors[2,0]-anchors[:2,0].mean())>=abs(anchors[2,1]-anchors[:2,1].mean()) else 'vertical',
        'angle_degrees':round(math.degrees(math.atan2(*(anchors[2]-anchors[:2].mean(axis=0))[::-1])),2),
        'confidence':.93,'source':meta['source']}
    return meta


def _observed_polarity_plates(mask,origin,bbox,pair):
    """Measure whole photographed plates, including split or curved strokes.

    The final conductor contacts supply the axis. Length is measured by
    projection of observed ink, so a lead attached halfway along a plate
    cannot split a long positive plate into two misleading short segments.
    This is used after wires only; legacy wire masking retains its geometry.
    """
    vector=pair[1]-pair[0];length=float(np.linalg.norm(vector))
    if length<5:return None
    unit=vector/length;normal=np.array([-unit[1],unit[0]])
    minimum=min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])
    bars=[line for line in _lines(mask,origin) if line['length']>=max(10,minimum*.18)
          and _angle_delta(line['angle'],math.atan2(normal[1],normal[0]))<math.radians(25)
          and bbox['x1']<=line['center'][0]<=bbox['x2'] and bbox['y1']<=line['center'][1]<=bbox['y2']]
    if not bars:return None
    angle=.5*math.atan2(sum(math.sin(2*l['angle'])*l['length'] for l in bars),
                         sum(math.cos(2*l['angle'])*l['length'] for l in bars))
    normal=np.array([math.cos(angle),math.sin(angle)])
    unit=np.array([-normal[1],normal[0]])
    if np.dot(unit,vector)<0:unit=-unit
    yy,xx=np.nonzero(mask)
    coords=np.stack((xx,yy),axis=1)+origin
    coords=coords[(coords[:,0]>=bbox['x1'])&(coords[:,0]<=bbox['x2'])&
                  (coords[:,1]>=bbox['y1'])&(coords[:,1]<=bbox['y2'])]
    if not len(coords):return None
    along=np.rint(coords@unit).astype(int);across=np.rint(coords@normal).astype(int)
    start,end=int(along.min()),int(along.max())
    left,right=int(across.min()),int(across.max())
    raster=np.zeros((end-start+1,right-left+1),np.uint8)
    raster[along-start,across-left]=1
    raster=cv2.morphologyEx(raster,cv2.MORPH_CLOSE,np.ones((1,3),np.uint8))
    band=max(2,min(5,int(minimum*.035)))
    profile=[];spans=[]
    for row in range(len(raster)):
        seen=np.any(raster[max(0,row-band):row+band+1],axis=0)
        edges=np.diff(np.r_[False,seen,False].astype(int))
        starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
        if not len(starts):profile.append(0);spans.append((0,0));continue
        best=int(np.argmax(ends-starts))
        profile.append(int(ends[best]-starts[best]));spans.append((starts[best]+left,ends[best]+left-1))
    threshold=max(12,minimum*.22,max(profile)*.25)
    groups=[]
    for row,size in enumerate(profile):
        if size<threshold:continue
        if not groups or row-groups[-1][-1]>2:groups.append([])
        groups[-1].append(row)
    plates=[]
    for group in groups:
        maximum=max(profile[row] for row in group)
        peak=[row for row in group if profile[row]>=maximum*.95]
        row=int(round(np.mean(peak)))
        middle=np.mean(spans[row])
        center=unit*(start+row)+normal*middle
        plates.append(dict(center=center,length=float(maximum)))
    if len(plates)<2:return None
    center=(plates[0]['center']+plates[-1]['center'])/2
    return dict(plates=plates,center=center,axis=unit)


def _source_polarity_center(mask,origin,bbox):
    area_box=(bbox['x2']-bbox['x1'])*(bbox['y2']-bbox['y1'])
    candidates=[]
    # Closing small gaps is confined to body interpretation, never to the
    # wire mask or final contact coordinates.
    for view in (mask,cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))):
        contours,_=cv2.findContours(view,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area=abs(cv2.contourArea(contour));perimeter=cv2.arcLength(contour,True)
            if len(contour)<5 or not .04*area_box<area<.8*area_box or not perimeter:continue
            (x,y),(a,b),_=cv2.fitEllipse(contour)
            if min(a,b)/max(a,b)<.6 or 4*math.pi*area/perimeter**2<.50:continue
            center=np.array([x,y])+origin
            if bbox['x1']<center[0]<bbox['x2'] and bbox['y1']<center[1]<bbox['y2']:
                candidates.append((area,center))
    return max(candidates,key=lambda item:item[0])[1] if candidates else None


def _polarized_plate_body(image,bbox,pair,kind):
    """Use the two lead-bearing outer plates; a '+' is not a third plate."""
    from backend.capacitor_geometry import capacitor_body_evidence
    ok,evidence=capacitor_body_evidence(image,bbox,maximum_gap_ratio=5 if kind=='battery' else 1.25)
    if ok:
        plates=[dict(center=np.asarray(p,float),length=float(length)) for p,length in
                zip(evidence['outer_plate_centers'],evidence['outer_plate_lengths'])]
        return dict(plates=plates,center=np.mean([p['center'] for p in plates],axis=0),
                    axis=np.asarray(evidence['electrical_axis'],float),source='outer_plates_with_opposed_leads')
    mask,origin=_crop(image,bbox)
    marks,labels=_marks(mask,origin,bbox)
    for mark in marks:
        if mark['kind']=='+':mask[labels==mark['label']]=0
    body=_observed_polarity_plates(mask,origin,bbox,pair)
    if body:body['source']='observed_plates_without_detached_polarity_glyphs'
    return body


def finalize_polarity_on_contacts(image,component,*,components=None,plate_body=None):
    """Name existing contacts from signs, battery lengths or digital plate shapes."""
    kind=component.get('type');terms=component.get('terminals') or []
    if kind not in POLARIZED_TWO_PIN_CLASSES or not terms:return None
    if len(terms)!=2:
        # A visible sign alone cannot certify a complete two-pole component.
        meta=dict(source='visible_polarity_on_final_wire_contacts',confirmed=False,
                  reason='opposite_polarized_lead_not_observed')
        component['terminal_identity_geometry']=meta
        for t in terms:t['identity_uncertain']=True
        for target in (component,component.setdefault('terminal_detection',{})):
            target['review_reasons']=list(dict.fromkeys((target.get('review_reasons') or [])+[meta['reason']]))
            target['review_required']=True
        return meta
    bbox=component['bbox'];pair=np.array([[t['position']['x'],t['position']['y']] for t in terms])
    mask,origin=_crop(image,bbox)
    minimum=min(bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1'])
    margin=max(8,min(28,minimum*.18))
    marks,_=_marks(mask,origin,bbox,include_attached=True,attached_margin=margin)
    if kind=='source_dc' and not any(m['kind']=='+' for m in marks):
        marks += _small_source_marks(mask,origin,bbox)
    if kind=='capacitor_polarized':
        from backend.symbol_semantics import visible_plus
        from backend.capacitor_sign_ownership import verified_plate_signs
        neighbors=[c['bbox'] for c in components or [] if c is not component and c.get('component_id')!=component.get('component_id') and c.get('bbox')]
        owned_signs=verified_plate_signs(image,bbox,visible_plus(image,bbox,neighbors))
        if owned_signs:
            # A nearby source may have another '+'. Use signs whose size,
            # position and ink ownership were verified against these plates.
            marks=[m for m in marks if m['kind']!='+']
        marks += [dict(m,kind='+',point=np.asarray(m['point'])) for m in owned_signs]
    plus=[m for m in marks if m['kind']=='+' and bbox['x1']-margin<=m['point'][0]<=bbox['x2']+margin
          and bbox['y1']-margin<=m['point'][1]<=bbox['y2']+margin]
    def owned(mark):
        x,y=mark['point']
        if bbox['x1']<=x<=bbox['x2'] and bbox['y1']<=y<=bbox['y2']:return True
        for other in components or []:
            if other is component or other.get('component_id')==component.get('component_id'):continue
            box=other.get('bbox') or {}
            if all(k in box for k in ('x1','y1','x2','y2')) and box['x1']<=x<=box['x2'] and box['y1']<=y<=box['y2']:
                return False
        return True
    plus=[m for m in plus if owned(m)]
    from backend.drawing_mode import CURRENT
    if kind=='source_dc' and not plus and CURRENT.get()=='handwritten':
        # Handwritten sources often put their sign above an external lead,
        # beyond the detector's circle box. Search a bounded neighbourhood
        # using four visible arms and ownership against neighbouring bodies.
        # Only electrical identity is changed; contact positions stay fixed.
        from backend.symbol_semantics import visible_plus
        neighbors=[c['bbox'] for c in components or [] if c is not component
            and c.get('component_id')!=component.get('component_id') and c.get('bbox')]
        for mark in visible_plus(image,bbox,neighbors):
            bounds=mark.get('bounds') or [*mark['point'],*mark['point']]
            if max(bounds[2]-bounds[0],bounds[3]-bounds[1])>max(12,minimum*.60):continue
            candidate=dict(mark,kind='+',point=np.asarray(mark['point'],float))
            if owned(candidate):plus.append(candidate)
    body = None
    if kind != 'source_dc':
        body = plate_body or _polarized_plate_body(image,bbox,pair,kind)
    center=body['center'] if body else _source_polarity_center(mask,origin,bbox) if kind=='source_dc' else None
    if center is None:center=pair.mean(axis=0)
    unit=body['axis'] if body else (pair[1]-pair[0])/max(1,np.linalg.norm(pair[1]-pair[0]))
    offsets=(pair-center)@unit
    candidates=[]
    for mark in plus:
        signed=float((mark['point']-center)@unit)
        if abs(signed)<max(2,minimum*.02):continue
        perpendicular=np.array([-unit[1],unit[0]])
        if mark.get('attached') and abs((mark['point']-center)@perpendicular)<5:
            if min(_support(mask,mark['point'],sign*unit,np.arange(10,22,2),origin) for sign in (-1,1))>=.65:
                continue  # A crossing along a wire is not a polarity sign.
        positive=int(np.argmax(offsets) if signed>0 else np.argmin(offsets))
        candidates.append((float(mark['score']),positive,mark))
    candidates.sort(key=lambda item:item[0],reverse=True)
    positive=None;reason='visible_polarity_mark_not_confirmed'
    if candidates:
        best=candidates[0]
        if not any(c[1]!=best[1] and c[0]>=best[0]-.08 for c in candidates[1:]):
            positive=best[1];reason='positive_terminal_from_visible_plus_after_wire_contacts'
    elif not plus and kind=='battery' and body:
        first,last=body['plates'][0],body['plates'][-1]
        difference=abs(first['length']-last['length'])
        if difference>=max(3,max(first['length'],last['length'])*.05):
            long=first if first['length']>last['length'] else last
            positive=int(np.argmax(offsets) if (long['center']-center)@unit>0 else np.argmin(offsets))
            reason='battery_long_plate_positive_after_wire_contacts'
    plate_evidence=None
    if positive is None and not plus and kind=='capacitor_polarized':
        from backend.printed_capacitor import plate_polarity,positive_contact
        plate_evidence=plate_polarity(image,bbox)
        positive=positive_contact(plate_evidence,pair)
        if positive is not None:reason=plate_evidence['method']+'_after_wire_contacts'
    meta=dict(source='visible_polarity_on_final_wire_contacts',confirmed=positive is not None,reason=reason,
              body_center=[round(float(v),2) for v in center],
              marks=[dict(kind='+',x=round(float(m['point'][0]),2),y=round(float(m['point'][1]),2)) for m in plus])
    if body:meta['plates']=[dict(length=round(p['length'],2),center=[round(float(v),2) for v in p['center']]) for p in body['plates']]
    if body:meta['plate_method']=body.get('source')
    if plate_evidence:meta['printed_plate_evidence']=plate_evidence
    if positive is None:
        component['terminal_identity_geometry']=meta
        for t in terms:t['identity_uncertain']=True
        for target in (component,component.setdefault('terminal_detection',{})):
            target['review_reasons']=list(dict.fromkeys((target.get('review_reasons') or [])+[reason]))
            target['review_required']=True
        return meta
    for index,terminal in enumerate(terms):
        pin='+' if index==positive else '-'
        terminal.update(pin=pin,pin_id=f"{component.get('component_id','COMP')}.{pin}",
                        electrical_role=ROLES[pin],index=1 if pin=='+' else 2,
                        identity_uncertain=False,polarity_source=meta['source'],confidence=max(.93,float(terminal.get('confidence') or 0)))
    # Keep the established '+' / '-' serialization order while preserving the
    # exact photographed contact positions and their wire evidence.
    component['terminals']=sorted(terms,key=lambda t:t['index'])
    component['terminal_identity_geometry']=meta
    td=component.setdefault('terminal_detection',{})
    polarity_reasons={'confirmar_identidade_eletrica_dos_lados','visible_polarity_mark_not_confirmed','opposite_polarized_lead_not_observed',
                      'modelo_multiclasse_baixa_confianca_revisar','orientacao_geometrica_ambigua',
                      'atribuicao_de_pinos_aos_fios_ambigua'}
    for target in (component,td):
        target['review_reasons']=[r for r in target.get('review_reasons',[]) if r not in polarity_reasons]
        target['review_required']=bool(target['review_reasons'])
    td['polarity_method']=meta['source']
    return meta
