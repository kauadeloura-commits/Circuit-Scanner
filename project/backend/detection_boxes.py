"""Strict component-box containment, independent of class and IoU thresholds."""

import copy
import math


def _coordinates(detection):
    box = detection.get('bbox') or {}
    try:
        coords = tuple(float(box[key]) for key in ('x1', 'y1', 'x2', 'y2'))
    except (KeyError, TypeError, ValueError):
        raise ValueError('Existe uma caixa de componente inválida.') from None
    if not all(math.isfinite(v) for v in coords) or coords[2] <= coords[0] or coords[3] <= coords[1]:
        raise ValueError('Existe uma caixa de componente inválida.')
    return coords


def fully_contains(outer, inner):
    """Inclusive edges; no tolerance that could turn partial overlap into containment."""
    ax1, ay1, ax2, ay2 = _coordinates(outer)
    bx1, by1, bx2, by2 = _coordinates(inner)
    return ax1 <= bx1 and ay1 <= by1 and ax2 >= bx2 and ay2 >= by2


def validate_no_nested_boxes(detections):
    """Reject manual containment before changing files, training or downstream artifacts."""
    for i, first in enumerate(detections):
        _coordinates(first)
        for j in range(i + 1, len(detections)):
            second = detections[j]
            if fully_contains(first, second) or fully_contains(second, first):
                raise ValueError(
                    f'As caixas {i + 1} ({first.get("class", "componente")}) e '
                    f'{j + 1} ({second.get("class", "componente")}) estão inteiramente '
                    'uma dentro da outra. Remova a detecção duplicada ou ajuste as caixas. '
                    'A sobreposição parcial é permitida.'
                )


def remove_nested_boxes(detections, *, renumber=True):
    """Keep complete accepted symbols; discard their fully contained child detections.

    Used after geometry/confidence validation and oversized-region rejection.
    Larger boxes win. Identical boxes prefer manual review, then confidence and
    consensus, with deterministic ties. Partial overlap is never filtered here.
    Returns independent objects and a rejection audit tied to final survivor IDs.
    """
    def priority(item):
        x1, y1, x2, y2 = _coordinates(item)
        return (-(x2-x1)*(y2-y1), -int(bool(item.get('manual_reviewed'))),
                -float(item.get('confidence', 0)), -int(item.get('consensus_votes', 0)),
                y1, x1, y2, x2, str(item.get('class', '')), str(item.get('detection_id', '')))

    kept, rejected = [], []
    for item in sorted(detections, key=priority):
        enclosing = next((outer for outer in kept if fully_contains(outer, item)), None)
        if enclosing is None:
            kept.append(copy.deepcopy(item))
        else:
            reject = copy.deepcopy(item)
            reject['rejection_reason'] = 'component_box_fully_inside_another'
            reject['enclosing_class'] = enclosing.get('class')
            reject['enclosing_bbox'] = copy.deepcopy(enclosing['bbox'])
            rejected.append((reject, enclosing))
    kept.sort(key=lambda item: (_coordinates(item)[1], _coordinates(item)[0]))
    if renumber:
        for i, item in enumerate(kept, 1):
            item['detection_id'] = f'DET_{i:03d}'
    for reject, enclosing in rejected:
        reject['enclosing_detection_id'] = enclosing['detection_id']
    return kept, [reject for reject, _ in rejected]


def merge_same_symbol_boxes(detections):
    """Deduplicate nearly identical same-class boxes after subtype corrections.

    Ordinary partial overlap remains allowed. Boxes must cover almost the same
    area and have nearly the same centre; neighbouring symbols are not merged.
    Manual boxes always take precedence and two reviewed boxes are never merged.
    """
    def priority(item):
        return (-int(bool(item.get('manual_reviewed'))),
                -float(item.get('best_learned_confidence', item.get('confidence', 0))),
                -int(item.get('consensus_votes', 0)), str(item.get('detection_id', '')))
    kept, rejected = [], []
    for item in sorted(detections, key=priority):
        a = _coordinates(item)
        winner = None
        for other in kept:
            if item.get('class') != other.get('class') or (item.get('manual_reviewed') and other.get('manual_reviewed')):
                continue
            b = _coordinates(other)
            area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
            inter = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
            iou = inter / max(1., area_a+area_b-inter)
            dx, dy = abs(a[0]+a[2]-b[0]-b[2])/2, abs(a[1]+a[3]-b[1]-b[3])/2
            same_body=False
            if item.get('class')=='resistor':
                ea=(item.get('geometry_validation') or {}).get('evidence') or {}
                eb=(other.get('geometry_validation') or {}).get('evidence') or {}
                if ea.get('opposed_leads') and eb.get('opposed_leads') and ea.get('body_corners') and eb.get('body_corners'):
                    ca,cb=ea.get('body_center'),eb.get('body_center')
                    if ca and cb:
                        corner_error=max(min(math.dist(p,q) for q in eb['body_corners']) for p in ea['body_corners'])
                        same_body=math.dist(ca,cb)<=2 and corner_error<=3
            if item.get('class')=='gnd':
                ea=(item.get('terminal_bar_ground') or {}).get('ground_body_bbox')
                eb=(other.get('terminal_bar_ground') or {}).get('ground_body_bbox')
                if ea and eb:
                    # A recovered ground and a learned ground may include
                    # different wire context. Their observed bars identify
                    # the same symbol independently of those proposal boxes.
                    ga=item['terminal_bar_ground'];gb=other['terminal_bar_ground']
                    turn=ga.get('source_quarter_turn')
                    distal=('y2','x1','y1','x2')[turn%4] if isinstance(turn,int) else None
                    error=max(abs(ea[k]-eb[k]) for k in ('x1','y1','x2','y2') if k!=distal)
                    same_body=(turn==gb.get('source_quarter_turn') and error<=3
                               and abs(ga.get('bar_length_px',0)-gb.get('bar_length_px',0))<=3)
            if same_body or iou >= .75 and dx <= .15*min(a[2]-a[0], b[2]-b[0]) and dy <= .15*min(a[3]-a[1], b[3]-b[1]):
                winner = other
                break
        if winner is None:
            kept.append(copy.deepcopy(item))
        else:
            reject = copy.deepcopy(item)
            reject.update(rejection_reason='duplicate_same_symbol_after_classification',
                          survivor_detection_id=winner.get('detection_id'), survivor_bbox=copy.deepcopy(winner['bbox']))
            rejected.append(reject)
    kept.sort(key=lambda item: (_coordinates(item)[1], _coordinates(item)[0]))
    return kept, rejected


def enclose_verified_resistor_body(learned_box, body_box, evidence):
    """Keep an observed rectangular/zigzag body when a learned crop cuts it.

    The geometry must have already found the body AND its opposed leads.
    Only a bounded extension of an overlapping proposal is allowed.
    """
    if not evidence.get('opposed_leads') or not (evidence.get('iec_rectangular_body') or evidence.get('alternating_connected_teeth')):
        return None
    a,b=_coordinates({'bbox':learned_box}),_coordinates({'bbox':body_box})
    aa,ab=(a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1])
    inter=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    if inter>=.85*ab or inter/max(1.,aa+ab-inter)<.30:return None
    x1,y1,x2,y2=min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])
    if (x2-x1)*(y2-y1)>1.8*aa:return None
    return dict(x1=x1,y1=y1,x2=x2,y2=y2,width=x2-x1,height=y2-y1,center_x=(x1+x2)/2,center_y=(y1+y2)/2)
