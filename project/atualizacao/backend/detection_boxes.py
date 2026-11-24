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
