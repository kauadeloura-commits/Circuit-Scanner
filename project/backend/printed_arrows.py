"""Two physical emission arrowheads, including heads attached to the body ink."""
from itertools import combinations
import cv2
import numpy as np
from backend.resistor_geometry import cached_vision


@cached_vision
def arrow_pairs(image, box):
    from backend.local_symbol_roles import local_view
    from backend.printed_capacitor import plate_ink
    from backend.resistor_geometry import _thin_symbol
    view, b, _, origin = local_view(image, box, 4)
    sk = (_thin_symbol(plate_ink(view)) > 0).astype(np.uint8)
    height, width = sk.shape
    scale = 4
    center = np.array([(b['x1'] + b['x2']) / 2, (b['y1'] + b['y2']) / 2])
    size = min(b['x2'] - b['x1'], b['y2'] - b['y1'])

    def neighbors(p):
        x, y = p
        # A staircase diagonal is one stroke, not a series of T junctions.
        return [(xx, yy) for yy in range(max(0, y-1), min(height, y+2))
                for xx in range(max(0, x-1), min(width, x+2))
                if (xx, yy) != p and sk[yy, xx]
                and not (xx != x and yy != y and (sk[y, xx] or sk[yy, x]))]

    degree = np.zeros(sk.shape, np.uint8)
    for y, x in np.column_stack(np.nonzero(sk)):
        degree[y, x] = len(neighbors((int(x), int(y))))
    count, labels, _, centers = cv2.connectedComponentsWithStats(((sk > 0) & (degree >= 3)).astype(np.uint8), 8)
    heads = []
    for j in range(1, count):
        region = {(int(x), int(y)) for y, x in np.column_stack(np.nonzero(labels == j))}
        head = centers[j]
        paths, seen = [], set(region)
        for seed in sorted({q for p in region for q in neighbors(p) if q not in region}):
            if seed in seen:
                continue
            seen.add(seed)
            previous = next((p for p in region if seed in neighbors(p)), None)
            current = seed
            for _ in range(scale * 22):
                choices = [p for p in neighbors(current) if p != previous and p not in seen]
                if not choices:
                    break
                following = min(choices, key=lambda p: np.linalg.norm(np.array(p) - head))
                if labels[following[1], following[0]] not in (0, j):
                    break
                previous, current = current, following
                seen.add(current)
                if len(choices) > 1 and degree[current[1], current[0]] >= 3:
                    break
            point = np.asarray(current, float)
            length = float(np.linalg.norm(point - head))
            if length >= scale * 1.2:
                paths.append((point, length))
        if len(paths) != 3:
            continue
        for tail in range(3):
            first, second = [paths[k] for k in range(3) if k != tail]
            shaft = paths[tail][0] - head
            direction = -shaft / max(1, np.linalg.norm(shaft))
            normal = np.array([-direction[1], direction[0]])
            u, v = (first[0] - head) / first[1], (second[0] - head) / second[1]
            if paths[tail][1] < max(first[1], second[1]) * .60 or max(first[1], second[1]) > size * .40:
                continue
            if not .30 < first[1] / second[1] < 3 or (u @ normal) * (v @ normal) > -.10 or (u @ direction) * (v @ direction) < -.10:
                continue
            away = head - center
            if np.linalg.norm(away) < size * .30 or direction @ away < np.linalg.norm(away) * .25:
                continue
            p = head / scale + origin
            bounds = np.stack([head, first[0], second[0], paths[tail][0]]) / scale + origin
            heads.append(dict(point=p, direction=direction, head_confirmed=True,
                              bounds=[float(bounds[:, 0].min()), float(bounds[:, 1].min()),
                                      float(bounds[:, 0].max()), float(bounds[:, 1].max())]))
    native_size = size / scale
    return [[a, b] for a, b in combinations(heads, 2)
            if a['direction'] @ b['direction'] >= .82
            and max(4, native_size * .05) <= np.linalg.norm(a['point'] - b['point']) <= max(16, native_size * .6)]
