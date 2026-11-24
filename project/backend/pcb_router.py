"""Clearance-aware routing, preferring one continuous bottom copper layer.

Try different net orders before resorting to top copper. Compare complete boards
by connectivity first, then by top-layer spans and vias. Plated pads join layers;
no physical jumper components or fictitious connections are silently inserted.
"""
import heapq
import math
import time
from collections import defaultdict

ROUTING_VERSION = 'single-layer-first-1'
GRID = .5
WIDTH = .25
CLEARANCE = .25
VIA_SIZE = .7
VIA_DRILL = .35
PRIMARY = 1  # B.Cu, convenient for through-hole boards.


def routing_is_current(manifest):
    return not manifest.get('pcb') or (manifest.get('routing') or {}).get('routing_version') == ROUTING_VERSION


def count_secondary_spans(tracks):
    """Count connected top-copper islands, including branches into old segments."""
    top = [t for t in tracks if t['layer']=='F.Cu']
    parent = list(range(len(top)))
    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def on(p, a, b):
        dx, dy = b[0]-a[0], b[1]-a[1]
        return (abs((p[0]-a[0])*dy-(p[1]-a[1])*dx)<1e-7
                and min(a[0], b[0])-1e-7<=p[0]<=max(a[0], b[0])+1e-7
                and min(a[1], b[1])-1e-7<=p[1]<=max(a[1], b[1])+1e-7)
    for i, a in enumerate(top):
        for j, b in enumerate(top[:i]):
            if a['net']==b['net'] and (any(on(p, b['start'], b['end']) for p in (a['start'], a['end']))
                    or any(on(p, a['start'], a['end']) for p in (b['start'], b['end']))):
                parent[root(i)] = root(j)
    return len({root(i) for i in range(len(top))})


def route_board(pads, width, height, max_seconds=20):
    started = time.monotonic()
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
               for v in (width, height)):
        raise ValueError('Dimensões de placa inválidas.')
    if not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or max_seconds < 0:
        raise ValueError('Tempo de roteamento inválido.')
    deadline = started + max_seconds
    nx, ny = int(width / GRID), int(height / GRID)
    groups = defaultdict(list)
    fixed = [{}, {}]
    invalid = []

    def cell(x, y):
        return round(x / GRID), round(y / GRID)

    def reserve(blocked, layer, x, y, radius, net):
        cx, cy = cell(x, y)
        radius += GRID * .71  # Covers edges between neighbouring cells.
        rad = math.ceil(radius / GRID)
        for dx in range(-rad, rad + 1):
            for dy in range(-rad, rad + 1):
                if math.hypot((cx + dx) * GRID - x, (cy + dy) * GRID - y) <= radius:
                    key = (cx + dx, cy + dy)
                    blocked[layer][key] = blocked[layer].get(key, frozenset()) | {net}

    for raw in pads:
        p = dict(raw)
        if not isinstance(p.get('net'), str) or not p['net'] or not isinstance(p.get('member'), str):
            raise ValueError('Pad sem identificação de terminal ou rede.')
        if not all(isinstance(p.get(k), (int, float)) and not isinstance(p[k], bool) and math.isfinite(p[k])
                   for k in ('x', 'y', 'size')) or p['size'] <= 0:
            raise ValueError('Geometria de pad inválida.')
        p['grid'] = cell(p['x'], p['y'])
        if not (1+p['size']/2 <= p['x'] <= width-1-p['size']/2
                and 1+p['size']/2 <= p['y'] <= height-1-p['size']/2):
            invalid.append(p['member'])
        groups[p['net']].append(p)
        radius = p['size'] * (.71 if p.get('rect') else .5) + CLEARANCE + WIDTH/2
        for layer in (0, 1):
            reserve(fixed, layer, p['x'], p['y'], radius, p['net'])

    def span(items):
        return max(p['x'] for p in items)-min(p['x'] for p in items) + max(p['y'] for p in items)-min(p['y'] for p in items)

    active = [n for n, items in groups.items() if len(items)>1]
    short_order = sorted(active, key=lambda n: (span(groups[n]), len(groups[n]), n))
    fanout_order = sorted(active, key=lambda n: (-len(groups[n]), span(groups[n]), n))
    orders = [short_order]
    if fanout_order != short_order:
        orders.append(fanout_order)
    reverse_order = list(reversed(short_order))
    if reverse_order not in orders:
        orders.append(reverse_order)

    def attempt(order, dual, stop):
        blocked = [dict(fixed[0]), dict(fixed[1])]
        tracks, vias, reports = [], [], {}
        via_keys = set()
        secondary_spans = 0

        def free(state, net):
            x, y, layer = state
            return (2 <= x <= nx-2 and 2 <= y <= ny-2
                    and not any(owner != net for owner in blocked[layer].get((x, y), ())))

        def search(source, goals, net):
            if time.monotonic() >= stop:
                return None
            targets = set(goals)
            points = {(x, y) for x, y, l in targets}
            rows = defaultdict(list); cols = defaultdict(list)
            for x, y in points:
                rows[y].append(x); cols[x].append(y)
            segments = []; covered = set()
            for mapping, vertical in ((rows, False), (cols, True)):
                for axis, values in mapping.items():
                    values.sort(); start = end = values[0]
                    for value in values[1:] + [math.inf]:
                        if value == end+1:
                            end = value; continue
                        if end > start:
                            segments.append((axis, axis, start, end) if vertical else (start, end, axis, axis))
                            covered.update((axis, v) if vertical else (v, axis) for v in range(start, end+1))
                        start = end = value
            segments.extend((x, x, y, y) for x, y in points-covered)
            heuristic = {}

            def h(x, y):
                key = x, y
                if key not in heuristic:
                    heuristic[key] = min(max(a-x, 0, x-b)+max(c-y, 0, y-d) for a,b,c,d in segments)
                return heuristic[key]

            queue, distance, previous = [], {}, {}
            for layer in ((PRIMARY, 1-PRIMARY) if dual else (PRIMARY,)):
                s = (*source, layer)
                if free(s, net):
                    cost = 8. if layer != PRIMARY else 0.
                    distance[s] = cost
                    heapq.heappush(queue, (cost+h(*source), cost, s))
            count = 0; via_cache = {}
            while queue:
                _, cost, state = heapq.heappop(queue)
                if cost != distance.get(state):
                    continue
                if state in targets:
                    path = [state]
                    while state in previous:
                        state = previous[state]; path.append(state)
                    return path[::-1]
                count += 1
                if count % 256 == 0 and time.monotonic() >= stop:
                    return None
                x, y, layer = state; choices = []
                for direction, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
                    nxt = (x+dx, y+dy, layer)
                    if free(nxt, net):
                        delta = 1. if layer == PRIMARY else 1.1
                        choices.append((nxt, delta))
                if dual:
                    key = (x, y)
                    if key not in via_cache:
                        r = math.ceil((VIA_SIZE/2+CLEARANCE+WIDTH/2)/GRID)
                        via_cache[key] = all(free((x+dx, y+dy, ll), net)
                            for ll in (0, 1) for dx in range(-r, r+1) for dy in range(-r, r+1)
                            if math.hypot(dx*GRID, dy*GRID) <= VIA_SIZE/2+CLEARANCE+WIDTH/2)
                    if via_cache[key]:
                        choices.append(((x, y, 1-layer), 24.))
                for nxt, delta in choices:
                    nc = cost+delta
                    if nc < distance.get(nxt, math.inf):
                        distance[nxt] = nc; previous[nxt] = state
                        heapq.heappush(queue, (nc+h(nxt[0], nxt[1]), nc, nxt))
            return None

        def add_track(a, b, layer, net):
            if a == b:
                return
            tracks.append(dict(start=a, end=b, layer='B.Cu' if layer == PRIMARY else 'F.Cu', net=net, width=WIDTH))
            steps = max(1, math.ceil(math.dist(a, b)/(.5*GRID)))
            for i in range(steps+1):
                x, y = a[0]+(b[0]-a[0])*i/steps, a[1]+(b[1]-a[1])*i/steps
                reserve(blocked, layer, x, y, WIDTH+CLEARANCE, net)

        def emit(path, net, source, roots):
            nonlocal secondary_spans
            add_track((source['x'], source['y']), (path[0][0]*GRID, path[0][1]*GRID), path[0][2], net)
            run = last = path[0]; direction = None
            if run[2] != PRIMARY:
                secondary_spans += 1
            for nxt in path[1:]:
                if nxt[2] != last[2]:
                    add_track((run[0]*GRID, run[1]*GRID), (last[0]*GRID, last[1]*GRID), last[2], net)
                    pos = (last[0]*GRID, last[1]*GRID); key = (pos, net)
                    if key not in via_keys:
                        vias.append(dict(position=pos, net=net, size=VIA_SIZE, drill=VIA_DRILL)); via_keys.add(key)
                        for layer in (0, 1):
                            reserve(blocked, layer, *pos, VIA_SIZE/2+CLEARANCE+WIDTH/2, net)
                    if nxt[2] != PRIMARY:
                        secondary_spans += 1
                    run = nxt; direction = None
                else:
                    d = (nxt[0]-last[0], nxt[1]-last[1])
                    if direction is not None and direction != d:
                        add_track((run[0]*GRID, run[1]*GRID), (last[0]*GRID, last[1]*GRID), last[2], net)
                        run = last
                    direction = d
                last = nxt
            add_track((run[0]*GRID, run[1]*GRID), (last[0]*GRID, last[1]*GRID), last[2], net)
            for root in roots:
                if root['grid'] == last[:2]:
                    add_track((root['x'], root['y']), (last[0]*GRID, last[1]*GRID), last[2], net)

        for net in order + sorted(set(groups)-set(order)):
            items = groups[net]
            mx = sorted(p['x'] for p in items)[len(items)//2]; my = sorted(p['y'] for p in items)[len(items)//2]
            root = min(items, key=lambda p: (abs(p['x']-mx)+abs(p['y']-my), p['member']))
            root_layers = [l for l in ((0, 1) if dual else (PRIMARY,)) if free((*root['grid'], l), net)]
            valid = root['member'] not in invalid and bool(root_layers)
            tree = {(*root['grid'], l) for l in root_layers} if valid else set()
            connected = [root['member']] if valid else []; roots = [root] if valid else []
            remaining = [p for p in items if p is not root]; unrouted = [] if valid else [root['member']]
            while remaining:
                p = min(remaining, key=lambda p: (min((abs(p['grid'][0]-q['grid'][0])+abs(p['grid'][1]-q['grid'][1]) for q in roots), default=0), p['member']))
                remaining.remove(p)
                path = search(p['grid'], tree, net) if tree and p['member'] not in invalid else None
                if path is None:
                    unrouted.append(p['member']); continue
                emit(path, net, p, roots); tree.update(path)
                # A plated pad provides an actual connection between both layers.
                tree.update((*p['grid'], l) for l in ((0, 1) if dual else (PRIMARY,)))
                roots.append(p); connected.append(p['member'])
            reports[net] = dict(net=net, pad_count=len(items), connected_members=connected, unrouted_members=unrouted)
        report = dict(nets=[reports[n] for n in sorted(reports)], tracks=len(tracks), vias=len(vias),
            complete=not any(r['unrouted_members'] for r in reports.values()), secondary_layer_spans=count_secondary_spans(tracks),
            total_length_mm=round(sum(math.dist(t['start'], t['end']) for t in tracks), 3),
            secondary_length_mm=round(sum(math.dist(t['start'], t['end']) for t in tracks if t['layer']=='F.Cu'), 3))
        return tracks, vias, report

    def score(result):
        r = result[2]
        return (sum(len(n['unrouted_members']) for n in r['nets']), r['secondary_layer_spans'], r['vias'], r['secondary_length_mm'], r['total_length_mm'])

    best = None; attempts = []
    # First retain the previous feasible solution as a safety bound. New route
    # orders may never sacrifice a completed connection to save a via.
    conflicting = [p['member'] for items in groups.values() for p in items
                   if any(n != p['net'] for n in fixed[0].get(p['grid'], ()))]
    if not invalid and not conflicting and max_seconds>0:
        from backend.pcb_router_fallback import route_board as fallback
        ft, fv, fr = fallback(pads, width, height, max_seconds=min(2., max_seconds*.18))
        if sum(math.dist(t['start'], t['end']) for t in ft if t['layer']=='F.Cu') > sum(math.dist(t['start'], t['end']) for t in ft if t['layer']=='B.Cu'):
            for t in ft:
                t['layer'] = 'B.Cu' if t['layer']=='F.Cu' else 'F.Cu'
        fr.update(secondary_layer_spans=count_secondary_spans(ft),
            secondary_length_mm=round(sum(math.dist(t['start'], t['end']) for t in ft if t['layer']=='F.Cu'), 3),
            total_length_mm=round(sum(math.dist(t['start'], t['end']) for t in ft), 3))
        best = ft, fv, fr
        attempts.append(dict(layers=2, feasible_safety_bound=True, unrouted=score(best)[0], vias=len(fv), secondary_layer_spans=fr['secondary_layer_spans']))
    plans = [(orders[0], False, .2)]
    if len(orders)>1:
        plans.append((orders[1], False, .35))
    plans.append((fanout_order, True, .7))
    if len(orders)>2:
        plans.append((orders[2], False, .82))
    plans.append((orders[0], True, 1.))
    for order, dual, fraction in plans:
        if best is not None and time.monotonic() >= deadline:
            break
        result = attempt(order, dual, min(deadline, started+max_seconds*fraction))
        attempts.append(dict(layers=2 if dual else 1, unrouted=score(result)[0], vias=result[2]['vias'], secondary_layer_spans=result[2]['secondary_layer_spans']))
        if best is None or score(result) < score(best):
            best = result
        if best[2]['complete'] and best[2]['secondary_layer_spans'] == 0:
            break
    tracks, vias, report = best
    report.update(strategy='single_layer_first_multi_order_astar', routing_version=ROUTING_VERSION,
        grid_mm=GRID, track_width_mm=WIDTH, clearance_mm=CLEARANCE, preferred_layer='B.Cu',
        single_layer=not any(t['layer']=='F.Cu' for t in tracks), attempts=attempts,
        duration_seconds=round(time.monotonic()-started, 3), invalid_members=invalid,
        budget_exhausted=time.monotonic()>=deadline, conflicting_members=conflicting)
    return tracks, vias, report
