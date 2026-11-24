"""Bounded two-layer A* routing between assigned through-hole pads.

Foreign pads and existing copper reserve clearance on both layers. Routes that
cannot be found remain explicitly unconnected; no fictitious completion flag.
"""
import heapq,math,time
from collections import defaultdict

GRID=.5
WIDTH=.25
CLEARANCE=.25
VIA_SIZE=.7
VIA_DRILL=.35

def route_board(pads,width,height,max_seconds=20):
    nx,ny=int(width/GRID),int(height/GRID);blocked=[defaultdict(set),defaultdict(set)]
    tracks=[];vias=[];groups=defaultdict(list);started=time.monotonic();reports=[]
    def cell(x,y):return round(x/GRID),round(y/GRID)
    def reserve(layer,x,y,radius,net):
        cx,cy=cell(x,y);rad=math.ceil((radius+GRID*.71)/GRID)
        for dx in range(-rad,rad+1):
            for dy in range(-rad,rad+1):
                if math.hypot((cx+dx)*GRID-x,(cy+dy)*GRID-y)<=radius+GRID*.71:blocked[layer][cx+dx,cy+dy].add(net)
    for p in pads:
        p=dict(p,grid=cell(p['x'],p['y']));groups[p['net']].append(p)
        radius=p['size']/2+CLEARANCE+WIDTH/2
        if p.get('rect'):radius=p['size']*.71+CLEARANCE+WIDTH/2
        for layer in (0,1):reserve(layer,p['x'],p['y'],radius,p['net'])
    def free(state,net):
        x,y,l=state
        return 2<=x<=nx-2 and 2<=y<=ny-2 and not (blocked[l].get((x,y),set())-{net})
    def search(source,goals,net):
        targets=set(goals)
        def h(state):return min(abs(state[0]-x)+abs(state[1]-y) for x,y,l in goals)
        queue=[];distance={};prev={}
        for layer in (0,1):
            s=(*source,layer)
            if free(s,net):distance[s]=0.;heapq.heappush(queue,(h(s),0.,s))
        count=0
        while queue:
            _,cost,state=heapq.heappop(queue)
            if cost!=distance[state]:continue
            if state in targets:
                path=[state]
                while state in prev:state=prev[state];path.append(state)
                return list(reversed(path))
            count+=1
            if count>100000 or (count%1024==0 and time.monotonic()-started>max_seconds):return None
            x,y,l=state
            choices=[((x+dx,y+dy,l),1.+(.2 if (dx!=0)==bool(l) else 0)) for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))]
            # A via needs clearance for its diameter on both copper layers.
            via_safe=all(not (blocked[ll].get((x+dx,y+dy),set())-{net}) for ll in (0,1) for dx,dy in ((0,0),(1,0),(-1,0),(0,1),(0,-1)))
            if via_safe:choices.append(((x,y,1-l),8.))
            for nxt,delta in choices:
                if not free(nxt,net):continue
                nc=cost+delta
                if nc<distance.get(nxt,math.inf):distance[nxt]=nc;prev[nxt]=state;heapq.heappush(queue,(nc+h(nxt),nc,nxt))
        return None
    def add_track(a,b,layer,net):
        if a==b:return
        tracks.append(dict(start=a,end=b,layer='F.Cu' if layer==0 else 'B.Cu',net=net,width=WIDTH))
        steps=max(1,math.ceil(math.dist(a,b)/(.5*GRID)))
        for i in range(steps+1):
            p=(a[0]+(b[0]-a[0])*i/steps,a[1]+(b[1]-a[1])*i/steps)
            reserve(layer,*p,WIDTH/2+CLEARANCE+WIDTH/2,net)
    # Difficult multi-pad nets first; all endpoints use the declared electrical net.
    for net,items in sorted(groups.items(),key=lambda pair:(-len(pair[1]),pair[0])):
        tree={(*items[0]['grid'],l) for l in (0,1)};connected=[items[0]['member']];remaining=list(items[1:]);unrouted=[]
        while remaining:
            p=min(remaining,key=lambda p:min(abs(p['grid'][0]-x)+abs(p['grid'][1]-y) for x,y,l in tree));remaining.remove(p)
            if time.monotonic()-started>=max_seconds:path=None
            else:path=search(p['grid'],tree,net)
            if not path:unrouted.append(p['member']);continue
            add_track((p['x'],p['y']),(path[0][0]*GRID,path[0][1]*GRID),path[0][2],net)
            run=path[0];last=run;direction=None
            for nxt in path[1:]:
                if nxt[2]!=last[2]:
                    add_track((run[0]*GRID,run[1]*GRID),(last[0]*GRID,last[1]*GRID),last[2],net)
                    pos=(last[0]*GRID,last[1]*GRID);vias.append(dict(position=pos,net=net,size=VIA_SIZE,drill=VIA_DRILL))
                    for layer in (0,1):reserve(layer,*pos,VIA_SIZE/2+CLEARANCE+WIDTH/2,net)
                    run=nxt;direction=None
                else:
                    d=(nxt[0]-last[0],nxt[1]-last[1])
                    if direction is not None and d!=direction:
                        add_track((run[0]*GRID,run[1]*GRID),(last[0]*GRID,last[1]*GRID),last[2],net);run=last
                    direction=d
                last=nxt
            add_track((run[0]*GRID,run[1]*GRID),(last[0]*GRID,last[1]*GRID),last[2],net)
            # The original root pad's exact coordinate is connected on both
            # layers by its plated hole, not merely the rounded grid position.
            for root in items:
                if root['grid']==(last[0],last[1]):add_track((root['x'],root['y']),(last[0]*GRID,last[1]*GRID),last[2],net)
            tree.update(path);connected.append(p['member'])
        reports.append(dict(net=net,pad_count=len(items),connected_members=connected,unrouted_members=unrouted))
    report=dict(strategy='two_layer_astar',grid_mm=GRID,track_width_mm=WIDTH,clearance_mm=CLEARANCE,
                tracks=len(tracks),vias=len(vias),nets=reports,complete=not any(r['unrouted_members'] for r in reports),
                duration_seconds=round(time.monotonic()-started,3))
    return tracks,vias,report
