"""Compact empty bands and route clean orthogonal connections."""
import heapq,math

class RoutingConflict(ValueError):pass

def separate_presentation(components,member_net):
    """Move only presentation geometry; electrical memberships are immutable."""
    placed=[];warnings=[]
    def bounds(c):
        points=list(c['pins'].values())
        x1,y1=min(p[0] for p in points),min(p[1] for p in points)
        x2,y2=max(p[0] for p in points),max(p[1] for p in points)
        # The lead envelope is not the body. Treating two nearby lead ends as
        # overlapping symbols displaced transistors far from their photo.
        if len(points)==2:
            if x2-x1>=y2-y1:x1,x2=x1+(x2-x1)*.2,x2-(x2-x1)*.2
            else:y1,y2=y1+(y2-y1)*.2,y2-(y2-y1)*.2
        elif len(points)==3:
            x1,x2=x1+(x2-x1)*.08,x2-(x2-x1)*.08
            y1,y2=y1+(y2-y1)*.08,y2-(y2-y1)*.08
        return x1-2,y1-2,x2+2,y2+2
    def collision(candidate):
        box=bounds(candidate)
        for other in placed:
            ob=bounds(other);shared=False
            for pin,p in candidate['pins'].items():
                for name,q in other['pins'].items():
                    same=member_net[candidate['id']+'.'+pin]==member_net[other['id']+'.'+name]
                    if max(abs(p[0]-q[0]),abs(p[1]-q[1]))<4 and not same:return True
                    if p==q and same:shared=True
            if shared:continue
            if max(box[0],ob[0])<min(box[2],ob[2]) and max(box[1],ob[1])<min(box[3],ob[3]):return True
        return False
    def move(c,dx,dy):
        return dict(c,pins={pin:(round(p[0]+dx,2),round(p[1]+dy,2)) for pin,p in c['pins'].items()},
                    center=(c['center'][0]+dx,c['center'][1]+dy),
                    lines=[[(p[0]+dx,p[1]+dy) for p in line] for line in c['lines']],
                    texts=[(x+dx,y+dy,text) for x,y,text in c['texts']])
    # Ground/power markers are compact and easier to place than a multi-pin body.
    for c in sorted(components,key=lambda c:(len(c['pins'])==1,components.index(c))):
        selected=c
        if collision(c):
            found=False
            for distance in range(8,257,8):
                for dx,dy in ((distance,0),(-distance,0),(0,distance),(0,-distance),(distance,distance),(-distance,distance),(distance,-distance),(-distance,-distance)):
                    shifted=move(c,dx,dy)
                    if min(min(p) for p in shifted['pins'].values())<4 or collision(shifted):continue
                    selected=shifted;found=True;break
                if found:break
            if not found:
                right=max((bounds(p)[2] for p in placed),default=0)+12
                selected=move(c,right-bounds(c)[0],max(0,8-bounds(c)[1]))
            warnings.append(c['id']+': posição do desenho ajustada para separar símbolos e redes; ligações preservadas.')
        placed.append(selected)
    by_id={c['id']:c for c in placed}
    return [by_id[c['id']] for c in components],warnings

def compact_axis(intervals,gap=24):
    merged=[]
    for lo,hi in sorted(intervals):
        if merged and lo<=merged[-1][1]+4:merged[-1][1]=max(hi,merged[-1][1])
        else:merged.append([lo,hi])
    gaps=[(a[1],b[0],b[0]-a[1]-gap) for a,b in zip(merged,merged[1:]) if b[0]-a[1]>gap]
    def convert(value):
        result=value
        for start,end,remove in gaps:
            if value>=end:result-=remove
            elif value>start:result-=remove*(value-start)/(end-start)
        return round(result,2)
    return convert

def aligned_pins(kind,pins):
    pins=dict(pins)
    if len(pins)==2:
        names=list(pins);a,b=(pins[n] for n in names);dx,dy=abs(a[0]-b[0]),abs(a[1]-b[1])
        if dx>=dy:
            y=2*round((a[1]+b[1])/4);pins={names[0]:(a[0],y),names[1]:(b[0],y)}
        else:
            x=2*round((a[0]+b[0])/4);pins={names[0]:(x,a[1]),names[1]:(x,b[1])}
    if kind=='opamp':
        plus,minus,out=(pins[k] for k in ('+','-','OUT'))
        cx,cy=(plus[0]+minus[0])/2,(plus[1]+minus[1])/2
        if abs(out[0]-cx)>=abs(out[1]-cy):
            x=2*round(cx/2);y=2*round(cy/2)
            pins={'+':(x,plus[1]),'-':(x,minus[1]),'OUT':(out[0],y)}
        else:
            x=2*round(cx/2);y=2*round(cy/2)
            pins={'+':(plus[0],y),'-':(minus[0],y),'OUT':(x,out[1])}
    if kind.startswith('transistor_') or kind.startswith('mosfet_'):
        control='B' if kind.startswith('transistor_') else 'G'
        names=[pin for pin in pins if pin!=control];a=pins[control];b,c=(pins[n] for n in names)
        cx,cy=2*round((b[0]+c[0])/4),2*round((b[1]+c[1])/4)
        if abs(cx-a[0])>=abs(cy-a[1]):
            pins={control:(a[0],cy),names[0]:(cx,b[1]),names[1]:(cx,c[1])}
        else:
            pins={control:(cx,a[1]),names[0]:(b[0],cy),names[1]:(c[0],cy)}
    return pins

def orthogonal(a,b,components,foreign=()):
    ignored={c['id'] for c in components if a in c['pins'].values() or b in c['pins'].values()}
    boxes=[(p[0]-1.5,p[1]-1.5,p[0]+1.5,p[1]+1.5) for p in foreign]
    for c in components:
        if c['id'] in ignored:continue
        pts=list(c['pins'].values());x1=min(p[0] for p in pts)-3;x2=max(p[0] for p in pts)+3;y1=min(p[1] for p in pts)-3;y2=max(p[1] for p in pts)+3
        boxes.append((x1,y1,x2,y2))
    def clear(p,q):
        for x1,y1,x2,y2 in boxes:
            if p[0]==q[0] and x1<p[0]<x2 and max(min(p[1],q[1]),y1)<min(max(p[1],q[1]),y2):return False
            if p[1]==q[1] and y1<p[1]<y2 and max(min(p[0],q[0]),x1)<min(max(p[0],q[0]),x2):return False
        return True
    def simplify(path):
        result=[]
        for p in path:
            if result and p==result[-1]:continue
            if len(result)>1 and (result[-2][0]==result[-1][0]==p[0] or result[-2][1]==result[-1][1]==p[1]):result.pop()
            result.append(p)
        return result
    for path in ([a,(b[0],a[1]),b],[a,(a[0],b[1]),b]):
        if all(clear(p,q) for p,q in zip(path,path[1:])):return simplify(path)
    xs=sorted({a[0],b[0],*[x for box in boxes for x in (2*math.floor((box[0]-2)/2),2*math.ceil((box[2]+2)/2))]});ys=sorted({a[1],b[1],*[y for box in boxes for y in (2*math.floor((box[1]-2)/2),2*math.ceil((box[3]+2)/2))]})
    # Small visibility grid; includes every obstacle's outside edges.
    start=(xs.index(a[0]),ys.index(a[1]));goal=(xs.index(b[0]),ys.index(b[1]));queue=[(0,start)];cost={start:0};prev={}
    while queue:
        d,u=heapq.heappop(queue)
        if d!=cost[u]:continue
        if u==goal:
            path=[b]
            while u in prev:u=prev[u];path.append((xs[u[0]],ys[u[1]]))
            return simplify(list(reversed(path)))
        for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
            v=(u[0]+di,u[1]+dj)
            if not(0<=v[0]<len(xs) and 0<=v[1]<len(ys)):continue
            p,q=(xs[u[0]],ys[u[1]]),(xs[v[0]],ys[v[1]])
            if not clear(p,q):continue
            nd=d+math.dist(p,q)
            if nd<cost.get(v,math.inf):cost[v]=nd;prev[v]=u;heapq.heappush(queue,(nd,v))
    raise RoutingConflict('Não foi possível organizar uma ligação sem cruzar outro componente.')
