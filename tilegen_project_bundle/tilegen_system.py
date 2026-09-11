import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
import zipfile

from PIL import Image, ImageDraw, ImageFont, ImageOps


def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def noise(seed:int, x:int, y:int, n:int=0)->float:
    z = (seed * 0x9E3779B1) ^ (x * 0x85EBCA77) ^ (y * 0xC2B2AE3D) ^ (n * 0x27D4EB2D)
    z &= 0xFFFFFFFF
    z ^= (z >> 16)
    z = (z * 0x7FEB352D) & 0xFFFFFFFF
    z ^= (z >> 15)
    z = (z * 0x846CA68B) & 0xFFFFFFFF
    z ^= (z >> 16)
    return z / 0xFFFFFFFF

class TileMap:
    def __init__(self, w:int, h:int, default_base='wall', default_overlay=''):
        self.w=w; self.h=h
        self.base=[[default_base for _ in range(w)] for _ in range(h)]
        self.overlay=[[default_overlay for _ in range(w)] for _ in range(h)]
    def inb(self,x,y): return 0<=x<self.w and 0<=y<self.h
    def set(self,x,y,base=None,overlay=None):
        if not self.inb(x,y): return
        if base is not None: self.base[y][x]=base
        if overlay is not None: self.overlay[y][x]=overlay
    def get(self,x,y):
        if not self.inb(x,y): return None,None
        return self.base[y][x], self.overlay[y][x]
    def fill(self, base=None, overlay=None):
        for y in range(self.h):
            for x in range(self.w):
                if base is not None: self.base[y][x]=base
                if overlay is not None: self.overlay[y][x]=overlay
    def rect(self,x,y,w,h,base=None,overlay=None):
        for yy in range(y,y+h):
            for xx in range(x,x+w):
                self.set(xx,yy,base,overlay)
    def outline_rect(self,x,y,w,h,base=None,overlay=None,th=1):
        for t in range(th):
            for xx in range(x+t, x+w-t):
                self.set(xx,y+t,base,overlay)
                self.set(xx,y+h-1-t,base,overlay)
            for yy in range(y+t, y+h-t):
                self.set(x+t,yy,base,overlay)
                self.set(x+w-1-t,yy,base,overlay)
    def circle(self,cx,cy,r,base=None,overlay=None,fill=True,th=1):
        r2=r*r
        for y in range(cy-r-1, cy+r+2):
            for x in range(cx-r-1, cx+r+2):
                if not self.inb(x,y): continue
                d=(x-cx)**2+(y-cy)**2
                if fill:
                    if d<=r2: self.set(x,y,base,overlay)
                else:
                    if r2-(2*r+1)*th <= d <= r2+(2*r+1)*th:
                        self.set(x,y,base,overlay)
    def ellipse(self,cx,cy,rx,ry,base=None):
        for y in range(cy-ry-1, cy+ry+2):
            for x in range(cx-rx-1, cx+rx+2):
                if not self.inb(x,y): continue
                v=((x-cx)/max(rx,1))**2+((y-cy)/max(ry,1))**2
                if v<=1.0: self.set(x,y,base,None)
    def ring(self,cx,cy,inner,outer,base=None):
        for y in range(cy-outer-1, cy+outer+2):
            for x in range(cx-outer-1, cx+outer+2):
                if not self.inb(x,y): continue
                d=(x-cx)**2+(y-cy)**2
                if inner*inner <= d <= outer*outer:
                    self.set(x,y,base,None)
    def square_ring(self,x,y,w,h,th,base=None):
        self.outline_rect(x,y,w,h,base,None,th)
    def stamp_disk(self,cx,cy,r,base=None,overlay=None):
        for y in range(cy-r, cy+r+1):
            for x in range(cx-r, cx+r+1):
                if (x-cx)**2+(y-cy)**2<=r*r:
                    self.set(x,y,base,overlay)
    def line(self,x0,y0,x1,y1,width=1,base=None,overlay=None):
        dx=x1-x0; dy=y1-y0
        steps=max(abs(dx),abs(dy),1)*2
        rr=max(0,width//2)
        for i in range(steps+1):
            t=i/steps
            x=round(x0+dx*t); y=round(y0+dy*t)
            if width<=1:
                self.set(x,y,base,overlay)
            else:
                self.stamp_disk(x,y,rr,base,overlay)
    def polyline(self,pts,width=1,base=None,overlay=None,closed=False):
        for a,b in zip(pts, pts[1:]):
            self.line(a[0],a[1],b[0],b[1],width,base,overlay)
        if closed and len(pts)>2:
            a,b=pts[-1],pts[0]
            self.line(a[0],a[1],b[0],b[1],width,base,overlay)
    def set_overlay(self,x,y,overlay):
        if self.inb(x,y):
            self.overlay[y][x]=overlay
    def area_points(self,x,y,w,h):
        return [(xx,yy) for yy in range(y,y+h) for xx in range(x,x+w) if self.inb(xx,yy)]
    def circle_points(self,cx,cy,r):
        pts=[]
        for y in range(cy-r, cy+r+1):
            for x in range(cx-r, cx+r+1):
                if self.inb(x,y) and (x-cx)**2+(y-cy)**2<=r*r:
                    pts.append((x,y))
        return pts
    def mirror_x(self): # vertical axis; copy left to right
        for y in range(self.h):
            for x in range(self.w//2):
                sx=x
                tx=self.w-1-x
                self.base[y][tx]=self.base[y][sx]
                self.overlay[y][tx]=self.overlay[y][sx]
    def mirror_y(self):
        for y in range(self.h//2):
            sy=y
            ty=self.h-1-y
            for x in range(self.w):
                self.base[ty][x]=self.base[sy][x]
                self.overlay[ty][x]=self.overlay[sy][x]
    def replace(self, old, new):
        for y in range(self.h):
            for x in range(self.w):
                if self.base[y][x]==old:
                    self.base[y][x]=new
    def neighbors4(self,x,y):
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            xx,yy=x+dx,y+dy
            if self.inb(xx,yy): yield xx,yy
    def carve_maze(self,x,y,w,h,base='floor',loopiness=0.1):
        # Ensure odd dimensions within bounds
        x1 = x + (w-1)
        y1 = y + (h-1)
        if x1>=self.w: x1=self.w-1
        if y1>=self.h: y1=self.h-1
        if (x1-x)%2==0: x1-=1
        if (y1-y)%2==0: y1-=1
        cells_w=((x1-x)//2)+1
        cells_h=((y1-y)//2)+1
        visited=set()
        rng=random.Random((x*73856093)^(y*19349663)^(w*83492791)^(h*2654435761))
        def mapc(cx,cy): return x+cx*2, y+cy*2
        stack=[(0,0)]
        visited.add((0,0))
        mx,my=mapc(0,0); self.set(mx,my,base,None)
        while stack:
            cx,cy=stack[-1]
            dirs=[]
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx,ny=cx+dx,cy+dy
                if 0<=nx<cells_w and 0<=ny<cells_h and (nx,ny) not in visited:
                    dirs.append((dx,dy,nx,ny))
            if dirs:
                dx,dy,nx,ny=rng.choice(dirs)
                ax,ay=mapc(cx,cy); bx,by=mapc(nx,ny)
                self.line(ax,ay,bx,by,1,base,None)
                visited.add((nx,ny))
                stack.append((nx,ny))
            else:
                stack.pop()
        # thicken open cells a bit
        for yy in range(y,y1+1):
            for xx in range(x,x1+1):
                if self.inb(xx,yy) and self.base[yy][xx]==base:
                    for nx,ny in self.neighbors4(xx,yy):
                        if self.base[ny][nx]!=base and rng.random()<0.12:
                            pass
        # loops
        for cy in range(cells_h):
            for cx in range(cells_w):
                if rng.random()<loopiness:
                    dirs=[]
                    for dx,dy in ((1,0),(0,1)):
                        nx,ny=cx+dx,cy+dy
                        if 0<=nx<cells_w and 0<=ny<cells_h:
                            dirs.append((nx,ny))
                    if dirs:
                        nx,ny=rng.choice(dirs)
                        ax,ay=mapc(cx,cy); bx,by=mapc(nx,ny)
                        self.line(ax,ay,bx,by,1,base,None)
    def scatter_obstacles(self, rng, count, tile='wall', shape='rect', radius=(1,2), bounds=None):
        if bounds is None:
            bounds=(1,1,self.w-2,self.h-2)
        x0,y0,x1,y1=bounds
        for _ in range(count):
            cx=rng.randint(x0,x1)
            cy=rng.randint(y0,y1)
            r=rng.randint(radius[0], radius[1])
            if shape=='circle':
                self.stamp_disk(cx,cy,r,tile,None)
            else:
                self.rect(cx-r, cy-r, r*2+1, r*2+1, tile,None)
    def blob(self, rng, cx,cy,rx,ry,base='floor', rough=0.25):
        for y in range(cy-ry-2, cy+ry+3):
            for x in range(cx-rx-2, cx+rx+3):
                if not self.inb(x,y): continue
                dx=(x-cx)/max(rx,1)
                dy=(y-cy)/max(ry,1)
                d=dx*dx+dy*dy
                t=noise(int(rng.random()*1e9),x,y)
                if d <= 1.0 + (t-0.5)*rough:
                    self.set(x,y,base,None)
    def to_int_layers(self):
        base_legend=[]
        base_map={}
        overlay_legend=['']
        overlay_map={'':0}
        base_layer=[]
        overlay_layer=[]
        for row in self.base:
            out=[]
            for b in row:
                if b not in base_map:
                    base_map[b]=len(base_legend)
                    base_legend.append(b)
                out.append(base_map[b])
            base_layer.append(out)
        for row in self.overlay:
            out=[]
            for o in row:
                if o not in overlay_map:
                    overlay_map[o]=len(overlay_legend)
                    overlay_legend.append(o)
                out.append(overlay_map[o])
            overlay_layer.append(out)
        return {'base_legend':base_legend,'overlay_legend':overlay_legend,'base':base_layer,'overlay':overlay_layer}

def corridor_path(points):
    return [(int(x),int(y)) for x,y in points]

def square_spiral_points(x0,y0,w,h,step=4):
    pts=[]
    left,top=x0,y0
    right=x0+w-1
    bottom=y0+h-1
    while left<=right and top<=bottom:
        pts += [(left,top),(right,top),(right,bottom),(left,bottom)]
        left += step; top += step; right -= step; bottom -= step
    # simplify adjacent duplicates
    out=[]
    for p in pts:
        if not out or out[-1]!=p:
            out.append(p)
    return out

def circular_spiral_points(cx,cy,start_r,end_r,turns,points=200):
    pts=[]
    for i in range(points+1):
        t=i/points
        ang = t*turns*2*math.pi
        r = start_r + (end_r-start_r)*t
        x = round(cx + math.cos(ang)*r)
        y = round(cy + math.sin(ang)*r)
        if not pts or pts[-1]!=(x,y):
            pts.append((x,y))
    return pts

def rooms_on_circle(cx,cy,radius,count,room_w,room_h):
    centers=[]
    for i in range(count):
        ang = (i/count)*2*math.pi
        x = round(cx + math.cos(ang)*radius)
        y = round(cy + math.sin(ang)*radius)
        centers.append((x,y))
    return [(x-room_w//2, y-room_h//2, room_w, room_h, x,y) for x,y in centers]

def random_room_positions(rng, w,h, count, room_w_range, room_h_range, margin=4, bounds=None):
    rooms=[]
    if bounds is None:
        x0,y0,x1,y1=margin,margin,w-margin-1,h-margin-1
    else:
        x0,y0,x1,y1=bounds
    attempts=0
    while len(rooms)<count and attempts<count*30:
        attempts+=1
        rw=rng.randint(*room_w_range); rh=rng.randint(*room_h_range)
        rx=rng.randint(x0, max(x0, x1-rw))
        ry=rng.randint(y0, max(y0, y1-rh))
        ok=True
        for x,y,ww,hh in rooms:
            if not (rx+rw+1 < x or x+ww+1 < rx or ry+rh+1 < y or y+hh+1 < ry):
                ok=False; break
        if ok:
            rooms.append((rx,ry,rw,rh))
    return rooms

def connect_room_centers(tm:TileMap, rooms, width=2, base='floor', mode='mst_loops', rng=None, extra=0.2):
    centers=[(x+w//2, y+h//2) for x,y,w,h in rooms]
    if not centers: return
    if rng is None: rng=random.Random(0)
    remaining=set(range(1,len(centers)))
    connected={0}
    edges=[]
    while remaining:
        best=None
        for i in connected:
            for j in remaining:
                d=abs(centers[i][0]-centers[j][0])+abs(centers[i][1]-centers[j][1])
                if best is None or d<best[0]:
                    best=(d,i,j)
        _,i,j=best
        connected.add(j); remaining.remove(j)
        edges.append((i,j))
    # extra loops
    for i in range(len(centers)):
        for j in range(i+1,len(centers)):
            if rng.random()<extra:
                edges.append((i,j))
    # draw L corridors
    for i,j in edges:
        x1,y1=centers[i]; x2,y2=centers[j]
        if rng.random()<0.5:
            tm.line(x1,y1,x2,y1,width,base,None)
            tm.line(x2,y1,x2,y2,width,base,None)
        else:
            tm.line(x1,y1,x1,y2,width,base,None)
            tm.line(x1,y2,x2,y2,width,base,None)

def add_doors_on_boundaries(tm:TileMap, rooms, overlay='door', every=1):
    for idx,(x,y,w,h) in enumerate(rooms):
        if idx % every !=0: continue
        pts=[(x+w//2,y),(x+w//2,y+h-1),(x,y+h//2),(x+w-1,y+h//2)]
        for px,py in pts[:2]:
            if tm.inb(px,py): tm.set_overlay(px,py,overlay)

def branch_tree_segments(rng, start, angle, length, depth, spread, step=5):
    segs=[]
    frontier=[(start[0],start[1],angle,length,depth)]
    while frontier:
        x,y,ang,l,dep=frontier.pop(0)
        if l<=0 or dep<0: continue
        nx = x + math.cos(ang)*step
        ny = y + math.sin(ang)*step
        segs.append(((round(x),round(y)), (round(nx),round(ny)), dep))
        if dep>0 and l>1:
            branch_count = 2 if rng.random()<0.55 else 1
            for b in range(branch_count):
                nang = ang + rng.uniform(-spread, spread)
                frontier.append((nx,ny,nang,l-1,dep-1))
        else:
            frontier.append((nx,ny,ang+rng.uniform(-0.3,0.3), l-1, dep))
    return segs

def gen_hub_spokes(spec, rng):
    size = spec.get('size', 64)
    bg = spec.get('background','wall')
    tm = TileMap(size,size,bg)
    cx=cy=size//2
    hub_r = spec.get('hub_r', size//8)
    spoke_len = spec.get('spoke_len', size//2 - hub_r - 6)
    spoke_w = spec.get('spoke_w', 2)
    spokes = spec.get('spokes', 6)
    hub_shape = spec.get('hub_shape','circle')
    if hub_shape=='circle':
        tm.circle(cx,cy,hub_r,'floor')
    else:
        hw = spec.get('hub_w', hub_r*2+2)
        hh = spec.get('hub_h', hub_r*2+2)
        tm.rect(cx-hw//2, cy-hh//2, hw, hh, 'floor')
    terminal = []
    for i in range(spokes):
        ang = spec.get('start_angle',0) + i*(2*math.pi/spokes)
        x2 = round(cx + math.cos(ang)*spoke_len)
        y2 = round(cy + math.sin(ang)*spoke_len)
        tm.line(cx,cy,x2,y2,spoke_w,'floor')
        terminal.append((x2,y2,ang))
    term_shape = spec.get('terminal','room')
    room_size = spec.get('terminal_size', 8)
    if term_shape!='none':
        for idx,(x,y,ang) in enumerate(terminal):
            if term_shape=='room':
                rw = room_size + (idx % 2)
                rh = room_size - (idx % 2)
                tm.rect(x-rw//2, y-rh//2, rw, rh, 'floor')
            elif term_shape=='circle':
                tm.circle(x,y,room_size//2,'floor')
            elif term_shape=='alcove':
                tm.rect(x-2,y-2,5,5,'floor')
    for idx,(x,y,ang) in enumerate(terminal):
        if spec.get('term_overlay'):
            tm.set_overlay(x,y,spec['term_overlay'])
        if spec.get('branch_rooms') and idx % 2 == 0:
            bx = round(x + math.cos(ang+math.pi/2)*8)
            by = round(y + math.sin(ang+math.pi/2)*8)
            tm.line(x,y,bx,by,2,'floor')
            tm.rect(bx-3,by-3,7,7,'floor')
    if spec.get('hidden_radials'):
        for ang in [0, math.pi/2, math.pi, 3*math.pi/2]:
            x=round(cx + math.cos(ang)*hub_r)
            y=round(cy + math.sin(ang)*hub_r)
            tm.set_overlay(x,y,'hidden')
            x2=round(cx + math.cos(ang)*(hub_r+10))
            y2=round(cy + math.sin(ang)*(hub_r+10))
            tm.line(x,y,x2,y2,1,'floor')
    if spec.get('outer_ring'):
        tm.ring(cx,cy,hub_r+10,hub_r+13,'floor')
    return tm


def gen_spiral(spec, rng):
    size=spec.get('size',64)
    bg=spec.get('background','wall')
    tm=TileMap(size,size,bg)
    cx=cy=size//2
    spiral_type=spec.get('spiral_type','circular')
    path_tile = spec.get('path_tile','floor')
    if spec.get('core')=='void':
        tm.circle(cx,cy,spec.get('core_r',5),'abyss')
    elif spec.get('core')=='chamber':
        tm.circle(cx,cy,spec.get('core_r',5),'floor')
    if spiral_type=='square':
        pts=square_spiral_points(6,6,size-12,size-12,step=spec.get('step',6))
    else:
        pts=circular_spiral_points(cx,cy,spec.get('start_r', size//2-8), spec.get('end_r', spec.get('core_r',4)+2), spec.get('turns',3.0), points=spec.get('points',240))
    tm.polyline(pts,width=spec.get('width',2),base=path_tile)
    if spec.get('platforms'):
        count=spec.get('platform_count',6)
        for i in range(count):
            t=(i+1)/(count+1)
            ang=t*spec.get('turns',3.0)*2*math.pi
            r=spec.get('start_r', size//2-8)+(spec.get('end_r', spec.get('core_r',4)+2)-spec.get('start_r', size//2-8))*t
            x=round(cx+math.cos(ang)*r)
            y=round(cy+math.sin(ang)*r)
            tm.rect(x-3,y-3,7,7,'platform')
            tm.set_overlay(x,y,spec.get('platform_overlay','stair'))
    if spec.get('landings'):
        for i,p in enumerate(pts[::max(1,len(pts)//6)]):
            x,y=p
            tm.rect(x-3,y-2,7,5,'platform')
            tm.set_overlay(x,y,'stair')
    if spec.get('outer_platform_ring'):
        tm.ring(cx,cy,spec.get('outer_platform_ring')[0], spec.get('outer_platform_ring')[1],'platform')
    return tm


def gen_grid_rooms(spec, rng):
    w=h=spec.get('size',64)
    tm=TileMap(w,h,spec.get('background','wall'))
    rows=spec.get('rows',4); cols=spec.get('cols',4)
    room_w=spec.get('room_w',8); room_h=spec.get('room_h',8)
    gap_x=spec.get('gap_x',4); gap_y=spec.get('gap_y',4)
    start_x=(w-(cols*room_w+(cols-1)*gap_x))//2
    start_y=(h-(rows*room_h+(rows-1)*gap_y))//2
    rooms=[]
    for r in range(rows):
        for c in range(cols):
            x=start_x+c*(room_w+gap_x)
            y=start_y+r*(room_h+gap_y)
            tm.rect(x,y,room_w,room_h,'floor')
            rooms.append((x,y,room_w,room_h,r,c))
    # connections
    for x,y,rw,rh,r,c in rooms:
        cx=x+rw//2; cy=y+rh//2
        if c<cols-1 and (not spec.get('skip_middle_col') or c!=cols//2-1):
            nx = x+rw+gap_x
            if spec.get('connection_mode','all')=='maze':
                if rng.random()<0.75:
                    tm.line(cx,cy,nx+rw//2,cy,spec.get('corridor_w',2),'floor')
            else:
                tm.line(cx,cy,nx+rw//2,cy,spec.get('corridor_w',2),'floor')
                if spec.get('locked_intersections') and rng.random()<spec.get('lock_rate',0.5):
                    mx=(cx+nx+rw//2)//2; my=cy
                    tm.set_overlay(mx,my,'locked')
        if r<rows-1:
            ny = y+rh+gap_y
            if spec.get('connection_mode','all')=='maze':
                if rng.random()<0.75:
                    tm.line(cx,cy,cx,ny+rh//2,spec.get('corridor_w',2),'floor')
            else:
                tm.line(cx,cy,cx,ny+rh//2,spec.get('corridor_w',2),'floor')
                if spec.get('locked_intersections') and rng.random()<spec.get('lock_rate',0.5):
                    mx=cx; my=(cy+ny+rh//2)//2
                    tm.set_overlay(mx,my,'locked')
    if spec.get('rotating'):
        for x,y,rw,rh,r,c in rooms:
            if (r+c)%2==0:
                tm.set_overlay(x+rw//2,y+rh//2,'rotate')
    if spec.get('maze_fill'):
        x0=start_x-1; y0=start_y-1
        ww=cols*(room_w+gap_x)-gap_x+2; hh=rows*(room_h+gap_y)-gap_y+2
        tm.carve_maze(x0,y0,ww,hh,'floor',loopiness=0.2)
    return tm


def gen_vertical_stack(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    cx=size//2
    rooms = spec.get('rooms',6)
    shaft_w = spec.get('shaft_w',3)
    y_positions=[int(size*0.12 + i*(size*0.76/max(1,rooms-1))) for i in range(rooms)]
    tm.rect(cx-shaft_w//2, 2, shaft_w, size-4, 'floor')
    for i,y in enumerate(y_positions):
        rw=spec.get('room_w',10)
        rh=spec.get('room_h',7)
        if spec.get('vary_room'):
            rw += (i%3)-1
            rh += ((i+1)%3)-1
        if spec.get('side_branches'):
            side=-1 if i%2==0 else 1
            x = cx + side*(shaft_w//2 + spec.get('branch_len',8))
            tm.line(cx, y, x, y, 2, 'floor')
            tm.rect(x-(rw//2 if side>0 else rw-rw//2), y-rh//2, rw, rh, 'floor')
            tm.set_overlay(cx,y,spec.get('shaft_overlay','ladder'))
            if spec.get('drop_shafts') and i%2==1:
                tm.set_overlay(cx,y,'shaft')
        else:
            tm.rect(cx-rw//2, y-rh//2, rw, rh, 'floor')
            tm.set_overlay(cx,y,spec.get('shaft_overlay','ladder'))
        if spec.get('challenge_rooms'):
            tm.set_overlay(cx,y,'challenge')
    if spec.get('narrow'):
        tm.rect(cx-1,2,3,size-4,'floor')
    return tm


def gen_branch_caves(spec, rng):
    size=spec.get('size',64)
    bg=spec.get('background','wall')
    tm=TileMap(size,size,bg)
    if spec.get('open_cavern'):
        tm.blob(rng, size//2, size//2, size//2-10, size//2-12, 'floor', rough=0.65)
        # irregular cutouts
        for _ in range(6):
            cx=rng.randint(10,size-11); cy=rng.randint(10,size-11)
            tm.blob(rng, cx, cy, rng.randint(3,6), rng.randint(3,6), 'wall', rough=0.55)
    start = spec.get('start', (size//2, size//2))
    width = spec.get('width', 3)
    segments = branch_tree_segments(rng, start, spec.get('angle', -math.pi/2), spec.get('length', 8), spec.get('depth', 4), spec.get('spread', 1.0), step=spec.get('step',5))
    endpoints=[]
    for (x1,y1),(x2,y2),dep in segments:
        tm.line(x1,y1,x2,y2,width,'floor')
        if rng.random()<spec.get('room_rate',0.35):
            tm.blob(rng, x2,y2, rng.randint(3,6), rng.randint(3,6), 'floor', rough=0.4)
        endpoints.append((x2,y2,dep))
    if spec.get('slanted'):
        for _ in range(6):
            x1=rng.randint(8,size-9); y1=rng.randint(8,size-9)
            x2=clamp(x1+rng.randint(-10,10),2,size-3); y2=clamp(y1+rng.randint(-10,10),2,size-3)
            tm.line(x1,y1,x2,y2,2,'floor')
    if spec.get('rubble'):
        for _ in range(spec.get('rubble_count',25)):
            x=rng.randint(2,size-3); y=rng.randint(2,size-3)
            if tm.base[y][x]=='floor':
                tm.base[y][x]='rubble'
                if rng.random()<0.2: tm.set_overlay(x,y,'collapse')
    if spec.get('shafts'):
        placed=0
        for x,y,dep in endpoints[::2]:
            tm.set_overlay(x,y,'shaft')
            placed += 1
        while placed < 6:
            x=rng.randint(8,size-9); y=rng.randint(8,size-9)
            if tm.base[y][x] in ('floor','rubble'):
                tm.set_overlay(x,y,'shaft')
                placed += 1
    if spec.get('river'):
        pts=[(2, rng.randint(size//3, 2*size//3))]
        cur=pts[0]
        for i in range(4):
            nx=int((i+1)*size/5)
            ny=clamp(cur[1]+rng.randint(-8,8),3,size-4)
            pts.append((nx,ny))
            cur=(nx,ny)
        pts.append((size-3, clamp(cur[1]+rng.randint(-6,6),3,size-4)))
        tm.polyline(pts, width=spec.get('river_w',4), base='water')
        for x,y,dep in endpoints[:3]:
            bx = clamp(x,2,size-3)
            by = y
            tm.line(bx-2,by,bx+2,by,2,'bridge')
    return tm


def gen_symmetrical_complex(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    half=size//2
    # left half plan
    center_w=spec.get('center_w',10); center_h=spec.get('center_h',8)
    tm.rect(half-center_w//2, size//2-center_h//2, center_w, center_h, 'floor')
    # left wing structures
    wing_rooms=spec.get('wing_rooms',3)
    base_x=half-center_w//2-4
    y0=size//2 - (wing_rooms*(spec.get('room_h',6)+4))//2
    prev=(half-center_w//2, size//2)
    for i in range(wing_rooms):
        rw=spec.get('room_w',8)+ (i%2)
        rh=spec.get('room_h',6)
        x=base_x - (i+1)*(rw+4)
        y=y0 + i*(rh+4)
        tm.rect(x,y,rw,rh,'floor')
        cx=x+rw//2; cy=y+rh//2
        tm.line(prev[0],prev[1],cx,cy,2,'floor')
        prev=(cx,cy)
    if spec.get('cross'):
        arm_len=spec.get('arm_len',18)
        arm_w=spec.get('arm_w',6)
        cx=cy=size//2
        tm.rect(cx-arm_w//2, cy-arm_len, arm_w, arm_len*2+1, 'floor')
        tm.rect(cx-arm_len, cy-arm_w//2, arm_len*2+1, arm_w, 'floor')
    tm.mirror_x()
    if spec.get('mirror_y'):
        tm.mirror_y()
    if spec.get('maze_wing'):
        tm.carve_maze(4,4,half-8,size-8,'floor',loopiness=0.08)
        tm.mirror_x()
    return tm


def gen_fortress(spec, rng):
    size=spec.get('size',64)
    bg=spec.get('background','wall')
    tm=TileMap(size,size,bg)
    rings=spec.get('rings',3)
    margin=spec.get('margin',4)
    step=spec.get('step',6)
    for i in range(rings):
        x=margin+i*step; y=margin+i*step
        w=size-2*(margin+i*step); h=size-2*(margin+i*step)
        tm.outline_rect(x,y,w,h,'floor',None,th=spec.get('corridor_th',3))
    if spec.get('courtyard',True):
        x=margin+step*(rings-1)+4
        y=margin+step*(rings-1)+4
        w=size-2*(margin+step*(rings-1)+4)
        h=size-2*(margin+step*(rings-1)+4)
        tm.rect(x,y,w,h,'courtyard')
    if spec.get('keep'):
        k=spec.get('keep',12)
        tm.rect(size//2-k//2,size//2-k//2,k,k,'floor')
    # connections inward
    entrances=spec.get('entrances',4)
    for i in range(entrances):
        side=i%4
        if side==0:
            x=size//2 + (i//4)*2; tm.line(x,0,x,size//2,2,'floor')
        elif side==1:
            x=size//2 + (i//4)*2; tm.line(x,size-1,x,size//2,2,'floor')
        elif side==2:
            y=size//2 + (i//4)*2; tm.line(0,y,size//2,y,2,'floor')
        else:
            y=size//2 + (i//4)*2; tm.line(size-1,y,size//2,y,2,'floor')
    if spec.get('perimeter_halls'):
        tm.outline_rect(6,6,size-12,size-12,'floor',None,th=2)
    if spec.get('guard_routes'):
        for y in [10,size-11]:
            tm.line(8,y,size-9,y,2,'floor')
        for x in [10,size-11]:
            tm.line(x,8,x,size-9,2,'floor')
    return tm


def gen_linear_gauntlet(spec, rng):
    size=spec.get('size',64)
    horizontal = spec.get('horizontal', True)
    tm=TileMap(size,size,spec.get('background','wall'))
    if horizontal:
        y=size//2
        tm.line(4,y,size-5,y,spec.get('corridor_w',3),'floor')
        chamber_every=spec.get('chamber_every',10)
        side=1
        for x in range(10,size-10,chamber_every):
            if rng.random() < spec.get('room_rate',0.8):
                rw=spec.get('room_w',7); rh=spec.get('room_h',6)
                oy = y + side*(spec.get('offset',8))
                tm.line(x,y,x,oy,2,'floor')
                tm.rect(x-rw//2, oy-rh//2, rw, rh, 'floor')
                if spec.get('gated'):
                    tm.set_overlay(x, (y+oy)//2, 'locked' if rng.random()<0.5 else 'door')
                if spec.get('challenge'):
                    tm.set_overlay(x,oy,'challenge')
                side*=-1
        if spec.get('chokepoints'):
            for x in range(12,size-12,12):
                tm.rect(x-1,y-2,3,5,'floor')
                tm.set_overlay(x,y,'door')
        if spec.get('detours'):
            for x in range(14,size-14,14):
                top = y-10 if (x//14)%2==0 else y+10
                tm.line(x,y,x,top,2,'floor')
                tm.line(x,top,x+6,top,2,'floor')
                tm.line(x+6,top,x+6,y,2,'floor')
    else:
        x=size//2
        tm.line(x,4,x,size-5,spec.get('corridor_w',3),'floor')
    return tm


def gen_loops_network(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    compact=spec.get('compact',False)
    if compact:
        rooms=random_room_positions(rng,size,size,spec.get('room_count',10),(4,7),(4,7),margin=8,bounds=(10,10,size-18,size-18))
    else:
        rooms=random_room_positions(rng,size,size,spec.get('room_count',12),(5,8),(5,8),margin=6)
    for x,y,w,h in rooms:
        tm.rect(x,y,w,h,'floor')
    connect_room_centers(tm, rooms, width=2, base='floor', rng=rng, extra=spec.get('extra_loops',0.25))
    # reconverging or dual path
    if spec.get('dual_path'):
        y1=size//3; y2=2*size//3
        tm.line(6,y1,size-7,y1,2,'floor')
        tm.line(6,y2,size-7,y2,2,'floor')
        for x in range(12,size-12,12):
            tm.line(x,y1,x,y2,2,'floor')
    if spec.get('central_hall'):
        tm.rect(size//2-4, 6, 9, size-12, 'floor')
        for side in (-1,1):
            for off in (-16,-4,8,20):
                x=size//2 + side*12
                y=size//2 + off
                tm.line(size//2,y,x,y,2,'floor')
                tm.rect(x-4,y-3,8,6,'floor')
                # loop back
                tm.line(x,y,x, size//2 + off//2,2,'floor')
    if spec.get('shifting'):
        for _ in range(8):
            x=rng.randint(8,size-9); y=rng.randint(8,size-9)
            if tm.base[y][x]=='floor':
                tm.set_overlay(x,y,'shift')
    if spec.get('overlap'):
        for _ in range(6):
            x1=rng.randint(8,size-9); y1=rng.randint(8,size-9)
            x2=rng.randint(8,size-9); y2=rng.randint(8,size-9)
            tm.line(x1,y1,x2,y2,1,'bridge')
            tm.set_overlay((x1+x2)//2,(y1+y2)//2,'ladder')
    return tm


def gen_pyramid_terraces(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    levels=spec.get('levels',5)
    margin=spec.get('margin',4)
    step=spec.get('step',5)
    for i in range(levels):
        x=margin+i*step
        y=margin+i*step
        w=size-2*(margin+i*step)
        h=size-2*(margin+i*step)
        if w<=2 or h<=2: break
        tm.outline_rect(x,y,w,h,'floor',None,th=spec.get('th',2))
        if spec.get('terraces'):
            tm.outline_rect(x+1,y+1,w-2,h-2,'platform',None,th=1)
        if i<levels-1:
            sx = x + w//2
            sy = y + h//2
            tm.set_overlay(sx, y+1, 'stair')
    if spec.get('arena'):
        tm.circle(size//2,size//2,spec.get('arena_r',8),'floor')
    return tm


def gen_flooded(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    theme=spec.get('theme','ruin')
    if theme=='sewer':
        # start with water channels
        tm.fill('wall')
        main_pts=[(6,size//2),(size//3,size//2-10),(2*size//3,size//2+8),(size-6,size//2)]
        tm.polyline(main_pts,width=5,base='water')
        for x in [size//3, 2*size//3]:
            tm.circle(x,size//2,5,'water')
        # walkways
        tm.line(6,size//2+8,size-6,size//2+8,2,'floor')
        for x in [size//3,2*size//3]:
            tm.line(x,size//2-5,x,size//2+13,2,'bridge')
    else:
        # submerged ruin/flooded labyrinth
        tm.rect(5,5,size-10,size-10,'water')
        dry_rooms=random_room_positions(rng,size,size,spec.get('room_count',8),(6,10),(5,9),margin=8)
        for x,y,w,h in dry_rooms:
            tm.rect(x,y,w,h,'floor')
        connect_room_centers(tm,dry_rooms,width=2,base='bridge' if spec.get('alternating') else 'floor',rng=rng,extra=0.15)
        if spec.get('alternating'):
            # convert some corridors to water
            for y in range(size):
                for x in range(size):
                    if tm.base[y][x]=='bridge' and ((x+y)%5 in (0,1)):
                        tm.base[y][x]='water'
            # add alternate dry bridges
            for _ in range(6):
                x1=rng.randint(8,size-9); y1=rng.randint(8,size-9)
                x2=clamp(x1+rng.randint(-12,12),2,size-3); y2=clamp(y1+rng.randint(-12,12),2,size-3)
                tm.line(x1,y1,x2,y2,2,'bridge')
        if spec.get('ruin'):
            for _ in range(20):
                x=rng.randint(6,size-7); y=rng.randint(6,size-7)
                if tm.base[y][x]=='floor':
                    tm.base[y][x]='rubble'
    return tm


def gen_city(spec, rng):
    size=spec.get('size',72)
    tm=TileMap(size,size,'wall')
    district_rows=spec.get('district_rows',3); district_cols=spec.get('district_cols',3)
    street=4
    block_w=(size-(district_cols+1)*street)//district_cols
    block_h=(size-(district_rows+1)*street)//district_rows
    districts=[]
    # streets first
    for x in range(0,size,block_w+street):
        tm.rect(x,0,street,size,'floor')
    for y in range(0,size,block_h+street):
        tm.rect(0,y,size,street,'floor')
    for r in range(district_rows):
        for c in range(district_cols):
            x=street+c*(block_w+street)
            y=street+r*(block_h+street)
            tm.rect(x,y,block_w,block_h,'floor')
            districts.append((x,y,block_w,block_h))
            # district wall border / district tag
            tm.outline_rect(x,y,block_w,block_h,'platform',None,th=1)
            tm.set_overlay(x+block_w//2, y+block_h//2, 'district')
            # inner buildings and alleys
            building_count=4+rng.randint(0,3)
            local_rooms=random_room_positions(rng, block_w, block_h, building_count, (4,6), (4,6), margin=1, bounds=(x+1,y+1,x+block_w-2,y+block_h-2))
            for bx,by,bw,bh in local_rooms:
                tm.rect(bx,by,bw,bh,'wall')
            # carve alleys
            tm.line(x+block_w//2,y+1,x+block_w//2,y+block_h-2,1,'floor')
            tm.line(x+1,y+block_h//2,x+block_w-2,y+block_h//2,1,'floor')
    return tm


def gen_fragmented(spec, rng):
    size=spec.get('size',64)
    bg=spec.get('background','wall')
    tm=TileMap(size,size,bg)
    fragments=spec.get('fragments',8)
    rooms=[]
    for _ in range(fragments):
        rw=rng.randint(6,10); rh=rng.randint(5,9)
        x=rng.randint(4,size-rw-4); y=rng.randint(4,size-rh-4)
        rooms.append((x,y,rw,rh))
        tm.rect(x,y,rw,rh,'floor')
        if spec.get('floating'):
            tm.outline_rect(x,y,rw,rh,'bridge',None,th=1)
    # broken connections
    for i in range(len(rooms)-1):
        x1,y1,w1,h1=rooms[i]; x2,y2,w2,h2=rooms[i+1]
        c1=(x1+w1//2,y1+h1//2); c2=(x2+w2//2,y2+h2//2)
        if spec.get('floating'):
            if rng.random()<0.35:
                tm.line(c1[0],c1[1],c2[0],c2[1],1,'bridge')
            else:
                tm.set_overlay(c1[0],c1[1],'teleport')
                tm.set_overlay(c2[0],c2[1],'teleport')
        else:
            if rng.random()<0.75:
                mid=((c1[0]+c2[0])//2 + rng.randint(-2,2), (c1[1]+c2[1])//2 + rng.randint(-2,2))
                tm.polyline([c1,mid,c2],2,'floor')
        if spec.get('collapsed'):
            for _ in range(20):
                x=rng.randint(3,size-4); y=rng.randint(3,size-4)
                if tm.base[y][x]=='floor':
                    tm.base[y][x]='rubble'
                    if rng.random()<0.1: tm.set_overlay(x,y,'collapse')
    return tm


def gen_canyon(spec, rng):
    size=spec.get('size',64)
    bg=spec.get('background','wall')
    tm=TileMap(size,size,bg)
    mode=spec.get('mode','zigzag')
    if mode=='cliffside':
        # central void, path hugging outer wall
        tm.rect(0,0,size,size,'wall')
        tm.rect(8,8,size-16,size-16,'abyss')
        pts=[(4,4),(size-5,4),(size-5,size//3),(size//3,size//3),(size//3,2*size//3),(size-5,2*size//3),(size-5,size-5)]
        tm.polyline(pts,3,'floor')
    elif mode=='bridged':
        tm.fill('abyss')
        # multi-path canyon floor ledges and bridges
        left=[(6,8),(12,20),(10,34),(14,50),(18,size-8)]
        right=[(size-8,10),(size-16,22),(size-12,38),(size-20,54),(size-18,size-10)]
        tm.polyline(left,4,'floor')
        tm.polyline(right,4,'floor')
        for y in [18,32,46]:
            tm.line(18,y,size-18,y,2,'bridge')
    else:
        pts=[(6,8),(size//3,16),(10,size//2),(size//3*2,size-18),(size-8,size-10)]
        if mode=='diagonal':
            pts=[(8,size-8),(20,size-20),(34,size-32),(48,18),(size-8,8)]
        tm.polyline(pts, spec.get('width',5), 'floor')
        # ledges/platforms
        for x,y in pts[1:-1]:
            tm.rect(x-4,y-3,8,6,'platform')
            if spec.get('bridges'):
                tm.line(x,y,x+rng.randint(-8,8),y+rng.randint(-8,8),1,'bridge')
    return tm


def gen_ring_core(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    cx=cy=size//2
    rings=spec.get('rings',2)
    start=spec.get('start_r',10)
    gap=spec.get('gap',6)
    for i in range(rings):
        inner=start+i*gap
        outer=inner+spec.get('th',2)
        tm.ring(cx,cy,inner,outer,'floor')
    if spec.get('core')=='arena':
        tm.circle(cx,cy,spec.get('core_r',7),'floor')
    elif spec.get('core')=='chamber':
        k=spec.get('core_w',10)
        tm.rect(cx-k//2,cy-k//2,k,k,'floor')
    if spec.get('radials',True):
        count=spec.get('radial_count',4)
        for i in range(count):
            ang = i*(2*math.pi/count) + spec.get('start_angle',0)
            x1=round(cx+math.cos(ang)*(start-2)); y1=round(cy+math.sin(ang)*(start-2))
            x2=round(cx+math.cos(ang)*(start+rings*gap+3)); y2=round(cy+math.sin(ang)*(start+rings*gap+3))
            tm.line(x1,y1,x2,y2,2,'floor')
            if spec.get('hidden') and i%2==1:
                tm.set_overlay(x1,y1,'hidden')
    if spec.get('ring_rooms'):
        for x,y,w,h,rcx,rcy in rooms_on_circle(cx,cy,start+rings*gap+4,spec.get('room_count',6),7,6):
            tm.rect(x,y,w,h,'floor')
            tm.line(rcx,rcy,cx,cy,1,'floor')
    if spec.get('diagonal_rooms'):
        for ang in [math.pi/4, 3*math.pi/4, 5*math.pi/4, 7*math.pi/4]:
            x=round(cx+math.cos(ang)*(start+rings*gap+2))
            y=round(cy+math.sin(ang)*(start+rings*gap+2))
            tm.rect(x-3,y-3,7,7,'floor')
            tm.line(x,y,cx+round(math.cos(ang)*start),cy+round(math.sin(ang)*start),1,'floor')
    if spec.get('maze_rings'):
        # gap openings to make maze-like
        for i in range(rings):
            r=start+i*gap+1
            for ang in [0.3,1.8,3.4,5.1]:
                ox=round(cx+math.cos(ang)*(r+1)); oy=round(cy+math.sin(ang)*(r+1))
                tm.rect(ox-1,oy-1,3,3,'wall')
    return tm


def gen_layered_overlap(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    # Lower floor
    tm.rect(6,10,size//2+8,size//2+10,'floor')
    tm.rect(size//3, size//3, size//2, size//2, 'platform')
    # Connect and overlap
    tm.line(10,size//2,size-10,size//2,2,'bridge')
    tm.line(size//2,10,size//2,size-10,2,'bridge')
    # hidden passages between levels
    for p in [(size//2,size//3),(size//3,size//2),(2*size//3,2*size//3)]:
        tm.set_overlay(*p, 'ladder' if not spec.get('hidden') else 'hidden')
    if spec.get('maze'):
        tm.carve_maze(8,8,size-16,size-16,'floor',loopiness=0.18)
        # offset second layer
        for y in range(12,size-12,10):
            tm.line(12,y,size-12,y+4,1,'platform')
            tm.set_overlay(size//2,y,'ladder')
    return tm


def gen_rotating_modular(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    cells=spec.get('cells',4)
    cell_size=(size-12)//cells
    margin=6
    for r in range(cells):
        for c in range(cells):
            x=margin+c*cell_size
            y=margin+r*cell_size
            tm.rect(x+1,y+1,cell_size-2,cell_size-2,'floor')
            tm.set_overlay(x+cell_size//2,y+cell_size//2,'rotate')
            # openings based on seed pattern
            dirs=[('N',0,-1),('S',0,1),('W',-1,0),('E',1,0)]
            for name,dx,dy in dirs:
                if rng.random()<spec.get('open_rate',0.6):
                    if dx!=0:
                        px=x + (cell_size-1 if dx>0 else 0)
                        py=y + cell_size//2
                        tm.rect(px-1,py-1,3,3,'floor')
                    else:
                        px=x + cell_size//2
                        py=y + (cell_size-1 if dy>0 else 0)
                        tm.rect(px-1,py-1,3,3,'floor')
    return tm


def gen_bridge_lattice(spec, rng):
    size=spec.get('size',64)
    hazard=spec.get('hazard','abyss')
    tm=TileMap(size,size,hazard)
    nodes=[]
    if spec.get('split'):
        # floor halves with crossings
        tm.rect(0,0,size//2-5,size,'floor')
        tm.rect(size//2+5,0,size//2-5,size,'floor')
        for y in [12, size//2, size-13]:
            tm.line(size//2-5,y,size//2+5,y,2,'bridge')
        return tm
    # lattice nodes
    for y in range(10,size-10,14):
        for x in range(10,size-10,14):
            nodes.append((x,y))
            tm.rect(x-2,y-2,5,5,'platform')
    for x1,y1 in nodes:
        for x2,y2 in nodes:
            if abs(x1-x2)+abs(y1-y2)==14 and rng.random()<0.9:
                tm.line(x1,y1,x2,y2,spec.get('width',1),'bridge')
    # some diagonals
    for _ in range(len(nodes)//2):
        a,b=rng.sample(nodes,2)
        if abs(a[0]-b[0])==14 and abs(a[1]-b[1])==14:
            tm.line(a[0],a[1],b[0],b[1],1,'bridge')
    return tm


def gen_honeycomb(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    r=spec.get('r',4)
    dx=r*3
    dy=int(r*2.6)
    centers=[]
    for row,y in enumerate(range(8,size-8,dy)):
        offset = 0 if row%2==0 else dx//2
        for x in range(8+offset,size-8,dx):
            centers.append((x,y))
    for cx,cy in centers:
        # approximate hex room
        pts=[(cx-r,cy),(cx-r//2,cy-r),(cx+r//2,cy-r),(cx+r,cy),(cx+r//2,cy+r),(cx-r//2,cy+r)]
        tm.polyline(pts,1,'floor',closed=True)
        tm.circle(cx,cy,r-1,'floor')
    # connect nearby centers
    for i,(x1,y1) in enumerate(centers):
        for j,(x2,y2) in enumerate(centers[i+1:], i+1):
            d=abs(x1-x2)+abs(y1-y2)
            if d<dx+dy//2:
                if rng.random()<0.35:
                    tm.line(x1,y1,x2,y2,1,'floor')
    return tm


def gen_diagonal_platforms(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','abyss' if spec.get('void_bg') else 'wall'))
    count=spec.get('count',8)
    for i in range(count):
        x=8 + i*spec.get('dx',6)
        y=size-12 - i*spec.get('dy',6)
        if spec.get('broken'):
            tm.rect(x,y,6,2,'platform')
            if i%2==0:
                tm.rect(x+4,y-4,2,2,'platform')
        else:
            tm.rect(x,y,8,4,'platform')
        if i>0:
            px=8 + (i-1)*spec.get('dx',6) + 4
            py=size-12 - (i-1)*spec.get('dy',6) + 2
            tm.line(px,py,x+4,y+2,1,'bridge' if spec.get('bridge_links') else 'floor')
            tm.set_overlay((px+x+4)//2,(py+y+2)//2,'ladder' if spec.get('ladders',True) else 'stair')
    if spec.get('balconies'):
        for i in range(5):
            y=10+i*10
            x=10 + (i%2)*12
            tm.rect(x,y,size-20-x,3,'platform')
    return tm


def gen_pillar_cavern(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    tm.blob(rng,size//2,size//2,size//2-6,size//2-8,'floor',rough=0.55)
    for _ in range(spec.get('pillars',22)):
        x=rng.randint(8,size-9); y=rng.randint(8,size-9)
        if tm.base[y][x]=='floor':
            tm.circle(x,y,rng.randint(1,2),'wall')
            tm.set_overlay(x,y,'pillar')
    return tm


def gen_zigzag_stairs(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    x1=10; x2=size-10
    y=8
    direction=1
    while y<size-8:
        nx = x2 if direction==1 else x1
        tm.line(x1 if direction==1 else x2, y, nx, y, 3, 'floor')
        tm.set_overlay(nx,y,'stair')
        y2=min(size-8, y+8)
        tm.line(nx,y,nx,y2,3,'floor')
        y=y2
        direction*=-1
    # landings rooms
    if spec.get('landings'):
        for yy in range(8,size-8,16):
            xx=x2 if (yy//8)%2==1 else x1
            tm.rect(xx-4,yy-2,8,5,'platform')
    return tm


def gen_stacked_arenas(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    cx=cy=size//2
    radii=[18,12,6]
    for idx,r in enumerate(radii):
        tm.ring(cx,cy,r-2,r,'floor')
        tm.set_overlay(cx+r//2,cy-idx*2,'ladder' if idx<2 else 'elevator')
    tm.circle(cx,cy,4,'floor')
    return tm


def gen_teleport_rooms(spec, rng):
    size=spec.get('size',64)
    bg=spec.get('background','abyss')
    tm=TileMap(size,size,bg)
    rooms=[(8,8,10,8),(size-18,8,10,8),(8,size-16,10,8),(size-18,size-16,10,8),(size//2-5,size//2-4,10,8)]
    for x,y,w,h in rooms:
        tm.rect(x,y,w,h,'floor')
        tm.set_overlay(x+w//2,y+h//2,'teleport')
    return tm


def gen_tree_deadends(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    root=(size//2,size-8)
    segs=branch_tree_segments(rng, root, -math.pi/2, spec.get('length',9), spec.get('depth',5), spec.get('spread',0.8), step=5)
    endpoints=[]
    for (x1,y1),(x2,y2),dep in segs:
        tm.line(x1,y1,x2,y2,2,'floor')
        if dep<=1:
            tm.rect(x2-2,y2-2,5,5,'floor')
            endpoints.append((x2,y2))
    for x,y in endpoints[::2]:
        tm.set_overlay(x,y,'deadend')
    return tm


def gen_patrol_routes(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    # perimeter corridors
    tm.outline_rect(6,6,size-12,size-12,'floor',None,th=3)
    tm.outline_rect(16,16,size-32,size-32,'floor',None,th=2)
    # branching guard routes
    for x in [size//3, 2*size//3]:
        tm.line(x,6,x,size-7,2,'floor')
    for y in [size//3, 2*size//3]:
        tm.line(6,y,size-7,y,2,'floor')
    if spec.get('interior_branches'):
        for _ in range(6):
            x=rng.choice([size//3,2*size//3]); y=rng.randint(10,size-11)
            tm.line(x,y,x+rng.choice([-8,8]),y+rng.choice([-6,6]),1,'floor')
    return tm


def gen_river_split(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    # rooms and paths first
    rooms=random_room_positions(rng,size,size,8,(6,9),(5,8),margin=6)
    for x,y,w,h in rooms:
        tm.rect(x,y,w,h,'floor')
    connect_room_centers(tm,rooms,width=2,base='floor',rng=rng,extra=0.15)
    # river
    pts=[(4,size//2-8),(size//3,size//2-2),(2*size//3,size//2+6),(size-4,size//2)]
    tm.polyline(pts,width=5,base='water')
    # bridges
    for x in [size//3, size//2, 2*size//3]:
        tm.line(x,size//2-5,x,size//2+5,2,'bridge')
    return tm


def gen_library(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    tm.rect(6,6,size-12,size-12,'floor')
    # shelves
    for y in range(12,size-12,8):
        tm.rect(12,y,size-24,3,'bookshelf')
    # stacked walkways
    for y in [18,size//2,size-18]:
        tm.line(8,y,size-9,y,2,'bridge')
    for x in [size//3,2*size//3]:
        tm.line(x,8,x,size-9,2,'ladder_tile')
        tm.set_overlay(x,size//2,'ladder')
    return tm


def gen_quadrants_core(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    tm.rect(6,6,size-12,size-12,'floor')
    cx=cy=size//2
    tm.rect(cx-2,6,4,size-12,'wall')
    tm.rect(6,cy-2,size-12,4,'wall')
    tm.rect(cx-5,cy-5,10,10,'floor')
    for dx in (-1,1):
        for dy in (-1,1):
            x=cx+dx*12; y=cy+dy*12
            tm.rect(x-5,y-5,10,10,'floor')
            tm.line(cx+dx*5,cy+dy*5,x,y,2,'floor')
    return tm


def gen_step_rooms(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    x=6; y=6
    for i in range(spec.get('steps',7)):
        rw=8+rng.randint(-1,2)
        rh=6+rng.randint(-1,2)
        tm.rect(x,y,rw,rh,'floor')
        if i>0:
            tm.line(prev[0],prev[1],x+rw//2,y+rh//2,2,'floor')
            tm.set_overlay((prev[0]+x+rw//2)//2,(prev[1]+y+rh//2)//2,'stair')
        prev=(x+rw//2,y+rh//2)
        x += spec.get('dx',6)
        y += spec.get('dy',6)
    return tm


def gen_dense_cluster(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    rooms=random_room_positions(rng,size,size,spec.get('room_count',22),(3,5),(3,5),margin=6,bounds=(8,8,size-16,size-16))
    for x,y,w,h in rooms:
        tm.rect(x,y,w,h,'floor')
    connect_room_centers(tm,rooms,width=1,base='floor',rng=rng,extra=0.45)
    return tm


def gen_identical_corridors(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,spec.get('background','wall'))
    # repetitive corridors
    for y in range(8,size-8,8):
        tm.line(8,y,size-9,y,2,'floor')
    for x in range(8,size-8,8):
        if x not in (size//2, size//2-8):
            tm.line(x,8,x,size-9,2,'floor')
    # subtle variations
    for _ in range(10):
        x=rng.choice(range(8,size-8,8)); y=rng.choice(range(8,size-8,8))
        tm.rect(x-2,y-2,5,5,'wall' if rng.random()<0.5 else 'floor')
    return tm


def gen_alt_interior_exterior(spec, rng):
    size=spec.get('size',64)
    tm=TileMap(size,size,'abyss')
    # tower rings
    tm.ring(size//2,size//2,10,13,'floor')  # interior ring
    tm.ring(size//2,size//2,18,21,'platform')  # exterior path
    for ang in [0, math.pi/2, math.pi, 3*math.pi/2]:
        x1=round(size//2+math.cos(ang)*13); y1=round(size//2+math.sin(ang)*13)
        x2=round(size//2+math.cos(ang)*18); y2=round(size//2+math.sin(ang)*18)
        tm.line(x1,y1,x2,y2,2,'bridge')
        tm.set_overlay((x1+x2)//2,(y1+y2)//2,'stair')
    tm.circle(size//2,size//2,6,'floor')
    return tm


def gen_multi_entrance_converge(spec, rng):
    size=spec.get('size',64)
    tm=gen_fortress({'size':size,'rings':3,'courtyard':False,'keep':12,'entrances':8,'background':'wall'}, rng)
    # extra converging diagonals
    for p in [(4,4),(size-5,4),(4,size-5),(size-5,size-5)]:
        tm.line(p[0],p[1],size//2,size//2,2,'floor')
    return tm


def matrix_to_map(matrix, open_tile='floor', closed_tile='wall'):
    h=len(matrix); w=len(matrix[0])
    tm=TileMap(w,h,closed_tile)
    for y,row in enumerate(matrix):
        for x,v in enumerate(row):
            if v==1:
                tm.set(x,y,open_tile,None)
            elif v==2:
                tm.set(x,y,'water',None)
            elif v==3:
                tm.set(x,y,'bridge',None)
            elif v==4:
                tm.set(x,y,'platform',None)
            else:
                tm.set(x,y,closed_tile,None)
    return tm


def make_spiral_matrix(n):
    # hand-tuned narrow spiral corridor; works best for odd n>=5
    g=[[0]*n for _ in range(n)]
    left=0; top=0; right=n-1; bottom=n-1
    while left<=right and top<=bottom:
        for x in range(left,right+1): g[top][x]=1
        top += 1
        for y in range(top,bottom+1): g[y][right]=1
        right -= 1
        if top<=bottom:
            for x in range(right,left-1,-1): g[bottom][x]=1
            bottom -= 1
        if left<=right:
            for y in range(bottom,top+1,-1): g[y][left]=1
            left += 1
        # carve blocking cuts one step inward to create a single corridor spiral
        if top-1 < n and left-1 < n and top-1>=0 and left-1>=0:
            g[top-1][left-1]=0
    c=n//2
    g[c][c]=1
    return g


def single_path_corner_to_corner(n):
    g=[[0]*n for _ in range(n)]
    x=y=0
    g[y][x]=1
    direction=1
    while y < n-1:
        target_x = n-1 if direction==1 else 0
        while x != target_x:
            x += direction
            g[y][x]=1
        if y < n-1:
            y += 1
            g[y][x]=1
        direction *= -1
    return g


def central_chamber_ring(n):
    g=[[0]*n for _ in range(n)]
    c=n//2
    # outer walls blocked, inner ring corridor at distance 2/3, central chamber 3x3 with doorways
    for y in range(1,n-1):
        for x in range(1,n-1):
            if max(abs(x-c),abs(y-c))==3:
                g[y][x]=1
            if abs(x-c)<=1 and abs(y-c)<=1:
                g[y][x]=1
    # doors between ring and chamber
    for x,y in [(c,c-2),(c,c+2),(c-2,c),(c+2,c)]:
        g[y][x]=1
    return g


def clustered_obstacles(n, seed=0):
    rng=random.Random(seed)
    g=[[1]*n for _ in range(n)]
    for _ in range(4):
        cx=rng.randint(1,n-2); cy=rng.randint(1,n-2)
        for y in range(cy-1, cy+2):
            for x in range(cx-1, cx+2):
                if 0<=x<n and 0<=y<n and rng.random()<0.6:
                    g[y][x]=0
    return g


def zigzag_barriers(n):
    g=[[1]*n for _ in range(n)]
    for i in range(1,n-1):
        if i%2==1:
            for y in range(i,n-1):
                g[y][i]=0
        else:
            for y in range(1,i+1):
                g[y][i]=0
    # cut openings to permit zigzag traversal
    for i in range(1,n-1):
        if i%2==1:
            g[n-2][i]=1
        else:
            g[1][i]=1
    return g


def concentric_square_alt(n):
    g=[[0]*n for _ in range(n)]
    c=n//2
    for y in range(n):
        for x in range(n):
            ring=max(abs(x-c),abs(y-c))
            g[y][x]=1 if ring%2==0 else 0
    return g


def hollow_border_room(n):
    g=[[1]*n for _ in range(n)]
    for i in range(n):
        g[0][i]=0; g[n-1][i]=0; g[i][0]=0; g[i][n-1]=0
    return g


def quadrants_cross_wall(n):
    g=[[1]*n for _ in range(n)]
    c=n//2
    for i in range(n):
        g[c][i]=0; g[i][c]=0
    return g


def vertical_stripes(n):
    return [[1 if x%2==0 else 0 for x in range(n)] for _ in range(n)]


def checkerboard(n, start=0):
    return [[1 if (x+y+start)%2 else 0 for x in range(n)] for y in range(n)]


def gen_micro_matrix(spec, rng):
    n=spec['size']
    pattern=spec.get('pattern')
    if 'matrix' in spec:
        m=spec['matrix']
    elif pattern=='spiral':
        m=make_spiral_matrix(n)
    elif pattern=='corner_path':
        m=single_path_corner_to_corner(n)
    elif pattern=='checkerboard':
        m=checkerboard(n, start=0)
    elif pattern=='hollow_border':
        m=hollow_border_room(n)
    elif pattern=='quadrants':
        m=quadrants_cross_wall(n)
    elif pattern=='concentric':
        m=concentric_square_alt(n)
    elif pattern=='stripes':
        m=vertical_stripes(n)
    elif pattern=='center_ring':
        m=central_chamber_ring(n)
    elif pattern=='clusters':
        m=clustered_obstacles(n, seed=spec.get('seed',0))
    elif pattern=='zigzag_barriers':
        m=zigzag_barriers(n)
    else:
        raise ValueError(pattern)
    return matrix_to_map(m)


layout_specs = [{'background': 'wall',
  'hub_r': 7,
  'id': 1,
  'name': 'A circular hub with radiating corridors like spokes',
  'size': 64,
  'spoke_len': 22,
  'spokes': 8,
  'template': 'hub_spokes',
  'terminal': 'room',
  'terminal_size': 7,
  'theme': 'dungeon'},
 {'background': 'wall',
  'core': 'chamber',
  'core_r': 4,
  'end_r': 7,
  'id': 2,
  'name': 'A descending spiral tower with layered platforms',
  'platform_count': 7,
  'platform_overlay': 'stair',
  'platforms': True,
  'size': 64,
  'spiral_type': 'circular',
  'start_r': 24,
  'template': 'spiral',
  'theme': 'tower',
  'turns': 3.3,
  'width': 2},
 {'background': 'wall',
  'cols': 4,
  'connection_mode': 'maze',
  'gap_x': 4,
  'gap_y': 4,
  'id': 3,
  'name': 'A grid-like labyrinth of uniform square rooms',
  'room_h': 8,
  'room_w': 8,
  'rows': 4,
  'size': 64,
  'template': 'grid_rooms',
  'theme': 'dungeon'},
 {'background': 'wall',
  'id': 4,
  'name': 'A vertical shaft with stacked chambers and ladders',
  'room_h': 6,
  'room_w': 10,
  'rooms': 6,
  'shaft_overlay': 'ladder',
  'shaft_w': 3,
  'size': 64,
  'template': 'vertical_stack',
  'theme': 'shaft'},
 {'angle': -1.5707963267948966,
  'background': 'wall',
  'depth': 5,
  'id': 5,
  'length': 9,
  'name': 'A branching cave system with irregular tunnels',
  'room_rate': 0.5,
  'size': 64,
  'spread': 1.1,
  'start': (32, 56),
  'template': 'branch_caves',
  'theme': 'cave',
  'width': 3},
 {'background': 'wall',
  'center_h': 10,
  'center_w': 12,
  'id': 6,
  'name': 'A symmetrical temple with mirrored wings',
  'room_h': 6,
  'room_w': 8,
  'size': 64,
  'template': 'symmetrical_complex',
  'theme': 'temple',
  'wing_rooms': 4},
 {'background': 'wall',
  'courtyard': True,
  'entrances': 4,
  'id': 7,
  'keep': 12,
  'name': 'A multi-floor fortress with central courtyard',
  'rings': 3,
  'size': 64,
  'template': 'fortress',
  'theme': 'fortress'},
 {'background': 'wall',
  'corridor_w': 3,
  'id': 8,
  'name': 'A narrow linear gauntlet with occasional side alcoves',
  'offset': 8,
  'room_h': 5,
  'room_rate': 0.8,
  'room_w': 6,
  'size': 64,
  'template': 'linear_gauntlet',
  'theme': 'gauntlet'},
 {'background': 'wall',
  'extra_loops': 0.35,
  'id': 9,
  'name': 'A maze of interlocking loops and shortcuts',
  'room_count': 12,
  'size': 64,
  'template': 'loops_network',
  'theme': 'dungeon'},
 {'background': 'wall',
  'id': 10,
  'levels': 5,
  'name': 'A tiered pyramid with ascending terraces',
  'size': 64,
  'step': 5,
  'template': 'pyramid_terraces',
  'terraces': True,
  'th': 2,
  'theme': 'pyramid'},
 {'background': 'wall',
  'id': 11,
  'name': 'A submerged ruin with interconnected waterlogged halls',
  'room_count': 8,
  'ruin': True,
  'size': 64,
  'template': 'flooded',
  'theme': 'ruin',
  'theme_variant': 'ruin'},
 {'background': 'wall',
  'district_cols': 3,
  'district_rows': 3,
  'id': 12,
  'name': 'A sprawling underground city with districts',
  'size': 72,
  'template': 'city',
  'theme': 'city'},
 {'background': 'wall',
  'collapsed': True,
  'fragments': 9,
  'id': 13,
  'name': 'A collapsed structure with fragmented pathways',
  'size': 64,
  'template': 'fragmented',
  'theme': 'ruin'},
 {'background': 'wall',
  'id': 14,
  'mode': 'zigzag',
  'name': 'A zigzagging canyon with elevated ledges',
  'size': 64,
  'template': 'canyon',
  'theme': 'canyon',
  'width': 5},
 {'background': 'wall',
  'core': 'chamber',
  'core_w': 10,
  'id': 15,
  'name': 'A ring-shaped corridor encircling a core chamber',
  'radials': False,
  'rings': 1,
  'size': 64,
  'start_r': 12,
  'template': 'ring_core',
  'th': 3,
  'theme': 'dungeon'},
 {'background': 'wall',
  'id': 16,
  'name': 'A layered dungeon where floors overlap vertically',
  'size': 64,
  'template': 'layered_overlap',
  'theme': 'dungeon'},
 {'background': 'wall',
  'cells': 4,
  'id': 17,
  'name': 'A modular set of rooms connected by rotating passages',
  'open_rate': 0.7,
  'size': 64,
  'template': 'rotating_modular',
  'theme': 'modular'},
 {'hazard': 'abyss',
  'id': 18,
  'name': 'A lattice of bridges suspended over void',
  'size': 64,
  'template': 'bridge_lattice',
  'theme': 'void'},
 {'background': 'wall',
  'id': 19,
  'name': 'A honeycomb structure of hexagonal chambers',
  'r': 4,
  'size': 64,
  'template': 'honeycomb',
  'theme': 'hive'},
 {'background': 'wall',
  'courtyard': False,
  'entrances': 4,
  'id': 20,
  'keep': 10,
  'name': 'A castle keep with concentric defensive layers',
  'rings': 4,
  'size': 64,
  'template': 'fortress',
  'theme': 'castle'},
 {'background': 'wall',
  'branch_len': 10,
  'id': 21,
  'name': 'A narrow shaft with branching side tunnels',
  'narrow': True,
  'room_h': 5,
  'room_w': 8,
  'rooms': 7,
  'shaft_overlay': 'ladder',
  'side_branches': True,
  'size': 64,
  'template': 'vertical_stack',
  'theme': 'mine'},
 {'background': 'wall',
  'branch_rooms': True,
  'hub_h': 10,
  'hub_shape': 'rect',
  'hub_w': 10,
  'id': 22,
  'name': 'A multi-wing complex with a central nexus',
  'size': 64,
  'spoke_len': 20,
  'spokes': 6,
  'template': 'hub_spokes',
  'terminal': 'room',
  'terminal_size': 8,
  'theme': 'complex'},
 {'background': 'abyss',
  'count': 8,
  'dx': 6,
  'dy': 6,
  'id': 23,
  'ladders': True,
  'name': 'A diagonal climb across staggered platforms',
  'size': 64,
  'template': 'diagonal_platforms',
  'theme': 'platform',
  'void_bg': True},
 {'alternating': True,
  'background': 'wall',
  'id': 24,
  'name': 'A flooded labyrinth with alternating dry and submerged paths',
  'room_count': 10,
  'ruin': False,
  'size': 64,
  'template': 'flooded',
  'theme': 'labyrinth'},
 {'background': 'wall',
  'core': 'void',
  'core_r': 6,
  'end_r': 8,
  'id': 25,
  'name': 'A spiral staircase wrapping a hollow core',
  'platforms': False,
  'size': 64,
  'spiral_type': 'circular',
  'start_r': 25,
  'template': 'spiral',
  'theme': 'tower',
  'turns': 3.0,
  'width': 2},
 {'background': 'wall',
  'extra_loops': 0.3,
  'id': 26,
  'name': 'A crisscrossing network of tunnels at multiple elevations',
  'overlap': True,
  'room_count': 10,
  'size': 64,
  'template': 'loops_network',
  'theme': 'caves'},
 {'background': 'wall',
  'cols': 5,
  'gap_x': 4,
  'gap_y': 4,
  'id': 27,
  'lock_rate': 0.7,
  'locked_intersections': True,
  'name': 'A rectangular grid with locked intersections',
  'room_h': 7,
  'room_w': 7,
  'rows': 4,
  'size': 64,
  'template': 'grid_rooms',
  'theme': 'dungeon'},
 {'background': 'wall',
  'id': 28,
  'name': 'A cavern with towering pillars forming natural corridors',
  'pillars': 26,
  'size': 64,
  'template': 'pillar_cavern',
  'theme': 'cave'},
 {'background': 'wall',
  'id': 29,
  'landings': True,
  'name': 'A descending zigzag staircase with landings',
  'size': 64,
  'template': 'zigzag_stairs',
  'theme': 'stair'},
 {'background': 'wall',
  'hub_r': 9,
  'id': 30,
  'name': 'A central arena with branching challenge rooms',
  'size': 64,
  'spoke_len': 18,
  'spokes': 6,
  'template': 'hub_spokes',
  'term_overlay': 'challenge',
  'terminal': 'room',
  'terminal_size': 7,
  'theme': 'arena'},
 {'background': 'wall',
  'id': 31,
  'maze_wing': True,
  'name': 'A mirrored maze where paths reflect each other',
  'size': 64,
  'template': 'symmetrical_complex',
  'theme': 'maze'},
 {'background': 'wall',
  'id': 32,
  'name': 'A tower of stacked circular arenas',
  'size': 64,
  'template': 'stacked_arenas',
  'theme': 'tower'},
 {'background': 'wall',
  'id': 33,
  'mode': 'cliffside',
  'name': 'A winding cliffside path hugging outer walls',
  'size': 64,
  'template': 'canyon',
  'theme': 'cliff'},
 {'background': 'abyss',
  'id': 34,
  'name': 'A set of isolated rooms connected by teleport nodes',
  'size': 64,
  'template': 'teleport_rooms',
  'theme': 'arcane'},
 {'background': 'wall',
  'depth': 5,
  'id': 35,
  'length': 9,
  'name': 'A branching tree-like structure with dead ends',
  'size': 64,
  'template': 'tree_deadends',
  'theme': 'roots'},
 {'background': 'wall',
  'id': 36,
  'name': 'A fortress wall interior with patrol corridors',
  'size': 64,
  'template': 'patrol_routes',
  'theme': 'fortress'},
 {'background': 'wall',
  'id': 37,
  'name': 'A subterranean river splitting pathways',
  'size': 64,
  'template': 'river_split',
  'theme': 'cave'},
 {'background': 'wall',
  'hidden': True,
  'id': 38,
  'name': 'A layered dungeon with hidden passages between floors',
  'size': 64,
  'template': 'layered_overlap',
  'theme': 'dungeon'},
 {'background': 'wall',
  'core': 'chamber',
  'core_w': 6,
  'gap': 5,
  'id': 39,
  'maze_rings': True,
  'name': 'A circular maze with concentric rings',
  'radial_count': 4,
  'radials': True,
  'rings': 3,
  'size': 64,
  'start_r': 8,
  'template': 'ring_core',
  'th': 2,
  'theme': 'maze'},
 {'background': 'abyss',
  'balconies': True,
  'count': 0,
  'id': 40,
  'name': 'A staggered series of balconies overlooking a void',
  'size': 64,
  'template': 'diagonal_platforms',
  'theme': 'void',
  'void_bg': True},
 {'background': 'wall',
  'compact': True,
  'extra_loops': 0.45,
  'id': 41,
  'name': 'A compact dungeon with dense interconnected shortcuts',
  'room_count': 14,
  'size': 64,
  'template': 'loops_network',
  'theme': 'dungeon'},
 {'background': 'wall',
  'gated': True,
  'id': 42,
  'name': 'A long corridor punctuated by gated chambers',
  'offset': 9,
  'room_rate': 0.9,
  'size': 64,
  'template': 'linear_gauntlet',
  'theme': 'gauntlet'},
 {'background': 'abyss',
  'bridge_links': True,
  'broken': True,
  'count': 9,
  'dx': 4,
  'dy': 6,
  'id': 43,
  'name': 'A vertical climb through broken platforms',
  'size': 64,
  'template': 'diagonal_platforms',
  'theme': 'platform',
  'void_bg': True},
 {'background': 'wall',
  'dual_path': True,
  'extra_loops': 0.15,
  'id': 44,
  'name': 'A dual-path layout that reconverges repeatedly',
  'room_count': 6,
  'size': 64,
  'template': 'loops_network',
  'theme': 'dungeon'},
 {'background': 'wall',
  'hub_r': 6,
  'id': 45,
  'name': 'A central shaft with radial elevators',
  'size': 64,
  'spoke_len': 20,
  'spokes': 8,
  'template': 'hub_spokes',
  'term_overlay': 'elevator',
  'terminal': 'alcove',
  'theme': 'shaft'},
 {'background': 'wall',
  'hub_r': 7,
  'id': 46,
  'name': 'A series of puzzle rooms branching from a hub',
  'size': 64,
  'spoke_len': 18,
  'spokes': 5,
  'template': 'hub_spokes',
  'term_overlay': 'challenge',
  'terminal': 'room',
  'terminal_size': 8,
  'theme': 'puzzle'},
 {'background': 'wall',
  'extra_loops': 0.3,
  'id': 47,
  'name': 'A looping corridor system with shifting walls',
  'room_count': 10,
  'shifting': True,
  'size': 64,
  'template': 'loops_network',
  'theme': 'dungeon'},
 {'hazard': 'lava',
  'id': 48,
  'name': 'A narrow bridge network over lava or abyss',
  'size': 64,
  'template': 'bridge_lattice',
  'theme': 'lava',
  'width': 1},
 {'angle': -0.8,
  'background': 'wall',
  'depth': 5,
  'id': 49,
  'length': 8,
  'name': 'A collapsed mine with slanted tunnels',
  'rubble': True,
  'size': 64,
  'slanted': True,
  'spread': 0.7,
  'start': (12, 52),
  'template': 'branch_caves',
  'theme': 'mine',
  'width': 2},
 {'background': 'wall',
  'cross': True,
  'id': 50,
  'mirror_y': False,
  'name': 'A symmetrical cross-shaped layout with four wings',
  'size': 64,
  'template': 'symmetrical_complex',
  'theme': 'temple'},
 {'background': 'wall',
  'core': 'void',
  'core_r': 4,
  'end_r': 5,
  'id': 51,
  'name': 'A descending funnel-shaped cavern',
  'outer_platform_ring': (24, 28),
  'size': 64,
  'spiral_type': 'circular',
  'start_r': 24,
  'template': 'spiral',
  'theme': 'cave',
  'turns': 2.4,
  'width': 4},
 {'arena': True,
  'arena_r': 9,
  'background': 'wall',
  'id': 52,
  'levels': 4,
  'name': 'A tiered arena with surrounding corridors',
  'size': 64,
  'step': 6,
  'template': 'pyramid_terraces',
  'terraces': False,
  'th': 2,
  'theme': 'arena'},
 {'background': 'wall',
  'courtyard': False,
  'entrances': 2,
  'id': 53,
  'keep': 10,
  'name': 'A rectangular fortress with perimeter hallways',
  'perimeter_halls': True,
  'rings': 2,
  'size': 64,
  'template': 'fortress',
  'theme': 'fortress'},
 {'background': 'wall',
  'id': 54,
  'name': 'A winding sewer system with circular junctions',
  'size': 64,
  'template': 'flooded',
  'theme': 'sewer'},
 {'background': 'wall',
  'id': 55,
  'name': 'A multi-level library with stacked walkways',
  'size': 64,
  'template': 'library',
  'theme': 'library'},
 {'background': 'wall',
  'id': 56,
  'name': 'A dungeon divided into quadrants around a core',
  'size': 64,
  'template': 'quadrants_core',
  'theme': 'dungeon'},
 {'background': 'wall',
  'dx': 6,
  'dy': 6,
  'id': 57,
  'name': 'A cascading series of rooms descending like steps',
  'size': 64,
  'steps': 7,
  'template': 'step_rooms',
  'theme': 'dungeon'},
 {'background': 'wall',
  'id': 58,
  'name': 'A dense cluster of small interconnected chambers',
  'room_count': 26,
  'size': 64,
  'template': 'dense_cluster',
  'theme': 'dungeon'},
 {'background': 'wall',
  'branch_len': 9,
  'drop_shafts': True,
  'id': 59,
  'name': 'A vertical labyrinth with ladders and drop shafts',
  'rooms': 8,
  'shaft_overlay': 'ladder',
  'side_branches': True,
  'size': 64,
  'template': 'vertical_stack',
  'theme': 'shaft',
  'vary_room': True},
 {'background': 'wall',
  'core': 'arena',
  'core_r': 7,
  'gap': 6,
  'hidden': True,
  'id': 60,
  'name': 'A circular arena surrounded by hidden passages',
  'radial_count': 4,
  'radials': True,
  'rings': 2,
  'size': 64,
  'start_r': 10,
  'template': 'ring_core',
  'th': 2,
  'theme': 'arena'},
 {'background': 'wall',
  'chokepoints': True,
  'id': 61,
  'name': 'A chain of rooms connected by narrow chokepoints',
  'offset': 7,
  'room_h': 6,
  'room_rate': 1.0,
  'room_w': 7,
  'size': 64,
  'template': 'linear_gauntlet',
  'theme': 'dungeon'},
 {'background': 'wall',
  'core': 'void',
  'core_r': 7,
  'end_r': 9,
  'id': 62,
  'landings': True,
  'name': 'A spiraling descent around a central pit',
  'size': 64,
  'spiral_type': 'circular',
  'start_r': 24,
  'template': 'spiral',
  'theme': 'tower',
  'turns': 3.8,
  'width': 2},
 {'background': 'abyss',
  'id': 63,
  'mode': 'bridged',
  'name': 'A multi-path canyon with crossing bridges',
  'size': 64,
  'template': 'canyon',
  'theme': 'canyon'},
 {'background': 'wall',
  'id': 64,
  'interior_branches': True,
  'name': 'A fortress interior with branching guard routes',
  'size': 64,
  'template': 'patrol_routes',
  'theme': 'fortress'},
 {'background': 'wall',
  'id': 65,
  'name': 'A maze of identical corridors with subtle variations',
  'size': 64,
  'template': 'identical_corridors',
  'theme': 'maze'},
 {'background': 'abyss',
  'bridge_links': True,
  'broken': False,
  'count': 10,
  'dx': 4,
  'dy': 5,
  'id': 66,
  'ladders': True,
  'name': 'A stacked set of platforms forming a vertical puzzle',
  'size': 64,
  'template': 'diagonal_platforms',
  'theme': 'platform',
  'void_bg': True},
 {'background': 'wall',
  'central_hall': True,
  'id': 67,
  'name': 'A central hall with branching wings that loop back',
  'room_count': 4,
  'size': 64,
  'template': 'loops_network',
  'theme': 'dungeon'},
 {'hazard': 'abyss',
  'id': 68,
  'name': 'A dungeon split by a massive chasm with crossings',
  'size': 64,
  'split': True,
  'template': 'bridge_lattice',
  'theme': 'chasm'},
 {'background': 'wall',
  'core': 'chamber',
  'core_w': 6,
  'gap': 6,
  'id': 69,
  'name': 'A series of concentric chambers decreasing inward',
  'radials': False,
  'rings': 3,
  'size': 64,
  'start_r': 8,
  'template': 'ring_core',
  'th': 3,
  'theme': 'dungeon'},
 {'background': 'wall',
  'cells': 5,
  'id': 70,
  'name': 'A grid of rooms with rotating or shifting connections',
  'open_rate': 0.55,
  'size': 64,
  'template': 'rotating_modular',
  'theme': 'modular'},
 {'background': 'abyss',
  'id': 71,
  'name': 'A tower with alternating interior and exterior paths',
  'size': 64,
  'template': 'alt_interior_exterior',
  'theme': 'tower'},
 {'angle': -1.5707963267948966,
  'background': 'wall',
  'depth': 4,
  'id': 72,
  'length': 8,
  'name': 'A layered cavern with natural vertical shafts',
  'open_cavern': True,
  'shafts': True,
  'size': 64,
  'spread': 0.9,
  'start': (32, 56),
  'template': 'branch_caves',
  'theme': 'cave',
  'width': 3},
 {'background': 'wall',
  'detours': True,
  'id': 73,
  'name': 'A linear descent punctuated by branching detours',
  'room_rate': 0.6,
  'size': 64,
  'template': 'linear_gauntlet',
  'theme': 'gauntlet'},
 {'background': 'wall',
  'core': 'chamber',
  'core_w': 8,
  'diagonal_rooms': True,
  'gap': 6,
  'id': 74,
  'name': 'A ring of rooms connected by diagonal corridors',
  'ring_rooms': True,
  'rings': 1,
  'room_count': 8,
  'size': 64,
  'start_r': 12,
  'template': 'ring_core',
  'th': 2,
  'theme': 'dungeon'},
 {'background': 'wall',
  'extra_loops': 0.4,
  'id': 75,
  'name': 'A network of tunnels forming overlapping loops',
  'overlap': True,
  'room_count': 11,
  'size': 64,
  'template': 'loops_network',
  'theme': 'caves'},
 {'background': 'wall',
  'id': 76,
  'name': 'A fortress with multiple entrances converging inward',
  'size': 64,
  'template': 'multi_entrance_converge',
  'theme': 'fortress'},
 {'background': 'wall',
  'challenge_rooms': True,
  'id': 77,
  'name': 'A vertical gauntlet of consecutive challenge rooms',
  'room_h': 6,
  'room_w': 11,
  'rooms': 7,
  'size': 64,
  'template': 'vertical_stack',
  'theme': 'gauntlet'},
 {'background': 'wall',
  'id': 78,
  'maze': True,
  'name': 'A staggered maze with offset pathways between levels',
  'size': 64,
  'template': 'layered_overlap',
  'theme': 'dungeon'},
 {'background': 'wall',
  'hidden_radials': True,
  'hub_r': 8,
  'id': 79,
  'name': 'A central chamber with hidden radial exits',
  'size': 64,
  'spoke_len': 14,
  'spokes': 4,
  'template': 'hub_spokes',
  'terminal': 'none',
  'theme': 'dungeon'},
 {'background': 'abyss',
  'floating': True,
  'fragments': 10,
  'id': 80,
  'name': 'A fragmented dungeon of floating, disconnected rooms',
  'size': 64,
  'template': 'fragmented',
  'theme': 'void'}]

micro_specs = [{'id': 81,
  'matrix': [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
  'name': 'A 3×3 grid with a solid border and a single open center tile',
  'size': 3,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 82,
  'name': 'A 3×3 checkerboard pattern alternating blocked and open tiles',
  'pattern': 'checkerboard',
  'size': 3,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 83,
  'matrix': [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
  'name': 'A 3×3 room where all edges are open but the center tile is blocked',
  'size': 3,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 84,
  'matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  'name': 'A 3×3 with a diagonal path from top-left to bottom-right',
  'size': 3,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 85,
  'matrix': [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
  'name': 'A 3×3 where only the four corner tiles are accessible',
  'size': 3,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 86,
  'matrix': [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
  'name': 'A 3×3 with a plus-shaped open path (center + orthogonal tiles)',
  'size': 3,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 87,
  'matrix': [[1, 1, 1], [0, 1, 1], [0, 1, 1]],
  'name': 'A 3×3 spiral path starting at an outer corner and ending in the center',
  'size': 3,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 88,
  'matrix': [[1, 0, 1], [1, 0, 1], [1, 0, 1]],
  'name': 'A 3×3 with two parallel vertical corridors and a blocked middle column',
  'size': 3,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 89,
  'matrix': [[1, 0, 1], [1, 1, 1], [1, 1, 1]],
  'name': 'A 3×3 fully open except for one random obstacle tile',
  'size': 3,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 90,
  'matrix': [[1, 1, 1], [1, 0, 0], [1, 0, 0]],
  'name': 'A 3×3 with an L-shaped accessible path hugging two edges',
  'size': 3,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 91,
  'name': 'A 9×9 room with a hollow square border and empty interior',
  'pattern': 'hollow_border',
  'size': 9,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 92,
  'matrix': [[1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 1, 1, 1, 1, 1, 1, 0, 1],
             [1, 0, 0, 0, 0, 0, 1, 0, 1],
             [1, 0, 1, 1, 1, 0, 1, 0, 1],
             [1, 0, 1, 0, 1, 0, 1, 0, 1],
             [1, 0, 1, 0, 1, 1, 1, 0, 1],
             [1, 0, 1, 0, 0, 0, 0, 0, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1]],
  'name': 'A 9×9 grid forming a spiral corridor winding inward',
  'size': 9,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 93,
  'name': 'A 9×9 with four quadrants separated by a cross-shaped wall',
  'pattern': 'quadrants',
  'size': 9,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 94,
  'name': 'A 9×9 checkerboard of alternating blocked and open tiles',
  'pattern': 'checkerboard',
  'size': 9,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 95,
  'name': 'A 9×9 with concentric square rings of alternating passability',
  'pattern': 'concentric',
  'size': 9,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 96,
  'matrix': [[1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 1]],
  'name': 'A 9×9 maze with a single winding path from one corner to the opposite',
  'size': 9,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 97,
  'name': 'A 9×9 divided into narrow vertical stripes of alternating accessibility',
  'pattern': 'stripes',
  'size': 9,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 98,
  'name': 'A 9×9 with a central 3×3 chamber surrounded by a ring corridor',
  'pattern': 'center_ring',
  'size': 9,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 99,
  'name': 'A 9×9 with scattered obstacles forming irregular clusters',
  'pattern': 'clusters',
  'seed': 99,
  'size': 9,
  'template': 'micro_matrix',
  'theme': 'micro'},
 {'id': 100,
  'name': 'A 9×9 featuring diagonal barriers creating zigzag traversal paths',
  'pattern': 'zigzag_barriers',
  'size': 9,
  'template': 'micro_matrix',
  'theme': 'micro'}]

PALETTE = {'abyss': (12, 12, 18),
 'bookshelf': (126, 84, 44),
 'bridge': (124, 86, 48),
 'courtyard': (76, 112, 72),
 'floor': (126, 128, 132),
 'ladder_tile': (138, 132, 120),
 'lava': (184, 88, 30),
 'platform': (152, 156, 164),
 'rubble': (98, 94, 92),
 'wall': (44, 48, 54),
 'water': (42, 90, 138)}
DEFAULT_FONT = ImageFont.load_default()

def vary_color(color, delta):
    return tuple(clamp(c+delta,0,255) for c in color)


def draw_tile(draw, x0, y0, s, tile, seed, cx, cy):
    base = PALETTE.get(tile, (128,128,128))
    delta = int((noise(seed,cx,cy)-0.5)*20)
    fill = vary_color(base, delta)
    draw.rectangle([x0,y0,x0+s-1,y0+s-1], fill=fill)
    # pattern
    n1 = noise(seed+13,cx,cy,1)
    n2 = noise(seed+29,cx,cy,2)
    if tile in ('floor','platform','ladder_tile'):
        # beveled edges and occasional cracks
        light=vary_color(fill, 12)
        dark=vary_color(fill, -12)
        draw.line([x0,y0,x0+s-1,y0], fill=light)
        draw.line([x0,y0,x0,y0+s-1], fill=light)
        draw.line([x0+s-1,y0,x0+s-1,y0+s-1], fill=dark)
        draw.line([x0,y0+s-1,x0+s-1,y0+s-1], fill=dark)
        if s>=6 and n1>0.55:
            draw.line([x0+1,y0+s//2,x0+s-2,y0+s//2 + (1 if n2>0.5 else -1)], fill=vary_color(fill,-25))
    elif tile=='wall':
        speck=vary_color(fill, 18)
        if s>=4:
            for i in range(2 if s<12 else 4):
                px = x0 + int(noise(seed+41,cx,cy,i)*max(1,s-2))
                py = y0 + int(noise(seed+57,cx,cy,i)*max(1,s-2))
                draw.point((px,py), fill=speck)
            if s>=6:
                yy=y0+s//2
                draw.line([x0,yy,x0+s-1,yy], fill=vary_color(fill,-10))
    elif tile=='water':
        wave=vary_color(fill, 30)
        for i in range(2):
            yy = y0 + (i+1)*s//3
            draw.arc([x0+1,yy-2,x0+s-2,yy+2], 0, 180, fill=wave)
    elif tile=='lava':
        glow=vary_color(fill, 35)
        draw.ellipse([x0+s//4,y0+s//4,x0+3*s//4,y0+3*s//4], outline=glow, width=1)
        if s>=6:
            draw.line([x0+1,y0+s-2,x0+s-2,y0+1], fill=vary_color(fill,50))
    elif tile=='abyss':
        star=vary_color((80,70,110), int(noise(seed+71,cx,cy)*40))
        for i in range(1 if s<10 else 2):
            px=x0+int(noise(seed+83,cx,cy,i)*max(1,s-1))
            py=y0+int(noise(seed+97,cx,cy,i)*max(1,s-1))
            draw.point((px,py), fill=star)
    elif tile=='bridge':
        plank=vary_color(fill,-18)
        if s>=4:
            step=max(2,s//3)
            for xx in range(x0, x0+s, step):
                draw.line([xx,y0,xx,y0+s-1], fill=plank)
    elif tile=='courtyard':
        grass=vary_color(fill, 18)
        for i in range(2 if s<10 else 4):
            px=x0+int(noise(seed+101,cx,cy,i)*max(1,s-2))
            py=y0+int(noise(seed+103,cx,cy,i)*max(1,s-2))
            draw.point((px,py), fill=grass)
    elif tile=='rubble':
        dark=vary_color(fill,-20)
        for i in range(3):
            px=x0+int(noise(seed+107,cx,cy,i)*max(1,s-2))
            py=y0+int(noise(seed+109,cx,cy,i)*max(1,s-2))
            draw.ellipse([px,py,px+1,py+1], fill=dark)
    elif tile=='bookshelf':
        wood=vary_color(fill,-18)
        for yy in range(y0+1, y0+s, max(2,s//3)):
            draw.line([x0,yy,x0+s-1,yy], fill=wood)
            if s>=6:
                draw.line([x0+s//2,yy-1,x0+s//2,yy+1], fill=vary_color(fill,20))


def draw_overlay(draw, x0,y0,s, overlay, seed, cx,cy):
    if not overlay:
        return
    c=(235,235,240)
    d=(20,20,20)
    mx=x0+s//2; my=y0+s//2
    if overlay=='ladder':
        draw.line([mx-s//4,my-s//3,mx-s//4,my+s//3], fill=c, width=1)
        draw.line([mx+s//4,my-s//3,mx+s//4,my+s//3], fill=c, width=1)
        for yy in range(my-s//4, my+s//4+1, max(1,s//5)):
            draw.line([mx-s//4,yy,mx+s//4,yy], fill=c)
    elif overlay=='teleport':
        draw.ellipse([mx-s//3,my-s//3,mx+s//3,my+s//3], outline=(120,240,255))
        draw.line([mx-s//4,my,mx+s//4,my], fill=(120,240,255))
        draw.line([mx,my-s//4,mx,my+s//4], fill=(120,240,255))
    elif overlay=='hidden':
        dash=max(1,s//6)
        for xx in range(x0+1,x0+s-1,dash*2):
            draw.line([xx,y0+1,min(xx+dash,x0+s-2),y0+1], fill=(230,210,120))
            draw.line([xx,y0+s-2,min(xx+dash,x0+s-2),y0+s-2], fill=(230,210,120))
        for yy in range(y0+1,y0+s-1,dash*2):
            draw.line([x0+1,yy,x0+1,min(yy+dash,y0+s-2)], fill=(230,210,120))
            draw.line([x0+s-2,yy,x0+s-2,min(yy+dash,y0+s-2)], fill=(230,210,120))
    elif overlay=='door':
        draw.line([mx,my-s//3,mx,my+s//3], fill=(180,140,70), width=1)
    elif overlay=='locked':
        draw.line([mx,my-s//3,mx,my+s//3], fill=(180,140,70), width=1)
        draw.rectangle([mx-2,my-1,mx+2,my+3], outline=(240,220,100))
        draw.arc([mx-2,my-4,mx+2,my], 180, 360, fill=(240,220,100))
    elif overlay=='elevator':
        draw.polygon([(mx,my-s//3),(mx-s//4,my-s//8),(mx+s//4,my-s//8)], outline=(200,240,255), fill=None)
        draw.polygon([(mx,my+s//3),(mx-s//4,my+s//8),(mx+s//4,my+s//8)], outline=(200,240,255), fill=None)
    elif overlay=='stair':
        for i in range(3):
            draw.line([x0+1+i*(s//5), y0+s-2-i*(s//5), x0+s-2, y0+s-2-i*(s//5)], fill=(240,240,240))
    elif overlay=='pillar':
        draw.ellipse([mx-s//4,my-s//4,mx+s//4,my+s//4], outline=(220,220,220))
    elif overlay=='shaft':
        draw.ellipse([mx-s//4,my-s//3,mx+s//4,my+s//3], outline=(200,200,220))
        draw.line([mx,my-s//4,mx,my+s//4], fill=(200,200,220))
    elif overlay=='challenge':
        draw.line([mx-s//3,my-s//3,mx+s//3,my+s//3], fill=(255,110,110))
        draw.line([mx+s//3,my-s//3,mx-s//3,my+s//3], fill=(255,110,110))
    elif overlay=='rotate':
        draw.arc([mx-s//3,my-s//3,mx+s//3,my+s//3], 30, 320, fill=(170,220,255))
        draw.polygon([(mx+s//4,my-s//4),(mx+s//3,my-s//2),(mx+1,my-s//5)], fill=(170,220,255))
    elif overlay=='shift':
        draw.line([mx-s//3,my,mx+s//3,my], fill=(255,220,120))
        draw.polygon([(mx-s//3,my),(mx-s//5,my-2),(mx-s//5,my+2)], fill=(255,220,120))
        draw.polygon([(mx+s//3,my),(mx+s//5,my-2),(mx+s//5,my+2)], fill=(255,220,120))
    elif overlay=='district':
        draw.rectangle([mx-s//4,my-s//4,mx+s//4,my+s//4], outline=(255,200,120))
    elif overlay=='deadend':
        draw.ellipse([mx-2,my-2,mx+2,my+2], outline=(255,120,120))
    elif overlay=='collapse':
        draw.line([mx-s//4,my-s//4,mx,my], fill=(255,180,100))
        draw.line([mx,my,mx+s//4,my+s//4], fill=(255,180,100))
        draw.line([mx-s//6,my+s//6,mx+s//6,my-s//6], fill=(255,180,100))


def render_map_image(tm:TileMap, spec:dict, max_map_px=360):
    tile_px = max(4, min(64, max_map_px // max(tm.w, tm.h)))
    if tm.w <= 9:
        tile_px = max(tile_px, 24 if tm.w==9 else 56)
    margin=8
    label_h=20
    W=tm.w*tile_px + margin*2
    H=tm.h*tile_px + margin*2 + label_h
    img=Image.new('RGB',(W,H),(18,20,24))
    draw=ImageDraw.Draw(img)
    seed=spec['id']*7919 + spec.get('seed',0)
    for y in range(tm.h):
        for x in range(tm.w):
            bx = margin + x*tile_px
            by = margin + y*tile_px
            draw_tile(draw,bx,by,tile_px,tm.base[y][x],seed,x,y)
    for y in range(tm.h):
        for x in range(tm.w):
            if tm.overlay[y][x]:
                bx=margin+x*tile_px
                by=margin+y*tile_px
                draw_overlay(draw,bx,by,tile_px,tm.overlay[y][x],seed,x,y)
    # Border and label
    draw.rectangle([margin-2,margin-2,margin+tm.w*tile_px+1,margin+tm.h*tile_px+1], outline=(80,84,92))
    label=f"{spec['id']:03d}"
    draw.rectangle([0,H-label_h,W,H], fill=(24,27,33))
    draw.text((8,H-label_h+4), label, font=DEFAULT_FONT, fill=(220,220,225))
    return img


GENERATORS = {
    'hub_spokes': gen_hub_spokes,
    'spiral': gen_spiral,
    'grid_rooms': gen_grid_rooms,
    'vertical_stack': gen_vertical_stack,
    'branch_caves': gen_branch_caves,
    'symmetrical_complex': gen_symmetrical_complex,
    'fortress': gen_fortress,
    'linear_gauntlet': gen_linear_gauntlet,
    'loops_network': gen_loops_network,
    'pyramid_terraces': gen_pyramid_terraces,
    'flooded': gen_flooded,
    'city': gen_city,
    'fragmented': gen_fragmented,
    'canyon': gen_canyon,
    'ring_core': gen_ring_core,
    'layered_overlap': gen_layered_overlap,
    'rotating_modular': gen_rotating_modular,
    'bridge_lattice': gen_bridge_lattice,
    'honeycomb': gen_honeycomb,
    'diagonal_platforms': gen_diagonal_platforms,
    'pillar_cavern': gen_pillar_cavern,
    'zigzag_stairs': gen_zigzag_stairs,
    'stacked_arenas': gen_stacked_arenas,
    'teleport_rooms': gen_teleport_rooms,
    'tree_deadends': gen_tree_deadends,
    'patrol_routes': gen_patrol_routes,
    'river_split': gen_river_split,
    'library': gen_library,
    'quadrants_core': gen_quadrants_core,
    'step_rooms': gen_step_rooms,
    'dense_cluster': gen_dense_cluster,
    'identical_corridors': gen_identical_corridors,
    'alt_interior_exterior': gen_alt_interior_exterior,
    'multi_entrance_converge': gen_multi_entrance_converge,
    'micro_matrix': gen_micro_matrix
}

all_specs = layout_specs + micro_specs

def make_layout(spec):
    seed = spec['id']*10007 + spec.get('seed', spec['id'])
    rng = random.Random(seed)
    tm = GENERATORS[spec['template']](spec, rng)
    return tm


def slugify(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text[:80]


def summarize_tiles(tm:TileMap):
    base_counts=Counter()
    overlay_counts=Counter()
    for row in tm.base:
        base_counts.update(row)
    for row in tm.overlay:
        overlay_counts.update([o for o in row if o])
    return dict(base_counts), dict(overlay_counts)


def export_map_json(tm:TileMap, spec:dict, path:Path):
    layers = tm.to_int_layers()
    base_counts, overlay_counts = summarize_tiles(tm)
    payload = {
        'id': spec['id'],
        'slug': slugify(spec['name']),
        'name': spec['name'],
        'template': spec['template'],
        'size': {'width': tm.w, 'height': tm.h},
        'theme': spec.get('theme',''),
        'background': spec.get('background','wall'),
        'base_counts': base_counts,
        'overlay_counts': overlay_counts,
        **layers,
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def build_contact_sheet(image_paths, labels, outpath, cols=5, cell_w=220, cell_h=240):
    rows=(len(image_paths)+cols-1)//cols
    sheet=Image.new('RGB',(cols*cell_w,rows*cell_h),(8,9,14))
    for idx,(img_path,label) in enumerate(zip(image_paths, labels)):
        im=Image.open(img_path).convert('RGB')
        thumb=ImageOps.contain(im,(cell_w-12,cell_h-24))
        canvas=Image.new('RGB',(cell_w,cell_h),(18,20,24))
        canvas.paste(thumb,((cell_w-thumb.width)//2,6))
        ImageDraw.Draw(canvas).text((8,cell_h-18),label,fill=(230,230,230),font=DEFAULT_FONT)
        sheet.paste(canvas,((idx%cols)*cell_w,(idx//cols)*cell_h))
    sheet.save(outpath)


def build_tileset_legend(outpath):
    tiles=['wall','floor','water','lava','abyss','bridge','courtyard','rubble','platform','bookshelf']
    overlays=['ladder','teleport','hidden','door','locked','elevator','stair','pillar','shaft','challenge','rotate','shift']
    cell=48
    cols=5
    rows=(len(tiles)+len(overlays)+cols-1)//cols
    img=Image.new('RGB',(cols*120, rows*100),(18,20,24))
    draw=ImageDraw.Draw(img)
    items=[('base',t) for t in tiles]+[('overlay',o) for o in overlays]
    for idx,(kind,name) in enumerate(items):
        x=(idx%cols)*120+10
        y=(idx//cols)*100+10
        draw.rectangle([x,y,x+cell,y+cell], fill=(30,32,36))
        if kind=='base':
            draw_tile(draw,x,y,cell,name,999,idx,0)
        else:
            draw_tile(draw,x,y,cell,'floor',999,idx,0)
            draw_overlay(draw,x,y,cell,name,999,idx,0)
        draw.text((x,y+cell+8), name, fill=(230,230,230), font=DEFAULT_FONT)
    img.save(outpath)


def build_viewer(catalog, outpath):
    # embed catalog for file:// compatibility
    embedded = json.dumps(catalog, ensure_ascii=False)
    html=f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Procedural Tile Generation System</title>
<style>
:root {{
  --bg:#0d1016; --panel:#161b24; --muted:#95a1b3; --text:#e6ebf4; --accent:#75b6ff;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:14px/1.4 system-ui, sans-serif; background:var(--bg); color:var(--text); }}
header {{ padding:20px 24px; border-bottom:1px solid #232a36; position:sticky; top:0; background:rgba(13,16,22,.95); backdrop-filter: blur(8px); z-index:10; }}
h1 {{ margin:0 0 8px; font-size:22px; }}
p {{ margin:0; color:var(--muted); }}
.controls {{ display:flex; gap:12px; margin-top:14px; flex-wrap:wrap; }}
input, select {{ background:#101521; color:var(--text); border:1px solid #273142; padding:8px 10px; border-radius:8px; }}
main {{ padding:20px 24px 32px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr)); gap:16px; }}
.card {{ background:var(--panel); border:1px solid #232a36; border-radius:14px; overflow:hidden; }}
.card img {{ width:100%; display:block; background:#0b0f16; }}
.card .meta {{ padding:12px; }}
.card .id {{ color:var(--accent); font-weight:700; letter-spacing:.04em; font-size:12px; }}
.card .name {{ margin-top:4px; font-weight:600; }}
.card .small {{ color:var(--muted); font-size:12px; margin-top:6px; }}
.card .links {{ display:flex; gap:10px; margin-top:8px; font-size:12px; }}
.card .links a {{ color:var(--accent); text-decoration:none; }}
footer {{ padding:20px 24px 32px; color:var(--muted); }}
.badge {{ display:inline-block; padding:2px 8px; border:1px solid #334155; border-radius:999px; font-size:11px; margin-right:6px; color:#bfd3ea; }}
</style>
</head>
<body>
<header>
  <h1>Procedural Tile Generation System</h1>
  <p>100 layouts generated from a data registry. Each card links to the render PNG and the raw tile matrix JSON.</p>
  <div class="controls">
    <input id="search" placeholder="Search layouts, templates, themes"/>
    <select id="kind">
      <option value="">All layouts</option>
      <option value="macro">Macro layouts</option>
      <option value="micro">Micro patterns</option>
    </select>
    <select id="template"><option value="">All templates</option></select>
  </div>
</header>
<main>
  <div id="grid" class="grid"></div>
</main>
<footer>Data-first export includes source specs, generated tile matrices, renders, contact sheet, and a procedural legend.</footer>
<script>
const catalog = {embedded};
const grid = document.getElementById('grid');
const search = document.getElementById('search');
const kind = document.getElementById('kind');
const template = document.getElementById('template');
const templates = [...new Set(catalog.map(x => x.template))].sort();
for (const t of templates) {{
  const opt = document.createElement('option');
  opt.value = t;
  opt.textContent = t;
  template.appendChild(opt);
}}
function render() {{
  const q = search.value.trim().toLowerCase();
  const kindVal = kind.value;
  const templateVal = template.value;
  const items = catalog.filter(item => {{
    if (kindVal && item.kind !== kindVal) return false;
    if (templateVal && item.template !== templateVal) return false;
    const hay = [item.id, item.name, item.template, item.theme, item.slug].join(' ').toLowerCase();
    if (q && !hay.includes(q)) return false;
    return true;
  }});
  grid.innerHTML = items.map(item => `
    <article class="card">
      <img src="${{item.render_png}}" alt="${{item.name}}">
      <div class="meta">
        <div class="id">${{String(item.id).padStart(3,'0')}}</div>
        <div class="name">${{item.name}}</div>
        <div class="small">
          <span class="badge">${{item.kind}}</span>
          <span class="badge">${{item.template}}</span>
          <span class="badge">${{item.theme || 'unthemed'}}</span>
        </div>
        <div class="small">${{item.width}}×${{item.height}} tiles</div>
        <div class="links">
          <a href="${{item.render_png}}">PNG</a>
          <a href="${{item.tile_json}}">Tile JSON</a>
        </div>
      </div>
    </article>
  `).join('');
}}
search.addEventListener('input', render);
kind.addEventListener('change', render);
template.addEventListener('change', render);
render();
</script>
</body>
</html>
"""
    outpath.write_text(html, encoding='utf-8')


def build_all(root):
    root=Path(root)
    if root.exists():
        # do not delete blindly; keep maybe
        pass
    renders=root/'renders'
    layouts_dir=root/'layouts'
    root.mkdir(parents=True, exist_ok=True)
    renders.mkdir(exist_ok=True)
    layouts_dir.mkdir(exist_ok=True)

    catalog=[]
    spec_export=[]
    image_paths=[]
    labels=[]
    for spec in all_specs:
        tm=make_layout(spec)
        slug=slugify(spec['name'])
        png_name=f"{spec['id']:03d}_{slug}.png"
        json_name=f"{spec['id']:03d}_{slug}.json"
        render_path=renders/png_name
        json_path=layouts_dir/json_name
        img=render_map_image(tm,spec)
        img.save(render_path)
        export_map_json(tm,spec,json_path)
        base_counts, overlay_counts = summarize_tiles(tm)
        rec={
            'id': spec['id'],
            'slug': slug,
            'name': spec['name'],
            'kind': 'macro' if spec['id']<=80 else 'micro',
            'template': spec['template'],
            'theme': spec.get('theme',''),
            'width': tm.w,
            'height': tm.h,
            'render_png': f"renders/{png_name}",
            'tile_json': f"layouts/{json_name}",
            'base_counts': base_counts,
            'overlay_counts': overlay_counts,
        }
        catalog.append(rec)
        spec_export.append(spec)
        image_paths.append(render_path)
        labels.append(f"{spec['id']:03d}")
    # data exports
    (root/'catalog.json').write_text(json.dumps(catalog, indent=2), encoding='utf-8')
    (root/'source_specs.json').write_text(json.dumps(spec_export, indent=2), encoding='utf-8')
    # media
    build_contact_sheet(image_paths, labels, root/'contact_sheet.png', cols=5)
    build_tileset_legend(root/'procedural_tileset_legend.png')
    build_viewer(catalog, root/'viewer.html')
    # README
    readme = f"""# Procedural Tile Generation System

This package contains a data-driven layout generator and batch render export for all 100 requested layouts.

## What is included

- `source_specs.json` — the registry of layout definitions
- `catalog.json` — generated catalog with file references and tile summaries
- `layouts/` — one raw tile matrix JSON per layout
- `renders/` — one rendered PNG per layout
- `contact_sheet.png` — overview of all 100 layouts
- `procedural_tileset_legend.png` — procedurally rendered tile/overlay legend
- `viewer.html` — local browser viewer

## Data-first structure

Each layout is defined by:
- `id`
- `name`
- `template`
- `size`
- theme/background params
- template-specific generator parameters

The generator registry turns that data into:
- base tile layer
- overlay layer
- render PNG
- exported JSON matrix

## Regenerate

Run:

```bash
python tilegen_system.py
```

Outputs will be written next to the script in `tilegen_output/`.

All textures, glyphs, and map renders are created procedurally in code. No external art assets are used.
"""
    (root/'README.md').write_text(readme, encoding='utf-8')
    return catalog



if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    output_root = script_dir / "tilegen_output"
    catalog = build_all(output_root)
    bundle_path = script_dir / "tilegen_output_bundle.zip"
    if bundle_path.exists():
        bundle_path.unlink()
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in output_root.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(script_dir))
    print(f"Built {len(catalog)} layouts into {output_root}")
    print(f"Bundle: {bundle_path}")
