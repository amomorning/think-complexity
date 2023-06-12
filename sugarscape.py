import taichi as ti
import numpy as np
import random

ti.init(ti.cpu, cpu_max_num_threads=1)

cell_size = 8
res = 100
r = ti.Vector([60, 40, 25, 15])
r1, r2 = 30, 70
pixels = ti.Vector.field(3, float, shape = (res*cell_size, res*cell_size))
cell = ti.field(int, shape=(res, res))
vis = ti.field(int, shape=(res, res))
gui = ti.GUI('Sugarscape', res*cell_size)

N = 400

pos = ti.field(float, shape=(N, 2))
dead = ti.field(int, shape=(N,))
agent = ti.Struct.field({
    "x":int,
    "y":int,
    "sugar": int,
    "metabolism": int,
    "vision": int
}, shape=(N,))
d4 = ti.Matrix([[-1, 0], [0, -1], [1, 0], [0, 1]])


@ti.kernel
def init():
    cell.fill(0)
    for i, j in cell:
        a = (i-r1)*(i-r1) + (j-r1)*(j-r1) 
        b = (i-r2)*(i-r2) + (j-r2)*(j-r2) 
        for k in ti.static(range(4)):
            if min(a, b) < r[k] * r[k]:
                cell[i, j] = k
    
    vis.fill(0)
    dead.fill(0)
    for i in range(N):
        x, y = ti.random(int) % res, ti.random(int) % res
        while vis[x, y] == 1:
            x, y = ti.random(int) % res, ti.random(int) % res
        vis[x, y] = 1
        agent[i].x = x #[0, 50)
        agent[i].y = y #[0, 50)
        agent[i].sugar = ti.random(int) % 21 + 5 #[5, 25]
        agent[i].metabolism = ti.random(int) % 4 + 1 #[1, 4]
        agent[i].vision = ti.random(int) % 6 + 1 #[1, 6]
        
@ti.kernel
def step():
    for i in range(N):
        if agent[i].sugar < 0: continue

        gate = ti.random(int)%2
        v = agent[i].vision
        mx = agent[i].x + gate * (ti.random(int)%(2*v)-v)
        my = agent[i].y + (1-gate) * (ti.random(int)%(2*v)-v)

        for j in range(v):
            for d in ti.static(range(4)):
                x, y = agent[i].x + j * d4[d, 0], agent[i].y + j * d4[d, 1]
                if cell[x, y] > cell[mx, my] and vis[x, y] == 0:
                    mx, my = x, y

        vis[agent[i].x, agent[i].y] = 0
        vis[mx, my] = 1
        agent[i].x = mx
        agent[i].y = my
        
        agent[i].sugar += cell[mx, my]
        cell[mx, my] = 0
        agent[i].sugar -= agent[i].metabolism
            
        
    
@ti.kernel
def draw():
    for i, j in pixels:
        i_, j_ = i // cell_size, j // cell_size
        pixels[i, j] = ti.Vector([1.0, 1.0, 1.0]) - ti.Vector([0.27, 0.76, 0.63]) * (cell[i_, j_]) / 5

    for i in range(N):
        pos[i, 0] = (agent[i].x+0.46)/res
        pos[i, 1] = (agent[i].y+0.46)/res

        dead[i] = ti.cast(agent[i].sugar < 0, int)
        
init()
draw()
while gui.running:
    for e in gui.get_events(gui.PRESS, gui.MOTION):
        if e.key == 'r':
            init()
            draw()
        elif e.key == 'n':
            step()
            draw()

    gui.set_image(pixels)
    gui.circles(pos.to_numpy(), (cell_size-1)/2, palette=[0x63B79D, 0xEAF1F2], palette_indices=dead)
    gui.show()
