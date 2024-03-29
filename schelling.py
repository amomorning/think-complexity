import taichi as ti
ti.init(ti.cpu, cpu_max_num_threads=1)

cell_size = 2
res = 400
pixels = ti.Vector.field(3, float, shape=(res*cell_size, res*cell_size))
cell = ti.field(int, shape=(res, res))
gui = ti.GUI('Schelling', res*cell_size)


prob = [0, 0.1, 0.4, 0.7]
color = [ti.Vector([0.95, 0.95, 0.93]), ti.Vector([0.61, 0.69, 0.79]), ti.Vector([0.91, 0.84, 0.39]), ti.Vector([0.91, 0.66, 0.38])]

num = int(res*res*prob[1])
indices = ti.Vector.field(2, int, shape=(num, ))

@ti.kernel
def init() -> int:
    cnt = 0
    for i, j in cell:
        for k in ti.static(range(4)):
            if ti.random() > prob[k]:
                cell[i, j] = k
        if cell[i, j] == 0:
            indices[cnt] = ti.Vector([i, j])
            cnt += 1
    return cnt

        
@ti.kernel
def run(p: float):
    for i, j in cell:
        if cell[i, j] == 0: continue
        same, tot = 0, 0
        for u in range(-1, 2):        
            for v in range(-1, 2):
                i_, j_ = i+u, j+v
                if i_ >= 0 and j_ >= 0 and i_ < res and j_ < res:
                    if cell[i_, j_] != 0:
                        tot += 1
                    if cell[i_, j_] == cell[i, j]:
                        same += 1
        if ti.cast(same-1, float)/(tot-1) < p:
            idx = ti.random(int) % num
            i_, j_ = indices[idx]
            indices[idx] = ti.Vector([i, j])
            cell[i, j], cell[i_, j_] = cell[i_, j_], cell[i, j]


    for i, j in pixels:
        i_, j_ = i // cell_size, j // cell_size
        if cell[i_, j_] == 0:
            pixels[i, j] = color[0]
        if cell[i_, j_] == 1:
            pixels[i, j] = color[1]
        if cell[i_, j_] == 2:
            pixels[i, j] = color[2]
        if cell[i_, j_] == 3:
            pixels[i, j] = color[3]

result_dir = "imgs/schelling-results"
# video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=True)

# gui.fps_limit = 1
pause = False
t = 0
num = init()
while gui.running:
    for e in gui.get_events(gui.PRESS, gui.MOTION):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.SPACE:
            pause = not pause
        elif e.key == 'r':
            num = init()

    if not pause:
        run(0.56)

    # video_manager.write_frame(pixels)
    gui.set_image(pixels)
    gui.show()
    t += 1
# video_manager.make_video(gif=True, mp4=True)


