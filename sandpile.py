from scipy.signal import correlate2d
import numpy as np
import taichi as ti
ti.init(ti.gpu)

# kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.int32)
kernel = np.array([[0, 1, 0], [2, -4, 1], [0, 0, 0]], dtype=np.int32)
N, M, K = 120, 120, 6
MX, R = 30, 20
sandpile_size = 6
image_size = (N * sandpile_size, M * sandpile_size)
pixels = ti.Vector.field(3, dtype=float, shape=image_size)
sandpile = ti.field(float, (N, M))
gui = ti.GUI('Sand Pile', image_size)

@ti.kernel
def init(out: ti.template(), MX: int):
    out.fill(0)
    # for i, j in out:
    #     out[i, j] = ti.random(int) % MX
    for i, j in out:
        tmp = ti.sqrt((i-N//5)**2 + (j-M//5)**2)
        if tmp < R:
            out[i, j] = MX-tmp


@ti.kernel
def correlate(out: ti.template(), 
              in1:ti.template(), 
              in2:ti.template(), fill:int):
    out.fill(0)
    us, vs = in2.shape
    for i, j in out:
        for u in range(us):
            for v in range(vs):
                p, q = i + u - us//2, j + v - vs//2
                if p < 0 or q < 0 or p >= in1.shape[0] or q >= in1.shape[1]:
                    out[i, j] += fill * in2[u, v]
                else:
                    out[i, j] += in1[p, q] * in2[u, v]

@ti.kernel
def init_toppling(out: ti.template(), arr: ti.template(), K:int) -> int:
    out.fill(0)
    sum = 0
    for i, j in out:
        if arr[i, j] > K:
            out[i, j] = 1
            sum += 1
    return sum
            
            
@ti.kernel
def add(a: ti.template(), b: ti.template()):
    for i, j in a:
        a[i, j] += b[i, j]

    for i, j in pixels:
        i_ = i // sandpile_size
        j_ = j // sandpile_size
        pixels[i, j] = ti.Vector([0.75, 0.78, 0.81]) * a[i_, j_] / K
    

def run():
    tot = 0
    toppling = ti.field(int, sandpile.shape)
    kernel0 = ti.field(int, kernel.shape)
    kernel0.from_numpy(kernel)
    
    out = ti.field(float, sandpile.shape)

    pixels.fill(0)
    init(sandpile, MX)

    while gui.running:

        
        num = init_toppling(toppling, sandpile, K)
        if num > 0:
            correlate(out, toppling, kernel0, 0)
            add(sandpile, out)
            tot += num

        gui.set_image(pixels)
        gui.show()
        
run()
