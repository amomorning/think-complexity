import taichi as ti
import networkx as nx
import matplotlib.pyplot as plt

ti.init(ti.gpu)
@ti.kernel
def generate_ring_lattice(n:int, k:int, out:ti.template()):
    out.fill(0)
    for i in range(n):
        for j in range(i+1, i+k//2+1):
            out[i, j%n] = 1

@ti.kernel
def rewrite(p:float, out:ti.template()):
    for i in range(n):
        for j in range(n):
            if out[i, j] == 1 and ti.random(float) < p:
                out[i, j] = 0
                k = ti.random(int) % n
                while out[i, k] == 1 or k == i:
                    k = ti.random(int) % n
                out[i, k] = 1

n = 15
k = 4

out = ti.field(int, (n, n))

fig = plt.figure(figsize=(13, 4))
for i, p in enumerate([0, 0.3, 1]):
    print(i+1, p)
    ax = fig.add_subplot(1, 3, i+1)
    generate_ring_lattice(n, k, out)
    rewrite(p, out)
    G = nx.from_numpy_array(out.to_numpy())
    nx.draw_circular(G, node_size=40, node_color='r')

plt.show()
    
