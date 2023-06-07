import taichi as ti
import numpy as np
import time
import networkx as nx
import matplotlib.pyplot as plt

n = 100
p = 0.06


ti.init(ti.gpu)
@ti.kernel
def calc(n:int, p:float, out:ti.template()):
    for i in range(n):
        for j in range(i):
            if ti.random() < p and i != j:
                out[i, j] = 1
    
ax1 = plt.subplot(121)

start_time = time.time()
G1 = nx.erdos_renyi_graph(n, p)
nx.draw(G1, node_size=20, node_color='r')
used_time = time.time() - start_time
print(f'networkx: er graph with {G1.number_of_edges()} edges in {used_time}s')
ax1.title.set_text(used_time)

ax2 = plt.subplot(122)
start_time = time.time()
out = ti.field(int, shape=(n, n))
calc(n, p, out)
used_time = time.time() - start_time
G2 = nx.from_numpy_array(out.to_numpy())
nx.draw(G2, node_size=20, node_color='r')
print(f'taichi: er graph with {G2.number_of_edges()} edges in {used_time}s')
ax2.title.set_text(used_time)
plt.show()
