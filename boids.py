import taichi as ti
import numpy as np
import math

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
arch = ti.cpu
ti.init(arch=arch)

N = 10
vs = [ti.Vector.field(3, dtype=ti.f32, shape=6) for _ in range(N)]

arr = [0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5]
fs = ti.field(dtype=ti.i32, shape=4*3)
for i, x in enumerate(arr):
    fs[i] = x

resolution = (768, 768)
show_window = True

def random_vector(a, b):
    coords = np.random.uniform(a, b, size=3)
    return ti.math.vec3(coords)

def project(a, b):
    return a - a.dot(b) / a.dot(a)
@ti.func
def project_func(a, b):
    return a - a.dot(b) / a.dot(a)

@ti.data_oriented
class Boid:
    def __init__(self, length=0.1, scale=0.3):
        self.u = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self.w = ti.Vector.field(3, dtype=ti.f32, shape=1)
        
        self.u[0] = random_vector(-1, 1).normalized()
        self.v[0] = random_vector(-1, 1)
        self.v[0] = (self.v[0] - project(self.v[0], self.u[0])).normalized()
        self.w[0] = self.u[0].cross(self.v[0])

        self.vs = ti.Vector.field(3, dtype=ti.f32, shape=6)
        self.vs[0] = random_vector(0, 1)
        self.vs[1] = self.vs[0] + length * self.u[0]
        self.vs[2] = self.vs[0] + scale * length * self.v[0]
        self.vs[3] = self.vs[0] + scale * length * self.w[0]
        self.vs[4] = self.vs[0] - scale * length * self.v[0]
        self.vs[5] = self.vs[0] - scale * length * self.w[0]

        self.goal =ti.Vector.field(3, dtype=ti.f32, shape=1)
        self.scale = scale
        self.length = length
    
    def get_neighbors(self, boids, radius, angle):
        neighbors = []
        for boid in boids:
            if boid is self:
                continue
        
            if (boid.vs[0]-self.vs[0]).norm() > radius:
                continue

            if isinstance(boid, Boid):
                cos = self.vs[0].dot(boid.vs[0]) / self.vs[0].norm() / boid.vs[0].norm()
                if ti.abs(ti.acos(cos)) > angle:
                    continue
            
            neighbors.append(boid)
        return neighbors
    
        
    def center(self, boids, radius=2, angle=1):
        neighbors = self.get_neighbors(boids, radius, angle)

        if len(neighbors) == 0:
            return ti.Vector([0, 0, 0], dt=ti.f32)
        center = ti.Vector([0, 0, 0], dt=ti.f32)
        for boid in neighbors:
            center += boid.vs[0]
        center /= len(neighbors)
        return center - self.vs[0]
    
    def avoid(self, boids, carrot, radius=0.5, angle=math.pi):
        objects = boids + [carrot]
        neighbors = self.get_neighbors(objects, radius, angle)
        center = ti.Vector([0, 0, 0], dt=ti.f32)
        if len(neighbors) == 0:
            return center
        for boid in neighbors:
            center += boid.vs[0]
        center /= len(neighbors)
        return self.vs[0] - center
    
    def align(self, boids, radius=0.8, angle=1):
        neighbors = self.get_neighbors(boids, radius, angle)
        center = ti.Vector([0, 0, 0], dt=ti.f32)
        if len(neighbors) == 0:
            return center
        for boid in neighbors:
            center += boid.u[0]
        center /= len(neighbors)
        return self.u[0] - center
    
    def love(self, carrot):
        return (carrot.vs[0] - self.vs[0]).normalized()


    def set_goal(self, boids, carrot):
        # self.goal = 
        w = [10, 3, 1, 10]
        #
        w = [1, 5, 10, 5]
        self.goal[0] = w[0] * self.center(boids) + w[1] * self.avoid(boids, carrot)\
                + w[2] * self.align(boids) + w[3] * self.love(carrot)
        self.goal[0] = self.goal[0].normalized()

    
    @ti.kernel
    def move(self, mu:ti.f32, dt:ti.f32):
        self.u[0] = (1.0-mu) * self.u[0] + mu*self.goal[0]
        self.u[0] = self.u[0].normalized()
        self.v[0] = (self.v[0] - project_func(self.v[0], self.u[0])).normalized()
        self.w[0] = (self.u[0].cross(self.v[0])).normalized()

        self.vs[0] += self.u[0] * dt
        self.vs[1] = self.vs[0] + self.length * self.u[0]
        self.vs[2] = self.vs[0] + self.scale * self.length * self.v[0]
        self.vs[3] = self.vs[0] + self.scale * self.length * self.w[0]
        self.vs[4] = self.vs[0] - self.scale * self.length * self.v[0]
        self.vs[5] = self.vs[0] - self.scale * self.length * self.w[0]
    
class Carrot:
    def __init__(self, pos):
        self.vs = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self.vs[0] = ti.Vector(pos)

class World:
    def __init__(self, axes_length = 1):
        self.axes = [ti.Vector.field(3, dtype=ti.f32, shape=2) for _ in range(3)]
        self.color = [None for _ in range(3)]
        for i in range(3):
            self.axes[i][1] = [axes_length if j == i else 0 for j in range(3)]
            self.color[i] = tuple([1 if j==i else 0 for j in range(3)])

        self.boids = [Boid() for _ in range(N)]
        self.carrot = Carrot([0, 0, 0])
    
    def draw_axes(self, scene):
        for i in range(3):
            scene.lines(self.axes[i], color=self.color[i], width = 5.0)

    def step(self, scene):
        for boid in self.boids:
            boid.set_goal(self.boids, self.carrot)
            boid.move(0.1, 0.05)
            scene.mesh(boid.vs, fs, color = (0.28, 0.68, 0.99))
        scene.particles(self.carrot.vs, radius = 0.1, color = (0.95, 0.43, 0.22))
 
result_dir = "imgs/boids-results"
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=True)


if __name__ == '__main__':

    window = ti.ui.Window('Boid', res = resolution, vsync = True, show_window = show_window)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(1, 2, 3)
    camera.lookat(0, 0, 0)
    world = World()


    while window.running:
        camera.track_user_inputs(window, movement_speed=0.3, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        world.draw_axes(scene)
        world.step(scene)

        canvas.scene(scene)
        if show_window:
            video_manager.write_frame(window.get_image_buffer_as_numpy())
            window.show()

