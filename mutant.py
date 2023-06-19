import taichi as ti
import random, copy
import numpy as np


ti.init(ti.cpu)

@ti.kernel
def random_loc(loc:ti.template()):
    for i in loc:
        loc[i] = ti.random(int) % 2

@ti.kernel
def copy(x:ti.template(), y:ti.template()):
    for I in ti.grouped(y):
        x[I] = y[I]

@ti.data_oriented
class FitnessLandscape:
    def __init__(self, N):
        self.N = N
        self.ones = ti.field(float, (N, ))
        self.zeros = ti.field(float, (N, ))
        self.set_value()
    
    @ti.kernel
    def set_value(self):
        for i in range(self.N):
            self.ones[i] = ti.random(float)
            self.zeros[i] = ti.random(float)

    @ti.kernel
    def fitness(self, loc:ti.template()) -> float:
        tot = 0.0
        for i in range(self.N):
            if loc[i] == 1:
                tot += self.ones[i]
            else:
                tot += self.zeros[i]
        return tot/self.N
    

    def distance(self, loc1:ti.template(), loc2:ti.template()):
        cnt = 0
        for i in range(self.N):
            if loc1[i] != loc2[i]:
                cnt += 1
        return cnt
            
class Agent:
    def __init__(self, loc, fit_land):
        self.N = loc.shape[0]
        self.loc = loc
        self.fit_land = fit_land
        self.fitness = fit_land.fitness(loc)
    
    def copy(self):
        return Agent(self.loc, self.fit_land)
    
    
class Mutant(Agent):
    def copy(self, prob=0.9):
        if random.random() > prob:
            loc = ti.field(int, (self.N, ))
            copy(loc, self.loc)
        else:
            pos = random.randint(0, self.N-1)
            loc = self.mutate(pos)
        return Mutant(loc, self.fit_land)
    
    def mutate(self, pos):
        loc = ti.field(int, (self.N, ))
        copy(loc, self.loc)
        loc[pos] ^= 1
        return loc
        
        

loc = ti.field(int, (3, ))
random_loc(loc)
fit = FitnessLandscape(3)

a = Mutant(loc, fit)
print('a', a.loc, a.fitness)

b = a.copy()
print('b', b.loc, b.fitness)
            
print(fit.distance(a.loc, b.loc))
    